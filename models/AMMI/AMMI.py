"""
AMMI: Adversarial Maximal Mutual Information
基于论文: Learning Discrete Structured Representations by Adversarially Maximizing Mutual Information (ICML 2020)

核心思想:
- 最大化互信息 I(Z;Y) = H(Z) - H(Z|Y)
- 使用对抗训练来估计 H(Z)
- 使用 Straight-Through Estimator 处理离散采样
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from textdata import SingleLabelTextDataset, MultiLabelTextDataset
from utils.utils import set_seed, retrieve_topk, compute_precision_at_k

set_seed()

# 获取当前文件所在目录
CURRENT_DIR = Path(__file__).parent
LOG_DIR = CURRENT_DIR / 'logs'
CHECKPOINT_DIR = CURRENT_DIR / 'checkpoints'

# 创建目录
LOG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ng20.tfidf')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_prior', type=float, default=0.01, help='先验网络学习率')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--bit', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--stop_iter', type=int, default=50, help='控制在效果没有提升多少次后停止运行')
    parser.add_argument('--entropy_weight', type=float, default=2.0, help='熵权重')
    parser.add_argument('--num_prior_steps', type=int, default=4, help='每步更新先验的次数')
    parser.add_argument('--dim_hidden', type=int, default=500, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='编码器层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout概率')
    parser.add_argument('--device', type=str, default='cuda', help='运行设备')
    parser.add_argument('--gpu', type=str, default='0', help='GPU设备号')
    return parser


def get_device(config):
    """获取运行设备"""
    if config['device'] == 'cuda' and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
        device = torch.device('cuda')
        print(f"使用 GPU: {config['gpu']}")
    else:
        device = torch.device('cpu')
        print("使用 CPU")
    return device


class FF(nn.Module):
    """前馈网络"""

    def __init__(self, dim_input, dim_hidden, dim_output, num_layers, dropout=0.0):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_dim = dim_input if i == 0 else dim_hidden
            layers.append(nn.Linear(in_dim, dim_hidden))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dim_hidden if num_layers > 0 else dim_input, dim_output))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Posterior(nn.Module):
    """
    后验网络 p(Z|Y)
    输出每个二值变量的条件概率
    """

    def __init__(self, vocab_size, dim_hidden, num_features, num_layers, dropout=0.0):
        super().__init__()
        self.num_features = num_features
        self.ff = FF(vocab_size, dim_hidden, num_features, num_layers, dropout)

    def forward(self, Y):
        # 输出 logits
        logits = self.ff(Y)  # B x m
        return logits


class Prior(nn.Module):
    """
    先验网络 q(Z)
    用于对抗训练估计 H(Z)
    """

    def __init__(self, num_features, dim_hidden, num_layers=1):
        super().__init__()
        self.num_features = num_features

        # 可学习的先验参数
        self.theta = nn.Parameter(torch.zeros(num_features))

        if num_layers > 0:
            # 使用MLP来学习更复杂的先验
            self.use_mlp = True
            self.embed = nn.Embedding(num_features, dim_hidden)
            self.ff = FF(dim_hidden, dim_hidden, 1, num_layers)
        else:
            self.use_mlp = False

    def forward(self):
        if self.use_mlp:
            # 每个位置有独立的先验
            logits = self.ff(self.embed.weight).squeeze(-1)  # m
        else:
            logits = self.theta
        return logits


class Decoder(nn.Module):
    """解码器 - 用于重建"""

    def __init__(self, num_features, vocab_size):
        super().__init__()
        self.E = nn.Embedding(num_features, vocab_size)
        self.b = nn.Parameter(torch.zeros(1, vocab_size))

    def forward(self, Z, targets):
        # Z: B x m (二值)
        # targets: B x V (词袋)
        scores = Z @ self.E.weight + self.b  # B x V
        log_probs = F.log_softmax(scores, dim=1)
        # 只对出现的词计算损失
        log_likelihood = (log_probs * (targets > 0).float()).sum(1).mean()
        return log_likelihood


class BernoulliStraightThrough(torch.autograd.Function):
    """伯努利采样 + Straight-Through Estimator"""

    @staticmethod
    def forward(ctx, probs, training=True):
        if training:
            samples = torch.bernoulli(probs)
        else:
            samples = (probs > 0.5).float()
        return samples

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class AMMI(nn.Module):
    """
    AMMI 模型
    通过对抗最大化互信息学习离散表示

    核心思想:
    - 最大化 I(Z;Y) = H(Z) - H(Z|Y)
    - H(Z|Y): 条件熵，由后验网络估计
    - H(Z): 边际熵，由对抗训练的先验网络估计

    损失函数:
    Loss = H(Z|Y) - entropy_weight * H(Z)
    """

    def __init__(self, dataset, vocab_size, num_features, device,
                 dim_hidden=500, num_layers=2, dropout=0.1):
        super(AMMI, self).__init__()

        self.dataset = dataset
        self.vocab_size = vocab_size
        self.num_features = num_features
        self.device = device

        # 后验网络 p(Z|Y)
        self.posterior = Posterior(vocab_size, dim_hidden, num_features, num_layers, dropout)

        # 先验网络 q(Z) - 用于对抗训练
        self.prior = Prior(num_features, dim_hidden // 2, num_layers=1)

        # 解码器 (可选，用于辅助训练)
        self.decoder = Decoder(num_features, vocab_size)

    def compute_conditional_entropy(self, P, Q_logits):
        """
        计算条件熵 H(Z|Y)
        使用变分下界估计

        P: 后验概率 [0,1]^{B x m}
        Q_logits: 先验logits R^{m}
        """
        # 扩展先验到batch维度
        Q = torch.sigmoid(Q_logits).unsqueeze(0).expand(P.size(0), -1)  # B x m

        # H(Z|Y) ≈ -E_p[log q(z)]
        # = -sum_i E[z_i log q_i + (1-z_i) log(1-q_i)]
        # = -sum_i [p_i log q_i + (1-p_i) log(1-q_i)]
        eps = 1e-10
        hZ_Y = -(P * torch.log(Q + eps) + (1 - P) * torch.log(1 - Q + eps))
        hZ_Y = hZ_Y.sum(dim=1).mean()

        return hZ_Y

    def compute_marginal_entropy(self, P):
        """
        计算边际熵 H(Z)
        使用平均概率估计

        P: 后验概率 [0,1]^{B x m}
        """
        # 估计边际分布 p(z) = E_Y[p(z|Y)]
        p_marginal = P.mean(dim=0)  # m

        # H(Z) = -sum_i [p_i log p_i + (1-p_i) log(1-p_i)]
        eps = 1e-10
        hZ = -(p_marginal * torch.log(p_marginal + eps) +
               (1 - p_marginal) * torch.log(1 - p_marginal + eps))
        hZ = hZ.sum()

        return hZ

    def sample_hashcode(self, probs, training=True):
        """采样二值哈希码"""
        return BernoulliStraightThrough.apply(probs, training)

    def forward(self, Y, update_prior=True, num_prior_steps=4, lr_prior=0.01):
        """
        前向传播

        Args:
            Y: 输入文档 B x V
            update_prior: 是否更新先验网络
            num_prior_steps: 先验更新步数
            lr_prior: 先验学习率
        """
        # 计算后验概率
        posterior_logits = self.posterior(Y)  # B x m
        P = torch.sigmoid(posterior_logits)   # B x m

        # 采样哈希码
        Z = self.sample_hashcode(P, self.training)

        # 对抗更新先验网络
        if update_prior and self.training:
            prior_optimizer = torch.optim.Adam(self.prior.parameters(), lr=lr_prior)
            for _ in range(num_prior_steps):
                prior_optimizer.zero_grad()
                Q_logits = self.prior()
                # 先验想要最小化条件熵（让预测更准确）
                hZ_Y_prior = self.compute_conditional_entropy(P.detach(), Q_logits)
                hZ_Y_prior.backward()
                nn.utils.clip_grad_norm_(self.prior.parameters(), 10.0)
                prior_optimizer.step()

        # 计算条件熵
        Q_logits = self.prior()
        hZ_Y = self.compute_conditional_entropy(P, Q_logits)

        # 计算边际熵
        hZ = self.compute_marginal_entropy(P)

        # 重建损失（辅助目标）
        recon_ll = self.decoder(Z, Y)

        return {
            'hZ_Y': hZ_Y,
            'hZ': hZ,
            'recon_ll': recon_ll,
            'Z': Z,
            'P': P
        }

    def get_name(self):
        return "AMMI"

    def encode_discrete(self, Y):
        """生成离散哈希码"""
        posterior_logits = self.posterior(Y)
        P = torch.sigmoid(posterior_logits)
        Z = (P > 0.5).float()
        return Z

    def get_binary_code(self, train_loader, test_loader):
        """生成二进制哈希码"""
        self.eval()

        train_codes = []
        train_labels = []
        for xb, yb in train_loader:
            xb = xb.to(self.device)
            Z = self.encode_discrete(xb)
            train_codes.append(Z.cpu())
            train_labels.append(yb)

        train_codes = torch.cat(train_codes, dim=0)
        train_labels = torch.cat(train_labels, dim=0)

        test_codes = []
        test_labels = []
        for xb, yb in test_loader:
            xb = xb.to(self.device)
            Z = self.encode_discrete(xb)
            test_codes.append(Z.cpu())
            test_labels.append(yb)

        test_codes = torch.cat(test_codes, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

        train_b = train_codes.type(torch.ByteTensor).to(self.device)
        test_b = test_codes.type(torch.ByteTensor).to(self.device)

        return train_b, test_b, train_labels, test_labels


def train_val(config):
    device = get_device(config)

    bit = config["bit"]
    dataset, data_fmt = config["dataset"].split('.')
    batch_size = config["batch_size"]
    entropy_weight = config["entropy_weight"]
    num_prior_steps = config["num_prior_steps"]
    lr_prior = config["lr_prior"]

    if dataset in ['reuters', 'tmc', 'rcv1']:
        single_label_flag = False
    else:
        single_label_flag = True

    # 使用数据集模块
    data_path = Path(__file__).parent.parent.parent / 'textdata'

    if single_label_flag:
        train_set = SingleLabelTextDataset(f'{data_path}/{dataset}', subset='train', bow_format=data_fmt)
        test_set = SingleLabelTextDataset(f'{data_path}/{dataset}', subset='test', bow_format=data_fmt)
        val_set = SingleLabelTextDataset(f'{data_path}/{dataset}', subset='cv', bow_format=data_fmt)
    else:
        train_set = MultiLabelTextDataset(f'{data_path}/{dataset}', subset='train', bow_format=data_fmt)
        test_set = MultiLabelTextDataset(f'{data_path}/{dataset}', subset='test', bow_format=data_fmt)
        val_set = MultiLabelTextDataset(f'{data_path}/{dataset}', subset='cv', bow_format=data_fmt)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

    num_features = train_set[0][0].size(0)
    model = AMMI(
        dataset, num_features, bit, device,
        dim_hidden=config["dim_hidden"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    )
    model.to(device)

    # 只优化后验网络和解码器
    posterior_params = list(model.posterior.parameters()) + list(model.decoder.parameters())
    optimizer = optim.Adam(posterior_params, lr=config["lr"])

    best_precision = 0
    prec = 0
    best_precision_epoch = 0
    step_count = 0

    # 日志保存
    log_file = LOG_DIR / f'data:{config["dataset"]}_bit:{config["bit"]}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=str(log_file),
        filemode='w'
    )

    # 模型保存路径
    checkpoint_path = CHECKPOINT_DIR / f'dataset:{dataset}_bit:{bit}.pth'

    for epoch in tqdm(range(config["epoch"])):
        model.train()
        avg_loss = []
        avg_hZ_Y = []
        avg_hZ = []
        avg_recon = []

        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)

            # 前向传播
            output = model(xb, update_prior=True, num_prior_steps=num_prior_steps, lr_prior=lr_prior)

            hZ_Y = output['hZ_Y']
            hZ = output['hZ']
            recon_ll = output['recon_ll']

            # 损失: 最小化条件熵 - 最大化边际熵 - 最大化重建似然
            # = 最小化 H(Z|Y) - entropy_weight * H(Z) - recon_ll
            loss = hZ_Y - entropy_weight * hZ - recon_ll

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(posterior_params, 10.0)
            optimizer.step()

            avg_loss.append(loss.item())
            avg_hZ_Y.append(hZ_Y.item())
            avg_hZ.append(hZ.item())
            avg_recon.append(recon_ll.item())

        # 验证
        model.eval()
        with torch.no_grad():
            train_b, val_b, train_y, val_y = model.get_binary_code(train_loader, val_loader)
            retrieved_indices = retrieve_topk(val_b.to(device), train_b.to(device), topK=100)
            prec = compute_precision_at_k(retrieved_indices, val_y.to(device), train_y.to(device),
                                          topK=100, is_single_label=single_label_flag)
            if prec.item() > best_precision:
                best_precision = prec.item()
                best_precision_epoch = epoch + 1
                step_count = 0
                torch.save(model, str(checkpoint_path))
            else:
                step_count += 1
            if step_count >= config["stop_iter"]:
                break

        tqdm.write(
            f'Epoch {epoch+1}/{config["epoch"]} - Loss: {np.mean(avg_loss):.4f} - '
            f'H(Z|Y): {np.mean(avg_hZ_Y):.4f} - H(Z): {np.mean(avg_hZ):.4f} - '
            f'Prec: {prec.item():.4f} - Best: {best_precision:.4f} [{best_precision_epoch}]')
        logging.info(
            f'Epoch {epoch+1}/{config["epoch"]} - Loss: {np.mean(avg_loss):.4f} - '
            f'H(Z|Y): {np.mean(avg_hZ_Y):.4f} - H(Z): {np.mean(avg_hZ):.4f} - '
            f'Prec: {prec:.4f} - Best: {best_precision:.4f} [{best_precision_epoch}]')

    # 加载最佳模型进行测试
    model = torch.load(str(checkpoint_path))
    model.eval()
    with torch.no_grad():
        train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
        retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
        prec = compute_precision_at_k(
            retrieved_indices,
            test_y.to(device),
            train_y.to(device),
            topK=100,
            is_single_label=single_label_flag)
        print(f'Test Precision: {prec:.4f}')
        logging.info(f'Test Precision: {prec:.4f}')


if __name__ == "__main__":
    argparser = get_argparser()
    args = argparser.parse_args()
    config = vars(args)
    train_val(config)
