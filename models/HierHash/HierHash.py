"""
HierHash: Document Hashing with Multi-Grained Prototype-Induced Hierarchical Generative Model
基于论文: Document Hashing with Multi-Grained Prototype-Induced Hierarchical Generative Model (EMNLP 2024)

核心思想:
1. 引入层次原型: 粗粒度原型捕捉高层语义，细粒度原型捕捉低层语义
2. 构建层次先验分布: 利用原型构建层次先验，替代标准高斯先验
3. 层次对比学习: 利用原型分配作为伪标签进行对比学习
"""

import argparse
import logging
import os
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--bit', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--stop_iter', type=int, default=10, help='控制在效果没有提升多少次后停止运行')

    # HierHash 特有参数
    parser.add_argument('--hidden_dim', type=int, default=500, help='隐藏层维度')
    parser.add_argument('--num_coarse', type=int, default=10, help='粗粒度原型数量')
    parser.add_argument('--num_fine', type=int, default=50, help='细粒度原型数量')
    parser.add_argument('--temperature', type=float, default=0.1, help='对比学习温度')
    parser.add_argument('--lambda_contrast', type=float, default=0.1, help='对比损失权重')
    parser.add_argument('--lambda_proto', type=float, default=0.1, help='原型损失权重')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout概率')

    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备: cuda 或 cpu')
    parser.add_argument('--gpu', type=str, default='0',
                        help='指定 GPU 设备号 (仅当 device=cuda 时有效)')
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


class HierHash(nn.Module):
    """
    HierHash 模型

    架构:
    - 编码器: 文档 -> (mu, logvar)
    - 层次原型: 粗粒度和细粒度原型
    - 解码器: 潜在编码 -> 文档重建
    """

    def __init__(self, vocab_size, hidden_dim, latent_dim,
                 num_coarse, num_fine, temperature, dropout, device):
        super(HierHash, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.temperature = temperature
        self.device = device

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid()
        )

        # 层次原型
        self.coarse_prototypes = nn.Parameter(torch.randn(num_coarse, latent_dim))
        self.fine_prototypes = nn.Parameter(torch.randn(num_fine, latent_dim))
        nn.init.xavier_uniform_(self.coarse_prototypes)
        nn.init.xavier_uniform_(self.fine_prototypes)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, vocab_size),
            nn.LogSoftmax(dim=1)
        )

    def encode(self, x):
        """编码器"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化"""
        std = torch.sqrt(torch.exp(logvar))
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_prototype_assignment(self, z):
        """计算文档到原型的软分配"""
        # 粗粒度分配
        coarse_sim = F.cosine_similarity(
            z.unsqueeze(1), self.coarse_prototypes.unsqueeze(0), dim=2
        )
        coarse_assign = F.softmax(coarse_sim / self.temperature, dim=1)

        # 细粒度分配
        fine_sim = F.cosine_similarity(
            z.unsqueeze(1), self.fine_prototypes.unsqueeze(0), dim=2
        )
        fine_assign = F.softmax(fine_sim / self.temperature, dim=1)

        return coarse_assign, fine_assign

    def get_hierarchical_prior(self, z, coarse_assign, fine_assign):
        """计算层次先验均值"""
        coarse_prior = torch.matmul(coarse_assign, self.coarse_prototypes)
        fine_prior = torch.matmul(fine_assign, self.fine_prototypes)
        return 0.5 * coarse_prior + 0.5 * fine_prior

    def forward(self, x):
        """前向传播"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        coarse_assign, fine_assign = self.get_prototype_assignment(mu)
        prior_mu = self.get_hierarchical_prior(mu, coarse_assign, fine_assign)
        recon = self.decoder(z)
        return recon, mu, logvar, prior_mu, coarse_assign

    def compute_loss(self, x, recon, mu, logvar, prior_mu, coarse_assign,
                     kl_weight, lambda_contrast, lambda_proto):
        """计算总损失"""
        # 1. 重建损失
        recon_loss = -torch.mean(torch.sum(recon * x, dim=1))

        # 2. KL散度 (与层次先验)
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + logvar - (mu - prior_mu).pow(2) - logvar.exp(), dim=1)
        )

        # 3. 对比损失 (使用粗粒度分配作为伪标签)
        contrast_loss = self.compute_contrastive_loss(mu, coarse_assign)

        # 4. 原型多样性损失
        proto_loss = self.compute_prototype_diversity()

        total_loss = recon_loss + kl_weight * kl_loss + \
            lambda_contrast * contrast_loss + lambda_proto * proto_loss

        return total_loss, recon_loss, kl_loss, contrast_loss, proto_loss

    def compute_contrastive_loss(self, z, coarse_assign):
        """对比损失: 同类样本拉近，不同类样本推远"""
        batch_size = z.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=z.device)

        pseudo_labels = torch.argmax(coarse_assign, dim=1)
        z_norm = F.normalize(z, dim=1)
        sim_matrix = torch.matmul(z_norm, z_norm.T) / self.temperature

        # 同类掩码
        labels_equal = pseudo_labels.unsqueeze(0) == pseudo_labels.unsqueeze(1)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=z.device)
        labels_equal = labels_equal & mask

        # InfoNCE
        exp_sim = torch.exp(sim_matrix)
        denominator = exp_sim.sum(dim=1) - exp_sim.diag()
        numerator = (exp_sim * labels_equal.float()).sum(dim=1)

        valid_mask = labels_equal.sum(dim=1) > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)

        loss = -torch.log(numerator[valid_mask] / (denominator[valid_mask] + 1e-8) + 1e-8)
        return loss.mean()

    def compute_prototype_diversity(self):
        """原型多样性损失"""
        coarse_norm = F.normalize(self.coarse_prototypes, dim=1)
        coarse_sim = torch.matmul(coarse_norm, coarse_norm.T)
        coarse_loss = (coarse_sim - torch.eye(self.num_coarse, device=coarse_sim.device)).pow(2).mean()

        fine_norm = F.normalize(self.fine_prototypes, dim=1)
        fine_sim = torch.matmul(fine_norm, fine_norm.T)
        fine_loss = (fine_sim - torch.eye(self.num_fine, device=fine_sim.device)).pow(2).mean()

        return coarse_loss + fine_loss

    def get_binary_code(self, train_loader, test_loader):
        """生成二进制哈希码"""
        self.eval()
        with torch.no_grad():
            train_z_list, train_y_list = [], []
            for batch in train_loader:
                xb, yb = batch[0], batch[1]
                xb = xb.to(self.device)
                mu, _ = self.encode(xb)
                train_z_list.append(mu)
                train_y_list.append(yb)
            train_z = torch.cat(train_z_list, dim=0)
            train_y = torch.cat(train_y_list, dim=0)

            test_z_list, test_y_list = [], []
            for batch in test_loader:
                xb, yb = batch[0], batch[1]
                xb = xb.to(self.device)
                mu, _ = self.encode(xb)
                test_z_list.append(mu)
                test_y_list.append(yb)
            test_z = torch.cat(test_z_list, dim=0)
            test_y = torch.cat(test_y_list, dim=0)

            mid_val, _ = torch.median(train_z, dim=0)
            train_b = (train_z > mid_val).type(torch.ByteTensor).to(self.device)
            test_b = (test_z > mid_val).type(torch.ByteTensor).to(self.device)

        return train_b, test_b, train_y, test_y


def train_val(config):
    device = get_device(config)

    bit = config["bit"]
    dataset, data_fmt = config["dataset"].split('.')
    batch_size = config["batch_size"]

    if dataset in ['reuters', 'tmc', 'rcv1']:
        single_label_flag = False
    else:
        single_label_flag = True

    data_path = Path(__file__).parent.parent.parent / 'textdata'

    if single_label_flag:
        train_set = SingleLabelTextDataset(f'{data_path}/{dataset}', subset='train', bow_format=data_fmt)
        test_set = SingleLabelTextDataset(f'{data_path}/{dataset}', subset='test', bow_format=data_fmt)
        val_set = SingleLabelTextDataset(f'{data_path}/{dataset}', subset='cv', bow_format=data_fmt)
    else:
        train_set = MultiLabelTextDataset(f'{data_path}/{dataset}', subset='train', bow_format=data_fmt)
        test_set = MultiLabelTextDataset(f'{data_path}/{dataset}', subset='test', bow_format=data_fmt)
        val_set = MultiLabelTextDataset(f'{data_path}/{dataset}', subset='cv', bow_format=data_fmt)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

    num_features = train_set[0][0].size(0)
    model = HierHash(
        vocab_size=num_features,
        hidden_dim=config['hidden_dim'],
        latent_dim=bit,
        num_coarse=config['num_coarse'],
        num_fine=config['num_fine'],
        temperature=config['temperature'],
        dropout=config['dropout'],
        device=device
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    kl_weight = 0.
    kl_step = 1 / 5000.

    best_precision = 0
    best_precision_epoch = 0
    bad_epochs = 0
    best_state_dict = None

    log_file = LOG_DIR / f'data:{config["dataset"]}_bit:{config["bit"]}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=str(log_file),
        filemode='w'
    )

    checkpoint_path = CHECKPOINT_DIR / f'dataset:{dataset}_bit:{bit}.pth'

    for epoch in tqdm(range(config["epoch"])):
        model.train()
        total_loss = total_recon = total_kl = total_contrast = total_proto = 0
        num_batches = 0

        for batch in train_loader:
            xb, yb = batch[0], batch[1]
            xb = xb.to(device)

            optimizer.zero_grad()
            recon, mu, logvar, prior_mu, coarse_assign = model(xb)

            loss, recon_loss, kl_loss, contrast_loss, proto_loss = model.compute_loss(
                xb, recon, mu, logvar, prior_mu, coarse_assign,
                kl_weight, config['lambda_contrast'], config['lambda_proto']
            )

            loss.backward()
            optimizer.step()

            kl_weight = min(kl_weight + kl_step, 1.)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_contrast += contrast_loss.item()
            total_proto += proto_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches
        avg_contrast = total_contrast / num_batches
        avg_proto = total_proto / num_batches

        model.eval()
        with torch.no_grad():
            train_b, val_b, train_y, val_y = model.get_binary_code(train_loader, val_loader)
            retrieved_indices = retrieve_topk(val_b.to(device), train_b.to(device), topK=100)
            prec = compute_precision_at_k(
                retrieved_indices, val_y.to(device), train_y.to(device),
                topK=100, is_single_label=single_label_flag
            )

            if prec.item() > best_precision:
                best_precision = prec.item()
                best_precision_epoch = epoch + 1
                bad_epochs = 0
                best_state_dict = deepcopy(model.state_dict())
                torch.save(model, str(checkpoint_path))
            else:
                bad_epochs += 1

        tqdm.write(
            f'Epoch {epoch+1}/{config["epoch"]} - Loss: {avg_loss:.4f} - '
            f'Recon: {avg_recon:.4f} - KL: {avg_kl:.4f} - '
            f'Prec: {prec.item():.4f} - Best: {best_precision:.4f} [{best_precision_epoch}]'
        )
        logging.info(
            f'Epoch {epoch+1}/{config["epoch"]} - Loss: {avg_loss:.4f} - '
            f'Recon: {avg_recon:.4f} - KL: {avg_kl:.4f} - Contrast: {avg_contrast:.4f} - '
            f'Proto: {avg_proto:.4f} - Prec: {prec:.4f} - Best: {best_precision:.4f}'
        )

        if bad_epochs >= config["stop_iter"]:
            print(f"提前停止: 连续 {bad_epochs} 个epoch没有提升")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    with torch.no_grad():
        train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
        retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
        test_prec = compute_precision_at_k(
            retrieved_indices, test_y.to(device), train_y.to(device),
            topK=100, is_single_label=single_label_flag
        )
        print(f'Test Precision: {test_prec:.4f}')
        logging.info(f'Test Precision: {test_prec:.4f}')


if __name__ == "__main__":
    argparser = get_argparser()
    args = argparser.parse_args()
    config = vars(args)
    train_val(config)
