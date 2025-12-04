"""
DSH: De-confusing Hard Samples for Text Semantic Hashing
基于论文: De-confusing Hard Samples for Text Semantic Hashing (ICASSP 2025)

论文核心方法:
1. Bernoulli VAE: 编码器-解码器网络，使用sign函数采样哈希码
2. 分类损失: 使用哈希码预测类别
3. 层次类别约束: 父级和子级对比损失（本实现简化为单层类别对比）
4. 困难样本约束:
   - 计算每个类别的中心和硬半径
   - 类内困难样本: 属于类别但距离中心 > 硬半径
   - 类间困难样本: 不属于类别但距离中心 < 硬半径
   - 去混淆损失拉近困难样本到自己类别中心

总损失: L = Lvae + β*Ly + γ*Lh + ξ*Ld
"""

import argparse
import logging
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
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
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--bit', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--stop_iter', type=int, default=10, help='控制在效果没有提升多少次后停止运行')

    # DSH 特有参数 (根据论文)
    parser.add_argument('--hidden_dim', type=int, default=1000, help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout概率')
    parser.add_argument('--beta', type=float, default=3.0, help='分类损失权重')
    parser.add_argument('--gamma', type=float, default=3.0, help='层次类别约束权重')
    parser.add_argument('--xi', type=float, default=1.0, help='去混淆损失权重')
    parser.add_argument('--margin', type=float, default=0.1, help='对比损失margin')
    parser.add_argument('--hard_margin', type=float, default=0.01, help='困难样本margin')
    parser.add_argument('--lambda_hard', type=float, default=1.0, help='硬半径系数 R = μ + λσ')
    parser.add_argument('--update_center_freq', type=int, default=3, help='更新类别中心的频率(epochs)')

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


class SignStraightThrough(torch.autograd.Function):
    """
    Sign函数的直通估计器 (Straight-Through Estimator)
    前向: sign(x - 0.5) 将[0,1]映射到{0,1}
    反向: 直接传递梯度
    """
    @staticmethod
    def forward(ctx, input):
        # 论文公式(1): hi = 1 if xi > 0.5 else 0
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class CategoryCenterManager:
    """
    类别中心管理器
    用于计算和存储每个类别的中心、均值距离、方差和硬半径
    """

    def __init__(self, num_classes, latent_dim, lambda_hard, device):
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.lambda_hard = lambda_hard
        self.device = device

        # 类别中心
        self.centers = torch.zeros(num_classes, latent_dim, device=device)
        # 均值距离
        self.mu = torch.zeros(num_classes, device=device)
        # 距离方差
        self.sigma = torch.zeros(num_classes, device=device)
        # 硬半径 R = μ + λσ
        self.hard_radius = torch.zeros(num_classes, device=device)

        self.initialized = False

    def update(self, embeddings, labels, single_label=True):
        """更新类别中心和硬半径"""
        with torch.no_grad():
            for c in range(self.num_classes):
                if single_label:
                    mask = (labels == c)
                else:
                    mask = (labels[:, c] > 0)

                if mask.sum() == 0:
                    continue

                class_embeddings = embeddings[mask]

                # 计算类别中心
                center = class_embeddings.mean(dim=0)
                self.centers[c] = center

                # 计算到中心的距离
                distances = torch.norm(class_embeddings - center, dim=1)

                # 计算均值和方差
                self.mu[c] = distances.mean()
                self.sigma[c] = distances.std() if len(distances) > 1 else torch.tensor(0.1)

                # 硬半径 R = μ + λσ
                self.hard_radius[c] = self.mu[c] + self.lambda_hard * self.sigma[c]

        self.initialized = True

    def get_hard_samples(self, embeddings, labels, single_label=True):
        """
        识别困难样本
        返回: intra_hard_mask, inter_hard_mask
        - intra_hard: 类内困难样本 (属于类别但距离 > R)
        - inter_hard: 类间困难样本 (不属于类别但距离 < R)
        """
        batch_size = embeddings.size(0)

        intra_hard_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        inter_hard_info = []  # 存储 (样本索引, 错误类别索引)

        for i in range(batch_size):
            emb = embeddings[i]

            if single_label:
                true_label = labels[i].item()
            else:
                true_label = labels[i].argmax().item()

            # 到自己类别中心的距离
            dist_to_own = torch.norm(emb - self.centers[true_label])

            # 类内困难: 距离 > 硬半径
            if dist_to_own > self.hard_radius[true_label]:
                intra_hard_mask[i] = True

            # 检查类间困难: 距离其他类别中心 < 该类别的硬半径
            for c in range(self.num_classes):
                if c == true_label:
                    continue
                dist_to_other = torch.norm(emb - self.centers[c])
                if dist_to_other < self.hard_radius[c]:
                    inter_hard_info.append((i, c))

        return intra_hard_mask, inter_hard_info


class DSH(nn.Module):
    """
    DSH 模型 (De-confusing Hard Samples for Text Semantic Hashing)

    架构:
    - 编码器: 文档 -> 连续向量 (每维是伯努利参数)
    - Sign函数: 二值化采样
    - 解码器: 重建文档
    - 分类器: 预测类别

    损失:
    - Lvae: 重建损失 + KL散度
    - Ly: 分类损失
    - Lh: 层次类别约束 (对比损失)
    - Ld: 去混淆损失 (困难样本约束)
    """

    def __init__(self, vocab_size, hidden_dim, latent_dim, num_classes, dropout, device):
        super(DSH, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = device

        # 编码器 (论文: MLP with 1000 neurons and leaky ReLU)
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Dropout(dropout),
            nn.Sigmoid()  # 输出伯努利参数 [0, 1]
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, vocab_size),
            nn.LogSoftmax(dim=1)
        )

        # 分类器 (论文公式3)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def encode(self, x):
        """编码器: 文档 -> 伯努利参数"""
        return self.encoder(x)

    def sample(self, prob):
        """采样: 使用Sign直通估计器"""
        if self.training:
            return SignStraightThrough.apply(prob)
        else:
            return (prob > 0.5).float()

    def decode(self, h):
        """解码器: 哈希码 -> 重建文档"""
        return self.decoder(h)

    def classify(self, h):
        """分类器: 哈希码 -> 类别预测"""
        return self.classifier(h)

    def forward(self, x):
        """前向传播"""
        # 编码
        prob = self.encode(x)

        # 采样哈希码
        h = self.sample(prob)

        # 解码
        recon = self.decode(h)

        # 分类
        logits = self.classify(h)

        return recon, prob, h, logits

    def compute_vae_loss(self, recon, x, prob):
        """
        计算VAE损失 (论文公式2)
        Lvae = -E[log p(x|h)] + KL(q(h|x) || p(h))
        """
        # 重建损失
        recon_loss = -torch.mean(torch.sum(recon * x, dim=1))

        # KL散度: KL(Bernoulli(prob) || Bernoulli(0.5))
        # 论文设置 ρ = 0.5 以最大化信息熵
        eps = 1e-8
        kl = prob * (torch.log(prob + eps) - torch.log(torch.tensor(0.5))) + \
            (1 - prob) * (torch.log(1 - prob + eps) - torch.log(torch.tensor(0.5)))
        kl_loss = torch.mean(torch.sum(kl, dim=1))

        return recon_loss + kl_loss, recon_loss, kl_loss

    def compute_category_loss(self, logits, labels):
        """
        计算分类损失 (论文公式4)
        Ly = CrossEntropy(ŷ, y)
        """
        return F.cross_entropy(logits, labels)

    def compute_contrastive_loss(self, embeddings, labels, margin, single_label):
        """
        计算对比损失 (论文公式5-8)
        Lh: 将同类样本拉近，不同类样本推远
        """
        batch_size = embeddings.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        loss = torch.tensor(0.0, device=embeddings.device)
        count = 0

        for i in range(batch_size):
            anchor = embeddings[i]

            if single_label:
                anchor_label = labels[i]
                pos_mask = (labels == anchor_label)
                neg_mask = (labels != anchor_label)
            else:
                anchor_label = labels[i]
                # 多标签: 有共同标签为正
                sim = (labels * anchor_label.unsqueeze(0)).sum(dim=1)
                pos_mask = sim > 0
                neg_mask = sim == 0

            pos_mask[i] = False  # 排除自身

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            # 随机选择一个正样本和负样本
            pos_indices = torch.where(pos_mask)[0]
            neg_indices = torch.where(neg_mask)[0]

            pos_idx = pos_indices[torch.randint(len(pos_indices), (1,))]
            neg_idx = neg_indices[torch.randint(len(neg_indices), (1,))]

            pos_emb = embeddings[pos_idx]
            neg_emb = embeddings[neg_idx]

            # 论文公式5: L(x, x+, x-, m) = relu(d(x, x+) - d(x, x-) + m)
            d_pos = torch.norm(anchor - pos_emb)
            d_neg = torch.norm(anchor - neg_emb)

            triplet_loss = F.relu(d_pos - d_neg + margin)
            loss = loss + triplet_loss
            count += 1

        return loss / max(count, 1)

    def compute_deconfusion_loss(self, embeddings, labels, center_manager,
                                 hard_margin, single_label):
        """
        计算去混淆损失 (论文公式9-11)
        Ld: 拉近困难样本到自己类别中心，远离其他类别中心
        """
        if not center_manager.initialized:
            return torch.tensor(0.0, device=embeddings.device)

        intra_hard_mask, inter_hard_info = center_manager.get_hard_samples(
            embeddings, labels, single_label
        )

        loss = torch.tensor(0.0, device=embeddings.device)
        count = 0

        # 类内困难样本: 拉近到自己类别中心
        if intra_hard_mask.sum() > 0:
            for i in torch.where(intra_hard_mask)[0]:
                emb = embeddings[i]
                if single_label:
                    true_label = labels[i].item()
                else:
                    true_label = labels[i].argmax().item()

                center = center_manager.centers[true_label]

                # 选择一个负类别中心
                neg_labels = [c for c in range(center_manager.num_classes) if c != true_label]
                if len(neg_labels) > 0:
                    neg_label = neg_labels[torch.randint(len(neg_labels), (1,)).item()]
                    neg_center = center_manager.centers[neg_label]

                    d_pos = torch.norm(emb - center)
                    d_neg = torch.norm(emb - neg_center)

                    loss = loss + F.relu(d_pos - d_neg + hard_margin)
                    count += 1

        # 类间困难样本: 推远离错误类别中心
        for (i, wrong_class) in inter_hard_info:
            emb = embeddings[i]
            if single_label:
                true_label = labels[i].item()
            else:
                true_label = labels[i].argmax().item()

            true_center = center_manager.centers[true_label]
            wrong_center = center_manager.centers[wrong_class]

            d_true = torch.norm(emb - true_center)
            d_wrong = torch.norm(emb - wrong_center)

            # 应该离真实中心近，离错误中心远
            loss = loss + F.relu(d_true - d_wrong + hard_margin)
            count += 1

        return loss / max(count, 1)

    def get_binary_code(self, train_loader, test_loader):
        """生成二进制哈希码"""
        self.eval()
        with torch.no_grad():
            # 训练集编码
            train_h_list = []
            train_y_list = []
            for batch in train_loader:
                xb, yb = batch[0], batch[1]
                xb = xb.to(self.device)
                prob = self.encode(xb)
                h = (prob > 0.5).byte()
                train_h_list.append(h)
                train_y_list.append(yb)
            train_h = torch.cat(train_h_list, dim=0)
            train_y = torch.cat(train_y_list, dim=0)

            # 测试集编码
            test_h_list = []
            test_y_list = []
            for batch in test_loader:
                xb, yb = batch[0], batch[1]
                xb = xb.to(self.device)
                prob = self.encode(xb)
                h = (prob > 0.5).byte()
                test_h_list.append(h)
                test_y_list.append(yb)
            test_h = torch.cat(test_h_list, dim=0)
            test_y = torch.cat(test_y_list, dim=0)

        return train_h, test_h, train_y, test_y


def train_val(config):
    device = get_device(config)

    bit = config["bit"]
    dataset, data_fmt = config["dataset"].split('.')
    batch_size = config["batch_size"]

    if dataset in ['reuters', 'tmc', 'rcv1']:
        single_label_flag = False
    else:
        single_label_flag = True

    # 加载数据
    data_path = Path(__file__).parent.parent.parent / 'textdata'

    if single_label_flag:
        train_set = SingleLabelTextDataset(f'{data_path}/{dataset}', subset='train', bow_format=data_fmt)
        test_set = SingleLabelTextDataset(f'{data_path}/{dataset}', subset='test', bow_format=data_fmt)
        val_set = SingleLabelTextDataset(f'{data_path}/{dataset}', subset='cv', bow_format=data_fmt)
        num_classes = train_set.num_classes()
    else:
        train_set = MultiLabelTextDataset(f'{data_path}/{dataset}', subset='train', bow_format=data_fmt)
        test_set = MultiLabelTextDataset(f'{data_path}/{dataset}', subset='test', bow_format=data_fmt)
        val_set = MultiLabelTextDataset(f'{data_path}/{dataset}', subset='cv', bow_format=data_fmt)
        num_classes = train_set[0][1].size(0)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=batch_size, shuffle=False
    )

    # 创建模型
    num_features = train_set[0][0].size(0)
    model = DSH(
        vocab_size=num_features,
        hidden_dim=config['hidden_dim'],
        latent_dim=bit,
        num_classes=num_classes,
        dropout=config['dropout'],
        device=device
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # 类别中心管理器
    center_manager = CategoryCenterManager(
        num_classes, bit, config['lambda_hard'], device
    )

    best_precision = 0
    best_precision_epoch = 0
    bad_epochs = 0
    best_state_dict = None

    # 日志设置
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
        total_loss = 0
        total_vae = 0
        total_cls = 0
        total_contrast = 0
        total_deconf = 0
        num_batches = 0

        # 收集所有embeddings用于更新类别中心
        all_embeddings = []
        all_labels = []

        for batch in train_loader:
            xb, yb = batch[0], batch[1]
            xb = xb.to(device)

            if single_label_flag:
                if not isinstance(yb, torch.Tensor):
                    yb = torch.tensor(yb)
                yb = yb.to(device)
            else:
                yb = yb.to(device)

            optimizer.zero_grad()

            # 前向传播
            recon, prob, h, logits = model(xb)

            # 使用prob (连续表示) 进行对比学习
            embeddings = prob

            # 计算各项损失
            vae_loss, recon_loss, kl_loss = model.compute_vae_loss(recon, xb, prob)

            if single_label_flag:
                cls_loss = model.compute_category_loss(logits, yb)
            else:
                cls_loss = F.binary_cross_entropy_with_logits(logits, yb)

            contrast_loss = model.compute_contrastive_loss(
                embeddings, yb, config['margin'], single_label_flag
            )

            deconf_loss = model.compute_deconfusion_loss(
                embeddings, yb, center_manager, config['hard_margin'], single_label_flag
            )

            # 总损失 (论文: L = Lvae + β*Ly + γ*Lh + ξ*Ld)
            loss = vae_loss + \
                config['beta'] * cls_loss + \
                config['gamma'] * contrast_loss + \
                config['xi'] * deconf_loss

            loss.backward()
            optimizer.step()

            # 收集embeddings
            all_embeddings.append(embeddings.detach())
            all_labels.append(yb.detach() if isinstance(yb, torch.Tensor) else torch.tensor(yb))

            total_loss += loss.item()
            total_vae += vae_loss.item()
            total_cls += cls_loss.item()
            total_contrast += contrast_loss.item()
            total_deconf += deconf_loss.item()
            num_batches += 1

        # 每隔一定epoch更新类别中心 (论文: every 3 epochs)
        if (epoch + 1) % config['update_center_freq'] == 0:
            all_embeddings = torch.cat(all_embeddings, dim=0)
            all_labels = torch.cat(all_labels, dim=0).to(device)
            center_manager.update(all_embeddings, all_labels, single_label_flag)

        avg_loss = total_loss / num_batches
        avg_vae = total_vae / num_batches
        avg_cls = total_cls / num_batches
        avg_contrast = total_contrast / num_batches
        avg_deconf = total_deconf / num_batches

        # 验证
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
            f'VAE: {avg_vae:.4f} - Cls: {avg_cls:.4f} - '
            f'Prec: {prec.item():.4f} - Best: {best_precision:.4f} [{best_precision_epoch}]'
        )
        logging.info(
            f'Epoch {epoch+1}/{config["epoch"]} - Loss: {avg_loss:.4f} - '
            f'VAE: {avg_vae:.4f} - Cls: {avg_cls:.4f} - Contrast: {avg_contrast:.4f} - '
            f'Deconf: {avg_deconf:.4f} - Prec: {prec:.4f} - Best: {best_precision:.4f}'
        )

        if bad_epochs >= config["stop_iter"]:
            print(f"提前停止: 连续 {bad_epochs} 个epoch没有提升")
            break

    # 加载最佳模型进行测试
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    with torch.no_grad():
        train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
        retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
        test_prec = compute_precision_at_k(
            retrieved_indices,
            test_y.to(device),
            train_y.to(device),
            topK=100,
            is_single_label=single_label_flag
        )
        print(f'Test Precision: {test_prec:.4f}')
        logging.info(f'Test Precision: {test_prec:.4f}')


if __name__ == "__main__":
    argparser = get_argparser()
    args = argparser.parse_args()
    config = vars(args)
    train_val(config)
