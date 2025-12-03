"""
RBSH: Ranking-Based Semantic Hashing
基于论文: Unsupervised Neural Generative Semantic Hashing (SIGIR 2019)

核心思想:
- 使用伯努利采样生成离散哈希码
- 使用弱监督的排序损失来约束相似文档的哈希码距离
- VAE风格的重建损失 + KL散度
- 噪声退火策略
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from textdata import SingleLabelTextDatasetDocID, MultiLabelTextDatasetDocID
from utils.utils import set_seed, retrieve_topk, compute_precision_at_k

set_seed()

# 获取当前文件所在目录
CURRENT_DIR = Path(__file__).parent
LOG_DIR = CURRENT_DIR / 'logs'
CHECKPOINT_DIR = CURRENT_DIR / 'checkpoints'
NEIGHBOR_DATA_DIR = CURRENT_DIR / 'neighbor_data'

# 创建目录
LOG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
NEIGHBOR_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ng20.tfidf')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--bit', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--stop_iter', type=int, default=50, help='控制在效果没有提升多少次后停止运行')
    parser.add_argument('--num_neighbors', type=int, default=20, help='邻居数量')
    parser.add_argument('--kl_weight_max', type=float, default=0.04, help='KL散度损失最大权重')
    parser.add_argument('--kl_anneal_epochs', type=int, default=5, help='KL权重退火epoch数')
    parser.add_argument('--rank_weight', type=float, default=0.5, help='排序损失权重')
    parser.add_argument('--rank_anneal_epochs', type=int, default=10, help='排序权重退火epoch数')
    parser.add_argument('--hinge_margin', type=float, default=1.0, help='Hinge loss margin')
    parser.add_argument('--sigma_max', type=float, default=1.0, help='噪声初始标准差')
    parser.add_argument('--sigma_min', type=float, default=0.0, help='噪声最小标准差')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout概率')
    parser.add_argument('--device', type=str, default='cuda', help='运行设备: cuda 或 cpu')
    parser.add_argument('--gpu', type=str, default='0', help='指定 GPU 设备号')
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


class TopDoc:
    """邻居文档索引类"""

    def __init__(self, data_fn, is_train=False):
        self.data_fn = data_fn
        self.is_train = is_train
        self.db = self._load(data_fn, is_train)

    def _load(self, fn, is_train):
        db = {}
        with open(fn) as in_data:
            for line in in_data:
                line = line.strip()
                first, rest = line.split(':')
                topk = list(map(int, rest.split(',')))
                doc_id = int(first)
                if is_train:
                    db[doc_id] = topk[1:]
                else:
                    db[doc_id] = topk
        return db

    def get_top_k(self, doc_id, top_k):
        if isinstance(doc_id, torch.Tensor):
            doc_id = doc_id.item()
        return self.db[doc_id][:top_k]


class TripletDataset(Dataset):
    """
    三元组数据集
    返回 (doc, doc1, doc2, sim1, sim2) 的三元组
    doc1 比 doc2 更相似于 doc (基于余弦相似度排序)
    """

    def __init__(self, base_dataset, neighbor_db, num_neighbors=20):
        self.base_dataset = base_dataset
        self.neighbor_db = neighbor_db
        self.num_neighbors = num_neighbors
        # 预加载所有文档
        self.all_docs = {}
        for i in range(len(base_dataset)):
            doc, idx, label = base_dataset[i]
            self.all_docs[idx] = (doc, label)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        doc, doc_id, label = self.base_dataset[idx]

        # 获取邻居列表 (按相似度排序)
        neighbors = self.neighbor_db.get_top_k(doc_id, self.num_neighbors)

        if len(neighbors) < 2:
            # 如果邻居不足，返回自身
            return doc, doc, doc, 1.0, 1.0, label

        # 随机选择两个邻居，确保 idx1 < idx2 (即 doc1 更相似)
        indices = np.random.choice(len(neighbors), size=2, replace=False)
        idx1, idx2 = sorted(indices)

        neighbor1_id = neighbors[idx1]
        neighbor2_id = neighbors[idx2]

        doc1, _ = self.all_docs[neighbor1_id]
        doc2, _ = self.all_docs[neighbor2_id]

        # 相似度分数 (基于排名，越小越相似)
        sim1 = 1.0 / (idx1 + 1)
        sim2 = 1.0 / (idx2 + 1)

        return doc, doc1, doc2, sim1, sim2, label


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


class RBSH(nn.Module):
    """
    RBSH 模型
    使用排序约束的无监督语义哈希

    架构:
    - 编码器: 多层全连接 + ReLU + Dropout -> Sigmoid (伯努利概率)
    - 伯努利采样: Straight-Through Estimator
    - 解码器: 词嵌入矩阵 + 重要性权重
    - 损失: 重建损失 + KL散度 + 排序损失
    """

    def __init__(self, dataset, vocab_size, latent_dim, device, dropout_prob=0.1, num_layers=2):
        super(RBSH, self).__init__()

        self.dataset = dataset
        self.hidden_dim = 1000
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.device = device
        self.num_layers = num_layers

        # 词重要性权重
        self.importance_weight = nn.Parameter(
            torch.FloatTensor(vocab_size).uniform_(0.1, 1.0)
        )

        # 词嵌入矩阵 (300维)
        self.word_embedding_full = nn.Parameter(
            torch.FloatTensor(vocab_size, 300).uniform_(-0.05, 0.05)
        )

        # 词嵌入降维层
        self.embedding_proj = nn.Linear(300, latent_dim)

        # 解码偏置
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))

        # 编码器
        encoder_layers = []
        input_dim = vocab_size
        for i in range(num_layers):
            encoder_layers.append(nn.Linear(input_dim, self.hidden_dim))
            encoder_layers.append(nn.ReLU(inplace=True))
            input_dim = self.hidden_dim

        encoder_layers.append(nn.Dropout(p=dropout_prob))
        encoder_layers.append(nn.Linear(input_dim, latent_dim))
        encoder_layers.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*encoder_layers)

        # 噪声标准差 (可退火)
        self.sigma = 1.0

    def encode(self, doc_mat):
        """编码器: 文档 -> 伯努利采样概率"""
        # 应用词重要性权重
        weighted_doc = doc_mat * self.importance_weight
        # 编码
        sampling_prob = self.encoder(weighted_doc)
        return sampling_prob

    def sample_hashcode(self, sampling_prob, training=True):
        """伯努利采样"""
        return BernoulliStraightThrough.apply(sampling_prob, training)

    def decode(self, hashcode, target_doc, sigma=0.0):
        """解码器: 使用词嵌入进行重建"""
        # 添加噪声
        if self.training and sigma > 0:
            noise = torch.randn_like(hashcode) * sigma
            noisy_hashcode = hashcode + noise
        else:
            noisy_hashcode = hashcode

        # 词嵌入降维
        word_emb = self.embedding_proj(self.word_embedding_full)  # [vocab_size, latent_dim]

        # 计算词概率: hashcode @ word_emb.T * importance + bias
        logits = torch.matmul(noisy_hashcode, word_emb.T) * self.importance_weight + self.decoder_bias
        log_probs = F.log_softmax(logits, dim=-1)

        # 计算重建损失 (只对出现的词计算)
        mask = (target_doc > 0).float()
        recon_loss = -torch.sum(log_probs * mask, dim=-1)

        return recon_loss, log_probs

    def compute_kl_loss(self, sampling_prob):
        """计算伯努利KL散度损失"""
        eps = 1e-10
        kl = sampling_prob * torch.log(torch.clamp(sampling_prob / 0.5, min=eps)) + \
            (1 - sampling_prob) * torch.log(torch.clamp((1 - sampling_prob) / 0.5, min=eps))
        kl = torch.sum(kl, dim=-1)
        return kl

    def compute_ranking_loss(self, hash_doc, hash_doc1, hash_doc2, sim1, sim2, margin=1.0):
        """
        计算排序损失
        约束: dist(doc, doc1) < dist(doc, doc2) 当 sim1 > sim2 时
        """
        # 计算距离 (欧氏距离的平方)
        dist1 = torch.sum((hash_doc - hash_doc1) ** 2, dim=-1)
        dist2 = torch.sum((hash_doc - hash_doc2) ** 2, dim=-1)

        # 确定哪个更相似 (sim1 > sim2 说明 doc1 更相似)
        more_similar = (sim1 > sim2).float()
        equal_similar = (torch.abs(sim1 - sim2) < 1e-10).float()

        # Hinge loss: max(0, margin + dist1 - dist2) 当 doc1 更相似时
        # 即要求 dist1 < dist2 - margin
        hinge_loss_unequal = F.relu(margin + dist1 - dist2) * more_similar * (1 - equal_similar)

        # 对于相似度相等的情况，约束距离相近
        equal_loss = torch.abs(dist1 - dist2) * equal_similar

        rank_loss = hinge_loss_unequal + equal_loss

        return rank_loss

    def forward(self, doc, doc1=None, doc2=None, sim1=None, sim2=None, sigma=0.0, margin=1.0):
        """前向传播"""
        # 编码主文档
        prob = self.encode(doc)
        hashcode = self.sample_hashcode(prob, self.training)

        # 重建损失
        recon_loss, _ = self.decode(hashcode, doc, sigma)

        # KL散度
        kl_loss = self.compute_kl_loss(prob)

        # 排序损失
        rank_loss = None
        doc1_recon_loss = None
        doc2_recon_loss = None

        if doc1 is not None and doc2 is not None:
            prob1 = self.encode(doc1)
            prob2 = self.encode(doc2)
            hashcode1 = self.sample_hashcode(prob1, self.training)
            hashcode2 = self.sample_hashcode(prob2, self.training)

            # 排序损失
            rank_loss = self.compute_ranking_loss(hashcode, hashcode1, hashcode2, sim1, sim2, margin)

            # doc1 和 doc2 的重建损失
            doc1_recon_loss, _ = self.decode(hashcode1, doc1, sigma)
            doc2_recon_loss, _ = self.decode(hashcode2, doc2, sigma)

        return recon_loss, kl_loss, rank_loss, doc1_recon_loss, doc2_recon_loss, hashcode, prob

    def get_name(self):
        return "RBSH"

    def get_binary_code(self, train_loader, test_loader):
        """生成二进制哈希码

        注意：train_loader 和 test_loader 应该使用普通数据集（不是triplet），
        返回格式为 (doc, doc_id, label)
        """
        self.eval()

        train_codes = []
        train_labels = []
        for xb, _, yb in train_loader:
            xb = xb.to(self.device)
            prob = self.encode(xb)
            code = (prob > 0.5).cpu()
            train_codes.append(code)
            train_labels.append(yb)

        train_codes = torch.cat(train_codes, dim=0)
        train_labels = torch.cat(train_labels, dim=0)

        test_codes = []
        test_labels = []
        for xb, _, yb in test_loader:
            xb = xb.to(self.device)
            prob = self.encode(xb)
            code = (prob > 0.5).cpu()
            test_codes.append(code)
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
    num_neighbors = config["num_neighbors"]

    if dataset in ['reuters', 'tmc', 'rcv1']:
        single_label_flag = False
    else:
        single_label_flag = True

    # 使用数据集模块
    data_path = Path(__file__).parent.parent.parent / 'textdata'

    if single_label_flag:
        train_set = SingleLabelTextDatasetDocID(f'{data_path}/{dataset}', subset='train', bow_format=data_fmt)
        test_set = SingleLabelTextDatasetDocID(f'{data_path}/{dataset}', subset='test', bow_format=data_fmt)
        val_set = SingleLabelTextDatasetDocID(f'{data_path}/{dataset}', subset='cv', bow_format=data_fmt)
    else:
        train_set = MultiLabelTextDatasetDocID(f'{data_path}/{dataset}', subset='train', bow_format=data_fmt)
        test_set = MultiLabelTextDatasetDocID(f'{data_path}/{dataset}', subset='test', bow_format=data_fmt)
        val_set = MultiLabelTextDatasetDocID(f'{data_path}/{dataset}', subset='cv', bow_format=data_fmt)

    # 加载邻居数据
    neighbor_file = NEIGHBOR_DATA_DIR / f'{dataset}_train_top101.txt'
    if not neighbor_file.exists():
        raise FileNotFoundError(
            f"邻居索引文件不存在: {neighbor_file}\n"
            f"请先运行 prepare_neighbor_data.py 生成邻居索引"
        )
    train_topk_docs_db = TopDoc(str(neighbor_file), is_train=True)

    # 创建三元组数据集（用于训练）
    triplet_train_set = TripletDataset(train_set, train_topk_docs_db, num_neighbors=num_neighbors)

    # 训练用loader（使用triplet数据集）
    train_loader = DataLoader(dataset=triplet_train_set, batch_size=batch_size, shuffle=True)

    # 评估用loader（使用普通数据集，不shuffle）
    train_eval_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

    num_features = train_set[0][0].size(0)
    model = RBSH(dataset, num_features, bit, device, dropout_prob=config["dropout"])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # 权重退火参数
    kl_weight = 0.0
    kl_weight_max = config["kl_weight_max"]
    kl_anneal_steps = config["kl_anneal_epochs"] * len(train_loader)
    kl_step = kl_weight_max / kl_anneal_steps if kl_anneal_steps > 0 else kl_weight_max

    rank_weight = config["rank_weight"]
    rank_weight_max = 30.0
    rank_anneal_steps = config["rank_anneal_epochs"] * len(train_loader)
    rank_step = (rank_weight_max - rank_weight) / rank_anneal_steps if rank_anneal_steps > 0 else 0

    # 噪声退火
    sigma = config["sigma_max"]
    sigma_min = config["sigma_min"]
    total_steps = config["epoch"] * len(train_loader)
    sigma_step = (sigma - sigma_min) / total_steps if total_steps > 0 else 0

    hinge_margin = config["hinge_margin"]

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
        avg_recon_loss = []
        avg_kl_loss = []
        avg_rank_loss = []

        for step, (doc, doc1, doc2, sim1, sim2, labels) in enumerate(train_loader):
            doc = doc.to(device)
            doc1 = doc1.to(device)
            doc2 = doc2.to(device)
            sim1 = sim1.to(device).float()
            sim2 = sim2.to(device).float()

            # 前向传播
            recon_loss, kl_loss, rank_loss, doc1_recon, doc2_recon, hashcode, prob = model(
                doc, doc1, doc2, sim1, sim2, sigma=sigma, margin=hinge_margin
            )

            # 计算总损失
            recon_loss_mean = torch.mean(recon_loss)
            kl_loss_mean = torch.mean(kl_loss)
            rank_loss_mean = torch.mean(rank_loss) if rank_loss is not None else 0
            doc1_recon_mean = torch.mean(doc1_recon) if doc1_recon is not None else 0
            doc2_recon_mean = torch.mean(doc2_recon) if doc2_recon is not None else 0

            # VAE loss: -ELBO = recon_loss + kl_weight * kl_loss
            vae_loss = recon_loss_mean + kl_weight * kl_loss_mean

            # 总损失
            loss = vae_loss + rank_weight * rank_loss_mean + doc1_recon_mean + doc2_recon_mean

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新权重
            kl_weight = min(kl_weight + kl_step, kl_weight_max)
            rank_weight = min(rank_weight + rank_step, rank_weight_max)
            sigma = max(sigma - sigma_step, sigma_min)

            avg_loss.append(loss.item())
            avg_recon_loss.append(recon_loss_mean.item())
            avg_kl_loss.append(kl_loss_mean.item())
            if rank_loss is not None:
                avg_rank_loss.append(rank_loss_mean.item())

        # 验证（使用评估用loader）
        model.eval()
        with torch.no_grad():
            train_b, val_b, train_y, val_y = model.get_binary_code(train_eval_loader, val_loader)
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
            f'Recon: {np.mean(avg_recon_loss):.2f} - KL: {np.mean(avg_kl_loss):.2f} - '
            f'Rank: {np.mean(avg_rank_loss) if avg_rank_loss else 0:.2f} - '
            f'Prec: {prec.item():.4f} - Best: {best_precision:.4f} [{best_precision_epoch}]')
        logging.info(
            f'Epoch {epoch+1}/{config["epoch"]} - Loss: {np.mean(avg_loss):.4f} - '
            f'Prec: {prec:.4f} - Best: {best_precision:.4f} [{best_precision_epoch}]')

    # 加载最佳模型进行测试
    model = torch.load(str(checkpoint_path))
    model.eval()
    with torch.no_grad():
        train_b, test_b, train_y, test_y = model.get_binary_code(train_eval_loader, test_loader)
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
