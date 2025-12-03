"""
PairRec: Unsupervised Semantic Hashing with Pairwise Reconstruction
基于论文: Unsupervised Semantic Hashing with Pairwise Reconstruction (SIGIR 2020)

核心思想:
- 使用伯努利采样生成离散哈希码
- 配对重建: 使用邻居文档的哈希码来重建原始文档
- 词嵌入解码器
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--bit', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--stop_iter', type=int, default=50, help='控制在效果没有提升多少次后停止运行')
    parser.add_argument('--num_neighbors', type=int, default=25, help='配对邻居数量')
    parser.add_argument('--pair_weight', type=float, default=1.0, help='配对重建损失权重')
    parser.add_argument('--kl_weight', type=float, default=0.0, help='KL散度损失权重')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout概率')
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


class TopDoc:
    """
    邻居文档索引类
    用于加载和查询预计算的邻居文档索引
    """

    def __init__(self, data_fn, is_train=False):
        self.data_fn = data_fn
        self.is_train = is_train
        self.db = self._load(data_fn, is_train)

    def _load(self, fn, is_train):
        """加载邻居索引文件"""
        db = {}
        with open(fn) as in_data:
            for line in in_data:
                line = line.strip()
                first, rest = line.split(':')
                topk = list(map(int, rest.split(',')))
                doc_id = int(first)
                # 训练集时排除自身（第一个是自身）
                if is_train:
                    db[doc_id] = topk[1:]
                else:
                    db[doc_id] = topk
        return db

    def get_top_k(self, doc_id, top_k):
        """获取文档的Top-K邻居"""
        if isinstance(doc_id, torch.Tensor):
            doc_id = doc_id.item()
        return self.db[doc_id][:top_k]

    def sample_neighbor(self, doc_id, num_neighbors):
        """随机采样一个邻居"""
        if isinstance(doc_id, torch.Tensor):
            doc_id = doc_id.item()
        neighbors = self.db[doc_id][:num_neighbors]
        return np.random.choice(neighbors)


class PairDataset(Dataset):
    """
    配对数据集
    返回 (doc1, doc2, doc1_id) 的配对
    """

    def __init__(self, base_dataset, neighbor_db, num_neighbors=25):
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
        doc1, doc1_id, label1 = self.base_dataset[idx]
        # 随机采样一个邻居
        neighbor_id = self.neighbor_db.sample_neighbor(doc1_id, self.num_neighbors)
        doc2, label2 = self.all_docs[neighbor_id]
        return doc1, doc2, doc1_id, label1


class BernoulliStraightThrough(torch.autograd.Function):
    """
    伯努利采样 + Straight-Through Estimator
    前向传播: 从伯努利分布采样
    反向传播: 梯度直接传递 (identity)
    """

    @staticmethod
    def forward(ctx, probs, training=True):
        if training:
            # 训练时随机采样
            samples = torch.bernoulli(probs)
        else:
            # 测试时使用阈值0.5
            samples = (probs > 0.5).float()
        return samples

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: 梯度直接传递
        return grad_output, None


class PairRec(nn.Module):
    """
    PairRec 模型
    使用配对重建的无监督语义哈希

    架构:
    - 编码器: 多层全连接 + ReLU + Dropout -> Sigmoid (伯努利概率)
    - 伯努利采样: 使用 Straight-Through Estimator
    - 解码器: 使用词嵌入矩阵 + 重要性权重
    - 损失: 自重建损失 + 配对重建损失 + KL散度
    """

    def __init__(self, dataset, vocab_size, latent_dim, device, dropout_prob=0.2, num_layers=2):
        super(PairRec, self).__init__()

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

        # 词嵌入矩阵 (用于解码)
        self.word_embedding = nn.Parameter(
            torch.FloatTensor(vocab_size, latent_dim).uniform_(-1, 1)
        )

        # 解码偏置
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))

        # 编码器
        encoder_layers = []
        input_dim = vocab_size
        for i in range(num_layers):
            output_dim = self.hidden_dim // (i + 1) if i > 0 else self.hidden_dim
            encoder_layers.append(nn.Linear(input_dim, output_dim))
            encoder_layers.append(nn.ReLU(inplace=True))
            input_dim = output_dim

        encoder_layers.append(nn.Dropout(p=dropout_prob))
        encoder_layers.append(nn.Linear(input_dim, latent_dim))
        encoder_layers.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*encoder_layers)

        # 噪声标准差 (用于VAE式的噪声注入)
        self.sigma = 1.0
        self.sigma_decay = 1e-6

    def encode(self, doc_mat):
        """
        编码器: 文档 -> 伯努利采样概率
        """
        # 应用词重要性权重
        weighted_doc = doc_mat * self.importance_weight
        # 编码得到采样概率
        sampling_prob = self.encoder(weighted_doc)
        return sampling_prob

    def sample_hashcode(self, sampling_prob, training=True):
        """
        从伯努利分布采样哈希码
        使用 Straight-Through Estimator 处理梯度
        """
        return BernoulliStraightThrough.apply(sampling_prob, training)

    def decode(self, hashcode, target_doc):
        """
        解码器: 使用词嵌入进行重建
        """
        # 添加噪声 (VAE风格)
        if self.training and self.sigma > 0:
            noise = torch.randn_like(hashcode) * self.sigma
            noisy_hashcode = hashcode + noise
        else:
            noisy_hashcode = hashcode

        # 计算词概率: hashcode @ word_embedding.T * importance_weight + bias
        logits = torch.matmul(noisy_hashcode, self.word_embedding.T) * self.importance_weight + self.decoder_bias
        log_probs = torch.log_softmax(logits, dim=-1)

        # 计算重建损失 (只对出现的词计算)
        mask = (target_doc > 0).float()
        recon_loss = -torch.sum(log_probs * mask, dim=-1)
        return recon_loss

    def compute_kl_loss(self, sampling_prob):
        """
        计算伯努利KL散度损失
        KL(Bernoulli(p) || Bernoulli(0.5))
        """
        # KL = p * log(p/0.5) + (1-p) * log((1-p)/0.5)
        eps = 1e-10
        kl = sampling_prob * torch.log(torch.clamp(sampling_prob / 0.5, min=eps)) + \
            (1 - sampling_prob) * torch.log(torch.clamp((1 - sampling_prob) / 0.5, min=eps))
        kl = torch.sum(kl, dim=-1)
        return kl

    def forward(self, doc1, doc2=None):
        """
        前向传播

        Args:
            doc1: 主文档
            doc2: 邻居文档 (可选)

        Returns:
            doc1_loss: 自重建损失
            pair_loss: 配对重建损失 (如果提供doc2)
            kl_loss: KL散度损失
            hashcode: 哈希码
        """
        # 编码
        prob1 = self.encode(doc1)
        hashcode1 = self.sample_hashcode(prob1, self.training)

        # 自重建损失
        doc1_loss = self.decode(hashcode1, doc1)

        # KL散度
        kl_loss = self.compute_kl_loss(prob1)

        pair_loss = None
        if doc2 is not None:
            # 编码邻居文档
            prob2 = self.encode(doc2)
            hashcode2 = self.sample_hashcode(prob2, self.training)
            # 用邻居的哈希码重建原始文档
            pair_loss = self.decode(hashcode2, doc1)

        return doc1_loss, pair_loss, kl_loss, hashcode1, prob1

    def get_name(self):
        return "PairRec"

    def update_sigma(self):
        """更新噪声标准差"""
        self.sigma = max(self.sigma - self.sigma_decay, 0)

    def get_binary_code(self, train_loader, test_loader):
        """生成二进制哈希码"""
        self.eval()

        train_codes = []
        train_labels = []
        for batch in train_loader:
            if len(batch) == 4:  # PairDataset
                xb, _, _, yb = batch
            else:  # 普通Dataset
                xb, _, yb = batch
            xb = xb.to(self.device)
            prob = self.encode(xb)
            code = (prob > 0.5).cpu()
            train_codes.append(code)
            train_labels.append(yb)

        train_codes = torch.cat(train_codes, dim=0)
        train_labels = torch.cat(train_labels, dim=0)

        test_codes = []
        test_labels = []
        for batch in test_loader:
            if len(batch) == 4:  # PairDataset
                xb, _, _, yb = batch
            else:  # 普通Dataset
                xb, _, yb = batch
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
    pair_weight = config["pair_weight"]
    kl_weight = config["kl_weight"]

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

    # 创建配对数据集
    pair_train_set = PairDataset(train_set, train_topk_docs_db, num_neighbors=num_neighbors)

    train_loader = DataLoader(dataset=pair_train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

    num_features = train_set[0][0].size(0)
    model = PairRec(dataset, num_features, bit, device, dropout_prob=config["dropout"])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

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
        avg_doc_loss = []
        avg_pair_loss = []
        avg_kl_loss = []

        for step, (doc1, doc2, doc_ids, labels) in enumerate(train_loader):
            doc1 = doc1.to(device)
            doc2 = doc2.to(device)

            # 前向传播
            doc_loss, pair_loss, kl_loss, hashcode, prob = model(doc1, doc2)

            # 计算总损失
            doc_loss_mean = torch.mean(doc_loss)
            pair_loss_mean = torch.mean(pair_loss) if pair_loss is not None else 0
            kl_loss_mean = torch.mean(kl_loss)

            loss = doc_loss_mean + pair_weight * pair_loss_mean + kl_weight * kl_loss_mean

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新噪声
            model.update_sigma()

            avg_loss.append(loss.item())
            avg_doc_loss.append(doc_loss_mean.item())
            if pair_loss is not None:
                avg_pair_loss.append(pair_loss_mean.item())
            avg_kl_loss.append(kl_loss_mean.item())

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
            f'Prec: {prec.item():.4f} - Best: {best_precision:.4f} [{best_precision_epoch}]')
        logging.info(
            f'Epoch {epoch+1}/{config["epoch"]} - Loss: {np.mean(avg_loss):.4f} - '
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
