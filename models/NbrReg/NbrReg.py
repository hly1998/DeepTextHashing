import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
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
    parser.add_argument('--nn_weight', type=float, default=1.0, help='邻居重建损失权重')
    parser.add_argument('--nn_top_k', type=int, default=20, help='邻居数量')
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
        return self.db[doc_id.item()][:top_k]

    def get_top_k_noisy(self, doc_id, top_k, top_candidates):
        """从候选中随机采样Top-K邻居（带噪声）"""
        candidates = self.db[doc_id][:top_candidates]
        candidates = np.random.permutation(candidates)
        return candidates[:top_k]


class NeighborDataLoader:
    """
    邻居数据加载器
    用于加载训练集数据以获取邻居文档的特征
    """

    def __init__(self, dataset_name, data_path):
        import pandas as pd
        self.train_df = pd.read_pickle(f"{data_path}/{dataset_name}/train.tfidf.df.pkl")
        # 预加载所有训练数据到numpy数组
        self.train = np.array([
            csr_data.toarray()[0] for csr_data in self.train_df.bow.tolist()
        ])

    def get_neighbor_docs(self, doc_indices):
        """获取指定索引的文档特征"""
        return self.train[doc_indices]


class NbrReg(nn.Module):
    """
    NbrReg 模型
    使用邻居正则化的变分自编码器进行文本哈希

    架构:
    - 编码器: 2层全连接 + ReLU + Dropout -> mu, logvar
    - 解码器: 两个分支
        - 原始文档重建: fc -> LogSoftmax
        - 邻居文档重建: fc -> LogSoftmax
    - 损失: 重建损失 + KL散度 + 邻居重建损失
    """

    def __init__(self, dataset, vocab_size, latent_dim, device, dropout_prob=0.):
        super(NbrReg, self).__init__()

        self.dataset = dataset
        self.hidden_dim = 1000
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.device = device

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(self.vocab_size, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob)
        )

        # 隐变量参数
        self.h_to_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.h_to_logvar = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.Sigmoid()
        )

        # 解码器 - 原始文档重建
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.vocab_size),
            nn.LogSoftmax(dim=1)
        )

        # 解码器 - 邻居文档重建
        self.nn_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.vocab_size),
            nn.LogSoftmax(dim=1)
        )

    def encode(self, doc_mat):
        """编码器: 文档 -> (mu, logvar)"""
        h = self.encoder(doc_mat)
        z_mu = self.h_to_mu(h)
        z_logvar = self.h_to_logvar(h)
        return z_mu, z_logvar

    def reparametrize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.sqrt(torch.exp(logvar))
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, document_mat):
        """前向传播"""
        mu, logvar = self.encode(document_mat)
        z = self.reparametrize(mu, logvar)
        prob_w = self.decoder(z)
        nn_prob_w = self.nn_decoder(z)
        return prob_w, nn_prob_w, mu, logvar

    def get_name(self):
        return "NbrReg"

    @staticmethod
    def calculate_KL_loss(mu, logvar):
        """计算KL散度损失"""
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element, dim=1)
        KLD = torch.mean(KLD).mul_(-0.5)
        return KLD

    @staticmethod
    def compute_reconstr_loss(logprob_word, doc_mat):
        """计算重建损失"""
        return -torch.mean(torch.sum(logprob_word * doc_mat, dim=1))

    @staticmethod
    def compute_nn_reconstr_loss(log_word_prob, batch_nn_docs, device):
        """
        计算邻居文档重建损失
        将邻居文档聚合后计算重建损失
        """
        # 聚合邻居文档: 将所有邻居的词向量相加
        # batch_nn_docs: (batch_size, num_neighbors, vocab_size)
        aggregated_nn_docs = np.sum(batch_nn_docs, axis=1)
        aggregated_nn_docs = (aggregated_nn_docs > 0).astype(np.float32)

        nn_loss = torch.tensor(0.0, device=device)

        for doc_idx, nn_doc in enumerate(aggregated_nn_docs):
            word_indices = np.nonzero(nn_doc)[0]
            if len(word_indices) == 0:
                continue
            word_indices = torch.LongTensor(word_indices).to(device)
            pred_logprob = torch.gather(log_word_prob[doc_idx], 0, word_indices)
            nn_loss = nn_loss - torch.sum(pred_logprob)

        return nn_loss / float(len(batch_nn_docs))

    def get_binary_code(self, train, test):
        """生成二进制哈希码"""
        train_zy = [(self.encode(xb.to(self.device))[0], yb) for xb, _, yb in train]
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = [(self.encode(xb.to(self.device))[0], yb) for xb, _, yb in test]
        test_z, test_y = zip(*test_zy)
        test_z = torch.cat(test_z, dim=0)
        test_y = torch.cat(test_y, dim=0)

        # 使用中位数作为阈值
        mid_val, _ = torch.median(train_z, dim=0)
        train_b = (train_z > mid_val).type(torch.ByteTensor).to(self.device)
        test_b = (test_z > mid_val).type(torch.ByteTensor).to(self.device)

        del train_z
        del test_z

        return train_b, test_b, train_y, test_y


def train_val(config):
    device = get_device(config)

    bit = config["bit"]
    dataset, data_fmt = config["dataset"].split('.')
    batch_size = config["batch_size"]
    nn_weight = config["nn_weight"]
    nn_top_k = config["nn_top_k"]

    if dataset in ['reuters', 'tmc', 'rcv1']:
        single_label_flag = False
    else:
        single_label_flag = True

    # 使用新的数据集模块
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

    # 加载邻居文档特征
    neighbor_data = NeighborDataLoader(dataset, str(data_path))

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

    num_features = train_set[0][0].size(0)
    model = NbrReg(dataset, num_features, bit, device, dropout_prob=0.1)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # KL权重退火
    kl_weight = 0.
    kl_step = 1 / 5000.

    # 邻居损失权重退火
    nn_loss_weight = 0.
    nn_loss_step = 1 / 1000.

    best_precision = 0
    prec = 0
    best_precision_epoch = 0
    step_count = 0

    # 日志保存到当前目录
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

        for step, (xb, ids, yb) in enumerate(train_loader):
            xb = xb.to(device)

            # 前向传播
            word_prob, nn_word_prob, mu, logvar = model(xb)

            # KL散度损失
            kl_loss = NbrReg.calculate_KL_loss(mu, logvar)

            # 原始文档重建损失
            reconstr_loss = NbrReg.compute_reconstr_loss(word_prob, xb)

            # 邻居文档重建损失
            batch_nn_docs = []
            for doc_id in ids:
                nn_doc_indices = train_topk_docs_db.get_top_k(doc_id, nn_top_k)
                nn_docs = neighbor_data.get_neighbor_docs(nn_doc_indices)
                batch_nn_docs.append(nn_docs)
            batch_nn_docs = np.stack(batch_nn_docs)
            nn_reconstr_loss = NbrReg.compute_nn_reconstr_loss(nn_word_prob, batch_nn_docs, device)

            # 总损失 = 重建损失 + KL权重 * KL损失 + 邻居权重 * 邻居损失
            loss = reconstr_loss + kl_weight * kl_loss + nn_loss_weight * nn_weight * nn_reconstr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新权重
            kl_weight = min(kl_weight + kl_step, 1.)
            nn_loss_weight = min(nn_loss_weight + nn_loss_step, 1.)
            avg_loss.append(loss.item())

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
            f'Epoch {epoch+1}/{config["epoch"]} - Prec: {prec.item():.4f} - '
            f'Best: {best_precision:.4f} [{best_precision_epoch}]')
        logging.info(
            f'Epoch {epoch+1}/{config["epoch"]} - Prec: {prec:.4f} - '
            f'Best: {best_precision:.4f} [{best_precision_epoch}]')

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
