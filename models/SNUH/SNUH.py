"""
SNUH: Semantic-Neighborhood Unified Hashing
基于论文: Integrating Semantics and Neighborhood Information with Graph-Driven Generative Models for Document Retrieval

核心思想:
1. 使用变分自编码器学习文档的潜在表示
2. 构建文档邻居图,利用邻居信息来约束潜在空间
3. KL散度损失分为两部分:
   - 节点KL: 标准VAE的KL散度
   - 边KL: 基于图结构的边KL散度,考虑邻居之间的相关性
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
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--bit', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--stop_iter', type=int, default=10, help='控制在效果没有提升多少次后停止运行')

    # SNUH特有参数
    parser.add_argument('--hidden_dim', type=int, default=500, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=0, help='FF网络层数')
    parser.add_argument('--num_neighbors', type=int, default=10, help='邻居数量')
    parser.add_argument('--num_trees', type=int, default=10, help='生成树数量')
    parser.add_argument('--alpha', type=float, default=0.1, help='邻居采样温度')
    parser.add_argument('--beta', type=float, default=0.05, help='KL损失权重')
    parser.add_argument('--temperature', type=float, default=0.1, help='二值化温度')
    parser.add_argument('--tau', type=float, default=0.99, help='先验相关系数')
    parser.add_argument('--clip', type=float, default=10, help='梯度裁剪')

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


class FF(nn.Module):
    """
    前馈神经网络模块
    支持可变层数的全连接网络
    """

    def __init__(self, dim_input, dim_hidden, dim_output, num_layers,
                 activation='relu', dropout_rate=0):
        super().__init__()

        assert num_layers >= 0  # 0 = 直接线性变换
        if num_layers > 0:
            assert dim_hidden > 0

        self.stack = nn.ModuleList()
        for l in range(num_layers):
            layer = []
            layer.append(nn.Linear(dim_input if l == 0 else dim_hidden, dim_hidden))
            layer.append({'tanh': nn.Tanh(), 'relu': nn.ReLU()}[activation])
            if dropout_rate > 0:
                layer.append(nn.Dropout(dropout_rate))
            self.stack.append(nn.Sequential(*layer))

        self.out = nn.Linear(dim_input if num_layers < 1 else dim_hidden, dim_output)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return self.out(x)


class VarEncoder(nn.Module):
    """
    变分编码器
    输入文档向量,输出高斯分布的均值和标准差
    使用temperature控制sigmoid的锐度,促进二值化
    """

    def __init__(self, dim_input, dim_hidden, dim_output, num_layers):
        super().__init__()
        self.dim_output = dim_output
        self.ff = FF(dim_input, dim_hidden, 2 * dim_output, num_layers)

    def forward(self, x, temperature):
        gaussian_params = self.ff(x)
        # mu使用sigmoid激活,temperature越小越接近二值
        mu = torch.sigmoid(gaussian_params[:, :self.dim_output] / temperature)
        # sigma使用softplus确保正值
        sigma = F.softplus(gaussian_params[:, self.dim_output:])
        return mu, sigma


class CorrEncoder(nn.Module):
    """
    相关性编码器
    输入两个文档向量,输出它们的相关系数
    用于计算边的KL散度
    """

    def __init__(self, dim_input, dim_hidden, dim_output, num_layers):
        super().__init__()
        self.dim_output = dim_output
        # 输入是两个文档的拼接
        self.ff = FF(2 * dim_input, dim_hidden, dim_output, num_layers)

    def forward(self, x1, x2):
        # 对称化: 同时处理 (x1, x2) 和 (x2, x1)
        net = torch.cat([torch.cat([x1, x2], dim=1), torch.cat([x2, x1], dim=1)], dim=0)
        corr_params = self.ff(net).reshape([2, -1, self.dim_output])
        # 取平均以确保对称性
        corr_params = (corr_params[0] + corr_params[1]) / 2.0
        # 相关系数范围 (-1, 1)
        correlation_coefficient = (1. - 1e-8) * (2. * torch.sigmoid(corr_params) - 1.)
        return correlation_coefficient


class Decoder(nn.Module):
    """
    解码器
    使用嵌入层将潜在编码映射回词汇空间
    """

    def __init__(self, dim_encoding, vocab_size):
        super().__init__()
        self.E = nn.Embedding(dim_encoding, vocab_size)
        self.b = nn.Parameter(torch.zeros(1, vocab_size))

    def forward(self, Z, targets):
        # Z: (B x m), targets: (B x V binary)
        scores = Z @ self.E.weight + self.b  # B x V
        log_probs = scores.log_softmax(dim=1)
        log_likelihood = (log_probs * targets).sum(1).mean()
        return log_likelihood


class SNUH(nn.Module):
    """
    SNUH模型

    架构:
    - VarEncoder: 文档 -> (mu, sigma)
    - CorrEncoder: (doc1, doc2) -> 相关系数
    - Decoder: 潜在编码 -> 文档重建

    损失:
    - log_likelihood: 重建似然
    - kl_node: 节点KL散度 (与标准高斯先验)
    - kl_edge: 边KL散度 (考虑邻居相关性)
    """

    def __init__(self, vocab_size, hidden_dim, latent_dim, num_layers,
                 temperature, tau, device):
        super(SNUH, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.temperature = temperature
        self.tau = torch.tensor([tau]).to(device)
        self.device = device

        # 编码器
        self.venc = VarEncoder(vocab_size, hidden_dim, latent_dim, num_layers)
        self.cenc = CorrEncoder(vocab_size, hidden_dim, latent_dim, num_layers)

        # 解码器
        self.dec = Decoder(latent_dim, vocab_size)

        # 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self, m, init_value=0.05):
        if init_value > 0.:
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight.data.uniform_(-init_value, init_value)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.fill_(0.)

    def forward(self, X, edge1, edge2, weight, num_node, num_edges):
        """
        前向传播

        Args:
            X: 当前batch的文档 (B x V)
            edge1: 边的第一个节点的文档 (num_edges_in_batch x V)
            edge2: 边的第二个节点的文档 (num_edges_in_batch x V)
            weight: 边的权重 (num_edges_in_batch,)
            num_node: 总节点数
            num_edges: 总边数
        """
        # 编码
        q_mu, q_sigma = self.venc(X, self.temperature)

        # 重参数化
        eps = torch.randn_like(q_mu)
        Z_st = q_mu + q_sigma * eps

        # 重建似然
        log_likelihood = self.dec(Z_st, X.sign())

        # KL散度
        kl = self.compute_kl(q_mu, q_sigma, edge1, edge2, weight, num_node, num_edges)

        return log_likelihood, kl

    def compute_kl(self, q_mu, q_sigma, edge1, edge2, weight, num_node, num_edges):
        """
        计算KL散度

        包含两部分:
        1. kl_node: 节点KL散度 - 与标准高斯先验的KL散度
        2. kl_edge: 边KL散度 - 利用邻居图结构的相关性约束
        """
        # 对边的两个端点进行编码
        q_mu1, q_sigma1 = self.venc(edge1, self.temperature)
        q_mu2, q_sigma2 = self.venc(edge2, self.temperature)

        # 节点KL散度: KL(q(z|x) || N(0,1))
        kl_node = torch.mean(torch.sum(
            q_mu**2 + q_sigma**2 - 1 - 2 * torch.log(q_sigma + 1e-8),
            dim=1
        ))

        # 边的相关系数 (通过CorrEncoder学习)
        gamma = self.cenc(edge1, edge2)

        # 边KL散度: 考虑邻居之间的相关性
        # 先验假设邻居的潜在变量之间有相关性 tau
        kl_edge = torch.mean(torch.sum(
            0.5 * (q_mu1**2 + q_mu2**2 + q_sigma1**2 + q_sigma2**2
                   - 2 * self.tau * gamma * q_sigma1 * q_sigma2
                   - 2 * self.tau * q_mu1 * q_mu2) / (1 - self.tau**2)
            - 0.5 * (q_mu1**2 + q_mu2**2 + q_sigma1**2 + q_sigma2**2)
            - 0.5 * torch.log(1 - gamma**2 + 1e-8)
            + 0.5 * torch.log(1 - self.tau**2),
            dim=1
        ) * weight)

        # 加权组合
        return kl_node + kl_edge * num_edges / num_node

    def encode(self, X):
        """编码文档为潜在表示"""
        mu, _ = self.venc(X, self.temperature)
        return mu

    def get_binary_code(self, train_loader, test_loader):
        """
        生成二进制哈希码
        使用训练集的中位数作为阈值
        """
        self.eval()
        with torch.no_grad():
            # 训练集编码
            train_z_list = []
            train_y_list = []
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                z = self.encode(xb)
                train_z_list.append(z)
                train_y_list.append(yb)
            train_z = torch.cat(train_z_list, dim=0)
            train_y = torch.cat(train_y_list, dim=0)

            # 测试集编码
            test_z_list = []
            test_y_list = []
            for xb, yb in test_loader:
                xb = xb.to(self.device)
                z = self.encode(xb)
                test_z_list.append(z)
                test_y_list.append(yb)
            test_z = torch.cat(test_z_list, dim=0)
            test_y = torch.cat(test_y_list, dim=0)

            # 使用中位数作为阈值
            mid_val, _ = torch.median(train_z, dim=0)
            train_b = (train_z > mid_val).type(torch.ByteTensor).to(self.device)
            test_b = (test_z > mid_val).type(torch.ByteTensor).to(self.device)

        return train_b, test_b, train_y, test_y


class NeighborGraph:
    """
    邻居图构建器
    使用cosine相似度找邻居,然后用spanning tree采样边
    """

    def __init__(self, X_train, num_neighbors):
        """
        Args:
            X_train: 训练集文档 (N x V)
            num_neighbors: 每个文档的邻居数量
        """
        self.X_train = X_train
        self.num_neighbors = num_neighbors
        self.num_nodes = X_train.shape[0]

        # 计算邻居
        self._compute_neighbors()

    def _compute_neighbors(self):
        """使用cosine相似度计算每个文档的Top-K邻居"""
        print("计算文档邻居...")
        documents = self.X_train.clone()

        # 归一化
        documents = documents / (torch.norm(documents, p=2, dim=-1, keepdim=True) + 1e-8)

        # 计算cosine相似度
        cos_sim_scores = torch.mm(documents, documents.T)

        # 获取Top-K邻居 (排除自身)
        scores, indices = torch.topk(cos_sim_scores, self.num_neighbors + 1, dim=1, largest=True)
        self.topK_scores = scores[:, 1:]  # 排除自身
        self.topK_indices = indices[:, 1:]

        print(f"邻居计算完成, 每个文档有 {self.num_neighbors} 个邻居")

    def get_spanning_trees(self, num_trees, alpha):
        """
        生成多棵生成树并合并边

        Args:
            num_trees: 生成树数量
            alpha: 邻居采样温度

        Returns:
            edges: (num_edges x 3) 数组, 每行是 [node1, node2, weight]
        """
        edges_indices = self.topK_indices.cpu().numpy()
        edges_scores = torch.softmax(self.topK_scores / alpha, dim=-1).cpu().numpy()

        N = self.num_nodes
        w_m = {}  # 边到权重的映射

        for _ in range(num_trees):
            visited = np.array([False for i in range(N)])
            while False in visited:
                # 从未访问节点开始
                init_node = np.random.choice(np.where(visited == False)[0], 1)[0]
                visited[init_node] = True
                queue = [init_node]

                while len(queue) > 0:
                    now = queue[0]
                    visited[now] = True

                    # 找未访问的邻居
                    edge_idx = np.where(visited[edges_indices[now]] == False)[0]
                    if len(edge_idx) == 0:
                        queue.pop(-1)
                        break

                    # 按概率采样下一个邻居
                    probs = edges_scores[now][edge_idx] / np.sum(edges_scores[now][edge_idx])
                    next_node = np.random.choice(edges_indices[now][edge_idx], 1, p=probs)[0]
                    visited[next_node] = True
                    queue.append(next_node)

                    # 记录边
                    edge_key = now * N + next_node
                    if edge_key not in w_m:
                        w_m[edge_key] = 1
                    else:
                        w_m[edge_key] += 1

        # 转换为数组: [node1, node2, weight]
        edges = [[key // N, key % N, val / num_trees] for key, val in w_m.items()]
        np.random.shuffle(edges)
        return np.array(edges)


class TrainDataset(torch.utils.data.Dataset):
    """训练数据集,包含文档和边信息"""

    def __init__(self, data, labels, edges):
        self.data = data
        self.labels = labels
        self.edges = edges
        self.edge_idx = 0

    def __getitem__(self, index):
        if self.edge_idx >= len(self.edges):
            self.edge_idx = 0

        text = self.data[index]
        labels = self.labels[index]
        edge1 = self.data[int(self.edges[self.edge_idx][0])]
        edge2 = self.data[int(self.edges[self.edge_idx][1])]
        weight = self.edges[self.edge_idx][2]
        self.edge_idx += 1

        return text, labels, edge1, edge2, weight

    def __len__(self):
        return len(self.data)


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
    else:
        train_set = MultiLabelTextDataset(f'{data_path}/{dataset}', subset='train', bow_format=data_fmt)
        test_set = MultiLabelTextDataset(f'{data_path}/{dataset}', subset='test', bow_format=data_fmt)
        val_set = MultiLabelTextDataset(f'{data_path}/{dataset}', subset='cv', bow_format=data_fmt)

    # 获取所有训练数据用于构建邻居图
    print("加载训练数据...")
    X_train = torch.stack([train_set[i][0] for i in range(len(train_set))])

    # 处理标签：单标签转换为 one-hot，多标签已经是张量
    if single_label_flag:
        num_classes = train_set.num_classes()
        labels = [train_set[i][1] for i in range(len(train_set))]
        Y_train = torch.zeros(len(train_set), num_classes)
        for i, label in enumerate(labels):
            Y_train[i, label] = 1
    else:
        Y_train = torch.stack([train_set[i][1] for i in range(len(train_set))])

    # 构建邻居图
    neighbor_graph = NeighborGraph(X_train, config['num_neighbors'])
    edges = neighbor_graph.get_spanning_trees(config['num_trees'], config['alpha'])
    num_nodes = X_train.shape[0]
    num_edges = edges.shape[0]

    print(f"节点数: {num_nodes}, 边数: {num_edges}")

    # 创建数据加载器
    train_dataset = TrainDataset(X_train, Y_train, edges)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    # 用于评估的数据加载器 (不包含边信息)
    eval_train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=batch_size, shuffle=False
    )

    # 创建模型
    num_features = train_set[0][0].size(0)
    model = SNUH(
        vocab_size=num_features,
        hidden_dim=config['hidden_dim'],
        latent_dim=bit,
        num_layers=config['num_layers'],
        temperature=config['temperature'],
        tau=config['tau'],
        device=device
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

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
        total_ll = 0
        total_kl = 0
        num_batches = 0

        for batch in train_loader:
            X, _, edge1, edge2, weight = batch
            X = X.to(device)
            edge1 = edge1.to(device)
            edge2 = edge2.to(device)
            weight = weight.float().to(device)

            optimizer.zero_grad()

            # 前向传播
            log_likelihood, kl = model(X, edge1, edge2, weight, num_nodes, num_edges)

            # 损失 = -似然 + beta * KL
            loss = -log_likelihood + config['beta'] * kl

            if torch.isnan(loss):
                print("警告: 损失为NaN, 跳过该batch")
                continue

            loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), config['clip'])

            optimizer.step()

            total_loss += loss.item()
            total_ll += log_likelihood.item()
            total_kl += kl.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_ll = total_ll / max(num_batches, 1)
        avg_kl = total_kl / max(num_batches, 1)

        # 验证
        model.eval()
        with torch.no_grad():
            train_b, val_b, train_y, val_y = model.get_binary_code(eval_train_loader, val_loader)
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
            f'LL: {avg_ll:.4f} - KL: {avg_kl:.4f} - '
            f'Prec: {prec.item():.4f} - Best: {best_precision:.4f} [{best_precision_epoch}]'
        )
        logging.info(
            f'Epoch {epoch+1}/{config["epoch"]} - Loss: {avg_loss:.4f} - '
            f'LL: {avg_ll:.4f} - KL: {avg_kl:.4f} - '
            f'Prec: {prec:.4f} - Best: {best_precision:.4f} [{best_precision_epoch}]'
        )

        if bad_epochs >= config["stop_iter"]:
            print(f"提前停止: 连续 {bad_epochs} 个epoch没有提升")
            break

    # 加载最佳模型进行测试
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    with torch.no_grad():
        train_b, test_b, train_y, test_y = model.get_binary_code(eval_train_loader, test_loader)
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
