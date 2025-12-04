"""
MISH: Multi-Index Semantic Hashing
基于论文: Unsupervised Multi-Index Semantic Hashing (WWW 2021)

核心思想:
- 将哈希码分成多个块（block），每个块可以独立索引
- 使用Memory Module存储历史哈希码
- Problem Pair Loss: 惩罚在block内完全匹配但整体不相似的文档对
- Bit Balance Loss: 减少block间比特的相关性
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
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--bit', type=int, default=32)
    parser.add_argument('--block_size', type=int, default=8, help='每个块的比特数')
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--stop_iter', type=int, default=50, help='控制在效果没有提升多少次后停止运行')
    parser.add_argument('--kl_weight', type=float, default=0.0, help='KL散度损失权重')
    parser.add_argument('--problem_pair_weight', type=float, default=1.0, help='问题配对损失权重 (alpha_1)')
    parser.add_argument('--bit_balance_weight', type=float, default=0.1, help='比特平衡损失权重')
    parser.add_argument('--memory_size', type=int, default=5000, help='Memory Module大小')
    parser.add_argument('--top_k', type=int, default=100, help='Top-K阈值')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout概率')
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


class GradientReversal(torch.autograd.Function):
    """梯度反转层"""

    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class MemoryModule:
    """
    Memory Module - 存储历史哈希码
    用于在训练时查找问题配对
    """

    def __init__(self, memory_size, num_bits, device):
        self.memory_size = memory_size
        self.num_bits = num_bits
        self.device = device
        self.ptr = 0
        self.is_full = False

        # 存储哈希码 (使用{-1, 1}格式)
        self.hashcodes = torch.zeros(memory_size, num_bits, dtype=torch.float32, device=device)

    def update(self, hashcodes):
        """更新memory"""
        batch_size = hashcodes.size(0)
        # 转换为{-1, 1}格式，使用 clone() 和 detach() 确保不影响计算图
        hashcodes_signed = (2 * hashcodes - 1).detach().clone()

        if self.ptr + batch_size <= self.memory_size:
            self.hashcodes[self.ptr:self.ptr + batch_size] = hashcodes_signed
            self.ptr += batch_size
        else:
            # 循环覆盖
            remaining = self.memory_size - self.ptr
            self.hashcodes[self.ptr:] = hashcodes_signed[:remaining]
            overflow = batch_size - remaining
            self.hashcodes[:overflow] = hashcodes_signed[remaining:]
            self.ptr = overflow
            self.is_full = True

        if self.ptr >= self.memory_size:
            self.ptr = 0
            self.is_full = True

    def get_valid_memory(self):
        """获取有效的memory内容（返回克隆避免inplace问题）"""
        if self.is_full:
            return self.hashcodes.clone()
        else:
            return self.hashcodes[:self.ptr].clone()


class BitBalanceLoss(nn.Module):
    """
    比特平衡损失 (pred_rev_block类型)

    核心思想：不同block之间的比特应该独立
    使用梯度反转：预测失败说明比特独立
    """

    def __init__(self, num_bits, block_size):
        super().__init__()
        self.num_bits = num_bits
        self.block_size = block_size
        self.num_blocks = num_bits // block_size

        # 为每对block创建预测器
        self.predictors = nn.ModuleDict()
        for i in range(self.block_size):
            for j in range(i + 1, self.block_size):
                key = f"from_{i}_to_{j}"
                self.predictors[key] = nn.Linear(self.num_blocks, self.num_blocks, bias=False)

    def forward(self, sampling_prob):
        """
        计算比特平衡损失

        按照论文的pred_rev_block方式：
        - 将比特重新组织为 block_size 组，每组 num_blocks 个比特
        - 预测一组能否预测另一组
        """
        batch_size = sampling_prob.size(0)

        # 居中
        centered = sampling_prob - 0.5

        # 重新组织：原来是 [B, num_bits]
        # 变成 block_size 组，每组 num_blocks 个比特
        # 第i组包含所有block的第i个比特
        blocks = []
        for i in range(self.block_size):
            # 收集每个block的第i个比特
            indices = [b * self.block_size + i for b in range(self.num_blocks)]
            block = centered[:, indices]  # B x num_blocks
            blocks.append(block)

        losses = []
        for i in range(self.block_size):
            for j in range(i + 1, self.block_size):
                key = f"from_{i}_to_{j}"
                # 使用梯度反转
                from_block = GradientReversal.apply(blocks[i])
                to_block = blocks[j].detach()

                pred = self.predictors[key](from_block)
                loss = F.mse_loss(pred, to_block)
                losses.append(loss)

        return torch.stack(losses).mean()


class MISH(nn.Module):
    """
    MISH 模型
    Multi-Index Semantic Hashing

    核心组件：
    1. 编码器：生成哈希码
    2. 解码器：重建文档
    3. Memory Module：存储历史哈希码
    4. Problem Pair Loss：惩罚block内匹配但整体不相似的文档对
    5. Bit Balance Loss：确保不同block的比特独立
    """

    def __init__(self, dataset, vocab_size, num_bits, block_size, device,
                 memory_size=5000, top_k=100, dropout_prob=0.2, num_layers=2):
        super(MISH, self).__init__()

        self.dataset = dataset
        self.hidden_dim = 1000
        self.vocab_size = vocab_size
        self.num_bits = num_bits
        self.block_size = block_size
        self.num_blocks = num_bits // block_size
        self.device = device
        self.top_k = top_k

        assert num_bits % block_size == 0, "num_bits 必须能被 block_size 整除"

        # 词重要性权重
        self.importance_weight = nn.Parameter(
            torch.FloatTensor(vocab_size).uniform_(0.1, 1.0)
        )

        # 词嵌入矩阵
        self.word_embedding = nn.Parameter(
            torch.FloatTensor(vocab_size, num_bits).uniform_(-1, 1)
        )

        # 解码偏置
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))

        # 编码器
        encoder_layers = []
        input_dim = vocab_size
        for i in range(num_layers):
            output_dim = self.hidden_dim // (i + 1) if i > 0 else self.hidden_dim
            encoder_layers.append(nn.Linear(input_dim, output_dim))
            encoder_layers.append(nn.ReLU(inplace=False))
            input_dim = output_dim

        encoder_layers.append(nn.Dropout(p=dropout_prob))
        encoder_layers.append(nn.Linear(input_dim, num_bits))
        encoder_layers.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*encoder_layers)

        # Memory Module
        self.memory = MemoryModule(memory_size, num_bits, device)

        # Bit Balance Loss模块
        self.bit_balance = BitBalanceLoss(num_bits, block_size)

        # 噪声退火
        self.sigma = 1.0
        self.sigma_decay = 1e-6

    def encode(self, doc_mat):
        """编码器"""
        weighted_doc = doc_mat * self.importance_weight
        sampling_prob = self.encoder(weighted_doc)
        return sampling_prob

    def sample_hashcode(self, sampling_prob, training=True):
        """伯努利采样"""
        return BernoulliStraightThrough.apply(sampling_prob, training)

    def decode(self, hashcode, target_doc):
        """解码器"""
        # 转为{-1, 1}
        hashcode_signed = 2 * hashcode - 1

        # 添加噪声
        if self.training and self.sigma > 0:
            noise = torch.randn_like(hashcode_signed) * self.sigma
            noisy_hashcode = hashcode_signed + noise
        else:
            noisy_hashcode = hashcode_signed

        # 计算词概率
        logits = torch.matmul(noisy_hashcode, self.word_embedding.T) * self.importance_weight + self.decoder_bias
        log_probs = F.log_softmax(logits, dim=-1)

        # 重建损失
        mask = (target_doc > 0).float()
        recon_loss = -torch.sum(log_probs * mask, dim=-1)

        return recon_loss

    def compute_kl_loss(self, sampling_prob):
        """KL散度损失"""
        eps = 1e-10
        kl = sampling_prob * torch.log(torch.clamp(sampling_prob / 0.5, min=eps)) + \
            (1 - sampling_prob) * torch.log(torch.clamp((1 - sampling_prob) / 0.5, min=eps))
        return torch.sum(kl, dim=-1)

    def find_problematic_pairs(self, hashcodes):
        """
        找到问题配对并计算损失

        问题配对定义：
        - 在某个block内完全匹配（距离=0）
        - 但整体相似度不在top-k内

        损失：惩罚这些配对的block内相似度，使它们不再完全匹配
        """
        memory = self.memory.get_valid_memory()
        if memory.size(0) < self.top_k:
            return torch.tensor(0.0, device=self.device), 0

        batch_size = hashcodes.size(0)
        mem_size = memory.size(0)

        # 转为{-1, 1}
        hashcodes_signed = 2 * hashcodes - 1  # B x num_bits

        # 计算整体相似度 (点积，范围 [-num_bits, num_bits])
        overall_sim = torch.matmul(hashcodes_signed, memory.T)  # B x mem_size

        # 找到top-k的阈值
        top_k = min(self.top_k, mem_size)
        top_k_values, _ = torch.topk(overall_sim, top_k, dim=1)
        threshold = top_k_values[:, -1:].detach()  # B x 1

        # 不在top-k内的mask
        not_in_topk = (overall_sim < threshold).float()  # B x mem_size

        losses = []
        num_valid_pairs = 0

        # 对每个block检查
        for block_idx in range(self.num_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size

            # 提取当前block
            query_block = hashcodes_signed[:, start:end]  # B x block_size
            memory_block = memory[:, start:end]  # mem_size x block_size

            # 计算block内相似度
            block_sim = torch.matmul(query_block, memory_block.T)  # B x mem_size

            # 完全匹配 = block_size (所有比特都一样)
            perfect_match = (block_sim == self.block_size).float()

            # 问题配对：block内完全匹配 但 整体不在top-k
            problem_pairs = perfect_match * not_in_topk  # B x mem_size

            if problem_pairs.sum() > 0:
                # 损失：惩罚block内相似度（希望减少相似度，不再完全匹配）
                pair_loss = (block_sim * problem_pairs).sum()
                losses.append(pair_loss)
                num_valid_pairs += problem_pairs.sum().item()

        if len(losses) > 0 and num_valid_pairs > 0:
            total_loss = torch.stack(losses).sum() / num_valid_pairs
        else:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        return total_loss, num_valid_pairs

    def compute_bit_balance_loss(self, sampling_prob):
        """比特平衡损失"""
        return self.bit_balance(sampling_prob)

    def forward(self, doc):
        """前向传播"""
        # 编码
        sampling_prob = self.encode(doc)
        hashcode = self.sample_hashcode(sampling_prob, self.training)

        # 重建损失
        recon_loss = self.decode(hashcode, doc)

        # KL散度
        kl_loss = self.compute_kl_loss(sampling_prob)

        # 问题配对损失
        problem_pair_loss, num_pairs = self.find_problematic_pairs(hashcode)

        # 比特平衡损失
        bit_balance_loss = self.compute_bit_balance_loss(sampling_prob)

        # 更新memory
        if self.training:
            self.memory.update(hashcode)

        return {
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'problem_pair_loss': problem_pair_loss,
            'bit_balance_loss': bit_balance_loss,
            'num_problem_pairs': num_pairs,
            'hashcode': hashcode,
            'sampling_prob': sampling_prob
        }

    def get_name(self):
        return "MISH"

    def update_sigma(self):
        """更新噪声标准差"""
        self.sigma = max(self.sigma - self.sigma_decay, 0)

    def get_binary_code(self, train_loader, test_loader):
        """生成二进制哈希码"""
        self.eval()

        train_codes = []
        train_labels = []
        for xb, yb in train_loader:
            xb = xb.to(self.device)
            prob = self.encode(xb)
            code = (prob > 0.5).cpu()
            train_codes.append(code)
            train_labels.append(yb)

        train_codes = torch.cat(train_codes, dim=0)
        train_labels = torch.cat(train_labels, dim=0)

        test_codes = []
        test_labels = []
        for xb, yb in test_loader:
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
    block_size = config["block_size"]
    dataset, data_fmt = config["dataset"].split('.')
    batch_size = config["batch_size"]

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
    model = MISH(
        dataset, num_features, bit, block_size, device,
        memory_size=config["memory_size"],
        top_k=config["top_k"],
        dropout_prob=config["dropout"]
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    kl_weight = config["kl_weight"]
    problem_pair_weight = config["problem_pair_weight"]
    bit_balance_weight = config["bit_balance_weight"]

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
        avg_recon = []
        avg_pp = []
        avg_bb = []
        total_problem_pairs = 0

        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)

            # 前向传播
            output = model(xb)

            recon_loss = output['recon_loss'].mean()
            kl_loss = output['kl_loss'].mean()
            problem_pair_loss = output['problem_pair_loss']
            bit_balance_loss = output['bit_balance_loss']
            total_problem_pairs += output['num_problem_pairs']

            # 总损失
            loss = recon_loss + kl_weight * kl_loss + \
                problem_pair_weight * problem_pair_loss + \
                bit_balance_weight * bit_balance_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新噪声
            model.update_sigma()

            avg_loss.append(loss.item())
            avg_recon.append(recon_loss.item())
            avg_pp.append(
                problem_pair_loss.item() if isinstance(
                    problem_pair_loss,
                    torch.Tensor) else problem_pair_loss)
            avg_bb.append(bit_balance_loss.item())

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
            f'Recon: {np.mean(avg_recon):.4f} - PP: {np.mean(avg_pp):.4f} ({total_problem_pairs:.0f}) - '
            f'BB: {np.mean(avg_bb):.4f} - Prec: {prec.item():.4f} - Best: {best_precision:.4f} [{best_precision_epoch}]')
        logging.info(
            f'Epoch {epoch+1}/{config["epoch"]} - Loss: {np.mean(avg_loss):.4f} - '
            f'PP: {np.mean(avg_pp):.4f} - Prec: {prec:.4f} - Best: {best_precision:.4f} [{best_precision_epoch}]')

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
