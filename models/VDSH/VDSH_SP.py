import argparse
import logging
import os
from pathlib import Path

import torch
from torch.autograd import Variable
import torch.nn as nn
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
    parser.add_argument('--bit', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--stop_iter', type=int, default=50, help='控制在效果没有提升多少次后停止运行')
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


class VDSH_SP(nn.Module):

    def __init__(self, dataset, vocabSize, latentDim, num_classes, device, dropoutProb=0.):
        super(VDSH_SP, self).__init__()

        self.dataset = dataset
        self.hidden_dim = 1000
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.dropoutProb = dropoutProb
        self.num_classes = num_classes
        self.device = device

        self.encoder = nn.Sequential(nn.Linear(self.vocabSize, self.hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=dropoutProb))

        self.h_to_mu = nn.Linear(self.hidden_dim, self.latentDim)
        self.h_to_logvar = nn.Sequential(nn.Linear(self.hidden_dim, self.latentDim),
                                         nn.Sigmoid())

        self.decoder = nn.Sequential(nn.Linear(self.latentDim, self.vocabSize),
                                     nn.LogSoftmax(dim=1))

        self.pred = nn.Sequential(nn.Linear(self.latentDim, self.num_classes))
        self.pred_loss = nn.CrossEntropyLoss()

        self.enc_v = nn.Sequential(nn.Linear(self.vocabSize, self.hidden_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(self.hidden_dim, self.hidden_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=dropoutProb))

        self.h_to_v_mu = nn.Linear(self.hidden_dim, self.latentDim)
        self.h_to_v_logvar = nn.Sequential(nn.Linear(self.hidden_dim, self.latentDim),
                                           nn.Sigmoid())

    def encode(self, doc_mat):
        h = self.encoder(doc_mat)
        z_mu = self.h_to_mu(h)
        z_logvar = self.h_to_logvar(h)

        h2 = self.enc_v(doc_mat)
        v_mu = self.h_to_v_mu(h2)
        v_logvar = self.h_to_v_logvar(h2)

        return z_mu, z_logvar, v_mu, v_logvar

    def reparametrize(self, mu, logvar):
        std = torch.sqrt(torch.exp(logvar))
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, document_mat):
        z_mu, z_logvar, v_mu, v_logvar = self.encode(document_mat)
        z = self.reparametrize(z_mu, z_logvar)
        v = self.reparametrize(v_mu, v_logvar)
        logprob_w = self.decoder(z + v)
        score_c = self.pred(z)
        return logprob_w, score_c, z_mu, z_logvar, v_mu, v_logvar

    def get_name(self):
        return "VDSH_SP"

    @staticmethod
    def calculate_KL_loss(mu, logvar):
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element, dim=1)
        KLD = torch.mean(KLD).mul_(-0.5)
        return KLD

    @staticmethod
    def compute_reconstr_loss(logprob_word, doc_mat):
        return -torch.mean(torch.sum(logprob_word * doc_mat, dim=1))

    def compute_prediction_loss(self, scores, labels):
        return self.pred_loss(scores, labels)

    def get_binary_code(self, train, test):
        train_zy = [(self.encode(xb.to(self.device))[0], yb) for xb, yb in train]
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = [(self.encode(xb.to(self.device))[0], yb) for xb, yb in test]
        test_z, test_y = zip(*test_zy)
        test_z = torch.cat(test_z, dim=0)
        test_y = torch.cat(test_y, dim=0)

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

    if dataset in ['reuters', 'tmc', 'rcv1']:
        single_label_flag = False
    else:
        single_label_flag = True

    # 使用新的数据集模块
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
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

    num_features = train_set[0][0].size(0)
    y_dim = train_set.num_classes()
    model = VDSH_SP(dataset, num_features, bit, y_dim, device, dropoutProb=0.1)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    kl_weight = 0.
    kl_step = 1 / 5000.
    pred_weight = 0.
    pred_weight_step = 1 / 1000.

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
        avg_loss = []
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            logprob_w, score_c, z_mu, z_logvar, v_mu, v_logvar = model(xb)
            z_kl_loss = VDSH_SP.calculate_KL_loss(z_mu, z_logvar)
            v_kl_loss = VDSH_SP.calculate_KL_loss(v_mu, v_logvar)
            reconstr_loss = VDSH_SP.compute_reconstr_loss(logprob_w, xb)

            if single_label_flag:
                pred_loss = model.compute_prediction_loss(score_c, yb)
            else:
                if len(yb.size()) == 1:
                    y_onehot = torch.zeros((xb.size(0), y_dim)).to(device)
                    y_onehot = y_onehot.scatter_(1, yb.unsqueeze(1), 1)
                    pred_loss = model.compute_prediction_loss(score_c, y_onehot)
                else:
                    pred_loss = model.compute_prediction_loss(score_c, yb)

            loss = reconstr_loss + (kl_weight * z_kl_loss) + (kl_weight * v_kl_loss) + pred_weight * pred_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            kl_weight = min(kl_weight + kl_step, 1.)
            pred_weight = min(pred_weight + pred_weight_step, 150.)
            avg_loss.append(loss.item())

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
