import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from textdata import SingleLabelTextDataset, MultiLabelTextDataset
from utils.utils import set_seed, retrieve_topk, compute_precision_at_k

set_seed()

CURRENT_DIR = Path(__file__).parent
LOG_DIR = CURRENT_DIR / 'logs'
CHECKPOINT_DIR = CURRENT_DIR / 'checkpoints'
LOG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ng20.tfidf')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--bit', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--stop_iter', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--gpu', type=str, default='0')
    return parser


def get_device(config):
    if config['device'] == 'cuda' and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
        device = torch.device('cuda')
        print(f"使用 GPU: {config['gpu']}")
    else:
        device = torch.device('cpu')
        print("使用 CPU")
    return device


class PreEncoder(nn.Module):
    def __init__(self, data_dim, layers, units, bn):
        super(PreEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.Linear(data_dim if len(self.layers) == 0 else units, units))
            if bn:
                self.layers.append(nn.BatchNorm1d(units))
            self.layers.append(nn.ReLU())
        self.output_layer = nn.Linear(units, units)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


class Generator(nn.Module):
    def __init__(self, Nb, data_dim, units, bn):
        super(Generator, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(Nb, units))
        self.output_layer = nn.Linear(units, data_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


class SSB_VAE(nn.Module):
    def __init__(self, vocabSize, latentDim, n_classes, device, bn=True, tau_ann=False, gamma=1.0, multilabel=False):
        super(SSB_VAE, self).__init__()
        self.hidden_dim = 500
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.device = device
        self.tau = Variable(torch.tensor(1.0 if tau_ann else 0.67))
        self.pre_encoder = PreEncoder(self.vocabSize, 2, self.hidden_dim, bn)
        self.generator = Generator(self.latentDim, self.vocabSize, self.hidden_dim, bn)
        self.logits_b_layer = nn.Linear(self.hidden_dim, self.latentDim)
        self.supervised_layer = nn.Linear(self.hidden_dim, n_classes)
        self.multilabel = multilabel
        self.gamma = gamma
        self.margin = self.latentDim / 3.0

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).to(self.device)
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def sampling(self, logits_b):
        b = logits_b + self.sample_gumbel(logits_b.size())
        return torch.sigmoid(b / self.tau)

    def forward(self, x, y_true):
        hidden = self.pre_encoder(x)
        logits_b = self.logits_b_layer(hidden)
        b_sampled = self.sampling(logits_b)
        output = self.generator(b_sampled)
        if self.multilabel:
            supervised_output = torch.sigmoid(self.supervised_layer(hidden))
        else:
            supervised_output = F.softmax(self.supervised_layer(hidden), dim=1)
        return output, supervised_output, b_sampled, logits_b

    def encode(self, x):
        hidden = self.pre_encoder(x)
        logits_b = self.logits_b_layer(hidden)
        return logits_b

    def rec_loss(self, x_true, x_pred):
        x_pred = torch.clamp(x_pred, 1e-9, 1)
        return -torch.sum(x_true * torch.log(x_pred), dim=-1)

    def bkl_loss(self, logits_b):
        p_b = torch.sigmoid(logits_b)
        Nb = p_b.size(1)
        ep = 1e-9
        return Nb * np.log(2) + torch.sum(p_b * torch.log(p_b + ep) + (1 - p_b) * torch.log(1 - p_b + ep), dim=1)

    def hamming_loss(self, y_true, y_pred, b_sampled, n_classes):
        r = torch.sum(b_sampled * b_sampled, 1).view(-1, 1)
        D = r - 2 * torch.mm(b_sampled, b_sampled.t()) + r.t()
        y_true_onehot = torch.nn.functional.one_hot(y_true, num_classes=n_classes).float()
        similar_mask = torch.mm(y_true_onehot, y_true_onehot.t())
        loss_hamming = (1.0 / self.latentDim) * torch.sum(similar_mask *
                                                          D + (1.0 - similar_mask) * F.relu(self.margin - D))
        return self.gamma * F.cross_entropy(y_pred, y_true) + loss_hamming

    def get_binary_code(self, train, test):
        train_zy = [(self.encode(xb.to(self.device)), yb) for xb, yb in train]
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = [(self.encode(xb.to(self.device)), yb) for xb, yb in test]
        test_z, test_y = zip(*test_zy)
        test_z = torch.cat(test_z, dim=0)
        test_y = torch.cat(test_y, dim=0)

        mid_val, _ = torch.median(train_z, dim=0)
        train_b = (train_z > mid_val).type(torch.ByteTensor).to(self.device)
        test_b = (test_z > mid_val).type(torch.ByteTensor).to(self.device)
        del train_z, test_z
        return train_b, test_b, train_y, test_y


def train_val(config):
    device = get_device(config)
    bit = config["bit"]
    dataset, data_fmt = config["dataset"].split('.')
    batch_size = config["batch_size"]

    single_label_flag = dataset not in ['reuters', 'tmc', 'rcv1']
    data_path = Path(__file__).parent.parent.parent / 'textdata'

    if single_label_flag:
        train_set = SingleLabelTextDataset(f'{data_path}/{dataset}', subset='train', bow_format=data_fmt)
        test_set = SingleLabelTextDataset(f'{data_path}/{dataset}', subset='test', bow_format=data_fmt)
    else:
        train_set = MultiLabelTextDataset(f'{data_path}/{dataset}', subset='train', bow_format=data_fmt)
        test_set = MultiLabelTextDataset(f'{data_path}/{dataset}', subset='test', bow_format=data_fmt)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    num_features = train_set[0][0].size(0)
    n_classes = train_set.num_classes()
    model = SSB_VAE(num_features, bit, n_classes, device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    best_precision = 0
    best_precision_epoch = 0
    step_count = 0

    log_file = LOG_DIR / f'data:{config["dataset"]}_bit:{config["bit"]}.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=str(log_file), filemode='w')
    checkpoint_path = CHECKPOINT_DIR / f'dataset:{dataset}_bit:{bit}.pth'

    for epoch in tqdm(range(config["epoch"])):
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)
            output, supervised_output, b_sampled, logits_b = model(xb, yb)
            rec_loss = model.rec_loss(xb, output)
            kl_loss = model.bkl_loss(logits_b)
            loss_hamming = model.hamming_loss(yb, supervised_output, b_sampled, n_classes)
            loss = torch.mean(rec_loss + 0.06250 * kl_loss + loss_hamming)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(test_b, train_b, topK=100)
            prec = compute_precision_at_k(
                retrieved_indices,
                test_y.to(device),
                train_y.to(device),
                topK=100,
                is_single_label=single_label_flag)
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
            f'Epoch {epoch+1}/{config["epoch"]} - Prec: {prec.item():.4f} - Best: {best_precision:.4f} [{best_precision_epoch}]')
        logging.info(
            f'Epoch {epoch+1}/{config["epoch"]} - Prec: {prec:.4f} - Best: {best_precision:.4f} [{best_precision_epoch}]')

    model = torch.load(str(checkpoint_path))
    model.eval()
    with torch.no_grad():
        train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
        retrieved_indices = retrieve_topk(test_b, train_b, topK=100)
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
