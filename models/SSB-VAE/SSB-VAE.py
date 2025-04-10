import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.nn import Parameter
from tqdm import tqdm
from utils.utils import *
from utils.datasets import *

set_seed()

def get_config():
    config = {
        "dataset": "ng20.tfidf",
        "bit": 8,
        "dropout": 0.1,
        "batch_size": 64,
        "epoch": 300,
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 0.001}},
        "stop_iter": 10
    }
    return config

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

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def sampling(logits_b, tau):
    b = logits_b + sample_gumbel(logits_b.size()).cuda()
    return torch.sigmoid(b / tau)

class SSB_VAE(nn.Module):
    def __init__(self, vocabSize, latentDim, n_classes, bn=True, summ=True, tau_ann=False, beta=0, alpha=1.0, gamma=1.0, multilabel=False):
        super(SSB_VAE, self).__init__()
        self.hidden_dim = 500
        self.vocabSize = vocabSize
        self.latentDim = latentDim

        self.tau = Variable(torch.tensor(1.0 if tau_ann else 0.67))
        self.pre_encoder = PreEncoder(self.vocabSize, 2, self.hidden_dim, bn)
        self.generator = Generator(self.latentDim, self.vocabSize, self.hidden_dim, bn)
        self.logits_b_layer = nn.Linear(self.hidden_dim, self.latentDim)
        self.supervised_layer = nn.Linear(self.hidden_dim, n_classes)
        self.multilabel = multilabel
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.margin = self.latentDim / 3.0

    def forward(self, x, y_true):
        hidden = self.pre_encoder(x)
        logits_b = self.logits_b_layer(hidden)
        b_sampled = sampling(logits_b, self.tau)
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

    def hamming_loss(self, y_true, y_pred, b_sampled):
        pred_loss = F.binary_cross_entropy_with_logits if self.multilabel else F.cross_entropy
        r = torch.sum(b_sampled * b_sampled, 1).view(-1, 1)
        D = r - 2 * torch.mm(b_sampled, b_sampled.t()) + r.t()
        y_true = torch.nn.functional.one_hot(y_true, num_classes=20).float()
        similar_mask = torch.mm(y_true, y_true.t())
        loss_hamming = (1.0 / self.latentDim) * torch.sum(similar_mask * D + (1.0 - similar_mask) * F.relu(self.margin - D))
        return self.gamma * pred_loss(y_true, y_pred) + loss_hamming

    def get_binary_code(self, train, test):
        train_zy = [(self.encode(xb.cuda()), yb) for xb, yb in train]
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = [(self.encode(xb.cuda()), yb) for xb, yb in test]
        test_z, test_y = zip(*test_zy)
        test_z = torch.cat(test_z, dim=0)
        test_y = torch.cat(test_y, dim=0)

        mid_val, _ = torch.median(train_z, dim=0)
        train_b = (train_z > mid_val).type(torch.cuda.ByteTensor)
        test_b = (test_z > mid_val).type(torch.cuda.ByteTensor)

        del train_z
        del test_z

        return train_b, test_b, train_y, test_y


def train_val(config):
    bit = config["bit"]
    dataset, data_fmt = config["dataset"].split('.')
    batch_size = config["batch_size"]

    if dataset in ['reuters', 'tmc', 'rcv1']:
        single_label_flag = False
    else:
        single_label_flag = True

    absolute_path = os.path.abspath(os.getcwd())
    data_path = absolute_path + '/../../datasets'
    if single_label_flag:
        train_set = SingleLabelTextDataset('{}/{}'.format(data_path, dataset), subset='train', bow_format=data_fmt, download=True)
        test_set = SingleLabelTextDataset('{}/{}'.format(data_path, dataset), subset='test', bow_format=data_fmt, download=True)
    else:
        train_set = MultiLabelTextDataset('{}/{}'.format(data_path, dataset), subset='train', bow_format=data_fmt, download=True)
        test_set = MultiLabelTextDataset('{}/{}'.format(data_path, dataset), subset='test', bow_format=data_fmt, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    num_features = train_set[0][0].size(0)
    model = SSB_VAE(num_features, bit, n_classes=20)
    model.cuda()
    optimizer = config["optimizer"]["type"](model.parameters(), lr=config["optimizer"]["optim_params"]["lr"])


    best_precision = 0
    prec = 0 
    best_precision_epoch = 0

    step_count = 0
    for epoch in tqdm(range(config["epoch"])):
        avg_loss = []
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.cuda()
            yb = yb.cuda()

            output, supervised_output, b_sampled, logits_b = model(xb, yb)
            rec_loss = model.rec_loss(xb, output)
            kl_loss = model.bkl_loss(logits_b)
            loss_hamming = model.hamming_loss(yb, supervised_output, b_sampled)
            
            loss = torch.mean(rec_loss + 0.06250 * kl_loss + loss_hamming)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())
            
        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(test_b.cuda(), train_b.cuda(), topK=100)
            prec = compute_precision_at_k(retrieved_indices, test_y.cuda(), train_y.cuda(), topK=100, is_single_label=single_label_flag)
            if prec.item() > best_precision:
                best_precision = prec.item()
                best_precision_epoch = epoch + 1
                step_count = 0
            else:
                step_count += 1
            if step_count >= config["stop_iter"]:
                break
        tqdm.write(f'Epoch {epoch+1}/{config["epoch"]} - Current Precision: {prec.item():.4f} - Best Precision: {best_precision:.4f} [{best_precision_epoch}]')

if __name__ == "__main__":
    config = get_config()
    train_val(config)
