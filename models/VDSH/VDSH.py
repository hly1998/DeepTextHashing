import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils.utils import *
from utils.datasets import *
import argparse
import logging

set_seed()

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ng20.tfidf')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--bit', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--stop_iter', type=int, default=50, help='控制在效果没有提升多少次后停止运行')
    return parser

# def get_config():
#     config = {
#         "dataset": args.dataset,
#         "bit": args.bit,
#         "dropout": 0.1,
#         "batch_size": args.batch_size,
#         "epoch": args.epoch,
#         "optimizer": {"type": optim.Adam, "optim_params": {"lr": args.lr}},
#         "stop_iter": args.stop_iter,
#     }
#     return config

class VDSH(nn.Module):
    
    def __init__(self, dataset, vocabSize, latentDim, dropoutProb=0.):
        super(VDSH, self).__init__()
        
        self.dataset = dataset
        self.hidden_dim = 1000
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.dropoutProb = dropoutProb
        
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
        
    def encode(self, doc_mat):
        h = self.encoder(doc_mat)
        z_mu = self.h_to_mu(h)
        z_logvar = self.h_to_logvar(h)
        return z_mu, z_logvar
        
    def reparametrize(self, mu, logvar):
        std = torch.sqrt(torch.exp(logvar))
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def forward(self, document_mat):
        mu, logvar = self.encode(document_mat)
        z = self.reparametrize(mu, logvar)
        prob_w = self.decoder(z)
        return prob_w, mu, logvar
    
    def get_name(self):
        return "VDSH"
    
    @staticmethod
    def calculate_KL_loss(mu, logvar):
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element, dim=1)
        KLD = torch.mean(KLD).mul_(-0.5)
        return KLD

    @staticmethod
    def compute_reconstr_loss(logprob_word, doc_mat):
        return -torch.mean(torch.sum(logprob_word * doc_mat, dim=1))
    
    def get_binary_code(self, train, test):
        train_zy = [(self.encode(xb.cuda())[0], yb) for xb, yb in train]
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = [(self.encode(xb.cuda())[0], yb) for xb, yb in test]
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
        val_set = SingleLabelTextDataset('{}/{}'.format(data_path, dataset), subset='cv', bow_format=data_fmt, download=True)
    else:
        train_set = MultiLabelTextDataset('{}/{}'.format(data_path, dataset), subset='train', bow_format=data_fmt, download=True)
        test_set = MultiLabelTextDataset('{}/{}'.format(data_path, dataset), subset='test', bow_format=data_fmt, download=True)
        val_set = MultiLabelTextDataset('{}/{}'.format(data_path, dataset), subset='cv', bow_format=data_fmt, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

    num_features = train_set[0][0].size(0)
    model = VDSH(dataset, num_features, bit, dropoutProb=0.1)
    model.cuda()
    # optimizer = config["optimizer"]["type"](model.parameters(), lr=config["optimizer"]["optim_params"]["lr"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    kl_weight = 0.
    kl_step = 1 / 5000.

    best_precision = 0
    prec = 0 
    best_precision_epoch = 0
    step_count = 0

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='../../logs/VDSH/data:{}_bit:{}.log'.format(config["dataset"], config["bit"]),
        filemode='w'
    )

    for epoch in tqdm(range(config["epoch"])):
        avg_loss = []
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.cuda()
            yb = yb.cuda()

            logprob_w, mu, logvar = model(xb)
            kl_loss = VDSH.calculate_KL_loss(mu, logvar)
            reconstr_loss = VDSH.compute_reconstr_loss(logprob_w, xb)
            
            loss = reconstr_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            kl_weight = min(kl_weight + kl_step, 1.)
            avg_loss.append(loss.item())
            
        with torch.no_grad():
            train_b, val_b, train_y, val_y = model.get_binary_code(train_loader, val_loader)
            retrieved_indices = retrieve_topk(val_b.cuda(), train_b.cuda(), topK=100)
            prec = compute_precision_at_k(retrieved_indices, val_y.cuda(), train_y.cuda(), topK=100, is_single_label=single_label_flag)
            if prec.item() > best_precision:
                best_precision = prec.item()
                best_precision_epoch = epoch + 1
                step_count = 0
                torch.save(model, '../../checkpoints/VDSH/dataset:{}_bit:{}.pth'.format(dataset, bit))
            else:
                step_count += 1
            if step_count >= config["stop_iter"]:
                break
        tqdm.write(f'Epoch {epoch+1}/{config["epoch"]} - Current Precision: {prec.item():.4f} - Best Precision: {best_precision:.4f} [{best_precision_epoch}]')
        logging.info(f'Epoch {epoch+1}/{config["epoch"]} - Current Precision: {prec:.4f} - Best Precision: {best_precision:.4f} [{best_precision_epoch}]')

    model = torch.load('../../checkpoints/VDSH/dataset:{}_bit:{}.pth'.format(dataset, bit))
    model.eval()
    with torch.no_grad():
        train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
        retrieved_indices = retrieve_topk(test_b.cuda(), train_b.cuda(), topK=100)
        prec = compute_precision_at_k(retrieved_indices, test_y.cuda(), train_y.cuda(), topK=100, is_single_label=single_label_flag)
        print(f'Test Precision: {prec:.4f}')
        logging.info(f'Test Precision: {prec:.4f}')

if __name__ == "__main__":
    # config = get_config()
    argparser = get_argparser()
    args = argparser.parse_args()
    config = vars(args)
    train_val(config)