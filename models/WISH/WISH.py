import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
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
#         "dataset": "ng20.tfidf",
#         "bit": 8,
#         "dropout": 0.1,
#         "batch_size": 64,
#         "epoch": 300,
#         "optimizer": {"type": optim.Adam, "optim_params": {"lr": 0.001}},
#         "stop_iter": 10
#     }
#     return config

class LBSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class WISH(nn.Module):
    def __init__(self,
                 dataset,
                 vocabSize,
                 latentDim,
                 topicDim,
                 topicNum,
                 dropoutProb=0.):
        super(WISH, self).__init__()

        self.dataset = dataset
        self.hidden_dim = 1000
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.topicDim = topicDim
        self.topicNum = topicNum
        self.dropoutProb = dropoutProb

        self.topicBook = nn.Parameter(
            torch.rand(self.topicNum, self.topicDim,
                       requires_grad=True).cuda())

        self.encoder = nn.Sequential(
            nn.Linear(self.vocabSize, self.hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(p=dropoutProb))

        self.h_to_mu = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latentDim), nn.Sigmoid())

        self.decoder = nn.Sequential(nn.Linear(self.topicDim, self.vocabSize),
                                     nn.LogSoftmax(dim=1))

    def encode(self, doc_mat, isStochastic):
        h = self.encoder(doc_mat)
        mu = self.h_to_mu(h)
        z = self.binarization(mu, isStochastic)
        return z, mu

    def binarization(self, mu, isStochastic):
        lb_sign = LBSign.apply
        if isStochastic:
            thresh = torch.FloatTensor(mu.size()).uniform_().cuda()
            return (lb_sign(mu - thresh) + 1) / 2
        else:
            return (lb_sign(mu - 0.5) + 1) / 2

    def forward(self, document_mat, isStochastic, integration='sum'):
        z, mu = self.encode(document_mat, isStochastic)
        if integration == 'sum':
            z_nor = z
        else:
            cnt = torch.sum(z, dim=-1)  # row sum
            cnt[cnt == 0] = 1.
            z_nor = z.div(cnt.view(-1, 1))

        topic_com = torch.mm(z_nor, self.topicBook)
        prob_w = self.decoder(topic_com)
        return prob_w, mu, torch.mm(self.topicBook, self.topicBook.t())

    def get_name(self):
        return "WISH"

    @staticmethod
    def calculate_KL_loss(mu):
        thresh = 1e-20 * torch.ones(mu.size()).cuda()
        KLD_element = mu * torch.log(torch.max(mu * 2, thresh)) + (
            1 - mu) * torch.log(torch.max((1 - mu) * 2, thresh))
        KLD = torch.sum(KLD_element, dim=1)
        KLD = torch.mean(KLD)
        return KLD

    @staticmethod
    def compute_reconstr_loss(logprob_word, doc_mat):
        return -torch.mean(torch.sum(logprob_word * doc_mat, dim=1))

    def get_binary_code(self, train, test, isStochastic):
        train_zy = [(self.encode(xb.cuda(), isStochastic)[0], yb)
                    for xb, yb in train]
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = [(self.encode(xb.cuda(), isStochastic)[0], yb)
                   for xb, yb in test]
        test_z, test_y = zip(*test_zy)
        test_z = torch.cat(test_z, dim=0)
        test_y = torch.cat(test_y, dim=0)

        train_b = train_z.type(torch.cuda.ByteTensor)
        test_b = test_z.type(torch.cuda.ByteTensor)

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

    num_bits = bit 
    num_features = train_set[0][0].size(0)

    model = WISH(dataset, num_features, num_bits, 100, num_bits, dropoutProb=0.2)
    model.cuda()

    num_epochs = config["epoch"] 

    # optimizer = config["optimizer"]["type"](model.parameters(), lr=config["optimizer"]["optim_params"]["lr"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    kl_weight = 0.
    kl_step = 5e-6
    alpha = 1.
    topicNum = num_bits
    I_matrix = torch.eye(topicNum).cuda()
    
    best_precision = 0
    best_precision_epoch = 0
    
    step_count = 0

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='../../logs/WISH/data:{}_bit:{}.log'.format(config["dataset"], config["bit"]),
        filemode='w'
    )


    for epoch in tqdm(range(config["epoch"])):
        avg_loss = []
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.cuda()
            yb = yb.cuda()
            logprob_w, mu, topicS = model(xb, True, integration='sum')
            kl_loss = WISH.calculate_KL_loss(mu)
            reconstr_loss = WISH.compute_reconstr_loss(logprob_w, xb)
            loss = reconstr_loss + kl_weight * kl_loss
            loss += torch.pow(torch.norm(topicS - I_matrix), 2) * alpha
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            kl_weight = min(kl_weight + kl_step, 1.)
            avg_loss.append(loss.item())

        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader, isStochastic=True)
            retrieved_indices = retrieve_topk(test_b.cuda(), train_b.cuda(), topK=100)
            prec = compute_precision_at_k(retrieved_indices, test_y.cuda(), train_y.cuda(), topK=100, is_single_label=single_label_flag)
            # print("precision at 100: {:.4f}".format(prec))

            if prec.item() > best_precision:
                best_precision = prec.item()
                best_precision_epoch = epoch + 1
                step_count = 0
                torch.save(model, '../../checkpoints/WISH/dataset:{}_bit:{}.pth'.format(dataset, bit))
            else:
                step_count += 1
            if step_count >= config["stop_iter"]:
                break

        tqdm.write(f'Epoch {epoch+1}/{config["epoch"]} - Current Precision: {prec.item():.4f} - Best Precision: {best_precision:.4f} [{best_precision_epoch}]')
        logging.info(f'Epoch {epoch+1}/{config["epoch"]} - Current Precision: {prec:.4f} - Best Precision: {best_precision:.4f} [{best_precision_epoch}]')
    
    model = torch.load('../../checkpoints/WISH/dataset:{}_bit:{}.pth'.format(dataset, bit))
    model.eval()
    with torch.no_grad():
        train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader, isStochastic=True)
        retrieved_indices = retrieve_topk(test_b.cuda(), train_b.cuda(), topK=100)
        prec = compute_precision_at_k(retrieved_indices, test_y.cuda(), train_y.cuda(), topK=100, is_single_label=single_label_flag)
        print(f'Test Precision: {prec:.4f}')
        logging.info(f'Test Precision: {prec:.4f}')

if __name__ == "__main__":
    argparser = get_argparser()
    args = argparser.parse_args()
    config = vars(args)
    train_val(config)