import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--bit', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--stop_iter', type=int, default=50, help='控制在效果没有提升多少次后停止运行')
    return parser

class Doc2Hash_S(nn.Module):
    
    def __init__(self, dataset, vocabSize, latentDim, num_classes, dropoutProb=0.):
        super(Doc2Hash_S, self).__init__()
        
        self.dataset = dataset
        self.hidden_dim = 500
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.dropoutProb = dropoutProb
        self.num_classes = num_classes
        
        self.encoder = nn.Sequential(nn.Linear(self.vocabSize, self.hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=dropoutProb),
                                     nn.Linear(self.hidden_dim, self.latentDim * 2),
                                     )
        
        # self.h_to_mu = nn.Linear(self.hidden_dim, self.latentDim)
        # self.h_to_logvar = nn.Sequential(nn.Linear(self.hidden_dim, self.latentDim),
        #                                  nn.Sigmoid())
        
        self.decoder = nn.Sequential(nn.Linear(self.latentDim*2, self.vocabSize),
                                     nn.LogSoftmax(dim=1))

        self.pred = nn.Sequential(nn.Linear(self.latentDim*2, self.num_classes))
        self.pred_loss = nn.CrossEntropyLoss()

    
    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).cuda()
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, latent_dim, categorical_dim = 2):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(-1, latent_dim*categorical_dim)
    
    def encode(self, doc_mat):
        h = self.encoder(doc_mat)
        z_mu = self.h_to_mu(h)
        z_logvar = self.h_to_logvar(h)
        return z_mu, z_logvar
    
    def forward(self, document_mat, tmp):
        q = self.encoder(document_mat)
        q_y = q.view(q.size(0), self.latentDim ,2)
        z = self.gumbel_softmax(q_y, tmp, latent_dim=self.latentDim)
        prob_w = self.decoder(z)
        score_c = self.pred(z)
        return prob_w, F.softmax(q), score_c
    
    def get_name(self):
        return "Doc2Hash_S"
    
    @staticmethod
    def calculate_KL_loss(qy, categorical_dim=2):
        log_qy = torch.log(qy+1e-20)
        g = Variable(torch.log(torch.Tensor([1.0/categorical_dim]).cuda()))
        KLD = torch.sum(qy*(log_qy-g), dim=-1).mean()
        return KLD

    @staticmethod
    def compute_reconstr_loss(logprob_word, doc_mat):
        return -torch.mean(torch.sum(logprob_word * doc_mat, dim=1))
    
    def compute_prediction_loss(self, scores, labels):
        return self.pred_loss(scores, labels)
    
    def get_binary_code(self, train, test):
        train_zy = []
        for xb, yb in train:
            q = self.encoder(xb.cuda())
            q_y = q.view(q.size(0), self.latentDim, 2)
            b = torch.argmax(q_y,dim=2)
            train_zy.append((b,yb))
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = []
        # tmp_z = []
        for xb, yb in test:
            q = self.encoder(xb.cuda())
            q_y = q.view(q.size(0), self.latentDim, 2)
            b = torch.argmax(q_y, dim=2)
            test_zy.append((b, yb))
            # tmp_z.append(q)
        test_z, test_y = zip(*test_zy)
        test_z = torch.cat(test_z, dim=0)
        test_y = torch.cat(test_y, dim=0)
        return train_z, test_z, train_y, test_y

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
    y_dim = train_set.num_classes()

    model = Doc2Hash_S(dataset, num_features, bit, y_dim, dropoutProb=0.1)
    model.cuda()
    # optimizer = config["optimizer"]["type"](model.parameters(), lr=config["optimizer"]["optim_params"]["lr"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    kl_weight = 2.
    # kl_step = 1 / 5000.
    tmp=1
    ar = 0.00003
    pred_weight = 0.
    pred_weight_step = 1 / 1000.
    
    best_precision = 0
    prec = 0 
    best_precision_epoch = 0
    step_count = 0

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='../../logs/Doc2Hash_S/data:{}_bit:{}.log'.format(config["dataset"], config["bit"]),
        filemode='w'
    )

    for epoch in tqdm(range(config["epoch"])):
        avg_loss = []
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.cuda()
            yb = yb.cuda()

            logprob_w, qy, score_c = model(xb,tmp)
            kl_loss = Doc2Hash_S.calculate_KL_loss(qy)
            reconstr_loss = Doc2Hash_S.compute_reconstr_loss(logprob_w, xb)

            if single_label_flag:
                pred_loss = model.compute_prediction_loss(score_c, yb)
            else:
                if len(yb.size()) == 1:
                    y_onehot = torch.zeros((xb.size(0), y_dim)).cuda()
                    y_onehot = y_onehot.scatter_(1, yb.unsqueeze(1), 1)
                    pred_loss = model.compute_prediction_loss(score_c, y_onehot)
                else:
                    pred_loss = model.compute_prediction_loss(score_c, yb)
            
            loss = reconstr_loss + kl_weight * kl_loss + pred_weight * pred_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tmp = max(tmp*0.96, 0.1)
            pred_weight = min(pred_weight + pred_weight_step, 150.)

            avg_loss.append(loss.item())
            
        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(test_b.cuda(), train_b.cuda(), topK=100)
            prec = compute_precision_at_k(retrieved_indices, test_y.cuda(), train_y.cuda(), topK=100, is_single_label=single_label_flag)
            if prec.item() > best_precision:
                best_precision = prec.item()
                best_precision_epoch = epoch + 1
                step_count = 0
                torch.save(model, '../../checkpoints/Doc2Hash_S/dataset:{}_bit:{}.pth'.format(dataset, bit))
            else:
                step_count += 1
            if step_count >= config["stop_iter"]:
                break
        tqdm.write(f'Epoch {epoch+1}/{config["epoch"]} - Current Precision: {prec.item():.4f} - Best Precision: {best_precision:.4f} [{best_precision_epoch}]')
        logging.info(f'Epoch {epoch+1}/{config["epoch"]} - Current Precision: {prec:.4f} - Best Precision: {best_precision:.4f} [{best_precision_epoch}]')

    model = torch.load('../../checkpoints/Doc2Hash_S/dataset:{}_bit:{}.pth'.format(dataset, bit))
    model.eval()
    with torch.no_grad():
        train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
        retrieved_indices = retrieve_topk(test_b.cuda(), train_b.cuda(), topK=100)
        prec = compute_precision_at_k(retrieved_indices, test_y.cuda(), train_y.cuda(), topK=100, is_single_label=single_label_flag)
        print(f'Test Precision: {prec:.4f}')
        logging.info(f'Test Precision: {prec:.4f}')

if __name__ == "__main__":
    argparser = get_argparser()
    args = argparser.parse_args()
    config = vars(args)
    train_val(config)