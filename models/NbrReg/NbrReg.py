import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from tqdm import tqdm
from utils.utils import *
from utils.datasets import *
from prepare_neighbor_data import Load_Dataset
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

class TopDoc(object):
    def __init__(self, data_fn, is_train=False):
        self.data_fn = data_fn
        self.is_train = is_train
        self.db = self.load(data_fn, is_train)
        
    def load(self, fn, is_train):
        db = {}
        with open(fn) as in_data:
            for line in in_data:
                line = line.strip()
                first, rest = line.split(':')

                topk = list(map(int, rest.split(',')))
                
                docId = int(first)
                if is_train:
                    db[docId] = topk[1:]
                else:
                    db[docId] = topk
        return db
    
    def getTopK(self, docId, topK):
        # print(docId.item())
        return self.db[docId.item()][:topK]

    def getTopK_Noisy(self, docId, topK, topCandidates):
        candidates = self.db[docId][:topCandidates]
        candidates = np.random.permutation(candidates)
        return candidates[:topK]

class NbrReg(nn.Module):
    
    def __init__(self, vocabSize, latentDim, dropoutProb=0.):
        super(NbrReg, self).__init__()
        
        self.hidden_dim = 1000
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        
        self.dtype = torch.cuda.FloatTensor

        self.fc1 = nn.Linear(self.vocabSize, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc31 = nn.Linear(self.hidden_dim, self.latentDim)
        self.fc32 = nn.Linear(self.hidden_dim, self.latentDim)
        self.dropout = nn.Dropout(p=dropoutProb)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.fc41 = nn.Linear(self.latentDim, self.vocabSize)
        self.nn_fc41 = nn.Linear(self.latentDim, self.vocabSize)

    # def encode(self, document_mat):
    def encode(self, documents):
        # documents = Variable(torch.from_numpy(document_mat).type(self.dtype))
        
        h1 = self.relu(self.fc1(documents))
        h2 = self.relu(self.fc2(h1))
        h3 = self.dropout(h2)
        
        z_mu = self.fc31(h3)
        z_logvar = self.sigmoid(self.fc32(h3))
        return z_mu, z_logvar
    
    def decode(self, Z):
        word_prob = self.fc41(Z)
        word_prob = self.log_softmax(word_prob)
        
        nn_word_prob = self.nn_fc41(Z)
        nn_word_prob = self.log_softmax(nn_word_prob)
        
        return word_prob, nn_word_prob
        
    def reparametrize(self, mu, logvar):
        std = torch.sqrt(torch.exp(logvar))
        
        if self.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def forward(self, document_mat):
        mu, logvar = self.encode(document_mat)
        z = self.reparametrize(mu, logvar)
        prob_w, nn_prob_w = self.decode(z)
        return prob_w, nn_prob_w, mu, logvar

    @staticmethod
    def calculate_KL_loss(mu, logvar):
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element, dim=1)
        KLD = torch.mean(KLD).mul_(-0.5)
        return KLD

    @staticmethod
    def compute_reconstr_loss(log_word_prob, document_mat):
        loss = None
        
        for idx, doc_vec in enumerate(document_mat):
            word_indices = doc_vec.nonzero()
            # word_indices = Variable(torch.from_numpy(word_indices[0]).type(torch.cuda.LongTensor))
            # print(word_indices[0])
            word_indices = Variable(word_indices[0].type(torch.cuda.LongTensor))

            pred_logprob = torch.gather(log_word_prob[idx], 0, word_indices)
            
            if loss is None:
                loss = -torch.sum(pred_logprob) 
            else:
                loss.add_(-torch.sum(pred_logprob))

        return loss / document_mat.shape[0]

    @staticmethod
    def batch_compute_NN_reconstr_loss(log_word_prob, batch_nn_docs):
        batch_nn_docs = np.sum(batch_nn_docs, axis=1)
        nn_loss = None
        
        for docIdx, nn_docs in enumerate(batch_nn_docs):
            word_indices = np.nonzero(nn_docs)
            word_indices = Variable(torch.cuda.LongTensor(word_indices[0]))
            pred_logprob = torch.gather(log_word_prob[docIdx], 0, word_indices)

            if nn_loss is None:
                nn_loss = -torch.sum(pred_logprob) 
            else:
                nn_loss.add_(-torch.sum(pred_logprob))
        
        return nn_loss / float(len(batch_nn_docs))

    def get_binary_code(self, train, test):
        train_zy = [(self.encode(xb.cuda())[0], yb) for xb, _, yb in train]
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = [(self.encode(xb.cuda())[0], yb) for xb, _, yb in test]
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

    data_for_neighbor = Load_Dataset(dataset)

    if single_label_flag:
        train_set = SingleLabelTextDatasetDocID('{}/{}'.format(data_path, dataset), subset='train', bow_format=data_fmt, download=True)
        test_set = SingleLabelTextDatasetDocID('{}/{}'.format(data_path, dataset), subset='test', bow_format=data_fmt, download=True)
        val_set = SingleLabelTextDatasetDocID('{}/{}'.format(data_path, dataset), subset='cv', bow_format=data_fmt, download=True)
    else:
        train_set = MultiLabelTextDatasetDocID('{}/{}'.format(data_path, dataset), subset='train', bow_format=data_fmt, download=True)
        test_set = MultiLabelTextDatasetDocID('{}/{}'.format(data_path, dataset), subset='test', bow_format=data_fmt, download=True)
        val_set = MultiLabelTextDatasetDocID('{}/{}'.format(data_path, dataset), subset='cv', bow_format=data_fmt, download=True)

    # load neighbor data
    train_topk_docs_db = TopDoc('./neighbor_data/{}_train_top101.txt'.format(dataset), is_train=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

    #########################################################################################################
    y_dim = train_set.num_classes()
    num_features = train_set[0][0].size(0)
    model = NbrReg(num_features, bit, dropoutProb=0.1)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    kl_weight = 0.
    kl_step = 1 / 5000.
    nn_TOP_K = 20
    # nn_TOP_Candidates = nn_TOP_K

    best_precision = 0
    best_precision_epoch = 0

    # logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='../../logs/NbrReg/data:{}_bit:{}.log'.format(config["dataset"], config["bit"]),
        filemode='w'
    )

    for epoch in tqdm(range(config["epoch"])):
        avg_loss = []
        for step, (xb, ids, yb) in enumerate(train_loader):
            xb = xb.cuda()
            yb = yb.cuda()

            word_prob, nn_word_prob, mu, logvar = model(xb)

            kl_loss = NbrReg.calculate_KL_loss(mu, logvar)

            reconstr_loss = NbrReg.compute_reconstr_loss(word_prob, xb)
            
            # compute nn reconstruction loss
            batch_nn_docs = []
            for docId in ids:
                nn_docList = train_topk_docs_db.getTopK(docId, nn_TOP_K)
                nn_docs = data_for_neighbor.train[nn_docList]
                batch_nn_docs.append(nn_docs)
            batch_nn_docs = np.stack(batch_nn_docs)
            nn_reconstr_loss = NbrReg.batch_compute_NN_reconstr_loss(nn_word_prob, batch_nn_docs)

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
            else:
                step_count += 1
            if step_count >= config["stop_iter"]:
                break
        tqdm.write(f'Epoch {epoch+1}/{config["epoch"]} - Current Precision: {prec.item():.4f} - Best Precision: {best_precision:.4f} [{best_precision_epoch}]')
        logging.info(f'Epoch {epoch+1}/{config["epoch"]} - Current Precision: {prec:.4f} - Best Precision: {best_precision:.4f} [{best_precision_epoch}]')

    model = torch.load('../../checkpoints/NbrReg/dataset:{}_bit:{}.pth'.format(dataset, bit))
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