import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils.utils import *
from utils.datasets import *

def get_config():
    config = {
        "device": torch.device("cuda:0"),
        "dataset": "ng20.tfidf",
        "bit": 8,
        "dropout": 0.1,
        "batch_size": 64,
        "epoch": 300,
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 0.001}},
        "stop_iter": 10
    }
    return config

class VDSH(nn.Module):
    
    def __init__(self, dataset, vocabSize, latentDim, device, dropoutProb=0.):
        super(VDSH, self).__init__()
        
        self.dataset = dataset
        self.hidden_dim = 1000
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.dropoutProb = dropoutProb
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
        train_zy = [(self.encode(xb.to(self.device))[0], yb) for xb, yb in train]
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = [(self.encode(xb.to(self.device))[0], yb) for xb, yb in test]
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
    device = config["device"]
    dataset, data_fmt = config["dataset"].split('.')
    batch_size = config["batch_size"]

    if dataset in ['reuters', 'tmc', 'rcv1']:
        single_label_flag = False
    else:
        single_label_flag = True

    if single_label_flag:
        train_set = SingleLabelTextDataset('datasets/{}'.format(dataset), subset='train', bow_format=data_fmt, download=True)
        test_set = SingleLabelTextDataset('datasets/{}'.format(dataset), subset='test', bow_format=data_fmt, download=True)
    else:
        train_set = MultiLabelTextDataset('datasets/{}'.format(dataset), subset='train', bow_format=data_fmt, download=True)
        test_set = MultiLabelTextDataset('datasets/{}'.format(dataset), subset='test', bow_format=data_fmt, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    num_features = train_set[0][0].size(0)
    model = VDSH(dataset, num_features, bit, dropoutProb=0.1, device=device)
    model.to(device)
    optimizer = config["optimizer"]["type"](model.parameters(), lr=config["optimizer"]["optim_params"]["lr"])
    kl_weight = 0.
    kl_step = 1 / 5000.

    best_precision = 0
    prec = 0
    best_precision_epoch = 0

    step_count = 0
    for epoch in tqdm(range(config["epoch"])):
        avg_loss = []
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)

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
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
            prec = compute_precision_at_k(retrieved_indices, test_y.to(device), train_y.to(device), topK=100, is_single_label=single_label_flag)
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