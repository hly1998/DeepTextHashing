import torch.autograd as autograd
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

class LBSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class NASH(nn.Module):
    def __init__(self, vocabSize, latentDim, device, dropoutProb=0.) :
        super(NASH, self).__init__()
        # self.hidden_dim = 1000
        # according to paper, we set the hidden_dim as 500
        self.hidden_dim = 500
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.dropoutProb = dropoutProb
        self.device = device
        
        # encoder network
        self.encoder = nn.Sequential(
            nn.Linear(self.vocabSize, self.hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(p=dropoutProb))

        self.h_to_mu = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latentDim), nn.Sigmoid())

        # decoder network
        self.decoder = nn.Sequential(nn.Linear(self.latentDim, self.vocabSize),
            nn.LogSoftmax(dim=1))

        # noise network
        self.sigma = nn.Sequential(
            nn.Linear(self.latentDim, self.latentDim)
            , nn.Sigmoid()
            , nn.Dropout(self.dropoutProb)
        )

        # self.deterministic = config.deterministic
        # self.mu = None
        # if not self.deterministic :
        #     self.mu = torch.rand(config.output_size)
        #     if config.use_cuda :
        #         self.mu = self.mu.cuda()

    def binarization(self, mu, isStochastic):
        lb_sign = LBSign.apply
        if isStochastic:
            thresh = torch.FloatTensor(mu.size()).uniform_().to(self.device)
            return (lb_sign(mu - thresh) + 1) / 2
        else:
            return (lb_sign(mu - 0.5) + 1) / 2
    
    def encode(self, doc_mat, isStochastic):
        h = self.encoder(doc_mat)
        mu = self.h_to_mu(h)
        z = self.binarization(mu, isStochastic)
        return z, mu

    def forward(self, document_mat, isStochastic, integration='sum'):
        z, mu = self.encode(document_mat, isStochastic)
        # add noise from mu
        z_noise = z + self.sigma(mu)
        prob_w = self.decoder(z_noise)
        prob_w = self.decoder(z)
        return prob_w, mu

    def get_name(self):
        return "NASH"
    
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
        train_zy = [(self.encode(xb.to(self.device), isStochastic)[0], yb)
                    for xb, yb in train]
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = [(self.encode(xb.to(self.device), isStochastic)[0], yb)
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

    num_bits = bit
    num_features = train_set[0][0].size(0)

    model = NASH(num_features, num_bits, dropoutProb=0.1, device=device)
    model.to(device)

    num_epochs = config["epoch"]

    optimizer = config["optimizer"]["type"](model.parameters(), lr=config["optimizer"]["optim_params"]["lr"])
    kl_weight = 0.
    kl_step = 5e-6

    best_precision = 0
    best_precision_epoch = 0
    
    step_count = 0
    for epoch in tqdm(range(config["epoch"])):
        avg_loss = []
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)
            logprob_w, mu = model(xb, isStochastic=True)

            kl_loss = NASH.calculate_KL_loss(mu)
            reconstr_loss = NASH.compute_reconstr_loss(logprob_w, xb)
            loss = reconstr_loss + kl_weight * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            kl_weight = min(kl_weight + kl_step, 1.)
            avg_loss.append(loss.item())

        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader, isStochastic=True)
            retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
            prec = compute_precision_at_k(retrieved_indices, test_y.to(device), train_y.to(device), topK=100, is_single_label=single_label_flag)
            print("precision at 100: {:.4f}".format(prec))

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