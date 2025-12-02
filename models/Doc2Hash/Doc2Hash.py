import argparse
import logging
import os
from pathlib import Path

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
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--bit', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--stop_iter', type=int, default=50)
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


class Doc2Hash(nn.Module):
    def __init__(self, dataset, vocabSize, latentDim, device, dropoutProb=0.):
        super(Doc2Hash, self).__init__()
        self.dataset = dataset
        self.hidden_dim = 500
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.dropoutProb = dropoutProb
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(self.vocabSize, self.hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(p=dropoutProb),
            nn.Linear(self.hidden_dim, self.latentDim * 2))
        self.decoder = nn.Sequential(nn.Linear(self.latentDim * 2, self.vocabSize), nn.LogSoftmax(dim=1))

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).to(self.device)
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, latent_dim, categorical_dim=2):
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(-1, latent_dim * categorical_dim)

    def forward(self, document_mat, tmp):
        q = self.encoder(document_mat)
        q_y = q.view(q.size(0), self.latentDim, 2)
        z = self.gumbel_softmax(q_y, tmp, latent_dim=self.latentDim)
        prob_w = self.decoder(z)
        return prob_w, F.softmax(q, dim=-1)

    def get_name(self):
        return "Doc2Hash"

    @staticmethod
    def calculate_KL_loss(qy, categorical_dim=2):
        log_qy = torch.log(qy + 1e-20)
        g = torch.log(torch.tensor(1.0 / categorical_dim))
        KLD = torch.sum(qy * (log_qy - g), dim=-1).mean()
        return KLD

    @staticmethod
    def compute_reconstr_loss(logprob_word, doc_mat):
        return -torch.mean(torch.sum(logprob_word * doc_mat, dim=1))

    def get_binary_code(self, train, test):
        train_zy = []
        for xb, yb in train:
            q = self.encoder(xb.to(self.device))
            q_y = q.view(q.size(0), self.latentDim, 2)
            b = torch.argmax(q_y, dim=2)
            train_zy.append((b, yb))
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = []
        for xb, yb in test:
            q = self.encoder(xb.to(self.device))
            q_y = q.view(q.size(0), self.latentDim, 2)
            b = torch.argmax(q_y, dim=2)
            test_zy.append((b, yb))
        test_z, test_y = zip(*test_zy)
        test_z = torch.cat(test_z, dim=0)
        test_y = torch.cat(test_y, dim=0)
        return train_z, test_z, train_y, test_y


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
        val_set = SingleLabelTextDataset(f'{data_path}/{dataset}', subset='cv', bow_format=data_fmt)
    else:
        train_set = MultiLabelTextDataset(f'{data_path}/{dataset}', subset='train', bow_format=data_fmt)
        test_set = MultiLabelTextDataset(f'{data_path}/{dataset}', subset='test', bow_format=data_fmt)
        val_set = MultiLabelTextDataset(f'{data_path}/{dataset}', subset='cv', bow_format=data_fmt)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

    num_features = train_set[0][0].size(0)
    model = Doc2Hash(dataset, num_features, bit, device, dropoutProb=0.1)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    kl_weight = 2.
    tmp = 1

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
            logprob_w, qy = model(xb, tmp)
            kl_loss = Doc2Hash.calculate_KL_loss(qy)
            reconstr_loss = Doc2Hash.compute_reconstr_loss(logprob_w, xb)
            loss = reconstr_loss + kl_weight * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tmp = max(tmp * 0.96, 0.1)

        with torch.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
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
