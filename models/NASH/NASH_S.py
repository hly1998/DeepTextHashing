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
    parser.add_argument('--stop_iter', type=int, default=50, help='控制在效果没有提升多少次后停止运行')
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


class LBSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class NASH_S(nn.Module):
    def __init__(self, vocabSize, latentDim, num_classes, device, dropoutProb=0.):
        super(NASH_S, self).__init__()
        # according to paper, we set the hidden_dim as 500
        self.hidden_dim = 500
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.dropoutProb = dropoutProb
        self.num_classes = num_classes
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
            nn.Linear(self.latentDim, self.latentDim),
            nn.Sigmoid(),
            nn.Dropout(self.dropoutProb))

        self.pred = nn.Sequential(nn.Linear(self.latentDim, self.num_classes))
        self.pred_loss = nn.CrossEntropyLoss()

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
        score_c = self.pred(z)
        return prob_w, mu, score_c

    def get_name(self):
        return "NASH_S"

    def calculate_KL_loss(self, mu):
        thresh = 1e-20 * torch.ones(mu.size()).to(self.device)
        KLD_element = mu * torch.log(torch.max(mu * 2, thresh)) + (
            1 - mu) * torch.log(torch.max((1 - mu) * 2, thresh))
        KLD = torch.sum(KLD_element, dim=1)
        KLD = torch.mean(KLD)
        return KLD

    @staticmethod
    def compute_reconstr_loss(logprob_word, doc_mat):
        return -torch.mean(torch.sum(logprob_word * doc_mat, dim=1))

    def compute_prediction_loss(self, scores, labels):
        return self.pred_loss(scores, labels)

    def get_binary_code(self, train, test, isStochastic=True):
        train_zy = [(self.encode(xb.to(self.device), isStochastic)[0], yb) for xb, yb in train]
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = [(self.encode(xb.to(self.device), isStochastic)[0], yb) for xb, yb in test]
        test_z, test_y = zip(*test_zy)
        test_z = torch.cat(test_z, dim=0)
        test_y = torch.cat(test_y, dim=0)

        train_b = train_z.type(torch.ByteTensor).to(self.device)
        test_b = test_z.type(torch.ByteTensor).to(self.device)

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

    model = NASH_S(num_features, bit, y_dim, device, dropoutProb=0.1)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    kl_weight = 0.
    kl_step = 5e-6
    pred_weight = 0.
    pred_weight_step = 1 / 1000.

    best_precision = 0
    best_precision_epoch = 0
    step_count = 0

    log_file = LOG_DIR / f'data:{config["dataset"]}_bit:{config["bit"]}.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=str(log_file), filemode='w')
    checkpoint_path = CHECKPOINT_DIR / f'dataset:{dataset}_bit:{bit}.pth'

    for epoch in tqdm(range(config["epoch"])):
        avg_loss = []
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)
            logprob_w, mu, score_c = model(xb, isStochastic=True)

            kl_loss = model.calculate_KL_loss(mu)
            reconstr_loss = NASH_S.compute_reconstr_loss(logprob_w, xb)

            if single_label_flag:
                pred_loss = model.compute_prediction_loss(score_c, yb)
            else:
                if len(yb.size()) == 1:
                    y_onehot = torch.zeros((xb.size(0), y_dim)).to(device)
                    y_onehot = y_onehot.scatter_(1, yb.unsqueeze(1), 1)
                    pred_loss = model.compute_prediction_loss(score_c, y_onehot)
                else:
                    pred_loss = model.compute_prediction_loss(score_c, yb)

            loss = reconstr_loss + kl_weight * kl_loss + pred_weight * pred_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            kl_weight = min(kl_weight + kl_step, 1.)
            pred_weight = min(pred_weight + pred_weight_step, 150.)
            avg_loss.append(loss.item())

        with torch.no_grad():
            train_b, val_b, train_y, val_y = model.get_binary_code(train_loader, val_loader, isStochastic=True)
            retrieved_indices = retrieve_topk(val_b, train_b, topK=100)
            prec = compute_precision_at_k(
                retrieved_indices,
                val_y.to(device),
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
