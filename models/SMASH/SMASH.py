import argparse
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from textdata import SingleLabelTextDatasetDocID, MultiLabelTextDatasetDocID
from utils.utils import set_seed, retrieve_topk, compute_precision_at_k

warnings.filterwarnings("ignore")
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
    parser.add_argument('--stop_iter', type=int, default=50)
    parser.add_argument('--lsc_weight', type=float, default=1)
    parser.add_argument('--bb_weight', type=float, default=1)
    parser.add_argument('--bd_weight', type=float, default=10)
    parser.add_argument('--em_alpha', type=float, default=0.3)
    parser.add_argument('--sigma', type=float, default=0.3)
    parser.add_argument('--n_sample', type=int, default=3)
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


class ExemplarMemory(Function):
    @staticmethod
    def forward(ctx, inputs, idxs, em, em_time_count, time_max, em_alpha, loss_change_term):
        ctx.save_for_backward(inputs, idxs, em, em_time_count)
        for idx in idxs:
            em_time_count[idx] = time_max
        em_out = em[(em_time_count > ((1 - loss_change_term) * (1 - em_alpha) * time_max)).nonzero()].squeeze(dim=1)
        return em_out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, idxs, em, em_time_count = ctx.saved_tensors
        for idx, z_code in zip(idxs, inputs):
            em[idx] = torch.sign(z_code)
        return None, None, None, None, None, None, None


class ExemplarMemory_warm_up(Function):
    @staticmethod
    def forward(ctx, inputs, idxs, em, em_time_count, time_max, em_alpha, loss_change_term):
        ctx.save_for_backward(inputs, idxs, em, em_time_count)
        for idx, z_code in zip(idxs, inputs):
            em[idx] = torch.sign(z_code)
            em_time_count[idx] = time_max
        em_out = em[(em_time_count > ((1 - loss_change_term) * (1 - em_alpha) * time_max)).nonzero()].squeeze(dim=1)
        return em_out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, idxs, em, em_time_count = ctx.saved_tensors
        return None, None, None, None, None, None, None


class SMASH(nn.Module):
    def __init__(self, dataset, vocabSize, latentDim, em_length, time_max, em_alpha, device, sigma=0.3, dropoutProb=0.):
        super(SMASH, self).__init__()
        self.dataset = dataset
        self.hidden_dim = 1000
        self.long_bit_code = 128
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.dropoutProb = dropoutProb
        self.time_max = time_max
        self.em_alpha = em_alpha
        self.sigma = sigma
        self.device = device
        self.em = nn.Parameter(torch.zeros([em_length, latentDim]))
        self.em_time_count = torch.ones(em_length) * time_max

        self.encoder = nn.Sequential(
            nn.Linear(self.vocabSize, self.hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(p=dropoutProb))
        self.encoder_long_bit_code = nn.Sequential(nn.Linear(self.hidden_dim, self.long_bit_code), nn.Tanh())
        self.longz_to_z = nn.Linear(self.long_bit_code, self.latentDim)
        self.decoder = nn.Sequential(nn.Linear(self.latentDim, self.vocabSize), nn.LogSoftmax(dim=1))

    def encode(self, doc_mat):
        h = self.encoder(doc_mat)
        long_z = self.encoder_long_bit_code(h)
        z = self.longz_to_z(long_z)
        z = torch.nn.functional.tanh(z)
        return z

    def forward(self, document_mat, idxs, epoch, loss_change_term, n_sample=3):
        document_mat_noise = self.add_noise(document_mat, n_sample)
        h = self.encoder(document_mat)
        noise_h = self.encoder(document_mat_noise)
        long_z_e = self.encoder_long_bit_code(h)
        noise_long_z_e = self.encoder_long_bit_code(noise_h)
        z = self.longz_to_z(long_z_e)
        z_noise = self.longz_to_z(noise_long_z_e)
        z = torch.nn.functional.tanh(z)
        z_noise = torch.nn.functional.tanh(z_noise)
        self.em_time_count = self.em_time_count - 1
        if epoch < 10:
            em_out = ExemplarMemory_warm_up.apply(
                z,
                idxs,
                self.em,
                self.em_time_count,
                self.time_max,
                self.em_alpha,
                loss_change_term)
        else:
            em_out = ExemplarMemory.apply(
                z,
                idxs,
                self.em,
                self.em_time_count,
                self.time_max,
                self.em_alpha,
                loss_change_term)
        prob_w = self.decoder(z)
        noise_prob_w = self.decoder(z_noise)
        return prob_w, noise_prob_w, z, z_noise, long_z_e, noise_long_z_e, em_out

    def add_noise(self, doc_mat, n_sample=3):
        x = torch.abs(torch.normal(torch.zeros([n_sample, doc_mat.shape[0], doc_mat.shape[1]]), 1 - self.sigma))
        x = torch.where(x > 1, torch.tensor(1.0), x)
        noise_matrix = torch.bernoulli(x).to(self.device)
        return torch.reshape(noise_matrix * doc_mat, [-1, doc_mat.shape[1]])

    def get_name(self):
        return "SMASH"

    def get_binary_code(self, train, test):
        train_zy = [(self.encode(xb.to(self.device)), yb) for xb, _, yb in train]
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = [(self.encode(xb.to(self.device)), yb) for xb, _, yb in test]
        test_z, test_y = zip(*test_zy)
        test_z = torch.cat(test_z, dim=0)
        test_y = torch.cat(test_y, dim=0)

        mid_val, _ = torch.median(train_z, dim=0)
        train_b = (train_z > mid_val).type(torch.ByteTensor).to(self.device)
        test_b = (test_z > mid_val).type(torch.ByteTensor).to(self.device)
        del train_z, test_z
        return train_b, test_b, train_y, test_y


def compute_reconstr_loss(logprob_word, doc_mat):
    return -torch.mean(torch.sum(logprob_word * doc_mat, dim=1))


def compute_reconstr_noise_loss(logprob_word, doc_mat, epsilon):
    logprob_word = logprob_word.reshape(int(logprob_word.shape[0] / doc_mat.shape[0]), -1, doc_mat.shape[1])
    return -torch.mean(epsilon.t() * torch.sum(logprob_word * doc_mat, dim=2))


def relevance_propagation_v1(z, long_z):
    a = torch.mm(z, z.t()) / z.size()[-1]
    b = torch.mm(long_z, long_z.t()) / long_z.size()[-1]
    c = 1 - torch.eye(z.size()[0]).to(z.device)
    a = a * c
    b = b * c
    dp_loss = torch.sum(torch.abs(a - b)) / (z.size()[0] * (z.size()[0] - 1))
    return dp_loss


def code_balance_global_v2(num_bits, em_out, batch, alpha=1.0, beta=1.0):
    device = batch.device
    batch_size = batch.shape[0]
    balance_w = torch.nn.functional.softmax(torch.abs(torch.sum(em_out, dim=0)), dim=0)
    bit_balance_loss = torch.sum(torch.abs(torch.sum(batch, dim=0)).mul(balance_w)) / num_bits
    I_matrix = torch.eye(num_bits).to(device)
    em_size = em_out.shape[0]
    uncorrelation_w = torch.nn.functional.softmax(torch.abs(em_out.t().mm(
        em_out) / em_size - I_matrix).view(-1), dim=0).view(num_bits, num_bits)
    bit_uncorrelation_loss = torch.pow(torch.norm(
        (batch.t().mm(batch) / batch_size - I_matrix).mul(uncorrelation_w)), 2) / (num_bits * num_bits)
    loss = alpha * bit_balance_loss + beta * bit_uncorrelation_loss
    return loss


def train_val(config):
    device = get_device(config)
    bit = config["bit"]
    dataset, data_fmt = config["dataset"].split('.')
    batch_size = config["batch_size"]

    single_label_flag = dataset not in ['reuters', 'tmc', 'rcv1']
    data_path = Path(__file__).parent.parent.parent / 'textdata'

    if single_label_flag:
        train_set = SingleLabelTextDatasetDocID(f'{data_path}/{dataset}', subset='train', bow_format=data_fmt)
        test_set = SingleLabelTextDatasetDocID(f'{data_path}/{dataset}', subset='test', bow_format=data_fmt)
    else:
        train_set = MultiLabelTextDatasetDocID(f'{data_path}/{dataset}', subset='train', bow_format=data_fmt)
        test_set = MultiLabelTextDatasetDocID(f'{data_path}/{dataset}', subset='test', bow_format=data_fmt)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    num_bits = config["bit"]
    n_sample = config["n_sample"]
    num_features = train_set[0][0].size(0)
    time_max = int(len(train_set) / batch_size)

    model = SMASH(
        dataset,
        num_features,
        num_bits,
        len(train_set),
        time_max,
        config["em_alpha"],
        device,
        sigma=config["sigma"],
        dropoutProb=0.1)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5e3, gamma=0.96)

    best_precision = 0
    best_precision_epoch = 0
    step_count = 0
    L_max = 0
    L_t_minus_1 = 0
    transfrom_flag = True

    log_file = LOG_DIR / f'data:{config["dataset"]}_bit:{config["bit"]}.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=str(log_file), filemode='w')
    checkpoint_path = CHECKPOINT_DIR / f'dataset:{dataset}_bit:{bit}.pth'

    for epoch in tqdm(range(config["epoch"])):
        total_loss = []
        model.train()
        for _, (xb, idxs, yb) in enumerate(train_loader):
            xb = xb.to(device)
            loss_change_term = np.abs(L_max - L_t_minus_1) / (L_max + 0.00001)
            logprob_w, logprob_w_noise, z, z_noise, long_z, noise_long_z_e, em_out = model(
                xb, idxs, epoch, loss_change_term, n_sample)
            noise_z_reshape = noise_long_z_e.reshape(xb.shape[0], 128, n_sample)
            mult = torch.matmul(long_z.unsqueeze(1), noise_z_reshape).squeeze(1)
            epsilon = torch.nn.functional.softmax(mult, dim=1)

            rec_loss = compute_reconstr_loss(logprob_w, xb) + compute_reconstr_noise_loss(logprob_w_noise, xb, epsilon)
            pro_loss = relevance_propagation_v1(z, long_z) + relevance_propagation_v1(z_noise, noise_long_z_e)
            loss = rec_loss + config["lsc_weight"] * pro_loss
            if transfrom_flag:
                ba_loss = code_balance_global_v2(
                    num_bits, em_out, z, alpha=config["bb_weight"], beta=config["bd_weight"])
                loss = loss + ba_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            L_t_minus_1 = loss.item()
            total_loss.append(loss.item())

        model.eval()
        if np.mean(total_loss) > L_max:
            L_max = np.mean(total_loss)

        with torch.no_grad():
            train_b, val_b, train_y, val_y = model.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(val_b, train_b, topK=100)
            prec = compute_precision_at_k(
                retrieved_indices,
                val_y.to(device),
                train_y.to(device),
                topK=100,
                is_single_label=single_label_flag)
            if prec > best_precision:
                best_precision = prec
                best_precision_epoch = epoch + 1
                step_count = 0
                torch.save(model, str(checkpoint_path))
            else:
                step_count += 1
            if step_count >= config["stop_iter"]:
                break

        tqdm.write(
            f'Epoch {epoch+1}/{config["epoch"]} - Prec: {prec:.4f} - Best: {best_precision:.4f} [{best_precision_epoch}]')
        logging.info(
            f'Epoch {epoch+1}/{config["epoch"]} - Prec: {prec:.4f} - Best: {best_precision:.4f} [{best_precision_epoch}]')


if __name__ == "__main__":
    argparser = get_argparser()
    args = argparser.parse_args()
    config = vars(args)
    train_val(config)
