import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import torch
from torch.autograd import Function
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.utils import *
from utils.datasets import *
import warnings
import argparse
import logging

warnings.filterwarnings("ignore")
set_seed()

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ng20.tfidf')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--bit', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--stop_iter', type=int, default=50, help='控制在效果没有提升多少次后停止运行')
    parser.add_argument('--lsc_weight', type=float, default=1)
    parser.add_argument('--bb_weight', type=float, default=1)
    parser.add_argument('--bd_weight', type=float, default=10)
    parser.add_argument('--em_alpha', type=float, default=0.3)
    parser.add_argument('--sigma', type=float, default=0.3)
    parser.add_argument('--n_sample', type=int, default=3)
    return parser

def get_config(args):
    config = {
        "dataset": args.dataset,
        "bit": args.bit,
        "dropout": 0.1,
        "batch_size": args.batch_size,
        "epoch": args.epoch,
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": args.lr}},
        "stop_iter": args.stop_iter,
        "lsc_weight": args.lsc_weight,
        "bb_weight": args.bb_weight,
        "bd_weight": args.bd_weight,
        "em_alpha": args.em_alpha,
        "sigma": args.sigma,
        "n_sample": args.n_sample,
    }
    return config

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
        # grad_inputs = grad_output.clone()[:inputs.shape[0], :]
        for idx, z_code in zip(idxs, inputs):
            em[idx] = torch.sign(z_code)
        # return grad_inputs, None, None
        return None, None, None, None, None, None, None


class ExemplarMemory_warm_up(Function):
    @staticmethod
    def forward(ctx, inputs, idxs, em, em_time_count, time_max, em_alpha, loss_change_term):
        ctx.save_for_backward(inputs, idxs, em, em_time_count)
        # em_time_count = em_time_count - 1
        for idx, z_code in zip(idxs, inputs):
            em[idx] = torch.sign(z_code)
            em_time_count[idx] = time_max
            # print(em_time_count[idx], idx)
        em_out = em[(em_time_count > ((1 - loss_change_term) * (1 - em_alpha) * time_max)).nonzero()].squeeze(dim=1)
        return em_out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, idxs, em, em_time_count = ctx.saved_tensors
        return None, None, None, None, None, None, None


class SMASH_S(nn.Module):
    def __init__(self,
                 dataset,
                 vocabSize,
                 latentDim,
                 em_length,
                 time_max,
                 em_alpha,
                 num_classes,
                 sigma=0.3,
                 dropoutProb=0.):
        super(SMASH_S, self).__init__()

        self.dataset = dataset
        self.hidden_dim = 1000
        self.long_bit_code = 128
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.dropoutProb = dropoutProb
        self.time_max = time_max
        self.em_alpha = em_alpha
        self.sigma = sigma
        self.em = nn.Parameter(torch.zeros([em_length, latentDim]))
        self.em_time_count = torch.ones(em_length) * time_max
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Linear(self.vocabSize, self.hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(p=dropoutProb))
        self.encoder_long_bit_code = nn.Sequential(
            nn.Linear(self.hidden_dim, self.long_bit_code), nn.Tanh())

        self.longz_to_z = nn.Linear(self.long_bit_code, self.latentDim)
        self.decoder = nn.Sequential(nn.Linear(self.latentDim, self.vocabSize),
                                     nn.LogSoftmax(dim=1))
        
        self.pred = nn.Sequential(nn.Linear(self.latentDim, self.num_classes))
        self.pred_loss = nn.CrossEntropyLoss()

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
            em_out = ExemplarMemory_warm_up.apply(z, idxs, self.em, self.em_time_count, self.time_max, self.em_alpha, loss_change_term)
        else:
            em_out = ExemplarMemory.apply(z, idxs, self.em, self.em_time_count, self.time_max, self.em_alpha, loss_change_term)
        prob_w = self.decoder(z)
        noise_prob_w = self.decoder(z_noise)
        score_c = self.pred(z)
        return prob_w, noise_prob_w, z, z_noise, long_z_e, noise_long_z_e, em_out, score_c

    def add_noise(self, doc_mat, n_sample=3):
        x = torch.abs(torch.normal(torch.zeros([n_sample, doc_mat.shape[0], doc_mat.shape[1]]), 1 - self.sigma))
        x = torch.where(x > 1, torch.tensor(1.0), x)
        noise_matrix = torch.bernoulli(x).cuda()
        return torch.reshape(noise_matrix * doc_mat, [-1, doc_mat.shape[1]])

    def get_name(self):
        return "SMASH_S"

    # def get_binary_code(self, train, test):
    #     train_zy = [(self.encode(xb.cuda()), yb)
    #                 for xb, _, yb in train]
    #     train_z, train_y = zip(*train_zy)
    #     train_z = torch.cat(train_z, dim=0)
    #     train_y = torch.cat(train_y, dim=0)

    #     test_zy = [(self.encode(xb.cuda()), yb) for xb, _, yb in test]
    #     test_z, test_y = zip(*test_zy)
    #     test_z = torch.cat(test_z, dim=0)
    #     test_y = torch.cat(test_y, dim=0)

    #     mid_val, _ = torch.median(train_z, dim=0)

    #     # train_b = (train_z > mid_val).type(torch.cuda.ByteTensor)
    #     # test_b = (test_z > mid_val).type(torch.cuda.ByteTensor)
    #     train_b = (train_z > mid_val).float()
    #     test_b = (test_z > mid_val).float()
        
    #     del train_z
    #     del test_z

    #     return train_b, test_b, train_y, test_y

    def compute_prediction_loss(self, scores, labels):
        return self.pred_loss(scores, labels)
    
    def get_binary_code(self, train, test):
        train_zy = [(self.encode(xb.cuda()), yb) for xb, _, yb in train]
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = [(self.encode(xb.cuda()), yb) for xb, _, yb in test]
        test_z, test_y = zip(*test_zy)
        test_z = torch.cat(test_z, dim=0)
        test_y = torch.cat(test_y, dim=0)

        mid_val, _ = torch.median(train_z, dim=0)
        train_b = (train_z > mid_val).type(torch.cuda.ByteTensor)
        test_b = (test_z > mid_val).type(torch.cuda.ByteTensor)

        del train_z
        del test_z

        return train_b, test_b, train_y, test_y

def compute_reconstr_loss(logprob_word, doc_mat):
    return -torch.mean(torch.sum(logprob_word * doc_mat, dim=1))

def compute_reconstr_noise_loss(logprob_word, doc_mat, epsilon):
    logprob_word = logprob_word.reshape(int(logprob_word.shape[0] / doc_mat.shape[0]), -1, doc_mat.shape[1])
    return -torch.mean(epsilon.t() * torch.sum(logprob_word * doc_mat, dim=2))

def relevance_propagation_v1(z, long_z):
    '''
    视两个向量都为哈希码的方式
    '''
    a = torch.mm(z, z.t()) / z.size()[-1]
    b = torch.mm(long_z, long_z.t()) / long_z.size()[-1]
    c = 1 - torch.eye(z.size()[0]).cuda()
    a = a * c
    b = b * c
    dp_loss = torch.sum(torch.abs(a - b)) / (z.size()[0] * (z.size()[0] - 1))
    return dp_loss

def code_balance_global_v2(num_bits, em_out, batch, alpha=1.0, beta=1.0):
    '''
    全局的code-balance 7.19修改 增加了系数 修改了bug
    '''
    # batch = 2 * batch - 1
    # 计算balance系数
    # balance_w = torch.nn.Softmax(torch.abs(torch.sum(em_out, dim=0)))
    batch_size = batch.shape[0]
    balance_w = torch.nn.functional.softmax(torch.abs(torch.sum(em_out, dim=0)))
    # print(balance_w)
    # bit_balance_loss = torch.sum(torch.abs(torch.sum(batch, dim=0)).mul(balance_w))
    # bit_balance_loss = torch.pow(torch.norm(torch.sum(batch, dim=0).mul(balance_w)), 2) / (num_bits)
    bit_balance_loss = torch.sum(torch.abs(torch.sum(batch, dim=0)).mul(balance_w)) / (num_bits)
    # bit_balance_loss = torch.pow(torch.norm(torch.sum(batch, dim=0).mul(balance_w)), 2)
    # 计算uncorrelation系数
    I_matrix = torch.eye(num_bits).cuda()
    em_size = em_out.shape[0]
    uncorrelation_w = torch.nn.functional.softmax(torch.abs(em_out.t().mm(em_out) / em_size - I_matrix))
    bit_uncorrelation_loss = torch.pow(torch.norm((batch.t().mm(batch) / batch_size - I_matrix).mul(uncorrelation_w)), 2) / (num_bits * num_bits)
    # bit_uncorrelation_loss = torch.pow(torch.norm((batch.t().mm(batch) - batch_size * I_matrix).mul(uncorrelation_w)), 2)
    loss = alpha * bit_balance_loss + beta * bit_uncorrelation_loss
    return loss

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
        train_set = SingleLabelTextDatasetDocID('{}/{}'.format(data_path, dataset),
                                           subset='train',
                                           bow_format=data_fmt,
                                           download=True)
        test_set = SingleLabelTextDatasetDocID('{}/{}'.format(data_path, dataset),
                                          subset='test',
                                          bow_format=data_fmt,
                                          download=True)
    else:
        train_set = MultiLabelTextDatasetDocID('{}/{}'.format(data_path, dataset),
                                          subset='train',
                                          bow_format=data_fmt,
                                          download=True)
        test_set = MultiLabelTextDatasetDocID('{}/{}'.format(data_path, dataset),
                                         subset='test',
                                         bow_format=data_fmt,
                                         download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batch_size,
                                              shuffle=True)
    num_bits = config["bit"]
    n_sample = config["n_sample"]
    num_features = train_set[0][0].size(0)
    best_precision = 0
    best_precision_epoch = 0
    time_max = int(len(train_set) / batch_size)
    y_dim = train_set.num_classes()

    model = SMASH_S(dataset,
                  num_features,
                  num_bits,
                  dropoutProb=0.1,
                  time_max=time_max,
                  em_alpha=config["em_alpha"],
                  num_classes=y_dim,
                  sigma=config["sigma"],
                  em_length=len(train_set))
    model.cuda()
    # optimizer = config["optimizer"]["type"](model.parameters(), lr=config["optimizer"]["optim_params"]["lr"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=5e3,
                                                gamma=0.96)
    L_max = 0
    L_t_minus_1 = 0
    transfrom_flag = True
    pred_weight = 0.
    pred_weight_step = 1 / 1000.

    # logging
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
        filename='./grid_search_results_S/data:{}_bit:{}_lr:{}_lsc_weight:{}_bb_weight:{}_bd_weight:{}_em_alpha:{}_sigma:{}_n_sample:{}.log'.format(config["dataset"], config["bit"], config["lr"], config["lsc_weight"], config["bb_weight"],config["bd_weight"],config["em_alpha"],config["sigma"],config["n_sample"]),  # 设置日志文件名
        filemode='w'  # 设置写入模式为覆盖
    )
    
    for epoch in tqdm(range(config["epoch"])):
        total_loss = []
        reconstr_loss = []
        propagation_loss = []
        balance_loss = []
        model.train()
        for _, (xb, idxs, yb) in enumerate(train_loader):
            xb = xb.cuda()
            yb = yb.cuda()
            loss_change_term = np.abs(L_max - L_t_minus_1) / (L_max + 0.00001)
            logprob_w, logprob_w_noise, z, z_noise, long_z, noise_long_z_e, em_out, score_c = model(xb, idxs, epoch, loss_change_term, n_sample)
            noise_z_reshape = noise_long_z_e.reshape(xb.shape[0], 128, n_sample)
            mult = torch.matmul(long_z.unsqueeze(1), noise_z_reshape).squeeze(1)
            epsilon = torch.nn.functional.softmax(mult, dim=1)

            rec_loss = compute_reconstr_loss(logprob_w, xb) + compute_reconstr_noise_loss(logprob_w_noise, xb, epsilon)
            pro_loss = relevance_propagation_v1(z, long_z) + relevance_propagation_v1(z_noise, noise_long_z_e)
            loss = rec_loss + config["lsc_weight"] * pro_loss
            if transfrom_flag:
                ba_loss = code_balance_global_v2(num_bits,
                                                    em_out,
                                                    z,
                                                    alpha=config["bb_weight"],
                                                    beta=config["bd_weight"])
                loss = loss + ba_loss
            
            if single_label_flag:
                pred_loss = model.compute_prediction_loss(score_c, yb)
            else:
                if len(yb.size()) == 1:
                    y_onehot = torch.zeros((xb.size(0), y_dim)).cuda()
                    y_onehot = y_onehot.scatter_(1, yb.unsqueeze(1), 1)
                    pred_loss = model.compute_prediction_loss(score_c, y_onehot)
                else:
                    pred_loss = model.compute_prediction_loss(score_c, yb)
            
            loss = loss + pred_weight * pred_loss

            pred_weight = min(pred_weight + pred_weight_step, 150.)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            L_t_minus_1 = loss.item()
            total_loss.append(loss.item())
            reconstr_loss.append(rec_loss.item())
            propagation_loss.append(pro_loss.item())
            if transfrom_flag:
                balance_loss.append(ba_loss.item())
            else:
                balance_loss.append(0)
        model.eval()
        if np.mean(total_loss) > L_max:
            L_max = np.mean(total_loss)

        with torch.no_grad():
            # train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
            # print(train_y)
            # retrieved_indices = retrieve_topk(test_b.cuda(), train_b.cuda(), topK=100)
            # prec = compute_precision_at_k(retrieved_indices, test_y.cuda(), train_y.cuda(), topK=100, is_single_label=single_label_flag)
            # prec = compute_precision_at_k_fast(test_b.cuda(), train_b.cuda(), test_y.cuda(), train_y.cuda(), topK=100)
            train_b, val_b, train_y, val_y = model.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(val_b.cuda(), train_b.cuda(), topK=100)
            prec = compute_precision_at_k(retrieved_indices, val_y.cuda(), train_y.cuda(), topK=100, is_single_label=single_label_flag)
            # if prec.item() > best_precision:
            if prec > best_precision:
                best_precision = prec
                best_precision_epoch = epoch + 1
                step_count = 0
            else:
                step_count += 1
            if step_count >= config["stop_iter"]:
                break
        tqdm.write(f'Epoch {epoch+1}/{config["epoch"]} - Current Precision: {prec:.4f} - Best Precision: {best_precision:.4f} [{best_precision_epoch}]')
        logging.info(f'Epoch {epoch+1}/{config["epoch"]} - Current Precision: {prec:.4f} - Best Precision: {best_precision:.4f} [{best_precision_epoch}]')

if __name__ == "__main__":
    argparser = get_argparser()
    args = argparser.parse_args()
    config = vars(args)
    train_val(config)