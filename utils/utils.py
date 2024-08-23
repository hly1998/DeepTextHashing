import numpy as np
import torch
from tqdm import tqdm
import pickle

def retrieve_topk(query_b, doc_b, topK, batch_size=100):
    n_bits = doc_b.size(1)
    n_train = doc_b.size(0)
    n_test = query_b.size(0)

    topScores = torch.cuda.ByteTensor(n_test, topK + batch_size).fill_(n_bits+1)
    topIndices = torch.cuda.LongTensor(n_test, topK + batch_size).zero_()

    testBinmat = query_b.unsqueeze(2)

    for batchIdx in tqdm(range(0, n_train, batch_size), ncols=0, leave=False):
        s_idx = batchIdx
        e_idx = min(batchIdx + batch_size, n_train)
        numCandidates = e_idx - s_idx

        trainBinmat = doc_b[s_idx:e_idx]
        trainBinmat.unsqueeze_(0)
        trainBinmat = trainBinmat.permute(0, 2, 1)
        trainBinmat = trainBinmat.expand(testBinmat.size(0), n_bits, trainBinmat.size(2))

        testBinmatExpand = testBinmat.expand_as(trainBinmat)

        scores = (trainBinmat ^ testBinmatExpand).sum(dim=1)
        indices = torch.arange(start=s_idx, end=e_idx, step=1).type(torch.cuda.LongTensor).unsqueeze(0).expand(n_test, numCandidates)

        topScores[:, -numCandidates:] = scores
        topIndices[:, -numCandidates:] = indices

        topScores, newIndices = topScores.sort(dim=1)
        topIndices = torch.gather(topIndices, 1, newIndices)

    return topIndices

def compute_precision_at_k(retrieved_indices, query_labels, doc_labels, topK, is_single_label):
    n_test = query_labels.size(0)
    
    Indices = retrieved_indices[:,:topK]
    if is_single_label:
        test_labels = query_labels.unsqueeze(1).expand(n_test, topK)
        topTrainLabels = [torch.index_select(doc_labels, 0, Indices[idx]).unsqueeze_(0) for idx in range(0, n_test)]
        topTrainLabels = torch.cat(topTrainLabels, dim=0)
        relevances = (test_labels == topTrainLabels).type(torch.cuda.ShortTensor)
    else:
        topTrainLabels = [torch.index_select(doc_labels, 0, Indices[idx]).unsqueeze_(0) for idx in range(0, n_test)]
        topTrainLabels = torch.cat(topTrainLabels, dim=0).type(torch.cuda.ShortTensor)
        test_labels = query_labels.unsqueeze(1).expand(n_test, topK, topTrainLabels.size(-1)).type(torch.cuda.ShortTensor)
        relevances = (topTrainLabels & test_labels).sum(dim=2)
        relevances = (relevances > 0).type(torch.cuda.ShortTensor)
        
    true_positive = relevances.sum(dim=1).type(torch.cuda.FloatTensor)
    true_positive = true_positive.div_(100)
    prec_at_k = torch.mean(true_positive)
    return prec_at_k

def int2bit(n, b):
    '''
    将实值转二值
    '''
    bit_list = [int(x) for x in list(bin(n)[2:])]
    bit_list = list(np.zeros(b - len(bit_list)).astype(int)) + bit_list
    return bit_list

def bit2int(n):
    '''
    将二值的list转10进制
    '''
    n = int(''.join([str(x) for x in n]),2)
    return n

# 使用hash table的方法测评结果--单哈希码
def precision_at_k_by_hash_table(query_b, doc_b, query_y, doc_y, neighbor, k=100):
    bit_num = query_b.shape[1]
    number = query_b.shape[0]
    query_b = query_b.cpu().numpy().tolist()
    doc_b = doc_b.cpu().numpy().tolist()
    hash_table = {}
    for code, label in zip(doc_b, doc_y):
        code_id = bit2int(code)
        if code_id not in hash_table.keys():
            hash_table[code_id]=[]
        hash_table[code_id].append(int(label))
    precision = 0
    for code, label in zip(query_b, query_y):
        code_id = bit2int(code)
        has_found_label_list = []
        for r in range(bit_num+1):
            has_found_code_list = neighbor[code_id][r]
            for c in has_found_code_list:
                if c not in hash_table.keys():
                    continue
                has_found_label_list = has_found_label_list + hash_table[c]
                if len(has_found_label_list) >= k:
                    has_found_label_list = has_found_label_list[:100]
                    break
            if len(has_found_label_list) >= k:
                break
        # 计算precision
        prec = np.sum(np.array(has_found_label_list) == label.item()) / k
        precision = precision + prec
    return precision /  number


# 使用hash table的方法测评结果--双哈希码
def precision_at_k_by_hash_table_pair(query_b_a, query_b_b, doc_b_a, doc_b_b, query_y, doc_y, neighbor, k=100):
    bit_num = query_b_a.shape[1]
    number = query_b_a.shape[0]
    query_b_a = query_b_a.cpu().numpy().tolist()
    query_b_b = query_b_b.cpu().numpy().tolist()
    doc_b_a = doc_b_a.cpu().numpy().tolist()
    doc_b_b = doc_b_b.cpu().numpy().tolist()
    hash_table_doc_id_a = {}
    hash_table_doc_id_b = {}
    id2label = {}
    for doc_id, (code_a, code_b, label) in enumerate(zip(doc_b_a, doc_b_b, doc_y)):
        code_a_id = bit2int(code_a)
        code_b_id = bit2int(code_b)
        if code_a_id not in hash_table_doc_id_a.keys():
            hash_table_doc_id_a[code_a_id]=[]
        hash_table_doc_id_a[code_a_id].append(int(doc_id))
        if code_b_id not in hash_table_doc_id_b.keys():
            hash_table_doc_id_b[code_b_id]=[]
        hash_table_doc_id_b[code_b_id].append(int(doc_id))
        id2label[doc_id] = label
    precision = 0
    for code_a, code_b, label in zip(query_b_a, query_b_b, query_y):
        code_id_a = bit2int(code_a)
        code_id_b = bit2int(code_b)
        has_found_doc_id_a = []
        has_found_doc_id_b = []
        for r in range(bit_num+1):
            neighbor_code_a_list = neighbor[code_id_a][r]
            neighbor_code_b_list = neighbor[code_id_b][r]
            intersection_ab = []
            for c_a, c_b in zip(neighbor_code_a_list, neighbor_code_b_list):
                if c_a not in hash_table_doc_id_a.keys():
                    continue
                if c_b not in hash_table_doc_id_b.keys():
                    continue
                has_found_doc_id_a = has_found_doc_id_a + hash_table_doc_id_a[c_a]
                has_found_doc_id_b = has_found_doc_id_b + hash_table_doc_id_b[c_b]
                intersection_ab = list(set(has_found_doc_id_a).intersection(set(has_found_doc_id_b)))
                if len(intersection_ab) >= k:
                    intersection_ab = intersection_ab[:100]
                    break
            if len(intersection_ab) >= k:
                break
        # 计算precision, 先将doc_id转化为对应的label
        label_list = [id2label[x] for x in intersection_ab]
        prec = np.sum(np.array(label_list) == label.item()) / k
        precision = precision + prec
    return precision /  number