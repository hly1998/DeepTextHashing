import numpy as np
import torch
from tqdm import tqdm
import pickle
import random

def set_seed():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

# def compute_precision_at_k_new(query_b, doc_b, query_labels, doc_labels, topK):
#     # 确保输入在 GPU 上
#     assert query_b.is_cuda and doc_b.is_cuda, "Tensors must be on GPU"
    
#     # 计算相似度（比如余弦相似度）
#     similarity = torch.matmul(query_b, doc_b.T)
    
#     # 获取每个查询向量的前K个最相似的文档索引
#     _, top_k_indices = torch.topk(similarity, topK, dim=1)
    
#     # 获取对应的文档标签
#     top_k_labels = doc_labels[top_k_indices]
    
#     # 计算 precision@K
#     correct = top_k_labels.eq(query_labels.unsqueeze(1)).sum(dim=1)
#     precision_at_k = correct.float() / topK
    
#     return precision_at_k.mean()

def compute_precision_at_k_fast(query_b, doc_b, query_labels, doc_labels, topK):
    """
    Computes precision@K for binary vectors using Hamming distance.

    Parameters:
    - query_b: torch.Tensor, binary query vectors of shape (num_queries, num_features)
    - doc_b: torch.Tensor, binary document vectors of shape (num_docs, num_features)
    - query_labels: torch.Tensor, labels for the queries of shape (num_queries,)
    - doc_labels: torch.Tensor, labels for the documents of shape (num_docs,)
    - topK: int, the number of top documents to consider for precision

    Returns:
    - precision_at_k: float, average precision@K over all queries
    """
    # Calculate Hamming distance
    hamming_dist = torch.cdist(query_b.float(), doc_b.float(), p=0)
    
    # Get topK indices with smallest Hamming distance
    topk_indices = torch.topk(hamming_dist, topK, largest=False).indices
    
    # Calculate precision@K
    precision_sum = 0.0
    for i, indices in enumerate(topk_indices):
        relevant_docs = (doc_labels[indices] == query_labels[i]).sum().item()
        precision_sum += relevant_docs / topK
    
    precision_at_k = precision_sum / query_b.size(0)
    return precision_at_k

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