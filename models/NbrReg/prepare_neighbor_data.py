import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

import numpy as np
import os
from tqdm import tqdm
import scipy.io
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap
import pandas as pd
import argparse

def get_config():
    config = {
        "dataset": "reuters",
        "usetrain": True,
    }
    return config

def Load_Dataset(dataset):
    # dataset = scipy.io.loadmat(filename)
    absolute_path = os.path.abspath(os.getcwd())
    data_path = absolute_path + '/../../datasets'

    train = pd.read_pickle("{}/{}/train.tfidf.df.pkl".format(data_path,dataset))
    test = pd.read_pickle("{}/{}/test.tfidf.df.pkl".format(data_path,dataset))
    cv = pd.read_pickle("{}/{}/cv.tfidf.df.pkl".format(data_path,dataset))
    
    # doc_bow = self.df.iloc[idx].bow
    # doc_bow = torch.from_numpy(doc_bow.toarray().squeeze().astype(
    #     np.float32))
    # label = self.df.iloc[idx].label

    # x_train = dataset['train']
    # x_test = dataset['test']
    # x_cv = dataset['cv']
    # y_train = dataset['gnd_train']
    # y_test = dataset['gnd_test']
    # y_cv = dataset['gnd_cv']

    # x_train = train.bow
    # y_train = train.label
    # print(y_train.tolist())
    # exit()
    # x_test = test.bow
    # y_test = test.label
    # x_cv = cv.bow
    # y_cv = cv.label
    
    x_train = np.array([csr_data.toarray()[0] for csr_data in train.bow.tolist()])
    x_test = np.array([csr_data.toarray()[0] for csr_data in train.bow.tolist()])
    x_cv = np.array([csr_data.toarray()[0] for csr_data in train.bow.tolist()])

    # x_test = dataset['test']
    # x_cv = dataset['cv']
    # y_train = dataset['gnd_train']
    # y_test = dataset['gnd_test']
    # y_cv = dataset['gnd_cv']
    
    data = DotMap()
    # data.n_trains = y_train.shape[0]
    # data.n_tests = y_test.shape[0]
    # data.n_cv = y_cv.shape[0]
    # data.n_tags = y_train.shape[1]
    data.n_feas = x_train.shape[1]

    ## Convert sparse to dense matricesimport numpy as np
    train = x_train
    nz_indices = np.where(np.sum(train, axis=1) > 0)[0]
    train = train[nz_indices, :]
    train_len = np.sum(train > 0, axis=1)
    train_len = np.squeeze(np.asarray(train_len))

    test = x_test
    test_len = np.sum(test > 0, axis=1)
    test_len = np.squeeze(np.asarray(test_len))

    if x_cv is not None:
        cv = x_cv
        cv_len = np.sum(cv > 0, axis=1)
        cv_len = np.squeeze(np.asarray(cv_len))
    else:
        cv = None
        cv_len = None
        
    # gnd_train = y_train[nz_indices, :]
    # gnd_test = y_test
    # gnd_cv = y_cv

    data.train = train
    data.test = test
    data.cv = cv
    data.train_len = train_len
    data.test_len = test_len
    data.cv_len = cv_len
    # data.gnd_train = gnd_train
    # data.gnd_test = gnd_test
    # data.gnd_cv = gnd_cv
    
    return data


def GetTopK_UsingCosineSim(outfn, queries, documents, TopK, queryBatchSize=10, docBatchSize=100):
    
    n_docs = documents.shape[0]
    n_queries = queries.shape[0]
    query_row = 0
    
    with open(outfn, 'w') as out_fn:
        for q_idx in tqdm(range(0, n_queries, queryBatchSize), desc='Query', ncols=0):
            query_batch_s_idx = q_idx
            query_batch_e_idx = min(query_batch_s_idx + queryBatchSize, n_queries)

            # queryMats = torch.cuda.FloatTensor(queries[query_batch_s_idx:query_batch_e_idx].toarray())
            queryMats = torch.cuda.FloatTensor(queries[query_batch_s_idx:query_batch_e_idx])
            queryNorm2 = torch.norm(queryMats, 2, dim=1)
            queryNorm2.unsqueeze_(1)
            queryMats.unsqueeze_(2)

            scoreList = []
            indicesList = []

            #print('{}: perform cosine sim ...'.format(q_idx))
            for idx in tqdm(range(0, n_docs, docBatchSize), desc='Doc', leave=False, ncols=0):
                batch_s_idx = idx
                batch_e_idx = min(batch_s_idx + docBatchSize, n_docs)
                n_doc_in_batch = batch_e_idx - batch_s_idx

                #if batch_s_idx > 1000:
                #    break

                # candidateMats = torch.cuda.FloatTensor(documents[batch_s_idx:batch_e_idx].toarray())
                candidateMats = torch.cuda.FloatTensor(documents[batch_s_idx:batch_e_idx])

                candidateNorm2 = torch.norm(candidateMats, 2, dim=1)
                candidateNorm2.unsqueeze_(0)

                candidateMats.unsqueeze_(2)
                candidateMats = candidateMats.permute(2, 1, 0)

                # compute cosine similarity
                queryMatsExpand = queryMats.expand(queryMats.size(0), queryMats.size(1), candidateMats.size(2))
                candidateMats = candidateMats.expand_as(queryMatsExpand)

                cos_sim_scores = torch.sum(queryMatsExpand * candidateMats, dim=1) / (queryNorm2 * candidateNorm2)

                K = min(TopK, n_doc_in_batch)
                scores, indices = torch.topk(cos_sim_scores, K, dim=1, largest=True)

                del cos_sim_scores
                del queryMatsExpand
                del candidateMats
                del candidateNorm2

                scoreList.append(scores)
                indicesList.append(indices + batch_s_idx)

            all_scores = torch.cat(scoreList, dim=1)
            all_indices = torch.cat(indicesList, dim=1)
            _, indices = torch.topk(all_scores, TopK, dim=1, largest=True)

            topK_indices = torch.gather(all_indices, 1, indices)
            #all_topK_indices.append(topK_indices)
            #all_topK_scores.append(scores)

            del queryMats
            del queryNorm2
            del scoreList
            del indicesList

            topK_indices = topK_indices.cpu().numpy()
            for row in topK_indices:
                out_fn.write("{}:".format(query_row))
                outtext = ','.join([str(col) for col in row])
                out_fn.write(outtext)
                out_fn.write('\n')
                query_row += 1

            torch.cuda.empty_cache()

#################################################################################################################

if __name__ == "__main__":
    config = get_config()
    dataset = config["dataset"]
    usetrain = config["usetrain"]    
    data = Load_Dataset(dataset)
    
    # print("num train:{} num tests:{} num cv:{}".format(data.n_trains, data.n_tests, data.cv_len))

    if usetrain:
        print("use train as a query corpus")
        query_corpus = data.train
        out_fn = "./neighbor_data/{}_train_top101.txt".format(dataset)
    else:
        print("use test as a query corpus")
        query_corpus = data.test
        out_fn = "./neighbor_data/{}_test_top101.txt".format(dataset)

    print("save the result to {}".format(out_fn))
    GetTopK_UsingCosineSim(out_fn, query_corpus, data.train, TopK=101, queryBatchSize=500, docBatchSize=100)
