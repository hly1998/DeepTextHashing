import pickle
import numpy as np
import glob
import time
import os
import multiprocessing as mp
import argparse
from scipy.io import savemat
import random

import matplotlib
import matplotlib.pyplot as plt

try:
    from .eval_helpers import eval_hashing, acc_top_k_iterative, eval_min_hashing_iterative, get_labels_and_indices, compute_hitrate, compute_hitrate_iterative
except:
    from eval_helpers import eval_hashing, acc_top_k_iterative, eval_min_hashing_iterative, get_labels_and_indices, compute_hitrate, compute_hitrate_iterative


os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

def compactbit(b, wordsize=8):
    nSamples, nbits = b.shape
    nwords = int(nbits/wordsize);

    cb = np.zeros((nwords, nSamples), dtype=np.uint8)

    for i in range(nSamples):
        for j in range(nwords):
            word = "".join([str(v) for v in b[i, (j*wordsize):((j+1)*wordsize)].tolist()])
            cb[j, i] = int("0b"+word, 2)

    return cb


def greedy_minwise_correlation_split(codes, bits=32, n_buckets=4):
    correlation_matrix = np.corrcoef(codes.transpose())
    total_corr = np.sum(np.abs(correlation_matrix), -1)

    sorted_args = np.argsort(-total_corr)
    buckets = []
    for i in range(n_buckets):
        buckets.append([])

    def compute_sub_corr(entries, correlation_matrix):
        if len(entries) == 0 or len(entries) == 1:
            return 0
        if len(entries) == 9:
            return np.inf

        x_index = []
        y_index = []
        for x_entry in entries:
            for y_entry in entries:
                if x_entry != y_entry:
                    x_index.append(x_entry)
                    y_index.append(y_entry)

        total_corr = np.mean(np.abs(correlation_matrix[x_index,y_index]))
        return total_corr

    for entry in sorted_args:
        lowest_correlation_increase = np.inf
        bucket_i = 5
        for i in range(len(buckets)):
            bucket = buckets[i]
            mean_corr = compute_sub_corr(bucket,correlation_matrix)
            bucket_new = bucket + [entry]
            new_mean_corr = compute_sub_corr(bucket_new,correlation_matrix)
            change_corr = new_mean_corr-mean_corr
            if change_corr < lowest_correlation_increase:
                bucket_i = i
                lowest_correlation_increase = change_corr

        buckets[int(bucket_i)].append(entry)
    buckets = [item for sublist in buckets for item in sublist]
    return buckets

def hitrate_run_job(train_codes,val_codes, bits, n_blocks, jobs):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    best_switch = (-1,-1)
    best_hitrate = np.inf

    for job in jobs:
        hitrate,violations, _ = compute_hitrate_iterative(train_codes, val_codes, bits, n_blocks, switch_i=job[0], switch_j=job[1])
        if hitrate < best_hitrate:
            best_hitrate = hitrate
            best_switch = (job[0],job[1])

    return (best_hitrate, best_switch)

def run_iteration(train_codes,val_codes, test_codes, bits, n_blocks, threads, report_test = True, rotate_order = None, max_switch=100):
    hitrate_initial, violations_initial, _ = compute_hitrate_iterative(train_codes, val_codes, bits, n_blocks)
    print("initial hitrate:", hitrate_initial, "initial violations:", violations_initial)
    if rotate_order is None:
        global_order = np.arange(bits)
    else:
        global_order = rotate_order

    train_codes = np.ndarray.astype(train_codes,dtype=np.float32)
    val_codes = np.ndarray.astype(val_codes,dtype=np.float32)
    if report_test:
        test_codes =np.ndarray.astype(test_codes,dtype=np.float32)

    jobs = []
    for i in range(bits):
        for j in range(i+1,bits):
            jobs.append((i,j))

    jobs = np.array_split(jobs,threads)
    pool = mp.Pool(threads)
    best_hitrate_global = hitrate_initial
    hitrate_test = -1
    for counter in range(max_switch):
        t = time.time()
        res = pool.starmap_async(hitrate_run_job, [[train_codes,val_codes, bits, n_blocks, jobs[i]] for i in range(len(jobs))])
        res = res.get()

        best_switch = (None, None)
        best_hitrate = best_hitrate_global
        found_better = False
        for hitrate, switch in res:
            if hitrate < best_hitrate:
                best_hitrate = hitrate
                best_switch = (switch[0], switch[1])
                found_better = True

        if found_better:
            order = np.arange(bits)
            order[best_switch[0]] = best_switch[1]
            order[best_switch[1]] = best_switch[0]

            train_codes = train_codes[:, order]
            val_codes = val_codes[:, order]

        if report_test:
            test_codes = test_codes[:, order]
        if best_hitrate_global>best_hitrate:
            best_hitrate_global=best_hitrate

            tmp = global_order[best_switch[0]]
            global_order[best_switch[0]] = global_order[best_switch[1]]
            global_order[best_switch[1]] = tmp
            if report_test:
                hitrate_test, _, _ = compute_hitrate_iterative(train_codes,test_codes,bits,n_blocks)
                print("current best val hitrate: ", best_hitrate_global, "current test hitrate: ", hitrate_test,"counter: ", counter)
            else:
                print("current best val hitrate: ", best_hitrate_global, "counter: ", counter)
        else:
            break
        print("time taken: ", time.time()-t)
    pool.close()
    pool.join()

    hitrate_val, violations_val, _ = compute_hitrate_iterative(train_codes, test_codes, bits, n_blocks)
    if report_test:
        hitrate_test, violations_test, _ = compute_hitrate_iterative(train_codes, test_codes, bits, n_blocks)
        print("best greedy hitrate:", hitrate_val, "hit rate at best val: ", hitrate_test)
        print("best greedy violations:", violations_val, "violations at best val: ", violations_test)
    else:
        print("best greedy hitrate:", hitrate_val, "violations val",violations_val )

    #return order, val, test
    return global_order, hitrate_val, hitrate_test

def loadLog(path):
    content = pickle.load(open(path + "res.pkl","rb"))
    best_embeddings, args, best_val_loss, train_count, vae_val, test_perf, val_perf, \
    totallosses, doclosses, traindists, valdists, testdists, val_iter_perf, test_iter_perf, val_hitrate, test_hitrate, \
    list_val_violations, list_test_violations, list_val_avg_topK, list_test_avg_topK = content[0], content[1], content[2], \
         content[3], content[4], content[5], content[6], content[7], content[8], content[9], content[10], \
         content[11], content[12], content[13], content[14], content[15], content[16], content[17], content[18], content[19]

    return  best_embeddings, args, best_val_loss, train_count, vae_val, test_perf, val_perf, \
    totallosses, doclosses, traindists, valdists, testdists, val_iter_perf, test_iter_perf, val_hitrate, test_hitrate, list_val_violations, list_test_violations, list_val_avg_topK, list_test_avg_topK


def postHocMinimiseHitrate(path, data_name, threads):
    labels, train_indices, val_indices, test_indices, data_text_vect, id2token, num_labels = get_labels_and_indices(data_name)

    best_embeddings, args, best_val_loss, train_count, vae_val, test_perf, val_perf, \
    totallosses, doclosses, traindists, valdists, testdists, val_iter_perf, test_iter_perf, val_hitrate, test_hitrate, list_val_violations, list_test_violations, list_val_avg_topK, list_test_avg_topK \
         = loadLog(path)

    train_embedding = best_embeddings[train_indices,:]
    val_embedding = best_embeddings[val_indices, :]
    test_embedding = best_embeddings[test_indices, :]


    bits = train_embedding.shape[1]
    n_blocks = int(bits/8)

    #print("initial hitrate val:", compute_hitrate(train_embedding,val_embedding,bits,n_blocks,switch_i=0,switch_j=0), "initial hitrate test:", compute_hitrate(train_embedding,test_embedding,bits,n_blocks,switch_i=0,switch_j=0))
    global_order, best_hitrate_global, hitrate_test = run_iteration(train_embedding,val_embedding,test_embedding, bits, n_blocks, threads=threads)
    return best_hitrate_global, hitrate_test

def easy_compute_hitrate(path, data_name, threads):
    labels, train_indices, val_indices, test_indices, data_text_vect, id2token, num_labels = get_labels_and_indices(data_name)

    best_embeddings, args, best_val_loss, train_count, vae_val, test_perf, val_perf, \
    totallosses, doclosses, traindists, valdists, testdists, val_iter_perf, test_iter_perf, val_hitrate, test_hitrate\
         = loadLog(path)

    train_embedding = best_embeddings[train_indices,:]
    #val_embedding = best_embeddings[val_indices, :]
    test_embedding = best_embeddings[test_indices, :]


    bits = train_embedding.shape[1]
    n_blocks = int(bits/8)
    hr = compute_hitrate(train_embedding,test_embedding,bits,n_blocks,switch_i=0,switch_j=0)
    return hr

def easy_compute_hitrate_iterative(path, data_name, threads):
    labels, train_indices, val_indices, test_indices, data_text_vect, id2token, num_labels = get_labels_and_indices(data_name)

    best_embeddings, args, best_val_loss, train_count, vae_val, test_perf, val_perf, \
    totallosses, doclosses, traindists, valdists, testdists, val_iter_perf, test_iter_perf, val_hitrate, test_hitrate\
         = loadLog(path)

    train_embedding = best_embeddings[train_indices,:]
    #val_embedding = best_embeddings[val_indices, :]
    test_embedding = best_embeddings[test_indices, :]


    bits = train_embedding.shape[1]
    n_blocks = int(bits/8)
    hr = compute_hitrate_iterative(train_embedding,test_embedding,bits,n_blocks)
    return hr
