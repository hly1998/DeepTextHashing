# 测试IndexBinaryHash和IndexBinaryMultiHash随着搜索半径的增加，检索时间的变化
import os
os.environ["OMP_NUM_THREADS"] = "4"
import faiss
import pickle
import numpy as np
import time

def make_binary_dataset(d, nt, nb, nq):
    '''
    生成测试数据
    d: dimension
    nt: test code num
    nb-nt: candidate code num
    nq-nb: query code num
    ===
    return:
    para1: the test code
    para2: the condidate code
    para3: the query code
    '''
    assert d % 8 == 0
    rs = np.random.RandomState(43)
    x = rs.randint(256, size=(nb + nq + nt, int(d / 8))).astype('uint8')
    return x[:nt], x[nt:-nq], x[-nq:]

# def IndexBinaryFlat_for_random_data():
#     '''
#     a description of IndexBinaryFlat(int d, int b)
#     使用前b个比特构建索引，哈希码的维度为d
#     '''
#     d = 16
#     nq = 100
#     nb = 2000
#     (_, xb, xq) = make_binary_dataset(d, 0, nb, nq)
#     index_ref = faiss.IndexBinaryFlat(d)
#     index_ref.add(xb)
#     radius = 55
#     # in Python, the results are returned as a triplet of 1D arrays lims, D, I, where result for query i is in I[lims[i]:lims[i+1]] (indices of neighbors), D[lims[i]:lims[i+1]] (distances).
#     Lref, Dref, Iref = index_ref.range_search(xq, radius)
#     print("nb res: ", Lref[-1])

def bit2int(c):
    n = 0
    for x in c[:-1]:
        n = (n + x) << 1
    n = n + c[-1]
    return n

def bit2int8(codes):
    '''
    提取8个bit转为0-256之间的值
    '''
    bits_num = codes.shape[1]
    int8len = int(bits_num / 8)
    new_codes = []
    for code in codes:
        new_code = []
        for i in range(int8len):
            new_code.append(bit2int(code[i:(i+1)*8]))
        new_codes.append(new_code)
    new_codes = np.array(new_codes).astype('uint8')
    return new_codes

def IndexBinaryHash_test(d, nq, nb, topk):
    '''
    Using the VDSH model to generate hash codes of ng20 dataset 
    '''
    # d = 16
    # nq = 100
    # nb = 2000
    (_, xb, xq) = make_binary_dataset(d, 0, nb, nq)
    print("data ok")
    # print(xb.shape)
    # exit()
    bit_num = d
    index = faiss.IndexBinaryHash(bit_num, bit_num)
    index.add(xb)
    stats = faiss.cvar.indexBinaryHash_stats
    index.nflip = 2
    stats.reset()
    s = time.time()
    # D[i, j] contains the distance from the i-th query vector to its j-th nearest neighbor.
    # I[i, j] contains the id of the j-th nearest neighbor of the i-th query vector.
    # D, I = index.search(xq, 100)
    # for q in xq:
    #     # index.range_search(np.expand_dims(q, axis=0), r)
    #     index.search(np.expand_dims(q, axis=0), topk)
    index.search(xq, topk)
    # index.range_search(xq, r)
    e = time.time()
    print("topk:", topk ,"time: ", (e - s)/nq)


def IndexBinaryMultiHash_test(d, nq, nb, topk):
    '''
    test the multi-index hashing
    '''
    # d = 8
    # nq = 3000
    # nb = 20000
    # # radius = 55
    # nfound = []
    # ndis = []
    (_, xb, xq) = make_binary_dataset(d, 0, nb, nq)
    nh = 4
    index = faiss.IndexBinaryMultiHash(d, nh, d // nh)
    index.add(xb)
    # index.display()
    stats = faiss.cvar.indexBinaryHash_stats
    # index.nflip = d // nh
    index.nflip = 2
    stats.reset()
    # Lnew, Dnew, Inew = index.range_search(xq, radius)
    # nfound.append(Lnew[-1])
    # ndis.append(stats.ndis)
    # print('nfound=', nfound)
    # print('ndis=', ndis)
    s = time.time()
    index.search(xq, topk)
    e = time.time()
    print("topk:", topk ,"time: ", (e - s)/nq)

IndexBinaryHash_test(16, 100, 10000000, 100)

# for r in [2,3,4,6,7]:
#     IndexBinaryHash_test(16, 100, 10000000, r)