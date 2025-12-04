import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
import multiprocessing as mp

import os
#os.environ['OPENBLAS_NUM_THREADS'] = '6'
#os.environ['MKL_NUM_THREADS'] = '6'


def compute_hitrate(train_codes,val_codes,bits,n_blocks,switch_i=0,switch_j=0):
    train_codes = np.copy(train_codes)
    val_codes = np.copy(val_codes)
    if switch_j != switch_i:
        order = np.arange(bits)
        order[switch_i] = switch_j
        order[switch_j] = switch_i
        train_codes = train_codes[:,order]
        val_codes = val_codes[:,order]

    n_bits_block = int(bits/n_blocks)
    train_samples, _ = train_codes.shape
    val_samples,_ = val_codes.shape

    collision = np.zeros((train_samples, val_samples),dtype=np.bool)
    for i in range(n_blocks):
        train = train_codes[:,(i*n_bits_block):((i+1)*n_bits_block)]
        val = val_codes[:, (i * n_bits_block):((i + 1) * n_bits_block)]
        tmp_collision = np.dot(train,val.transpose()) == n_bits_block
        collision = np.logical_or(collision, tmp_collision)

    hitrate = np.sum(collision) / (train_samples*val_samples)
    return hitrate

#simulate a retrieval system where we retrieve uptil dist_k and then sort as best as possible.
def compute_hit_prec_at_dist_k(train_codes, val_codes, bits, n_blocks, dist_k):
    pass

def threaded_compute_hitrate_iterative(train_codes, val_codes, bits, n_blocks, topK = 100, max_search = 4, threads = 4):
    splitted_val = np.array_split(val_codes, threads)
    splitted_val_lengths = [x.shape[0] for x in splitted_val]

    pool = mp.Pool(threads)
    res = pool.starmap_async(call_compute_hitrate_iterative,
                             [[train_codes, splitted_val[i], bits, n_blocks, topK, max_search]
                              for i in range(len(splitted_val))])
    res = res.get()

    total_samples = val_codes.shape[0]
    hitrate = 0
    violations = 0
    sort_distance = 0
    #hitrate, violations, sort_distance
    for i in range(threads):
        hitrate += res[i][0] * splitted_val_lengths[i] / (total_samples)
        violations += res[i][1] * splitted_val_lengths[i] / (total_samples)
        sort_distance += res[i][2] * splitted_val_lengths[i] / (total_samples)

    pool.close()
    pool.join()
    return hitrate, violations, sort_distance

def call_compute_hitrate_iterative(train_codes, val_codes, bits, n_blocks, topK, max_search):
    return compute_hitrate_iterative(train_codes, val_codes, bits, n_blocks, topK=topK, max_search=max_search)

def compute_hitrate_iterative(train_codes, val_codes, bits, n_blocks, topK = 100, switch_i=0,switch_j=0,
                              split_job=1, max_search = 4, skip_hitrate_comp = False):
    #everything below can be done less memory intense by splitting the val codes
    if switch_j != switch_i:
        train_codes = np.copy(train_codes)
        val_codes = np.copy(val_codes)
        order = np.arange(bits)
        order[switch_i] = switch_j
        order[switch_j] = switch_i
        train_codes = train_codes[:,order]
        val_codes = val_codes[:,order]

    n_bits_block = int(bits / n_blocks)
    train_samples, _ = train_codes.shape

    splitted_val = np.array_split(val_codes,split_job)
    splitted_val_lengths = [x.shape[0] for x in splitted_val]

    hitrate_l = []
    violations_l =[]
    sort_distance_l = []
    for val_codes in splitted_val:
        val_samples, _ = val_codes.shape

        similarity = np.dot(val_codes, train_codes.T)
        distance = (bits-similarity)/2
        sort_distance = np.sort(distance)
        distance100 = sort_distance[:,topK]

        #currently only go up to number of blocks... fix later !
        #distance100[distance100 >= n_blocks] = n_blocks-1
        distance100 = np.expand_dims(distance100,-1)
        violations = np.mean(distance100 > (n_blocks*max_search-1))

        if not skip_hitrate_comp:
            collision = np.zeros((val_samples, train_samples),dtype=np.bool)
            #for i in range(n_blocks):
            block_dist = np.zeros((val_codes.shape[0], train_codes.shape[0],n_blocks))
            for i in range(n_blocks):
                train = train_codes[:, (i * n_bits_block):((i + 1) * n_bits_block)]
                val = val_codes[:, (i * n_bits_block):((i + 1) * n_bits_block)]
                block_dist[:, :, i] = np.dot(val, train.transpose())

            for j in range(max_search):
                for i in range(n_blocks):
                    '''
                    train = train_codes[:,(i*n_bits_block):((i+1)*n_bits_block)]
                    val = val_codes[:, (i * n_bits_block):((i + 1) * n_bits_block)]
                    tmp_collision = np.dot(val,train.transpose()) == (n_bits_block-j*2)
                    '''
                    tmp_collision = block_dist[:, :, i] == (n_bits_block-j*2)
                    tmp_collision = (distance100>=(i+n_blocks*j))*tmp_collision
                    collision = np.logical_or(collision, tmp_collision)

            hitrate = np.sum(collision) / (train_samples*val_samples)
        else:
            hitrate = -1

        hitrate_l.append(hitrate)
        violations_l.append(violations)
        sort_distance_l.append(np.mean(sort_distance[:,topK]))

    splitted_val_lengths_sum = np.sum(splitted_val_lengths)

    hitrate=0
    violations=0
    sort_distance=0
    for i in range(len(hitrate_l)):
        hitrate += hitrate_l[i]*(splitted_val_lengths[i]/splitted_val_lengths_sum)
        violations += violations_l[i] * (splitted_val_lengths[i] / splitted_val_lengths_sum)
        sort_distance += sort_distance_l[i] * (splitted_val_lengths[i] / splitted_val_lengths_sum)

    return hitrate, violations, sort_distance


def compute_hitrate_iterative_count():
    pass

def compute_hitrate_topk(train_codes,val_codes,bits,n_blocks,topK):
    train_codes = np.copy(train_codes)
    val_codes = np.copy(val_codes)

    n_bits_block = int(bits/n_blocks)
    train_samples, _ = train_codes.shape
    val_samples,_ = val_codes.shape

    collision = np.zeros((train_samples, val_samples),dtype=np.bool)
    for i in range(n_blocks):
        train = train_codes[:,(i*n_bits_block):((i+1)*n_bits_block)]
        val = val_codes[:, (i * n_bits_block):((i + 1) * n_bits_block)]
        tmp_collision = np.dot(train,val.transpose()) == n_bits_block
        collision = np.logical_or(collision, tmp_collision)

    hitrate = np.sum(collision) / (train_samples*val_samples)
    return hitrate

def get_labels_and_indices(dname):
    collection = pickle.load(open("../data/" + dname + "_collections", "rb"))
    _, training, _, validation, testing, _, data_text_vect, labels, _, id2token = collection

    all_labels = []
    for tmp in labels:
        all_labels += tmp
    all_labels = list(set(all_labels))
    label_dict = {}
    for i, label in enumerate(all_labels):
        label_dict[label] = str(i)

    for i in range(len(labels)):
        labels[i] = [label_dict[v] for v in labels[i]]

    train_indices = training[-1]
    val_indices = validation[-1]
    test_indices = testing[-1]

    if "reuters" in dname:
        num_labels = 88
    elif "TMC" in dname:
        num_labels = 22
    elif "20news" in dname:
        num_labels = 20
    elif "agnews" in dname:
        num_labels = 4
    else:
        raise Exception("unknown dname", dname)

    return labels, train_indices, val_indices, test_indices, data_text_vect, id2token, num_labels

def eval_hashing(train_vectors, train_labels, val_vectors, val_labels, medianTrick=False):

    train_vectors = np.array(train_vectors)
    val_vectors = np.array(val_vectors)

    if medianTrick:
        medians = np.median(train_vectors, 0)
        train_vectors = (train_vectors > medians).astype(int)
        val_vectors = (val_vectors > medians).astype(int)

    upto = 100
    top100_precisions = []

    #for vali in range(len(val_vectors)):
    knn = NearestNeighbors(n_neighbors=upto, metric="manhattan", n_jobs=2)#, algorithm="brute")
    used_train_vectors = train_vectors
    use_val_vector = val_vectors
    use_val_labels = val_labels

    #used_train_vectors = used_train_vectors * use_val_vector[0]
    knn.fit(used_train_vectors)

    dists, nns = knn.kneighbors(use_val_vector, upto, return_distance=True)
    for i, nn_indices in enumerate(nns):
        eval_label = use_val_labels[i]
        matches = np.zeros(upto)
        for j, idx in enumerate(nn_indices):
            if any([label in train_labels[idx] for label in eval_label]):
                matches[j] = 1
        top100_precisions.append(np.mean(matches))

    return top100_precisions, np.mean(np.mean(dists, 0))



def threaded_acc_top_k_tie_aware(train_vectors, train_labels, val_vectors, val_labels, topK, num_labels, threads = 4, medianTrick = False):
    splitted_val = np.array_split(val_vectors,threads)
    splitted_val_labels =  np.array_split(val_labels,threads)
    splitted_val_lengths = [x.shape[0] for x in splitted_val]

    pool = mp.Pool(threads)
    res = pool.starmap_async(acc_top_k_tie_aware, [[train_vectors, train_labels, splitted_val[i], splitted_val_labels[i], topK, num_labels] for i in range(len(splitted_val))])
    res = res.get()

    total_samples = val_vectors.shape[0]
    tie_aware_hitrate = 0
    lower_bound_hitrate = 0
    for i in range(threads):
        tie_aware_hitrate += res[i][0] * splitted_val_lengths[i]/(total_samples)
        lower_bound_hitrate += res[i][1] * splitted_val_lengths[i] / (total_samples)
    pool.close()
    pool.join()
    return tie_aware_hitrate, lower_bound_hitrate

def acc_top_k_tie_aware(train_vectors, train_labels, val_vectors, val_labels, topK, num_labels, medianTrick = False):
    train_vectors = np.asarray(train_vectors,dtype=np.int32)
    val_vectors = np.asarray(val_vectors, dtype=np.int32)

    topK = topK[0]

    np_train_labels = np.zeros((len(train_labels), num_labels),np.int32)
    np_val_labels = np.zeros((len(val_labels), num_labels), np.int32)

    bits = train_vectors.shape[1]
    for i,e in enumerate(train_labels):
        for j in e:
            np_train_labels[i,int(j)] = 1
    for i,e in enumerate(val_labels):
        for j in e:
            np_val_labels[i,int(j)] = 1

    train_labels = np_train_labels
    val_labels = np_val_labels


    if medianTrick:
        medians = np.median(train_vectors, 0)
        train_vectors = (train_vectors > medians).astype(int)
        val_vectors = (val_vectors > medians).astype(int)


    dist = (bits - np.dot(val_vectors,train_vectors.transpose()))/2
    match = (np.dot(val_labels,train_labels.transpose()) >= 1) *1.0
    sort_idx = np.argsort(dist)

    dist_sorted = np.take_along_axis(dist, sort_idx, 1)
    match_sorted = np.take_along_axis(match, sort_idx, 1)
    dist_100 = np.expand_dims(dist_sorted[:,(topK-1)],-1)

    #no ties:
    no_ties = dist_sorted < (dist_100)
    no_ties_count = np.sum(no_ties,-1)
    no_ties_hit = np.sum(match_sorted*no_ties,-1)
    no_ties_count_is_0 = no_ties_count == 0

    #the below equation is only to handle edge case when count is 0, in this case the no_ties_hit will be 0 so 0/1 still 0
    no_ties_hitrate = no_ties_hit/(no_ties_count+no_ties_count_is_0)

    #ties, there must be atleast one with this distance, as the k element have this distance
    ties = dist_sorted == dist_100
    ties_count =  np.sum(ties,-1)
    ties_hit = np.sum(match_sorted*ties,-1)
    ties_hitrate = ties_hit/ties_count

    tie_aware_hitrate = no_ties_hitrate * (no_ties_count / topK) + ties_hitrate * (1 - no_ties_count / topK)


    #first calculate how much of the tie could be filled out with non ties
    bad_matches_ties = 1 - (ties_count * (1 - ties_hitrate)) / (topK - no_ties_count)
    #this is lower bounded by 0:
    bad_matches_ties = (bad_matches_ties>0)*bad_matches_ties
    lower_bound_hitrate = no_ties_hitrate * (no_ties_count / topK) + (1 - no_ties_count / topK) * bad_matches_ties

    return np.mean(tie_aware_hitrate), np.mean(lower_bound_hitrate),

def acc_top_k_iterative(train_vectors, train_labels, val_vectors, val_labels, topK, num_labels, medianTrick = False):
    train_vectors = train_vectors > 0
    val_vectors = val_vectors > 0
    train_vectors = np.asarray(train_vectors,dtype=np.int32)
    val_vectors = np.asarray(val_vectors, dtype=np.int32)


    np_train_labels = np.zeros((len(train_labels), num_labels),np.int32)
    np_val_labels = np.zeros((len(val_labels), num_labels), np.int32)

    for i,e in enumerate(train_labels):
        for j in e:
            np_train_labels[i,int(j)] = 1
    for i,e in enumerate(val_labels):
        for j in e:
            np_val_labels[i,int(j)] = 1

    train_labels = np_train_labels
    val_labels = np_val_labels


    if medianTrick:
        medians = np.median(train_vectors, 0)
        train_vectors = (train_vectors > medians).astype(int)
        val_vectors = (val_vectors > medians).astype(int)

    #some hardcodings for what is feasible to minwise hash
    bits = train_vectors.shape[1]
    if bits==32:
        initial_search_radii = 3
        maximum_search_radii = np.inf
    elif bits==64:
        initial_search_radii = 7
        maximum_search_radii = np.inf
    elif bits==128:
        initial_search_radii = 15
        maximum_search_radii = np.inf
    else:
        raise Exception("Not implemented this amount of bits")

    #first gaurentee that we have all occurences with the same distance as the last in topK
    upto = 4000
    while (1):
        knn = NearestNeighbors(n_neighbors=upto, metric="manhattan", n_jobs=3)
        knn.fit(train_vectors)
        nns_dist, nns = knn.kneighbors(val_vectors, upto, return_distance=True)

        to_con=True
        for k in topK:
            if np.logical_or.reduce(nns_dist[:, k - 1] == nns_dist[:, upto - 1]):
                to_con = to_con and False

        if to_con or upto >= train_vectors.shape[0]:
            break

        # in case we did not get all in sarch radii, just increase it
        upto = np.min([upto * 2,train_vectors.shape[0]])


    #we now know that we have all the necesarry points
    topK_precission = np.zeros((len(nns),len(topK)))
    topK_lower_precission = np.zeros((len(nns), len(topK)))
    topK_hits = np.zeros((len(nns),len(topK)))
    topK_exceed_minimum = np.zeros((len(nns),len(topK)))
    for i in range(len(nns)):
        eval_label = val_labels[i]
        nn_indices = nns[i,:]
        nn_dist = nns_dist[i,:]

        for j,k in enumerate(topK):
            #first check if we can find enough within initial_search_radii
            distance_at_eval = nn_dist[k - 1]
            if distance_at_eval <= initial_search_radii:
                pass #all good
            elif distance_at_eval <= maximum_search_radii:
                topK_exceed_minimum[i,j] = 1
            else:
                topK_exceed_minimum[i,j] = -1 #we do not succed in finding enough hits.

            if topK_exceed_minimum[i,j] >= 0:
                #first find the accuracy where there is not any potential ties. Edge case for distance_at_eval=0
                if distance_at_eval==0: #here we just find the avg precision for all with distance 0
                    labels_nn = train_labels[nn_indices[nn_dist <= distance_at_eval]]
                    prec = np.mean(np.sum(eval_label * labels_nn, 1) > 0)
                    if np.isnan(prec): #in case all distances are equal distance at eval.
                        prec = 0
                    number_of_hits_tie = len(labels_nn)

                    #compute worst case, where the tie breaks give worst possible outcome
                    lower_prec = np.max([1-(number_of_hits_tie*(1-prec))/k,0])
                else:
                    #fist find precission for all distance_at_eval-1
                    labels_nn = train_labels[nn_indices[nn_dist < distance_at_eval]]
                    prec_no_tie = np.mean(np.sum(eval_label * labels_nn, 1) > 0)
                    number_of_hits_no_tie = len(labels_nn)
                    if np.isnan(prec_no_tie): #in case all distances are equal distance at eval.
                        prec_no_tie = 0

                    #then find the precission for all the ties
                    labels_nn = train_labels[nn_indices[nn_dist == distance_at_eval]]
                    prec_tie = np.mean(np.sum(eval_label * labels_nn, 1) > 0)
                    number_of_hits_tie = len(labels_nn)

                    #compute precision as interpolation
                    prec = prec_no_tie*(number_of_hits_no_tie/k) + prec_tie*(1-number_of_hits_no_tie/k)
                    #compute worst case, where the tie breaks give worst possible outcome
                    lower_prec = prec_no_tie*(number_of_hits_no_tie/k) + (1-number_of_hits_no_tie/k) * np.max([1 - (number_of_hits_tie * (1 - prec_tie)) / (k-number_of_hits_no_tie), 0])

                topK_hits[i, j] = k #we know we have enough here, otherwise we would be in the else clause
                topK_precission[i,j] = prec
                topK_lower_precission[i,j] = lower_prec

            else:
                raise Exception('error in evaluation of hitrate, legacy code')
                '''
                #we can only go up to the maximum search radii, so we do not find enough hits
                labels_nn = train_labels[nn_indices[nn_dist <= maximum_search_radii]]
                prec = np.sum((np.sum(eval_label * labels_nn, 1) > 0),-1) / k
                if np.isnan(prec):  # in case there is not hits at maximum search radii
                    prec = 0
                topK_hits[i, j] = len(labels_nn)
                topK_precission[i,j] = prec
                topK_lower_precission[i,j] = prec
                '''

    return topK_precission, topK_lower_precission, topK_hits, topK_exceed_minimum

def eval_min_hashing_iterative(train_vectors, val_vectors, topK, medianTrick=False):
    #train_vectors = np.array(train_vectors)
    #val_vectors = np.array(val_vectors)
    train_vectors = train_vectors > 0
    val_vectors = val_vectors > 0
    train_vectors = np.asarray(train_vectors,dtype=np.int32)
    val_vectors = np.asarray(val_vectors, dtype=np.int32)

    if medianTrick:
        medians = np.median(train_vectors, 0)
        train_vectors = (train_vectors > medians).astype(int)
        val_vectors = (val_vectors > medians).astype(int)

    #some hardcodings for what is feasible to minwise hash
    bits = train_vectors.shape[1]
    if bits==32:
        initial_diff_per_block = 0
        maximum_diff_per_block = 1
        n_blocks = 4
        block_size = 8
    elif bits==64:
        initial_diff_per_block = 0
        maximum_diff_per_block = 1
        n_blocks = 8
        block_size = 8
    else:
        raise Exception("Not implemented this amount of bits")



    # total number of comparisons without min_hashing is just brute force search over all train
    brute_force_n = train_vectors.shape[0] * val_vectors.shape[0]


    def make_block_dits(max_distance_in_block):
        # make hash maps for each block:
        block_dicts = {}
        for i in range(n_blocks):
            block_dicts[i] = {}

        # make a hash map for each id
        # first find all the bit strings
        bit_codes = ['']
        for i in range(block_size):
            tmp = []
            for e in bit_codes:
                tmp.append(e + "0")
                tmp.append(e + "1")
            bit_codes = tmp

        # setup all the hash tables
        for block_dict in block_dicts.values():
            for bit_code in bit_codes:
                block_dict[bit_code] = []

        # only interested in the raw count at the moment for each table
        # run over train to get the population for each entry in the hash maps
        for c, bit_code in enumerate(train_vectors):
            for block_n in range(n_blocks):
                sub_bit_code = bit_code[block_n * block_size:(block_n + 1) * block_size]
                sub_bit_code = "".join([str(int(e)) for e in sub_bit_code])
                sub_bit_codes = [sub_bit_code]
                # each iteration find all permutations that is 1 further away
                for j in range(max_distance_in_block):
                    sub_bit_codes = list(set(
                        [s[:i] + to_replace + s[i + 1:] for to_replace in ["0", "1"] for i in range(block_size) for s in
                         sub_bit_codes]))

                for sub_bit_code in sub_bit_codes:
                    block_dicts[block_n][sub_bit_code].append(c)
        return block_dicts

    block_dict_initial = make_block_dits(initial_diff_per_block)
    maximum_diff_per_block = make_block_dits(maximum_diff_per_block)



    # run over the val set to see the number of comparisons needed:
    number_of_comparisons = np.zeros(len(topK))
    number_with_max = np.zeros(len(topK))
    number_still_not_satisfied = np.zeros(len(topK))
    for bit_code in val_vectors:
        to_compare_initial = []
        to_compare_maximum = []
        for block_n in range(n_blocks):
            sub_bit_code = bit_code[block_n * block_size:(block_n + 1) * block_size]
            sub_bit_code = "".join([str(int(e)) for e in sub_bit_code])

            initial_hits = block_dict_initial[block_n][sub_bit_code]
            maximum_hits = maximum_diff_per_block[block_n][sub_bit_code]

            to_compare_initial = to_compare_initial + initial_hits
            to_compare_maximum = to_compare_maximum + maximum_hits

        to_compare_initial = set(to_compare_initial)
        to_compare_maximum = set(to_compare_maximum)

        n_to_compare_initial = len(to_compare_initial)
        n_to_compare_maximum = len(to_compare_maximum)

        for i,k in enumerate(topK):
            if n_to_compare_initial < k:
                number_of_comparisons[i] += n_to_compare_maximum
                number_with_max[i] += 1

                if n_to_compare_maximum< k:
                    number_still_not_satisfied[i] += 1

            else:
                number_of_comparisons[i] += n_to_compare_initial

    return number_of_comparisons, number_with_max, number_still_not_satisfied, brute_force_n

def eval_correlation_block_level(binary, n_blocks):
    correlation_matrix = np.corrcoef(binary.transpose())
    n_bits = binary.shape[1]
    block_size = int(n_bits / n_blocks)

    #compute the correlation within block, and between blocks. Remove diagonals.
    diagonal_blocks = []
    off_diagonal_blocks = []
    for block_i in range(n_blocks):
        for block_j in range(n_blocks):
            block_correlation = np.abs(correlation_matrix[block_size*block_i:block_size*(1+block_i), block_size*block_j:block_size*(1+block_j)])
            if block_i == block_j:
                diagonal_blocks.append(block_correlation)
            else:
                off_diagonal_blocks.append(block_correlation)

    #remove diagonal in diagonal blocks
    tmp = []
    for block in diagonal_blocks:
        remove_diagonal_block = block[~np.eye(block.shape[0], dtype=bool)].reshape(block.shape[0], -1)
        tmp.append(remove_diagonal_block)
    diagonal_blocks = tmp

    #compute avg correlation in blocks and between blocks
    mean_diagonal_blocks = []
    for block in diagonal_blocks:
        block =block[~np.isnan(block)]
        mean_diagonal_blocks.append(np.mean(block))
    mean_diagonal_blocks = np.mean(mean_diagonal_blocks)

    mean_off_diagonal_blocks = []
    for block in off_diagonal_blocks:
        block =block[~np.isnan(block)]
        mean_off_diagonal_blocks.append(np.mean(block))
    mean_off_diagonal_blocks = np.mean(mean_off_diagonal_blocks)

    return mean_diagonal_blocks, mean_off_diagonal_blocks

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





if __name__ == "__main__":
    train_codes = np.ones((500, 32))
    val_codes = np.ones((100, 32))
    compute_hitrate_iterative(train_codes, val_codes, 32, 4, topK=100)
