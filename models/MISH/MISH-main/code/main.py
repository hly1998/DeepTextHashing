import tensorflow as tf
import pickle
import numpy as np
import glob
import argparse
from sklearn.neighbors import NearestNeighbors

try:
    from .nn_helpers import generator
except:
    from nn_helpers import generator
try:
    from .model import SemiHash
except:
    from model import SemiHash
try:
    from .eval_helpers import eval_hashing, acc_top_k_iterative, eval_min_hashing_iterative, get_labels_and_indices,\
        compute_hitrate_iterative, acc_top_k_tie_aware,threaded_acc_top_k_tie_aware, threaded_compute_hitrate_iterative
except:
    from eval_helpers import eval_hashing, acc_top_k_iterative, eval_min_hashing_iterative, get_labels_and_indices,\
        compute_hitrate_iterative, acc_top_k_tie_aware,threaded_acc_top_k_tie_aware, threaded_compute_hitrate_iterative

try:
    from .run_over_log import run_iteration
except:
    from run_over_log import run_iteration

import time

import os
#os.environ['OPENBLAS_NUM_THREADS'] = '6'
#os.environ['MKL_NUM_THREADS'] = '6'

def extract_vectors_labels(dist_opposite, dist_same, top_k_prec, sess, handle, specific_handle, num_samples, batch_placeholder, is_training, sigma_anneal_vae,
                           loss, emb_update, eval_batchsize, indices, labels, model, loss_doc_only, super_doc_only, rank_loss, nbr_loss, bit_balance_loss, prob_pair_loss,
                           problem_pair, problem_pair_val, rotate_order, val_rotate_order, mem_updates=None):
    total = num_samples
    done = False
    losses = []
    losses_doc = []

    sup_losses_doc = []
    rank_losses_doc = []
    nbr_losses_doc = []
    top_k_prec_vals = []

    bit_balance_loss_vals_l = []
    prob_pair_loss_vals_l = []


    dist_opposite_list = []
    dist_same_list = []


    start = time.time()
    while not done:
        #print(total)
        if mem_updates is None:
            dist_opposite_val, dist_same_val, top_k_prec_val, lossvals, _, loss_doc_only_vals, sup_losses_doc_vals, rank_losses_doc_vals, nbr_losses_doc_vals, bit_balance_loss_vals, prob_pair_loss_vals =\
                sess.run([dist_opposite, dist_same, top_k_prec, loss, emb_update, loss_doc_only, super_doc_only, rank_loss, nbr_loss, bit_balance_loss, prob_pair_loss],
                         feed_dict={handle: specific_handle, batch_placeholder: min(total, eval_batchsize),
                                                    is_training: False, sigma_anneal_vae: 0, problem_pair: problem_pair_val,
                                                    rotate_order: val_rotate_order})
        else:
            top_k_prec_val, lossvals, _, loss_doc_only_vals, sup_losses_doc_vals, rank_losses_doc_vals, nbr_losses_doc_vals, bit_balance_loss_vals, _, _, _, _ = sess.run(
                [top_k_prec, loss, emb_update, loss_doc_only, super_doc_only, rank_loss, nbr_loss, bit_balance_loss] + mem_updates[0],
                feed_dict={handle: specific_handle, batch_placeholder: min(total, eval_batchsize),
                           is_training: False, sigma_anneal_vae: 0})
            sess.run(mem_updates[1])
        losses += lossvals.tolist()
        losses_doc += loss_doc_only_vals.tolist()

        sup_losses_doc += sup_losses_doc_vals.tolist()
        rank_losses_doc += rank_losses_doc_vals.tolist()
        nbr_losses_doc += nbr_losses_doc_vals.tolist()
        top_k_prec_vals += top_k_prec_val.tolist()

        dist_opposite_list += dist_opposite_val.tolist()
        dist_same_list += dist_same_val.tolist()

        bit_balance_loss_vals_l.append(bit_balance_loss_vals)
        prob_pair_loss_vals_l.append(prob_pair_loss_vals)

        total -= len(lossvals)
        if total <= 0:
            done = True

    #print("time", time.time() - start)
    losses = np.mean(losses)
    losses_doc = np.mean(losses_doc)
    sup_losses_doc = np.mean(sup_losses_doc)
    rank_losses_doc = np.mean(rank_losses_doc)
    nbr_losses_doc = np.mean(nbr_losses_doc)
    top_k_prec_vals = np.mean(top_k_prec_vals)
    dist_opposite_list = np.mean(dist_opposite_list)
    dist_same_list = np.mean(dist_same_list)
    bit_balance_loss_vals_l = np.mean(bit_balance_loss_vals_l)
    prob_pair_loss_vals_l = np.mean(prob_pair_loss_vals_l)

    embedding = sess.run(model.get_hashcodes())

    extracted_hashcodes = embedding[indices]
    print("ones:",np.sum(extracted_hashcodes)/np.sum(extracted_hashcodes>-2))
    extracted_labels = [labels[i] for i in indices]

    return losses, extracted_hashcodes, extracted_labels, losses_doc, sup_losses_doc, rank_losses_doc, nbr_losses_doc, top_k_prec_vals, dist_opposite_list, dist_same_list,bit_balance_loss_vals_l,prob_pair_loss_vals_l

def myprint(f,*x):
    res = (" ".join(str(y) for y in x))
    print(res)
    res = res + "\n"
    f.write(res)

def main():
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--block_size", default=16, type=int)
    parser.add_argument("--problem_pair", default=1, type=float) # alpha_1
    parser.add_argument("--new_top_k_pair", default=-1, type=float) # alpha_2, sample one with high distance in top 100, that is hard to reach with multi index.

    #scale topk with mem size vs train size
    parser.add_argument("--scale_top_k", default=1, type=int)
    parser.add_argument("--new_top_k_pair_apply_at_block_distance", default=2, type=int) #the distance in the block where the loss should try to contract around anchor. valid values either 1 or 2
    
    parser.add_argument("--memsize", default=20000, type=int)
    parser.add_argument("--numnbr", default=10, type=int)

    parser.add_argument("--batchsize", default=40, type=int)
    parser.add_argument("--bits", default=32, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--dname", default="reuters", type=str)
    parser.add_argument("--max_seq_size", default=512, type=int)
    parser.add_argument("--save_folder", default="test_01_04_evening", type=str)
    parser.add_argument("--eval_every", default=2000, type=int)
    parser.add_argument("--maxiter", default=30000, type=int)

    parser.add_argument("--agg_loss", default=1, type=int) #-1 = do nothing, 0 = mean_loss, 1 = sum_loss
    parser.add_argument("--layersize", default=1000, type=int)
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.8, type=float)
    parser.add_argument("--doc2weight", default=1.0, type=float) # nbr loss weight
    parser.add_argument("--KLweight", default=0.0, type=float)

    #dont touch below
    parser.add_argument("--anchor_problem_pair", default=1, type=int)
    parser.add_argument("--problem_pair_less_than", default=1, type=int) #right now what we sample has to be exact distance, not it just need to be equal or less to have a problem

    parser.add_argument("--type_bb", default="pred_rev_block", type=str) #pred_rev_block pred_one_block pred_one_all none
    parser.add_argument("--coef_bb", default=-1, type=float)
    parser.add_argument("--problem_pair_choose_last", default=1, type=int) #choose the pair with largetst distanc
    parser.add_argument("--top_k_pair", default=0, type=float) #tries to pull the top 100 more together, works as the scale of the loss
    parser.add_argument("--top_k_pair_violate_extra_cost", default=0, type=float) #should be kept above 1 if we want to primarily sample bad violators
    parser.add_argument("--top_k_pair_hard_extra_cost", default=0,type=float)  # should be either 0 or 1, depending on if we want to avoid having to do anything else than direct comparisons and punish this
    parser.add_argument("--mask_equal_bits", default=0, type=int) #only tries to push bits together that are not equal
    parser.add_argument("--recon_pairs", default=0, type=float) #weight of reconstruction of pair documents, normal doc is weigthed 1. Should never be above 1.

    parser.add_argument("--sample_blocks", default=1, type=int)

    parser.add_argument("--problem_pair_batches", default=0, type=int)

    parser.add_argument("--rotate", default=-1, type=int)
    parser.add_argument("--threads_greedy", default=4, type=int)
    parser.add_argument("--down_sample_train", default=8000, type=int)
    parser.add_argument("--down_sample_val", default=1000, type=int)

    parser.add_argument("--use_importance_encoder", default=-1, type=int)
    parser.add_argument("--embedding_size", default=-1, type=int) #if below 0, use bit size as embedding size, and dont use projection

    parser.add_argument("--refresh_memory", default=-1, type=int) #number of batches between a full update of memory module
    parser.add_argument("--refresh_memory_batch_size", default=1000, type=int)

    parser.add_argument("--rankweight", default=0.0, type=float)
    parser.add_argument("--simoppositeweight", default=0.0, type=float)
    parser.add_argument("--superweight", default=0.0, type=float)
    parser.add_argument("--semisupervised", default=0, type=int)
    parser.add_argument("--datasettype", default="1000", type=str)

    args = parser.parse_args()
    print(args)
    eval_batchsize = args.batchsize

    os.makedirs("../results/" + args.save_folder, exist_ok=True)

    savename = "../results/" + args.save_folder + "/" + "_".join([str(v) for v in [args.dname, args.bits, args.batchsize, args.lr, args.KLweight, args.agg_loss,
                                                       args.numnbr, args.eval_every, args.dropout, args.layers, args.layersize, args.maxiter, args.datasettype, args.semisupervised, args.memsize,
                                                       args.superweight, args.rankweight]]) + "_" + str(np.random.randint(1000000000)) + str(np.random.randint(1000000000)) + str(np.random.randint(1000000000))

    os.makedirs(savename)
    args = vars(args)

    basepath = lambda v1 : "../data/datasets/" + args["dname"] + "/" + str(v1)

    trainfiles = glob.glob(basepath("train_only*"))
    valfiles = glob.glob(basepath("val*"))
    testfiles = glob.glob(basepath("test*"))
    semitrainfiles = glob.glob(basepath("train_semi" + "_" + str(args["datasettype"] + "*")))

    print([len(v) for v in [trainfiles, valfiles, testfiles, semitrainfiles]])
    print(trainfiles)
    labels, train_indices, val_indices, test_indices, data_text_vect, id2token, num_labels = get_labels_and_indices(args["dname"])
    args["num_labels"] = num_labels
    num_dataset_samples = len(labels)
    bowlen = data_text_vect.shape[1]
    args["bowlen"] = bowlen
    args["ndocs"] = num_dataset_samples

    print("----", bowlen)

    num_train_samples = sum([sum(1 for _ in tf.python_io.tf_record_iterator(file)) for file in trainfiles])
    num_val_samples = sum([sum(1 for _ in tf.python_io.tf_record_iterator(file)) for file in valfiles])
    num_test_samples = sum([sum(1 for _ in tf.python_io.tf_record_iterator(file)) for file in testfiles])
    num_semitrain_samples = sum([sum(1 for _ in tf.python_io.tf_record_iterator(file)) for file in semitrainfiles])

    print(num_train_samples, num_val_samples, num_test_samples)

    print("num train", num_train_samples)

    if args["semisupervised"]:
        check_num_samples = num_semitrain_samples
    else:
        check_num_samples = num_train_samples
    args["check_num_samples"] = check_num_samples


    if check_num_samples < args["memsize"]:
        args["memsize"] = check_num_samples
    print("!! memsize", check_num_samples)

    tf.reset_default_graph()
    with tf.Session() as sess:
        handle = tf.placeholder(tf.string, shape=[], name="handle_iterator")
        training_handle, train_iter, gen_iter = generator(sess, args["max_seq_size"], handle, args["batchsize"], trainfiles, 0, bowlen, num_labels)
        #training_handle, train_iter, gen_iter = generator(sess, args["max_seq_size"], handle, args["batchsize"], trainfiles, 0, bowlen, num_labels)
        training_single_handle, training_single_iter, _ = generator(sess, args["max_seq_size"], handle, eval_batchsize, trainfiles, 1, bowlen, num_labels)
        val_handle, val_iter, _ = generator(sess, args["max_seq_size"], handle, eval_batchsize, valfiles, 1, bowlen, num_labels)
        test_handle, test_iter, _ = generator(sess, args["max_seq_size"], handle, eval_batchsize, testfiles, 1, bowlen, num_labels)
        semitrain_handle, semitrain_iter, _ = generator(sess, args["max_seq_size"], handle, eval_batchsize, semitrainfiles, 1, bowlen, num_labels)

        if args['refresh_memory'] > 0:
            training_refresh_handle, training_refresh_iter, _ = generator(sess, args["max_seq_size"], handle,
                                                                        args['refresh_memory_batch_size'], trainfiles, 0, bowlen,
                                                                        num_labels)
            sess.run(training_refresh_iter.initializer)


        sample = gen_iter.get_next()

        batch_placeholder = tf.placeholder(tf.int32, name="batch_placeholder") # use this to specify batchsize (for val/test it needs to be smaller in the last batch)
        is_training = tf.placeholder(tf.bool, name="is_training")
        sigma_anneal_vae = tf.placeholder(tf.float32, name="anneal_val", shape=())
        problem_pair = tf.placeholder(tf.float32, name="problem_pair", shape=())
        rotate_order = tf.placeholder(tf.int32, name="rotate_order", shape=(args['bits']))
        val_rotate_order = np.arange(0,args['bits'],dtype=np.int32)


        model = SemiHash(sample, args, batch_placeholder, is_training, sigma_anneal_vae, num_dataset_samples, problem_pair, rotate_order, num_train_samples)

        train_op, loss, emb_update, loss_doc_only, mem_updates, update_mem_indices, \
        supervised_doc_loss, rank_loss, nbr_loss, dist_opposite, dist_same, top_k_prec, top_k_sims,  =\
            model.make_network(data_text_vect)

        valid_pairs_problem_pair, valid_pairs_top_k_pair = model.return_valid_pairs()

        if args['refresh_memory'] > 0:
            refresh_mem_updates, refresh_update_mem_indices = model.refresh_memory_module(sample, args['refresh_memory_batch_size'])

        loss_bit_balance = model.return_bit_balance_loss()
        prob_pair_loss,_ = model.return_prob_pair_loss()

        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(train_iter.initializer)

        running = True
        losses = []
        train_count = 0
        buffer_count = 0
        vae_val = 1.0
        vae_val_reduction = 1e-6

        patience_max = 5
        patience_current = 0
        best_val_loss = 100000000

        best_embeddings_list = []

        test_perf = []
        val_perf = []
        times = []
        top_k_prec_list = []
        val_losses_list = []
        val_losses_doc_list = []

        list_val_opposite = []
        list_val_same = []

        list_val_iter_perf = []
        list_test_iter_perf = []

        list_val_hitrate = []
        list_test_hitrate = []

        list_val_violations = []
        list_test_violations = []

        list_val_avg_topK = []
        list_test_avg_topK = []

        valid_pairs_problem_pair_l = []
        valid_pairs_top_k_pair_l = []

        time_before_problem_loss = args["problem_pair_batches"]
        if time_before_problem_loss == 0:
            problem_pair_val = args["problem_pair"]
        else:
            problem_pair_val = 0

        with open(savename + "/run_log", "a") as file_handle:
            while running:
                start_time = time.time()

                # fill up memory buffers first, run for 1 epoch
                if buffer_count*args["batchsize"]/check_num_samples < 1.0:
                    sess.run(mem_updates, feed_dict={handle: training_handle,
                                                      batch_placeholder: args["batchsize"],
                                                      is_training: True, sigma_anneal_vae: vae_val,
                                                     rotate_order: val_rotate_order})
                    sess.run(update_mem_indices)
                    buffer_count += 1
                    continue

                val_top_k_sims, val_top_k_prec, val_opposite, val_same, _, _, lossval, valid_pairs_problem_pair_r, valid_pairs_top_k_pair_r, _, _, _, _ = sess.run([top_k_sims, top_k_prec, dist_opposite, dist_same, train_op, emb_update, loss, valid_pairs_problem_pair, valid_pairs_top_k_pair] + mem_updates, feed_dict={handle: training_handle,
                                                                                  batch_placeholder: args["batchsize"],
                                                                                  is_training: True, sigma_anneal_vae: vae_val, problem_pair : problem_pair_val,
                                                                                    rotate_order: val_rotate_order})

                sess.run(update_mem_indices)

                losses+= lossval.tolist()
                train_count += 1
                vae_val = max(vae_val - vae_val_reduction, 0)
                top_k_prec_list += val_top_k_prec.tolist()

                list_val_opposite.append(np.mean(val_opposite))
                list_val_same.append(np.mean(val_same))

                times.append(time.time() - start_time)

                valid_pairs_problem_pair_l.append(valid_pairs_problem_pair_r)
                valid_pairs_top_k_pair_l.append(valid_pairs_top_k_pair_r)

                time_before_problem_loss-=1
                if time_before_problem_loss == 0:
                    problem_pair_val = args['problem_pair']
                    print("UPDATED PROBLEM PAIR COEF TO BE USED !!")

                #update memory with fresh codes
                if args['refresh_memory'] > 0 and train_count > 0 and train_count % args['refresh_memory'] == 0:
                    num_refresh_updates = int(np.ceil(args['memsize']/ args['refresh_memory_batch_size']))
                    for i in range(num_refresh_updates):
                        sess.run([refresh_mem_updates], feed_dict={handle: training_refresh_handle, is_training: False})
                        sess.run(refresh_update_mem_indices)

                if train_count > 0 and train_count % args["eval_every"] == 0:
                    myprint(file_handle,"patience", patience_current)
                    myprint(file_handle,"Training", np.mean(losses), "vae_val", vae_val, "epochs", train_count*args["batchsize"]/num_train_samples, "top_k_prec", np.mean(top_k_prec_list))
                    myprint(file_handle,"% of valid pairs", np.mean(valid_pairs_problem_pair_l), np.mean(valid_pairs_top_k_pair_l))
                    myprint(file_handle,"mean train time", np.mean(times), "opposite", np.mean(list_val_opposite), "same", np.mean(list_val_same))


                    times = []
                    top_k_prec_list = []
                    losses = losses[-(num_train_samples):]

                    problem_pair_dist_0_val_l = []

                    valid_pairs_problem_pair_l = []
                    valid_pairs_top_k_pair_l = []

                    sess.run([train_iter.initializer, training_single_iter.initializer, val_iter.initializer, test_iter.initializer, semitrain_iter.initializer])

                    trainloss, train_hashcodes, train_labels, train_losses_doc, \
                    train_sup_losses_doc, train_rank_losses_doc, train_nbr_losses_doc, topkprec_train, \
                    dist_opposite_list_train, dist_same_list_train, train_bit_balance_loss, train_prob_pair_loss =\
                        extract_vectors_labels(dist_opposite, dist_same, top_k_prec, sess, handle, training_single_handle,
                                                                                      num_train_samples,
                                                                                      batch_placeholder, is_training,
                                                                                      sigma_anneal_vae, loss, emb_update,
                                                                                      eval_batchsize, train_indices, labels,
                                                                                      model, loss_doc_only, supervised_doc_loss, rank_loss, nbr_loss, loss_bit_balance,
                                                                                                prob_pair_loss,problem_pair,problem_pair_val,
                                                                                        rotate_order, val_rotate_order, mem_updates=None)

                    valloss, val_hashcodes, val_labels, val_losses_doc, \
                    val_sup_losses_doc, val_rank_losses_doc, val_nbr_losses_doc, topkprec_val, \
                    dist_opposite_list_val, dist_same_list_val, val_bit_balance_loss, val_prob_pair_loss =\
                        extract_vectors_labels(dist_opposite, dist_same, top_k_prec, sess, handle, val_handle,
                                                                                      num_val_samples,
                                                                                      batch_placeholder, is_training,
                                                                                      sigma_anneal_vae, loss, emb_update,
                                                                                      eval_batchsize, val_indices, labels,
                                                                                      model, loss_doc_only, supervised_doc_loss, rank_loss, nbr_loss, loss_bit_balance,
                                                                                              prob_pair_loss,problem_pair,problem_pair_val,
                                                                                        rotate_order, val_rotate_order, mem_updates=None)

                    testloss, test_hashcodes, test_labels, test_losses_doc, \
                    test_sup_losses_doc, test_rank_losses_doc, test_nbr_losses_doc, topkprec_test, \
                    dist_opposite_list_test, dist_same_list_test, test_bit_balance_loss, test_prob_pair_loss =\
                        extract_vectors_labels(dist_opposite, dist_same, top_k_prec, sess, handle, test_handle,
                                                                                      num_test_samples,
                                                                                      batch_placeholder, is_training,
                                                                                      sigma_anneal_vae, loss, emb_update,
                                                                                      eval_batchsize, test_indices, labels,
                                                                                      model, loss_doc_only, supervised_doc_loss, rank_loss, nbr_loss, loss_bit_balance,
                                                                                             prob_pair_loss,problem_pair,problem_pair_val,
                                                                                       rotate_order, val_rotate_order,mem_updates=None)


                    if args['rotate'] > 0:
                        print("START FINDING NEW ORDER")
                        time_start = time.time()
                        perm_train = np.random.permutation(len(train_hashcodes))[:args['down_sample_train']]
                        train_hashcodes = train_hashcodes[perm_train]
                        perm_val = np.random.permutation(len(val_hashcodes))[:args['down_sample_val']]
                        val_hashcodes = val_hashcodes[perm_val]

                        val_rotate_order, _, _ = run_iteration(train_hashcodes, val_hashcodes, None, bits=args['bits'],
                                                         n_blocks=int(args['bits']/args['block_size']),
                                                         threads=args['threads_greedy'], report_test=False,
                                                         rotate_order=val_rotate_order, max_switch=1)

                        print("new order: ", val_rotate_order, "time taken: ", time.time() - time_start)

                        sess.run([train_iter.initializer, training_single_iter.initializer, val_iter.initializer,
                                  test_iter.initializer, semitrain_iter.initializer])

                        trainloss, train_hashcodes, train_labels, train_losses_doc, \
                        train_sup_losses_doc, train_rank_losses_doc, train_nbr_losses_doc, topkprec_train, \
                        dist_opposite_list_train, dist_same_list_train, train_bit_balance_loss, train_prob_pair_loss = \
                            extract_vectors_labels(dist_opposite, dist_same, top_k_prec, sess, handle,
                                                   training_single_handle,
                                                   num_train_samples,
                                                   batch_placeholder, is_training,
                                                   sigma_anneal_vae, loss, emb_update,
                                                   eval_batchsize, train_indices, labels,
                                                   model, loss_doc_only, supervised_doc_loss, rank_loss, nbr_loss,
                                                   loss_bit_balance,
                                                   prob_pair_loss, problem_pair, problem_pair_val,
                                                   rotate_order, val_rotate_order, mem_updates=None)

                        valloss, val_hashcodes, val_labels, val_losses_doc, \
                        val_sup_losses_doc, val_rank_losses_doc, val_nbr_losses_doc, topkprec_val, \
                        dist_opposite_list_val, dist_same_list_val, val_bit_balance_loss, val_prob_pair_loss = \
                            extract_vectors_labels(dist_opposite, dist_same, top_k_prec, sess, handle, val_handle,
                                                   num_val_samples,
                                                   batch_placeholder, is_training,
                                                   sigma_anneal_vae, loss, emb_update,
                                                   eval_batchsize, val_indices, labels,
                                                   model, loss_doc_only, supervised_doc_loss, rank_loss, nbr_loss,
                                                   loss_bit_balance,
                                                   prob_pair_loss, problem_pair, problem_pair_val,
                                                   rotate_order, val_rotate_order, mem_updates=None)

                        testloss, test_hashcodes, test_labels, test_losses_doc, \
                        test_sup_losses_doc, test_rank_losses_doc, test_nbr_losses_doc, topkprec_test, \
                        dist_opposite_list_test, dist_same_list_test, test_bit_balance_loss, test_prob_pair_loss = \
                            extract_vectors_labels(dist_opposite, dist_same, top_k_prec, sess, handle, test_handle,
                                                   num_test_samples,
                                                   batch_placeholder, is_training,
                                                   sigma_anneal_vae, loss, emb_update,
                                                   eval_batchsize, test_indices, labels,
                                                   model, loss_doc_only, supervised_doc_loss, rank_loss, nbr_loss,
                                                   loss_bit_balance,
                                                   prob_pair_loss, problem_pair, problem_pair_val,
                                                   rotate_order, val_rotate_order, mem_updates=None)


                    val_prec100, valdists = eval_hashing(train_hashcodes, train_labels, val_hashcodes, val_labels)
                    test_prec100, testdists = eval_hashing(train_hashcodes, train_labels, test_hashcodes, test_labels)

                    #minwise hitrate
                    t = time.time()
                    '''
                    val_hitrate, val_violations, val_avg_topK = threaded_compute_hitrate_iterative(train_hashcodes,val_hashcodes,args['bits'],int(args['bits']/args['block_size']))
                    test_hitrate, test_violations, test_avg_topK = threaded_compute_hitrate_iterative(train_hashcodes,test_hashcodes,args['bits'],int(args['bits']/args['block_size']))
                    '''
                    val_hitrate, val_violations, val_avg_topK = compute_hitrate_iterative(train_hashcodes,val_hashcodes,args['bits'],int(args['bits']/args['block_size']), skip_hitrate_comp=True)
                    test_hitrate, test_violations, test_avg_topK = compute_hitrate_iterative(train_hashcodes,test_hashcodes,args['bits'],int(args['bits']/args['block_size']), skip_hitrate_comp=True)
                    print("time for hitrate:", time.time()-t)

                    #iterative prec100, that resolve ties by averaging the performance

                    t=time.time()
                    #val_prec100_iter, _, _, _ = acc_top_k_iterative(train_hashcodes, train_labels, val_hashcodes, val_labels, [100], args["num_labels"])
                    val_prec100_iter, _ = threaded_acc_top_k_tie_aware(train_hashcodes, train_labels, val_hashcodes, val_labels, [100], args["num_labels"])
                    val_prec100_iter = np.mean(val_prec100_iter)

                    #test_prec100_iter, _, _, _ = acc_top_k_iterative(train_hashcodes, train_labels, test_hashcodes, test_labels, [100], args["num_labels"])
                    test_prec100_iter, _ = threaded_acc_top_k_tie_aware(train_hashcodes, train_labels, test_hashcodes,test_labels, [100], args["num_labels"])
                    test_prec100_iter = np.mean(test_prec100_iter)
                    print("time for top_k:", time.time()-t)


                    myprint(file_handle,"Train", trainloss, train_losses_doc, train_sup_losses_doc, train_rank_losses_doc, train_nbr_losses_doc, train_bit_balance_loss, train_prob_pair_loss)
                    myprint(file_handle,"Val", valdists, valloss, np.mean(val_prec100), val_losses_doc, val_sup_losses_doc, val_rank_losses_doc, val_nbr_losses_doc, val_bit_balance_loss, val_prob_pair_loss)
                    myprint(file_handle,"Testing", testdists, testloss, np.mean(test_prec100), test_losses_doc, test_sup_losses_doc, test_rank_losses_doc, test_nbr_losses_doc, test_bit_balance_loss, test_prob_pair_loss)
                    myprint(file_handle,"topkprec_network", topkprec_train, topkprec_val, topkprec_test, "top iter 100", val_prec100_iter, test_prec100_iter)
                    myprint(file_handle,"dist_opposites",  dist_opposite_list_train, dist_opposite_list_val, dist_opposite_list_test)
                    myprint(file_handle,"dist_same",  dist_same_list_train, dist_same_list_val, dist_same_list_test)
                    myprint(file_handle,"hit rate", val_hitrate, test_hitrate)
                    myprint(file_handle, "violations", val_violations, test_violations, "avg tokK", val_avg_topK, test_avg_topK)


                    if best_val_loss > val_losses_doc:
                        emb_matrix = sess.run(model.get_hashcodes())
                        best_embeddings = emb_matrix
                        
                        best_embeddings_list.append(best_embeddings)
                        best_val_loss = val_losses_doc
                        patience_current = 0
                        test_perf.append(test_prec100)
                        val_perf.append(np.mean(val_prec100))

                        val_losses_list.append(valloss)
                        val_losses_doc_list.append(val_losses_doc)

                        list_val_iter_perf.append(val_prec100_iter)
                        list_test_iter_perf.append(test_prec100_iter)

                        list_val_hitrate.append(val_hitrate)
                        list_test_hitrate.append(test_hitrate)

                        list_val_violations.append(val_violations)
                        list_test_violations.append(test_violations)


                        list_val_avg_topK.append(val_avg_topK)
                        list_test_avg_topK.append(test_avg_topK)



                        pickle.dump([best_embeddings_list, args, best_val_loss, train_count, vae_val, test_perf, val_perf,
                                     val_losses_list, val_losses_doc_list,
                                     [dist_opposite_list_train,dist_same_list_train], [dist_opposite_list_val,dist_same_list_val], [dist_opposite_list_test,dist_same_list_test],
                                     list_val_iter_perf, list_test_iter_perf, list_val_hitrate, list_test_hitrate, list_val_violations, list_test_violations,
                                     list_val_avg_topK, list_test_avg_topK],
                                    open(savename + "/res.pkl", "wb"))
                    else:
                        patience_current += 1

                    if patience_current >= patience_max or train_count > args["maxiter"]: #or ((time.time() - start_time) > (60*60*args["hours"])):
                        running = False
                file_handle.flush()

if __name__ == '__main__':
    main()


