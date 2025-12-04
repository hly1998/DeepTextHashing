
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

import numpy as np

from tensorflow.losses import compute_weighted_loss, Reduction

def hinge_loss_eps(labels, logits, epsval, weights=1.0, scope=None,
               loss_collection=ops.GraphKeys.LOSSES,
               reduction=Reduction.NONE):#Reduction.SUM_BY_NONZERO_WEIGHTS):
  if labels is None:
    raise ValueError("labels must not be None.")
  if logits is None:
    raise ValueError("logits must not be None.")
  with ops.name_scope(scope, "hinge_loss", (logits, labels, weights)) as scope:
    logits = math_ops.to_float(logits)
    labels = math_ops.to_float(labels)
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    # We first need to convert binary labels to -1/1 labels (as floats).
    all_eps = array_ops.ones_like(labels)*epsval
    all_ones = array_ops.ones_like(labels)

    labels = math_ops.subtract(2 * labels, all_ones)
    losses = nn_ops.relu(
        math_ops.subtract(all_eps, math_ops.multiply(labels, logits)))
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


@tf.custom_gradient
def rev_grad_layer(x):
    def grad(dy):
        return -dy
    return tf.identity(x), grad

class SemiHash():
    def __init__(self, sample, args, batchsize, is_training, sigma_anneal_vae, num_dataset_samples, problem_pair, rotate_order, num_train_samples):
        self.sample = sample
        self.args = args

        self.batchsize = batchsize
        self.is_training = is_training
        self.sigma_anneal_vae = sigma_anneal_vae
        self.problem_pair_coef = problem_pair

        self.rotate_order = rotate_order

        self.num_dataset_samples = num_dataset_samples
        self.num_train_samples = num_train_samples



    #################### Bernoulli Sample #####################
    ## ref code_31_03: https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
    @staticmethod
    def bernoulliSample(x,do_det):
        """
        Uses a tensor whose values are in [0,1] to sample a tensor with values in {0, 1},
        using the straight through estimator for the gradient.
        E.g.,:
        if x is 0.6, bernoulliSample(x) will be 1 with probability 0.6, and 0 otherwise,
        and the gradient will be pass-through (identity).
        """
        g = tf.get_default_graph()
        with ops.name_scope("BernoulliSample") as name:
            with g.gradient_override_map({"Ceil": "Identity", "Sub": "BernoulliSample_ST"}):
                train_fn = lambda: tf.minimum(tf.ones(tf.shape(x)), tf.random_uniform(tf.shape(x)))
                eval_fn = lambda: tf.minimum(tf.ones(tf.shape(x)), tf.ones(tf.shape(x)) * 0.5)

                mus = tf.cond(do_det, train_fn, eval_fn)
                return tf.ceil(x - mus, name=name)

    @ops.RegisterGradient("BernoulliSample_ST")
    def bernoulliSample_ST(op, grad):
        return [grad, tf.zeros(tf.shape(op.inputs[1]))]

    ###################################################
    ############## BIT BALANCE ########################
    ###################################################

    @staticmethod
    def within_block_prediction_loss_one_all(sampling_prob, block_size, bits, sample_size):
        n_blocks = int(bits / block_size)
        blocks = []
        for i in range(n_blocks):
            block = sampling_prob[:, i * block_size:(i + 1) * block_size] - 0.5  # -0.5 to center the data
            blocks.append(block)

        pred_block_losses = []
        for i in range(n_blocks):
            for j in range(block_size):
                from_entry = rev_grad_layer(tf.expand_dims(blocks[i][:, j], -1))
                to_block = tf.stop_gradient(blocks[i])

                pred_block = tf.layers.dense(from_entry, units=block_size, name="pred_block" + str(i) + str(j),
                                             use_bias=False, reuse=tf.AUTO_REUSE)
                to_mask = np.ones((1, block_size), dtype=np.float32)
                to_mask[:, j] = 0
                batch_adjusted_mask = tf.tile(to_mask, [tf.cast(sample_size, dtype=tf.int32), 1])
                pred_block_loss = tf.losses.mean_squared_error(to_block, pred_block, weights=batch_adjusted_mask)
                pred_block_losses.append(pred_block_loss)

        pred_block_losses = tf.stack(pred_block_losses)
        pred_block_losses = tf.reduce_mean(pred_block_losses)
        return pred_block_losses

    @staticmethod
    def prediction_loss_one_all(sampling_prob, bits, sample_size):
        pred_losses = []
        for i in range(bits):
            block = sampling_prob[:, i:i + 1]
            from_entry = rev_grad_layer(block)
            to_block = tf.stop_gradient(sampling_prob)

            pred_block = tf.layers.dense(from_entry, units=bits, name="pred_block" + str(i), use_bias=False,
                                         reuse=tf.AUTO_REUSE)
            to_mask = np.ones((1, bits), dtype=np.float32)
            to_mask[:, i] = 0
            batch_adjusted_mask = tf.tile(to_mask, [tf.cast(sample_size, dtype=tf.int32), 1])
            pred_block_loss = tf.losses.mean_squared_error(to_block, pred_block, weights=batch_adjusted_mask)
            pred_losses.append(pred_block_loss)

        pred_losses = tf.stack(pred_losses)
        pred_losses = tf.reduce_mean(pred_losses)
        return pred_losses

    # REMEMBER IF YOU USE THIS, YOU NEED TO REORGINIZE THE BIT VECTOR !
    @staticmethod
    def small_block_split_out(sampling_prob, block_size, bits):
        n_blocks = int(bits / block_size)
        blocks = []
        # We need as many blocks as block_size, of size n_blocks (remember to construct bit code_31_03 correctly)
        for i in range(block_size):
            block = sampling_prob[:, i * n_blocks:(i + 1) * n_blocks] - 0.5  # -0.5 to center the data
            blocks.append(block)

        pred_block_losses = []
        for i in range(block_size):
            for j in range(i + 1, block_size):
                # compute with static midpoint of 0.5, so a translated 2. moment. Makes the prop goes toward 0.5 just as KL loss
                pred_block = tf.layers.dense(rev_grad_layer(blocks[i]), units=n_blocks,
                                             name="from_to_" + str(i) + "_" + str(j), use_bias=False,
                                             reuse=tf.AUTO_REUSE)
                pred_block_loss = tf.losses.mean_squared_error(tf.stop_gradient(blocks[j]), pred_block)
                pred_block_losses.append(pred_block_loss)

        pred_block_losses = tf.stack(pred_block_losses)
        pred_block_losses = tf.reduce_mean(pred_block_losses)

        return pred_block_losses

    @staticmethod
    def reorginize_bit_vector(bit_vector, block_size, bits):
        n_blocks = int(bits / block_size)
        blocks = []
        # We need as many blocks as block_size, of size n_blocks (remember to construct bit code_31_03 correctly)
        for i in range(block_size):
            block = bit_vector[:, i * n_blocks:(i + 1) * n_blocks]
            blocks.append(block)

        reorginized_blocks = []
        for i in range(n_blocks):
            block = []
            for j in range(block_size):
                block.append(blocks[j][:, i])
            block = tf.stack(block, -1)
            reorginized_blocks.append(block)
        reorginized_blocks = tf.concat(reorginized_blocks, -1)
        return reorginized_blocks

    def compute_correlation_loss(self, bit_prop, bit_vector):
        if self.args['coef_bb'] > 0:
            if self.args['type_bb'] == "pred_rev_block":
                block_correlation_loss = self.small_block_split_out(bit_prop, self.args['block_size'],
                                                                                     self.args['bits'])
                bit_vector = self.reorginize_bit_vector(bit_vector, self.args['block_size'], self.args['bits'])
            elif self.args['type_bb'] == "pred_one_block":
                block_correlation_loss = self.within_block_prediction_loss_one_all(bit_prop, self.args['block_size'],
                                                                                   self.args['bits'], self.batchsize)
            elif self.args['type_bb'] == "pred_one_all":
                block_correlation_loss = self.prediction_loss_one_all(bit_prop, self.args['bits'], self.batchsize)
            elif self.args['type_bb'] == "none":
                block_correlation_loss = tf.zeros(1)
            else:
                raise Exception("invalid correlation type!")
        else:
            block_correlation_loss = tf.zeros(1)

        return block_correlation_loss, bit_vector

    ###################################################
    ############## REST ###############################
    ###################################################
    @staticmethod
    def rearrange_bits(embedding, order):
        embedding = tf.transpose(tf.gather(tf.transpose(embedding), order))
        return embedding

    def encoder(self, docbow):
        #tf.concat((docbow, bert_output), axis=-1)
        #doc_layer = tf.nn.dropout(bert_output, tf.cond(self.is_training, lambda: 0.8, lambda: 1.0))

        if self.args['use_importance_encoder'] > 0:
            docbow = docbow * self.importance_emb_matrix

        doc_layer = tf.layers.dense(docbow, self.args["layersize"], name="encode_layer0", reuse=tf.AUTO_REUSE, activation=tf.nn.relu)

        for i in range(1, self.args["layers"]):
            doc_layer = tf.layers.dense(doc_layer, int(self.args["layersize"]/i), name="encode_layer" + str(i),reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
            #doc_layer = tf.nn.dropout(doc_layer, tf.cond(self.is_training, lambda: 0.8, lambda: 1.0))

        doc_layer = tf.nn.dropout(doc_layer, tf.cond(self.is_training, lambda: self.args["dropout"], lambda: 1.0))
        #doc_layer = tf.layers.batch_normalization(doc_layer, training=self.is_training)
        sampling_vector = tf.layers.dense(doc_layer, self.args["bits"], name="last_encode", reuse=tf.AUTO_REUSE, activation=tf.nn.sigmoid)

        bit_vector = self.bernoulliSample(sampling_vector, self.is_training)
        bit_vector_det = self.bernoulliSample(sampling_vector, tf.constant(False))#tf.ceil(sampling_vector - tf.ones(tf.shape(sampling_vector))*0.5)
        bit_vector_det = 2*bit_vector_det - 1
        bit_vector = 2*bit_vector - 1

        #some discrete optimization have provided a better bit balancing, that we should run with.
        if self.args['rotate'] > 0:
            bit_vector_det = self.rearrange_bits(bit_vector_det, self.rotate_order)

        return bit_vector, sampling_vector, bit_vector_det

    def make_noisy_hashcode(self, hashcode):
        e = tf.random.normal([self.batchsize, self.args["bits"]])
        return tf.math.multiply(e, self.sigma_anneal_vae) + hashcode

    def compute_KL(self, sampling_vector):
        loss_kl = tf.multiply(sampling_vector, tf.math.log(tf.maximum(sampling_vector / 0.5, 1e-10))) + \
                  tf.multiply(1 - sampling_vector, tf.math.log(tf.maximum((1 - sampling_vector) / 0.5, 1e-10)))
        loss_kl = tf.reduce_sum(tf.reduce_sum(loss_kl, 1), axis=0)
        return loss_kl

    def decoder(self, hashcode, sampling_vector, target):
        noisy_hashcode = self.make_noisy_hashcode(hashcode)
        # decode_layer = tf.layers.dense(noisy_hashcode, target.shape[1], name="decode0", reuse=tf.AUTO_REUSE)
        kl_loss = self.compute_KL(sampling_vector)
        # sqr_diff = tf.math.pow(decode_layer - target, 2)
        # mse = tf.reduce_mean(sqr_diff, axis=-1)
        # loss = mse + self.args["KLweight"]*kl_loss

        if self.args['embedding_size']>0:
            embedding = tf.layers.dense(self.word_emb_matrix, self.args["bits"], name="lower_dim_embedding_layer", reuse=tf.AUTO_REUSE)
        else:
            embedding = self.word_emb_matrix
        dot_emb_vector = tf.linalg.matmul(noisy_hashcode,tf.transpose(embedding * tf.expand_dims(self.importance_emb_matrix,-1))) + self.softmax_bias
        softmaxed = tf.nn.softmax(dot_emb_vector)
        logaritmed = tf.math.log(tf.maximum(softmaxed, 1e-10))
        logaritmed = tf.multiply(logaritmed, tf.cast(target > 0, tf.float32))
        loss_recon = tf.reduce_sum(logaritmed, 1)
        loss = -(loss_recon - self.args["KLweight"]*kl_loss)
        return loss

    def supervised_decoder(self, hashcode, target):
        noisy_hashcode = self.make_noisy_hashcode(hashcode)
        prediction_logits3 = tf.layers.dense(noisy_hashcode, self.args["num_labels"], reuse=tf.AUTO_REUSE, name="dec3")

        sup_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(target, tf.float32), logits=prediction_logits3)
        sup_loss = tf.reduce_sum(sup_loss, -1)

        return sup_loss

    def get_hashcodes(self):
        return self.hashcode_embedding

    def make_data_tensor(self, bow_sparse_matrix):
        indices = []
        for i, row in enumerate(bow_sparse_matrix):
            for idx in row.indices:
                indices.append([idx,i])
        data_tensor = tf.SparseTensor(indices=indices, values=bow_sparse_matrix.data.astype(np.float32), dense_shape=[bow_sparse_matrix.shape[1], bow_sparse_matrix.shape[0]])
        return data_tensor

    def get_doc_bow(self, doc_idx):
        doc_idx_onehot = tf.transpose(tf.one_hot(doc_idx, self.args["ndocs"], dtype=tf.float32))
        doc_bow = tf.transpose(tf.sparse_tensor_dense_matmul(self.data_tensor, doc_idx_onehot))
        return doc_bow

    def get_most_similar_idx(self, memory, item):
        STH_sums = tf.transpose(tf.matmul(memory, tf.transpose(item)))
        STH_sim, STH_num = tf.nn.top_k(STH_sums, k=self.args['numnbr'])
        STH_num = STH_num[:, 1:]

        STH_num = tf.transpose(STH_num)
        STH_num = tf.random_shuffle(STH_num) # random_shuffle only supports shuffling first dimension
        STH_num = tf.transpose(STH_num)[:, 0]

        STH_nbr_idx = tf.nn.embedding_lookup(self.doc_idx_memory, STH_num)
        return STH_nbr_idx

    def similar_doc_opposite_label(self, hashcode_mem, label_mem, hashcode, label):
        code_sums = tf.transpose(tf.matmul(hashcode_mem, tf.transpose(tf.cast(hashcode, tf.int32)))) # dot product of hash does

        negative_label_sums = -tf.transpose(tf.matmul(label_mem, tf.transpose(label))) # we want to find a sample with an oppose label (not simple for multilabels)
        is_opposite_label = tf.cast(negative_label_sums > -1, tf.int32)

        sortvalues = code_sums * is_opposite_label
        sims, nums = tf.nn.top_k(sortvalues, k=self.args['numnbr'])
        nums = tf.transpose(nums)
        nums = tf.random_shuffle(nums) # random_shuffle only supports shuffling first dimension
        nums = tf.transpose(nums)[:, 0]

        nbr_idx = tf.nn.embedding_lookup(self.doc_idx_memory, nums)
        return nbr_idx

    def dissimilar_doc_same_label(self, hashcode_mem, label_mem, hashcode, label):
        code_sums = -tf.transpose(tf.matmul(hashcode_mem, tf.transpose(tf.cast(hashcode, tf.int32))))  # dot product of hash does

        label_sums = tf.transpose(tf.matmul(label_mem, tf.transpose(label)))  # we want to find a sample with an oppose label (not simple for multilabels)
        similar_label = tf.cast(label_sums > 0, tf.int32)

        sortvalues = code_sums * similar_label
        sims, nums = tf.nn.top_k(sortvalues, k=self.args['numnbr'])
        nums = tf.transpose(nums)
        nums = tf.random_shuffle(nums) # random_shuffle only supports shuffling first dimension
        nums = tf.transpose(nums)[:, 0]

        nbr_idx = tf.nn.embedding_lookup(self.doc_idx_memory, nums)
        return nbr_idx

    def top_k_idx(self, hashcode_mem, label_mem, hashcode, label, kk=100):
        hashcode = tf.cond(self.is_training, lambda : self.make_noisy_hashcode(hashcode), lambda : hashcode)
        code_sums = tf.transpose(tf.matmul(hashcode_mem, tf.transpose(tf.cast(hashcode, tf.int32))))  # dot product of hash does

        rettt, top_k_idx = tf.nn.top_k(code_sums, k=kk)
        top_k_hashcodes = tf.nn.embedding_lookup(hashcode_mem, top_k_idx)
        top_k_labels = tf.nn.embedding_lookup(label_mem, top_k_idx)

        top_k_dists = tf.reduce_sum(top_k_hashcodes * tf.expand_dims(tf.cast(hashcode, tf.int32), 1), -1)
        top_k_labelsum = tf.reduce_sum(top_k_labels * tf.expand_dims(label, 1), -1)

        top_k_similar_label = tf.cast(top_k_labelsum > 0, tf.int32)
        top_k_opposite_label = tf.cast((-top_k_labelsum) > -1, tf.int32)

        same_sims, same_nums = tf.nn.top_k((self.args["bits"]*2 - top_k_dists) * top_k_similar_label, k=1) # find sample with same label, which is far away
        opposite_sims, opposite_nums = tf.nn.top_k(top_k_dists * top_k_opposite_label, k=1) # find sample with different label, which is close

        same_onehot = tf.one_hot(same_nums[:, 0], kk, dtype=tf.int32)
        same_idx = tf.reduce_sum(same_onehot * top_k_idx, -1)

        opposite_onehot = tf.one_hot(opposite_nums[:, 0], kk, dtype=tf.int32)
        opposite_idx = tf.reduce_sum(opposite_onehot * top_k_idx, -1)

        opposite_exists = tf.cast(tf.reduce_sum(top_k_opposite_label, -1) > 0, tf.float32)
        same_exists = tf.cast(tf.reduce_sum(top_k_similar_label, -1) > 0, tf.float32)

        return same_idx, opposite_idx, same_exists, opposite_exists, tf.reduce_mean(tf.cast(top_k_similar_label, tf.float32), -1), same_sims


    def find_problematic_pair(self, hashcode_mem,hashcode, doc_idx_memory, kk=100):
        #kk is only correct if the memsize contains all documents in collections. This is not possible for
        #larger datasets, and kk should therefore be adjusted based on the collection size vs memory size
        #if the memory size is much muich smaller than collection size (1-5% or similar), a 2 step process should
        #be used. Where the network is trained in 1 step, and valid pairs are constructed in the 2. step.

        #update kk
        if self.args['scale_top_k'] > 0:
            fraction_mem_dataset = self.memsize / self.num_train_samples
            kk = int(kk*fraction_mem_dataset)

        hashcode = tf.cast(hashcode, tf.int32)
        hashcode_mem = hashcode_mem
        code_sums = tf.transpose(tf.matmul(hashcode_mem, tf.transpose(hashcode)))  # dot product of hash does

        #the top K should be adjusted based on the size of the memory module, vs actual dataset
        #in case of extreme large datasets, the topK retrieval should be done as a seperate step outside the network to
        #generate the pairs.
        distances, top_k_idx = tf.nn.top_k(code_sums, k=kk)
        dist100 = distances[:,-1]

        block_size = self.args['block_size']
        blocks = int(self.args['bits']/block_size)
        const = np.zeros((blocks,self.args['bits']))

        for i in range(blocks):
            const[i,(i)*block_size:(i+1)*block_size] = 1
        const = tf.constant(const, dtype=tf.int32)

        block_to_look_into = tf.random.uniform([self.batchsize], minval=0, maxval=blocks, dtype=tf.dtypes.int32)
        block_to_look_into = tf.one_hot(block_to_look_into,blocks, dtype=tf.int32)
        block_to_look_into = tf.matmul(block_to_look_into,const)

        masked_hashcode = hashcode * block_to_look_into

        distance_in_block = tf.transpose(tf.matmul(hashcode_mem, tf.transpose(masked_hashcode)))
        distance_in_block_0 = tf.cast(tf.equal(distance_in_block,block_size),tf.float32)
        distance_over_thresh = tf.cast(code_sums < tf.expand_dims(dist100, -1), tf.float32)

        #based on the distance to top k, 1 distance can also be problematic.
        distance_in_block_1 = tf.cast(tf.equal(distance_in_block, block_size-2), tf.float32)


        if self.args['problem_pair_choose_last'] > 0:
            print("choosing last document as problem pair")
            #for this function distance over thresh is not if its over thresh, bat rather if its the last element with 0 in block
            #minus 32 so scale it to 0 to -64 and then negate so most similar have highest value (get dissimilarity instead of similarity score)
            to_sort_for_prob_pair = distance_in_block_0 * distance_over_thresh * -(tf.cast(code_sums,dtype=tf.float32)-self.args['bits']) + tf.random.normal(
                [self.batchsize, self.memsize], mean=0.0)
        else:
            to_sort_for_prob_pair = distance_in_block_0 * distance_over_thresh * tf.random.normal(
                [self.batchsize, self.memsize], mean=10.0)

        #add random noise so its not the same always that is chosen in topk later
        prob_pair_is_0, prob_pair_idx = tf.nn.top_k(to_sort_for_prob_pair, k=1)
        #if there is no hit prob_pair_dist can be 0.. update block_to_look_into to handle this case
        #i now checks the actual hash codes for 0 distance, so this is no longer necesarry
        #block_to_look_into = tf.cast(masked_hashcode,dtype=tf.float32) * tf.cast(prob_pair_dist>1,dtype=tf.float32)
        prob_pairs = tf.nn.embedding_lookup(doc_idx_memory, prob_pair_idx)

        if self.args['top_k_pair'] > 0:
            #prioritize pick those that can not be retrived (distance higher than n_blocks * 2 - 1)
            #after that prioritise those in the range n_blocks-1 to n_blocks*2-1
            #if there is none of these, all are ok and should not do anything.
            dist_sums = (self.args['bits']-code_sums)/2 #convert to distances instead of similarities
            dist100 = (self.args['bits']-dist100)/2

            #conditions for the sample to hold
            is_below_dist100 = tf.cast(dist_sums < tf.expand_dims(dist100, -1), tf.float32)
            is_above_n_block = tf.cast(tf.greater_equal(distance_in_block,blocks-1),tf.float32) # there are not optimal
            is_above_2n_block = tf.cast(tf.greater_equal(distance_in_block, 2*blocks - 1), tf.float32) #these can not be retrieved
            #add randomness that will still keep overall order
            potential_pos_pair = (is_below_dist100*(is_above_n_block+is_above_2n_block)) *  tf.random.normal([self.batchsize, self.memsize],mean=2, stddev=0.1) #sample the most problematic pairs first
            _, closests_top_k_idx = tf.nn.top_k(potential_pos_pair, k=1)
            return prob_pairs, block_to_look_into, tf.cast(prob_pair_is_0>0,dtype=tf.float32), closests_top_k_idx
        else:
            return prob_pairs, block_to_look_into, tf.cast(prob_pair_is_0>0,dtype=tf.float32)


    def find_problematic_pair_updated(self, hashcode_mem,hashcode, doc_idx_memory, kk=100):
        #kk is only correct if the memsize contains all documents in collections. This is not possible for
        #larger datasets, and kk should therefore be adjusted based on the collection size vs memory size
        #if the memory size is much muich smaller than collection size (1-5% or similar), a 2 step process should
        #be used. Where the network is trained in 1 step, and valid pairs are constructed in the 2. step.
        #update kk
        if self.args['scale_top_k'] > 0:
            fraction_mem_dataset = self.memsize / self.num_train_samples 
            kk = int(kk*fraction_mem_dataset)

        block_size = self.args['block_size']
        blocks = int(self.args['bits']/block_size)

        hashcode = tf.cast(hashcode, tf.int32)
        hashcode_mem = hashcode_mem
        code_sums = tf.transpose(tf.matmul(hashcode_mem, tf.transpose(hashcode)))  # dot product of hash does

        #the top K should be adjusted based on the size of the memory module, vs actual dataset
        #in case of extreme large datasets, the topK retrieval should be done as a seperate step outside the network to
        #generate the pairs.
        distances, top_k_idx = tf.nn.top_k(code_sums, k=kk)
        dist100 = distances[:,-1]


        threshed_dist100 = tf.cast((self.args['bits']-dist100)/2,dtype=tf.float32) # convert to dist instead of sim.
        id_block = tf.expand_dims(tf.constant(np.arange(0, self.args['bits'] + 1),dtype=tf.float32),0)
        id_block = tf.tile(id_block,(self.batchsize,1))
        id_block = tf.less_equal(id_block,tf.expand_dims(threshed_dist100,-1))
        id_block = tf.cast(id_block,dtype=tf.float32) + tf.random.normal([self.batchsize, self.args['bits']+1],0,stddev=0.1) #add random noise, so the block is random which is biggest
        _,block_to_look_into = tf.nn.top_k(id_block,k=1)
        #block_to_look_into = tf.squeeze(block_to_look_into)
        block_to_look_into = tf.reshape(block_to_look_into,[self.batchsize])
        distance_to_look_into = block_to_look_into // blocks
        block_to_look_into = block_to_look_into % blocks

        #block_to_look_into = tf.random.uniform([self.batchsize], minval=0, maxval=blocks, dtype=tf.dtypes.int32)

        const = np.zeros((blocks,self.args['bits']))
        for i in range(blocks):
            const[i,(i)*block_size:(i+1)*block_size] = 1
        const = tf.constant(const, dtype=tf.int32)

        block_to_look_into = tf.one_hot(block_to_look_into,blocks, dtype=tf.int32)
        block_to_look_into = tf.matmul(block_to_look_into,const)

        masked_hashcode = hashcode * block_to_look_into

        distance_in_block = tf.transpose(tf.matmul(hashcode_mem, tf.transpose(masked_hashcode)))
        distance_in_block_0 = tf.cast(tf.equal(distance_in_block,block_size-tf.expand_dims(distance_to_look_into,-1)*2),tf.float32)
        distance_over_thresh = tf.cast(code_sums < tf.expand_dims(dist100, -1), tf.float32)
        if self.args['problem_pair_choose_last'] > 0:
            print("choosing last document as problem pair")
            #for this function distance over thresh is not if its over thresh, bat rather if its the last element with 0 in block
            #minus 32 so scale it to 0 to -64 and then negate so most similar have highest value (get dissimilarity instead of similarity score)
            to_sort_for_prob_pair = distance_in_block_0 * distance_over_thresh * -(tf.cast(code_sums,dtype=tf.float32)-self.args['bits']) + tf.random.normal(
                [self.batchsize, self.memsize], mean=0.0)
        else:
            to_sort_for_prob_pair = distance_in_block_0 * distance_over_thresh * tf.random.normal(
                [self.batchsize, self.memsize], mean=10.0)

        #add random noise so its not the same always that is chosen in topk later
        _, prob_pair_idx = tf.nn.top_k(to_sort_for_prob_pair, k=1)
        #if there is no hit prob_pair_dist can be 0.. update block_to_look_into to handle this case
        #i now checks the actual hash codes for 0 distance, so this is no longer necesarry
        #block_to_look_into = tf.cast(masked_hashcode,dtype=tf.float32) * tf.cast(prob_pair_dist>1,dtype=tf.float32)
        prob_pairs = tf.nn.embedding_lookup(doc_idx_memory, prob_pair_idx)

        if self.args['top_k_pair'] > 0:
            #prioritize pick those that can not be retrived (distance higher than n_blocks * 2 - 1)
            #after that prioritise those in the range n_blocks-1 to n_blocks*2-1
            #if there is none of these, all are ok and should not do anything.
            dist_sums = (self.args['bits']-code_sums)/2 #convert to distances instead of similarities
            dist100 = (self.args['bits']-dist100)/2
            #conditions for the sample to hold
            is_below_dist100 = tf.cast(dist_sums < tf.expand_dims(dist100, -1), tf.float32)
            is_above_n_block = tf.cast(tf.greater_equal(distance_in_block,blocks-1),tf.float32) # there are not optimal
            is_above_2n_block = tf.cast(tf.greater_equal(distance_in_block, 2*blocks - 1), tf.float32) #these can not be retrieved
            #add randomness that will still keep overall order
            potential_pos_pair = (is_below_dist100*(is_above_n_block+is_above_2n_block)) *  tf.random.normal([self.batchsize, self.memsize],mean=2, stddev=0.1) #sample the most problematic pairs first
            _, closests_top_k_idx = tf.nn.top_k(potential_pos_pair, k=1)
            return prob_pairs, block_to_look_into, distance_to_look_into, closests_top_k_idx
        elif self.args['new_top_k_pair'] > 0:
            # convert to distances instead of similarities
            dist_sums = (self.args['bits']-code_sums)/2
            dist100 = (self.args['bits']-dist100)/2
            is_below_equal_dist100 = tf.cast(dist_sums <= tf.expand_dims(dist100, -1), tf.float32)
            expensive_distance = self.args['new_top_k_pair_apply_at_block_distance'] * blocks - 1
            is_above_expensive_thresh = tf.cast(dist_sums > expensive_distance, tf.float32)
            dist_sums = tf.cast(dist_sums,tf.float32)
            print(dist_sums,is_below_equal_dist100, is_above_expensive_thresh)
            potential_pos_pair = dist_sums*is_below_equal_dist100*is_above_expensive_thresh + tf.random.normal([self.batchsize, self.memsize], mean=0.0, stddev=0.05) #just for tie breaking
            _, close_top_k_idx = tf.nn.top_k(potential_pos_pair, k=1)
            #the cost is primarily caused by the distance at top 100, the only way to reduce the cost is to reduce the distance to top 100
            return prob_pairs, block_to_look_into, distance_to_look_into, close_top_k_idx, expensive_distance


        else:
            return prob_pairs, block_to_look_into, distance_to_look_into



    def get_memory_updates(self, doc_hashcode, emb_dtype, doc_STH, doc_idx, label, batchsize = None):
        if batchsize is None:
            batchsize= self.args['batchsize']
        mem_indices = tf.Variable(np.arange(batchsize, dtype=np.int32), dtype=tf.int32)
        update_mem_indices = mem_indices.assign( (mem_indices + self.args["batchsize"]) % self.memsize )
        hashcode_embedding_memory_update = tf.scatter_update(self.hashcode_embedding_memory, mem_indices, tf.cast(doc_hashcode, emb_dtype))
        STHcodes_memory_update = tf.scatter_update(self.STHcodes_memory, mem_indices, tf.cast(doc_STH, emb_dtype))
        doc_idx_memory_update = tf.scatter_update(self.doc_idx_memory, mem_indices, doc_idx)
        label_memory_update = tf.scatter_update(self.label_memory, mem_indices, label)
        all_updates = [hashcode_embedding_memory_update, STHcodes_memory_update, doc_idx_memory_update, label_memory_update]
        return all_updates, mem_indices, update_mem_indices


    def refresh_memory_module(self, refresh_sample, refresh_batchsize):
        emb_dtype = tf.int32
        _, doc_idx, doc_STH, label = refresh_sample
        doc_bow = self.get_doc_bow(doc_idx)
        _, _, doc_hashcode_det = self.encoder(doc_bow)
        mem_updates, _, update_mem_indices = self.get_memory_updates(doc_hashcode_det, emb_dtype, doc_STH, doc_idx, label, refresh_batchsize)
        return mem_updates, update_mem_indices


    def make_network(self, bow_sparse_matrix):
        emb_size = self.args["bits"]
        emb_dtype = tf.int32

        #init all necesarry Vars for memory and embeddings
        self.data_tensor = self.make_data_tensor(bow_sparse_matrix)
        self.importance_emb_matrix = tf.Variable(tf.random_uniform(shape=[self.args["bowlen"]], minval=0.1, maxval=1),
                        trainable=True, name="importance_embedding")

        if self.args['embedding_size']>0:
            self.word_emb_matrix = tf.Variable(tf.random_uniform(shape=[self.args["bowlen"], self.args["embedding_size"]], minval=-1, maxval=1),
                                                 trainable=True, name="word_embedding")
        else:
            self.word_emb_matrix = tf.Variable(tf.random_uniform(shape=[self.args["bowlen"], self.args["bits"]], minval=-1, maxval=1),
                                                 trainable=True, name="word_embedding")

        self.hashcode_embedding = tf.Variable(tf.zeros(shape=[self.num_dataset_samples, emb_size], dtype=emb_dtype), trainable=False, name="hashcode_emb")

        self.memsize = self.args["memsize"] #tf.cond(self.is_training, lambda : self.args["check_num_samples"] - self.args["check_num_samples"] % self.args["batchsize"], lambda : self.batchsize)

        self.hashcode_embedding_memory = tf.Variable(tf.ones(shape=[self.memsize, emb_size], dtype=emb_dtype), trainable=False, name="hashcode_emb_mem")
        self.STHcodes_memory = tf.Variable(tf.zeros(shape=[self.memsize, 64], dtype=tf.int32), trainable=False, name="STHcodes_mem")
        self.doc_idx_memory = tf.Variable(tf.zeros(shape=[self.memsize,], dtype=tf.int32), trainable=False, name="doc_idx_mem")
        self.label_memory = tf.Variable(tf.zeros(shape=[self.memsize, self.args["num_labels"]], dtype=tf.int32), trainable=False, name="label_mem")
        self.softmax_bias = tf.Variable(tf.zeros(self.args["bowlen"]), name="softmax_bias")


        #read sample
        _, doc_idx, doc_STH, label = self.sample
        doc_bow = self.get_doc_bow(doc_idx)

        #encode sample
        doc_hashcode, doc_sampling_hashcode, doc_hashcode_det = self.encoder(doc_bow)

        # find nearest STH
        label_nbr_idx = self.get_most_similar_idx(self.STHcodes_memory, doc_STH)
        label_nbr_bow = self.get_doc_bow(label_nbr_idx)
        label_nbr_hashcode, label_nbr_sampling_hashcode, label_nbr_hashcode_det = self.encoder(label_nbr_bow)


        # find triplet samples. The same-label with the longest distance and different-label with closest distance.
        same_idx, opposite_idx, same_exists, opposite_exists, top_k_prec, top_k_sims = self.top_k_idx(self.hashcode_embedding_memory, self.label_memory, doc_hashcode_det, label)
        same_idx_bow = self.get_doc_bow(same_idx)
        same_idx_hashcode, same_idx_sampling_hashcode, _ = self.encoder(same_idx_bow)
        opposite_idx_bow = self.get_doc_bow(opposite_idx)
        opposite_idx_hashcode, opposite_idx_sampling_hashcode, _ = self.encoder(opposite_idx_bow)


        #find problematic pairs for minwise hashing:
        if self.args['problem_pair'] >= 0 or self.args['new_top_k_pair'] > 0:

            if self.args['sample_blocks'] > 0:
                print("sampling from all blocks")
                fun_to_call = self.find_problematic_pair_updated
            else:
                print("only find 0 blocks")
                fun_to_call = self.find_problematic_pair


            if self.args['top_k_pair'] > 0:
                prob_pairs_dox_idx, block_mask, distance_to_look_into, closests_top_k_idx = \
                    fun_to_call(self.hashcode_embedding_memory, doc_hashcode_det, self.doc_idx_memory, kk=100)
            elif self.args['new_top_k_pair'] > 0:
                prob_pairs_dox_idx, block_mask, distance_to_look_into,  close_top_k_idx, expensive_distance = \
                    fun_to_call(self.hashcode_embedding_memory, doc_hashcode_det, self.doc_idx_memory, kk=100)
            else:
                prob_pairs_dox_idx, block_mask, distance_to_look_into = fun_to_call(self.hashcode_embedding_memory, doc_hashcode_det, self.doc_idx_memory, kk=100)
            print("##############")
            print(prob_pairs_dox_idx, block_mask, distance_to_look_into)
            print("##############")
            prob_pair_doc_bow = self.get_doc_bow(tf.squeeze(prob_pairs_dox_idx))
            prob_pair_doc_hashcode, prob_pair_doc_sampling_hashcode, prob_pair_doc_hashcode_det = self.encoder(prob_pair_doc_bow)
            block_mask = tf.cast(block_mask,tf.float32)
            prob_pair_doc_hashcode_det_masked = prob_pair_doc_hashcode_det*block_mask
            doc_hashcode_det_masked = doc_hashcode_det*block_mask
            #Check if the problem still persist (the hash codes come from memory, that may no longer be valid)
            #if same, should be equal to block size

            if self.args['anchor_problem_pair'] > 0:
                doc_hashcode_det_masked = tf.stop_gradient(doc_hashcode_det_masked)

            sim = tf.reduce_sum(prob_pair_doc_hashcode_det_masked * doc_hashcode_det_masked, -1)
            if self.args['sample_blocks'] > 0:
                if self.args['problem_pair_less_than'] >0:
                    is_0_dist = tf.stop_gradient(
                        tf.cast((tf.greater_equal(tf.cast(sim, tf.int32), self.args['block_size'] - distance_to_look_into * 2)),
                                dtype=tf.float32))
                else:
                    is_0_dist = tf.stop_gradient(
                        tf.cast((tf.equal(tf.cast(sim, tf.int32), self.args['block_size']-distance_to_look_into*2)), dtype=tf.float32))
            else:
                is_0_dist = tf.stop_gradient(
                    tf.cast((tf.equal(tf.cast(sim, tf.int32), self.args['block_size'])),
                            dtype=tf.float32))

            valid_pairs_problem_pair = tf.reduce_sum(is_0_dist) / tf.cast(self.batchsize,tf.float32)
            prob_pair_loss = tf.reduce_sum(sim*is_0_dist)
            if  self.args['problem_pair'] > 0:
                self.prob_pair_loss = prob_pair_loss*self.problem_pair_coef
            else:
                self.prob_pair_loss = tf.zeros(1)

            #problematic loss
            if self.args['top_k_pair'] > 0:
                raise Exception("No longer supported top_k_pair, use new_top_k_pair!!")
            #new loss for keeping top 100 closer, to reduce retrieval cost
            if self.args['new_top_k_pair'] > 0:
                close_pair_doc_bow = self.get_doc_bow(tf.squeeze(close_top_k_idx))
                _, _, close_pair_doc_hashcode_det = self.encoder(close_pair_doc_bow)

                if self.args['anchor_problem_pair'] > 0:
                    doc_hashcode_det_to_use = tf.stop_gradient(doc_hashcode_det)
                else:
                    doc_hashcode_det_to_use = doc_hashcode_det

                sim_close = tf.reduce_sum(close_pair_doc_hashcode_det * doc_hashcode_det_to_use,axis=-1)
                dist_close = (self.args['bits']-sim_close)/2

                is_above_expensive_thresh = tf.cast(dist_close >= expensive_distance, dtype=tf.float32)

                close_pair_loss = tf.reduce_sum(-sim_close*is_above_expensive_thresh)
                self.prob_pair_loss =  self.prob_pair_loss + close_pair_loss * self.args['new_top_k_pair'] 

                valid_pairs_top_k_pair = tf.reduce_sum(is_above_expensive_thresh) / tf.cast(self.batchsize,tf.float32)


            if self.args['recon_pairs'] > 0:
                prob_pair_recon_loss = self.decoder(prob_pair_doc_hashcode, prob_pair_doc_sampling_hashcode, prob_pair_doc_bow)
                pair_recon_loss = prob_pair_recon_loss * self.args['recon_pairs']
                if self.args["top_k_pair"] > 0:
                    close_pair_recon_loss = self.decoder(close_pair_doc_hashcode, close_pair_doc_sampling_hashcode, close_pair_doc_bow)
                    pair_recon_loss = pair_recon_loss + close_pair_recon_loss * self.args['recon_pairs']
            else:
                pair_recon_loss = tf.zeros(1)

        else:
            self.prob_pair_loss = tf.zeros(1)
            self.problem_pair_dist_0 = tf.zeros(1)
            pair_recon_loss = tf.zeros(1)


        # decode
        doc_loss = self.decoder(doc_hashcode, doc_sampling_hashcode, doc_bow)
        label_nbr_loss = self.decoder(label_nbr_hashcode, label_nbr_sampling_hashcode, doc_bow)
        supervised_doc_loss = tf.zeros(1)#self.supervised_decoder(doc_hashcode, label)

        dist_opposite = tf.reduce_sum(doc_hashcode * opposite_idx_hashcode, 1)
        dist_same = tf.reduce_sum(doc_hashcode * same_idx_hashcode, 1)

        rank_loss = opposite_exists*hinge_loss_eps(labels=tf.ones(self.batchsize), logits=(dist_same - dist_opposite), epsval=1.0)

        loss = doc_loss + pair_recon_loss
        if self.args["simoppositeweight"] > 0.000001:
            print("use sim opposite weight", self.args["simoppositeweight"])
            loss = opposite_exists*(-(self.args["simoppositeweight"] * dist_opposite) + self.args["simoppositeweight"]  * dist_same * same_exists)

        if self.args["rankweight"] > 0.000001:
            loss += self.args["rankweight"] * rank_loss
        else:
            rank_loss = tf.zeros(1)
        if self.args["superweight"] > 0.000001:
            loss += self.args["superweight"] * supervised_doc_loss
        else:
            supervised_doc_loss = tf.zeros(1)
        if self.args["doc2weight"] > 0.000001:
            loss += self.args["doc2weight"] * label_nbr_loss
        else:
            label_nbr_loss = tf.zeros(1)

        # update memory
        mem_updates, mem_indices, update_mem_indices = self.get_memory_updates(doc_hashcode_det, emb_dtype, doc_STH, doc_idx, label)

        train_op = tf.train.AdamOptimizer(learning_rate=self.args["lr"],name="AdamOptimizer")

        if self.args["agg_loss"] == -1:
            oploss = loss
        elif self.args["agg_loss"] == 0:
            oploss = tf.reduce_mean(loss, -1)
        elif self.args["agg_loss"] == 1:
            oploss = tf.reduce_sum(loss, -1)
        else:
            exit(-1)

        #add bit balance loss
        bit_balance_loss, doc_hashcode_det = self.compute_correlation_loss(doc_sampling_hashcode, doc_hashcode_det)
        nbr_bit_balance_loss, label_nbr_hashcode_det = self.compute_correlation_loss(label_nbr_sampling_hashcode, label_nbr_hashcode_det)
        self.bit_balance_loss = (bit_balance_loss + nbr_bit_balance_loss) * self.args['coef_bb']
        oploss = oploss + self.bit_balance_loss + self.prob_pair_loss

        train_op = train_op.minimize(oploss)
        emb_update = tf.scatter_update(self.hashcode_embedding, doc_idx, tf.cast(doc_hashcode_det, emb_dtype)) 


        #some logging of number of valid samples
        if self.args['problem_pair'] > 0:
            self.valid_pairs_problem_pair = valid_pairs_problem_pair
        else:
            self.valid_pairs_problem_pair = tf.zeros(1)-1

        if self.args['new_top_k_pair'] > 0:
            self.valid_pairs_top_k_pair = valid_pairs_top_k_pair
        else:
            self.valid_pairs_top_k_pair = tf.zeros(1)-1



        return train_op, loss, emb_update, doc_loss, mem_updates, update_mem_indices, supervised_doc_loss, rank_loss, label_nbr_loss, \
               -((dist_opposite-self.args["bits"])/2), -((dist_same-self.args["bits"])/2), top_k_prec, same_exists


    def return_bit_balance_loss(self):
        return self.bit_balance_loss

    def return_prob_pair_loss(self):
        return self.prob_pair_loss, self.valid_pairs_problem_pair

    def return_valid_pairs(self):
        return self.valid_pairs_problem_pair, self.valid_pairs_top_k_pair