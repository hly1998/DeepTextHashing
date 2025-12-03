import tensorflow as tf

def generator(sess, maxsize, handle, batchsize, record_paths, is_test, bowlen, labellen):
    def extract_fn(data_record):
        '''
        features["doc_bow_data"] = create_float_feature(doc_bow_data.flatten())
        features["doc_bow_indices"] = create_int_feature(doc_bow_indices.flatten())
        features["doc_idx"] = create_int_feature([int(doc_idx)])
        features["label"] = create_int_feature(labels[doc_idx].flatten())
        '''
        features = {
            'doc_bow_data': tf.VarLenFeature(tf.float32),
            'doc_bow_indices': tf.VarLenFeature(tf.int64),
            'doc_idx': tf.FixedLenFeature([1], tf.int64),
            'STHcode': tf.FixedLenFeature([64], tf.int64),
            'label': tf.FixedLenFeature([labellen], tf.int64)
        }

        sample = tf.parse_single_example(data_record, features)

        for key in ['doc_idx']:
            sample[key] = tf.squeeze(sample[key], -1)

        for key in ['doc_idx', 'label', 'STHcode']:
            sample[key] = tf.cast(sample[key], tf.int32)

        doc_bow_index = tf.expand_dims(sample['doc_bow_indices'].values, -1)# tf.reshape(sample['doc_bow_indices'].values, [-1, 1])
        doc_bow_values = sample['doc_bow_data'].values
        doc_bow = tf.sparse.SparseTensor(indices=doc_bow_index, values=doc_bow_values, dense_shape=[bowlen,])
        #doc_bow = tf.sparse.to_dense(doc_bow, )

        # print("###", doc_bow, tf.sparse.to_dense(doc_bow, ))
        sample["doc_bow"] = doc_bow

        feature_order = ['doc_bow', 'doc_idx', 'STHcode', 'label']

        return tuple([sample[key] for key in feature_order])

    output_t = [tf.int32 for _ in range(4)]
    output_t[0] = tf.float32
    output_t = tuple(output_t)

    default_shape = tf.TensorShape([None,])
    bow_shape = tf.TensorShape([None, bowlen,])
    label_shape = tf.TensorShape([None, labellen,])
    sth_shape = tf.TensorShape([None, 64])
    output_s = [bow_shape, default_shape, sth_shape, label_shape]
    output_s = tuple(output_s)

    dataset = tf.data.Dataset.from_tensor_slices(record_paths)
    if not is_test:
        dataset = dataset.repeat()
    dataset = dataset.shuffle(100)
    dataset = dataset.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(extract_fn, num_parallel_calls=2)
    if not is_test:
        dataset = dataset.shuffle(30000)
    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(100)
    iterator = dataset.make_initializable_iterator()

    generic_iter = tf.data.Iterator.from_string_handle(handle, output_t, output_s, tuple([tf.SparseTensor, tf.Tensor, tf.Tensor, tf.Tensor]))
    specific_handle = sess.run(iterator.string_handle())

    return specific_handle, iterator, generic_iter