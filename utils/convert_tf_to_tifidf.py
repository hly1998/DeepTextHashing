import os
import numpy as np
import pandas as pd
import pickle
import argparse
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix, csr_matrix, vstack

######################################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="Name of the dataset.")

args = parser.parse_args()
    
if not args.dataset:
    parser.error("Need to provide the dataset.")

dataset = args.dataset
data_dir = '../datasets/{}'.format(dataset)
    
######################################################################################################################################

print('Transform to TFIDF format ...')

train_df = pd.read_pickle(os.path.join(data_dir, 'train.tf.df.pkl'))
test_df = pd.read_pickle(os.path.join(data_dir, 'test.tf.df.pkl'))
cv_df = pd.read_pickle(os.path.join(data_dir, 'cv.tf.df.pkl'))

def convert_to_dataframe(source_df, doc_id_list, bow_list, label_list):
    df = pd.DataFrame({'doc_id': doc_id_list, 'bow': bow_list, 'label': label_list})
    df.set_index('doc_id', inplace=True)
    return df

train_tf = vstack(list(train_df.bow))
test_tf = vstack(list(test_df.bow))
cv_tf = vstack(list(cv_df.bow))

transformer = TfidfTransformer(sublinear_tf=True)
train_tfidf = transformer.fit_transform(train_tf)
test_tfidf = transformer.transform(test_tf)
cv_tfidf = transformer.transform(cv_tf)

train_tfidf_df = convert_to_dataframe(train_df, list(train_df.index), [bow for bow in train_tfidf], list(train_df.label))
test_tfidf_df = convert_to_dataframe(test_df, list(test_df.index), [bow for bow in test_tfidf], list(test_df.label))
cv_tfidf_df = convert_to_dataframe(cv_df, list(cv_df.index), [bow for bow in cv_tfidf], list(cv_df.label))

save_dir = '../datasets/{}'.format(args.dataset)
print('save tf dataset to {} ...'.format(save_dir))

train_tfidf_df.to_pickle(os.path.join(save_dir, 'train.tfidf.df.pkl'))
test_tfidf_df.to_pickle(os.path.join(save_dir, 'test.tfidf.df.pkl'))
cv_tfidf_df.to_pickle(os.path.join(save_dir, 'cv.tfidf.df.pkl'))

######################################################################################################################################
# Binary format
print('Transform to Binary format ...')

def create_dataframe(doc_tf, doc_targets):
    docs = []
    for i, bow in enumerate(doc_tf):
        d = {'doc_id': i, 'bow': bow, 'label': doc_targets[i]}
        docs.append(d)
    df = pd.DataFrame.from_dict(docs)
    df.set_index('doc_id', inplace=True)
    return df

def create_bin_matrix(doc_tf_df):
    # create TFIDF
    doc_bin = []
    for index, row in doc_tf_df.iterrows():
        bow = (row.bow.toarray().squeeze() > 0).astype(np.float64)
        bow = csr_matrix(bow)
        doc_bin.append(bow)
    return vstack(doc_bin)

train_bin_df = create_dataframe(create_bin_matrix(train_df), list(train_df.label))
test_bin_df = create_dataframe(create_bin_matrix(test_df), list(test_df.label))
cv_bin_df = create_dataframe(create_bin_matrix(cv_df), list(cv_df.label))

# save the dataframes
train_bin_df.to_pickle(os.path.join(save_dir, 'train.bin.df.pkl'))
test_bin_df.to_pickle(os.path.join(save_dir, 'test.bin.df.pkl'))
cv_bin_df.to_pickle(os.path.join(save_dir, 'cv.bin.df.pkl'))

print('Done.')