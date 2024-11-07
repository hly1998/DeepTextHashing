import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import argparse
from sklearn.utils import shuffle
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
#from nltk.stem import PorterStemmer
from pathlib import Path

##################################################################################################

home = str(Path.home())

##################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="Name of the dataset.")
parser.add_argument("-v", "--vocab_size", type=int, default=10000, help="The number of vocabs.")
parser.add_argument("--num_train", type=int, default=0, help="The number of training samples.")
parser.add_argument("--num_test", type=int, default=0, help="The number of testing and cv samples.")

parser.add_argument("--max_df", default=0.8, type=float)
parser.add_argument("--min_df", default=3, type=int)
parser.add_argument('--remove_short_docs', dest='remove_short_docs', action='store_true', help='Remove any document that has a length less than 5 words.')
parser.add_argument('--remove_long_docs', dest='remove_long_docs', action='store_true', help='Remove any document that has a length more than 500 words.')
parser.set_defaults(remove_short_docs=True)
parser.set_defaults(remove_long_docs=True)

args = parser.parse_args()
    
if not args.dataset:
    parser.error("Need to provide the dataset.")

##################################################################################################
remove_short_document = args.remove_short_docs
remove_long_document = args.remove_long_docs

if args.dataset == 'ng20':
    train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes')) 
    test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    train_docs = train.data
    train_tags = train.target
    test_docs = test.data
    test_tags = test.target
elif args.dataset == 'dbpedia':
    root_dir = os.path.join(home, 'datasets/dbpedia')
    train_fn = os.path.join(root_dir, 'train.csv')
    df = pd.read_csv(train_fn, header=None)
    df.columns = ['label', 'title', 'body']
    train_docs = list(df.body)
    train_tags = list(df.label - 1)

    del df
    
    test_fn = os.path.join(root_dir, 'test.csv')
    df = pd.read_csv(test_fn, header=None)
    df.columns = ['label', 'title', 'body']
    test_docs = list(df.body)
    test_tags = list(df.label - 1)
    
    del df
elif args.dataset == 'agnews':
    root_dir = os.path.join(home, 'datasets/agnews')
    train_fn = os.path.join(root_dir, 'train.csv')
    df = pd.read_csv(train_fn, header=None)
    df.columns = ['label', 'title', 'body']
    train_docs = list(df.body)
    train_tags = list(df.label - 1)

    del df
    
    test_fn = os.path.join(root_dir, 'test.csv')
    df = pd.read_csv(test_fn, header=None)
    df.columns = ['label', 'title', 'body']
    test_docs = list(df.body)
    test_tags = list(df.label - 1)
    
    del df

elif args.dataset == 'yahooanswer':
    root_dir = os.path.join(home, 'datasets/yahooanswer')
    train_fn = os.path.join(root_dir, 'train.csv')
    df = pd.read_csv(train_fn, header=None)
    df.columns = ['label', 'title', 'body', 'answer']
    train_docs = list(df.title)
    train_tags = list(df.label - 1)

    test_fn = os.path.join(root_dir, 'test.csv')
    df = pd.read_csv(test_fn, header=None)
    df.columns = ['label', 'title', 'body', 'answer']
    test_docs = list(df.title)
    test_tags = list(df.label - 1)
    
    del df

##################################################################################################

count_vect = CountVectorizer(stop_words='english', max_features=args.vocab_size, max_df=args.max_df, min_df=args.min_df)
train_tf = count_vect.fit_transform(train_docs)
test_tf = count_vect.transform(test_docs)

def create_dataframe(doc_tf, doc_targets):
    docs = []
    for i, bow in enumerate(doc_tf):
        d = {'doc_id': i, 'bow': bow, 'label': doc_targets[i]}
        docs.append(d)
    df = pd.DataFrame.from_dict(docs)
    df.set_index('doc_id', inplace=True)
    return df

train_df = create_dataframe(train_tf, train_tags)
if args.num_train < 0:
    parser.error("The number of training samples must be positive.")
if args.num_train > len(train_df):
    parser.error("The number of training samples must not exceed the total number of samples.")

if args.num_train > 0:
    train_df = train_df.sample(n=args.num_train)
    
test_df = create_dataframe(test_tf, test_tags)
if args.num_test < 0:
    parser.error("The number of testing samples must be positive.")
if args.num_test * 2 > len(test_df):
    parser.error("The number of testing samples must not exceed the half of the total number of samples. We will use another half for CV set.")

if args.num_test > 0:
    test_df = test_df.sample(n=args.num_test * 2)

print('Before filtering: num train: {} num test: {}'.format(len(train_df), len(test_df)))
##################################################################################################

def get_doc_length(doc_bow):
    return doc_bow.sum()

# def get_num_word(doc_bow):
#     return doc_bow.nonzero()[1].shape[0]

# remove an empty document
train_df = train_df[train_df.bow.apply(get_doc_length) > 0]
test_df = test_df[test_df.bow.apply(get_doc_length) > 0]

print('num train: {} num test: {}'.format(len(train_df), len(test_df)))

if remove_short_document:
    print('remove any short document that has less than 5 words.')
    train_df = train_df[train_df.bow.apply(get_doc_length) > 5]
    test_df = test_df[test_df.bow.apply(get_doc_length) > 5]
    print('num train: {} num test: {}'.format(len(train_df), len(test_df)))

if remove_long_document:
    print('remove any long document that has more than 500 words.')
    train_df = train_df[train_df.bow.apply(get_doc_length) <= 500]
    test_df = test_df[test_df.bow.apply(get_doc_length) <= 500]
    print('num train: {} num test: {}'.format(len(train_df), len(test_df)))
    
##################################################################################################

# split test and cv
num_train = len(train_df)
num_test = len(test_df) // 2
num_cv = len(test_df) - num_test

print('train: {} test: {} cv: {}'.format(num_train, num_test, num_cv))

test_df = shuffle(test_df)
cv_df = test_df.iloc[:num_cv]
test_df = test_df.iloc[num_cv:]

##################################################################################################

# save the dataframes
save_dir = '../datasets/{}'.format(args.dataset)
print('save tf dataset to {} ...'.format(save_dir))

train_df.to_pickle(os.path.join(save_dir, 'train.tf.df.pkl'))
test_df.to_pickle(os.path.join(save_dir, 'test.tf.df.pkl'))
cv_df.to_pickle(os.path.join(save_dir, 'cv.tf.df.pkl'))

# save vocab
with open('../datasets/{}/vocab.pkl'.format(args.dataset), 'wb') as handle:
    pickle.dump(count_vect.vocabulary_, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print('Done.')