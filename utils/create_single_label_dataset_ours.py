# 2024.10.16 by hly
import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import argparse
from sklearn.utils import shuffle
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
#from nltk.stem import PorterStemmer
from pathlib import Path

##################################################################################################

# the path to this project
home = "/data/lyhe/DeepTextHashing/"

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

elif args.dataset == 'agnews':
    root_dir = os.path.join(home, 'datasets/agnews')
    train_fn = os.path.join(root_dir, 'train.csv')
    df = pd.read_csv(train_fn, header=None)
    df.columns = ['label', 'body']
    train_docs = list(df.body)
    train_tags = list(df.label - 1)

    del df
    
    test_fn = os.path.join(root_dir, 'test.csv')
    df = pd.read_csv(test_fn, header=None)
    df.columns = ['label', 'body']
    test_docs = list(df.body)
    test_tags = list(df.label - 1)
    
    del df

# remove any short document that has less than 5 words
# remove any long document that has more than 500 words
# train_docs,test_docs,train_tags,test_tags:list
print(type(train_docs), type(test_docs))
docs = np.concatenate((train_docs, test_docs))
tags = np.concatenate((train_tags, test_tags))

new_docs = []
new_tags = []

print("the original docs number:", len(docs))
for d,t in zip(docs,tags):
    if len(d.split(" ")) < 10:
        continue
    if len(d.split(" ")) >= 500:
        continue
    new_docs.append(d)
    new_tags.append(t)
print("after process:", len(new_docs))

# process to tf
count_vect = CountVectorizer(stop_words='english', max_features=args.vocab_size, max_df=args.max_df, min_df=args.min_df)
docs_tf = count_vect.fit_transform(new_docs)

def create_dataframe(doc_tf, doc_targets):
    docs = []
    for i, bow in enumerate(doc_tf):
        d = {'doc_id': i, 'bow': bow, 'label': doc_targets[i]}
        docs.append(d)
    df = pd.DataFrame.from_dict(docs)
    df.set_index('doc_id', inplace=True)
    return df

docs_df = create_dataframe(docs_tf, new_tags)

print('Before filtering: num train: {}'.format(len(docs_df)))

def get_doc_length(doc_bow):
    return doc_bow.sum()
# remove empty document
docs_df = docs_df[docs_df.bow.apply(get_doc_length) > 0]

print('num train: {}'.format(len(docs_df)))

train_df, temp_df = train_test_split(docs_df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print('train: {} test: {} cv: {}'.format(len(train_df), len(val_df), len(test_df)))

##################################################################################################

# save the dataframes
save_dir = '../datasets/{}'.format(args.dataset)
print('save tf dataset to {} ...'.format(save_dir))

train_df.to_pickle(os.path.join(save_dir, 'train.tf.df.pkl'))
test_df.to_pickle(os.path.join(save_dir, 'test.tf.df.pkl'))
val_df.to_pickle(os.path.join(save_dir, 'cv.tf.df.pkl'))

# save vocab
with open('../datasets/{}/vocab.pkl'.format(args.dataset), 'wb') as handle:
    pickle.dump(count_vect.vocabulary_, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print('Done.')