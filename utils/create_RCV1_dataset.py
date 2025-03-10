import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_rcv1

##################################################################################################

home = str(Path.home())
rcv1 = fetch_rcv1()

##################################################################################################
num_train = 100000
num_test = 20000

num_labels = 40
num_vocabs = 15000
remove_short_document = True
remove_long_document = True

##################################################################################################
feature_indices = np.argsort(-rcv1.target.sum(axis=0), axis=1)[0, :num_labels]
feature_indices = np.asarray(feature_indices).squeeze()
targets = rcv1.target[:, feature_indices]

word_indices = np.argsort(-rcv1.data.sum(axis=0), axis=1)[0, :num_vocabs]
word_indices = np.asarray(word_indices).squeeze()
documents = rcv1.data[:, word_indices]

targets = [t for t in targets]
documents = [d for d in documents]

df = pd.DataFrame({'doc_id': rcv1.sample_id.tolist(), 'bow': documents, 'label': targets})
df.set_index('doc_id', inplace=True)
print('total docs: {}'.format(len(df)))

##################################################################################################
# remove any empty labels
def count_num_tags(target):
    return target.sum()

df = df[df.label.apply(count_num_tags) > 0]
print('after filter: total docs: {}'.format(len(df)))

def get_num_word(bow):
    return bow.count_nonzero()

df = df[df.bow.apply(get_num_word) > 0]
print('after filter: total docs: {}'.format(len(df)))

# remove any empty documents
if remove_short_document:
    print('remove any short document that has less than 5 words.')
    df = df[df.bow.apply(get_num_word) > 5]
    print('num docs: {}'.format(len(df)))

if remove_long_document:
    print('remove any long document that has more than 500 words.')
    df = df[df.bow.apply(get_num_word) <= 500]
    print('num docs: {}'.format(len(df)))

df = df.reindex(np.random.permutation(df.index))

##################################################################################################
sampled_df = df.sample(num_train + num_test)
train_df = sampled_df.iloc[:num_train]

test_df = sampled_df.iloc[num_train:]
cv_df = test_df[:num_test//2]
test_df = test_df[num_test//2:]

print('num train: {} num test: {} num cv: {}'.format(len(train_df), len(test_df), len(cv_df)))

##################################################################################################
# save the dataframes
save_dir = '../datasets/rcv1'
print('save tfidf dataset to {} ...'.format(save_dir))

train_df.to_pickle(os.path.join(save_dir, 'train.tfidf.df.pkl'))
test_df.to_pickle(os.path.join(save_dir, 'test.tfidf.df.pkl'))
cv_df.to_pickle(os.path.join(save_dir, 'cv.tfidf.df.pkl'))