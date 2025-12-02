"""
数据预处理脚本
将原始数据转换为模型可用的格式 (TF, TF-IDF, Binary)

使用方法:
    # 处理单标签数据集
    python preprocess.py -d ng20 -v 10000

    # 处理多标签数据集
    python preprocess.py -d tmc -v 10000

    # 转换为 TF-IDF 格式
    python preprocess.py -d ng20 --convert-tfidf
"""

import argparse
import pickle
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split


# ============================================================================
# 配置
# ============================================================================

TEXTDATA_DIR = Path(__file__).parent

SINGLE_LABEL_DATASETS = ['ng20', 'agnews', 'dbpedia', 'yahooanswer']
MULTI_LABEL_DATASETS = ['reuters', 'tmc', 'rcv1']


# ============================================================================
# 数据加载函数
# ============================================================================

def load_ng20() -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """加载 20 Newsgroups 数据集"""
    from sklearn.datasets import fetch_20newsgroups

    print("加载 20 Newsgroups 数据集...")
    train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    return train.data, train.target, test.data, test.target


def load_agnews(data_dir: Path) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """加载 AG News 数据集"""
    print("加载 AG News 数据集...")

    train_df = pd.read_csv(data_dir / 'train.csv', header=None)
    train_df.columns = ['label', 'title', 'body']
    train_docs = list(train_df.body)
    train_tags = np.array(train_df.label - 1)

    test_df = pd.read_csv(data_dir / 'test.csv', header=None)
    test_df.columns = ['label', 'title', 'body']
    test_docs = list(test_df.body)
    test_tags = np.array(test_df.label - 1)

    return train_docs, train_tags, test_docs, test_tags


def load_dbpedia(data_dir: Path) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """加载 DBpedia 数据集"""
    print("加载 DBpedia 数据集...")

    train_df = pd.read_csv(data_dir / 'train.csv', header=None)
    train_df.columns = ['label', 'title', 'body']
    train_docs = list(train_df.body)
    train_tags = np.array(train_df.label - 1)

    test_df = pd.read_csv(data_dir / 'test.csv', header=None)
    test_df.columns = ['label', 'title', 'body']
    test_docs = list(test_df.body)
    test_tags = np.array(test_df.label - 1)

    return train_docs, train_tags, test_docs, test_tags


def load_yahooanswer(data_dir: Path) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """加载 Yahoo Answer 数据集"""
    print("加载 Yahoo Answer 数据集...")

    train_df = pd.read_csv(data_dir / 'train.csv', header=None)
    train_df.columns = ['label', 'title', 'body', 'answer']
    train_docs = list(train_df.title)
    train_tags = np.array(train_df.label - 1)

    test_df = pd.read_csv(data_dir / 'test.csv', header=None)
    test_df.columns = ['label', 'title', 'body', 'answer']
    test_docs = list(test_df.title)
    test_tags = np.array(test_df.label - 1)

    return train_docs, train_tags, test_docs, test_tags


def load_reuters() -> Tuple[List[str], csr_matrix, List[str], csr_matrix]:
    """加载 Reuters 数据集 (多标签)"""
    import nltk
    try:
        nltk.download('reuters', quiet=True)
    except Exception:
        print("下载 reuters 数据集时出错")
    from nltk.corpus import reuters

    print("加载 Reuters 数据集...")

    train_docs, test_docs = [], []
    train_tags_raw, test_tags_raw = [], []

    for doc_id in reuters.fileids():
        if doc_id.startswith("train"):
            train_docs.append(reuters.raw(doc_id))
            train_tags_raw.append(' '.join(reuters.categories(doc_id)))
        else:
            test_docs.append(reuters.raw(doc_id))
            test_tags_raw.append(' '.join(reuters.categories(doc_id)))

    # 将标签转换为二值向量
    num_labels = 20
    label_tf = CountVectorizer(binary=True, max_features=num_labels)
    train_tags = csr_matrix(label_tf.fit_transform(train_tags_raw), dtype='int')
    test_tags = csr_matrix(label_tf.transform(test_tags_raw), dtype='int')

    return train_docs, train_tags, test_docs, test_tags


def load_tmc(data_dir: Path) -> Tuple[List[str], csr_matrix, List[str], csr_matrix]:
    """加载 TMC 数据集 (多标签)"""
    print("加载 TMC 数据集...")

    train_docs = []
    with open(data_dir / 'TrainingData.txt') as f:
        for line in f:
            train_docs.append(line.strip()[2:])

    test_docs = []
    with open(data_dir / 'TestData.txt') as f:
        for line in f:
            test_docs.append(line.strip()[2:])

    with open(data_dir / 'TrainCategoryMatrix.csv') as f:
        y_train = [[(int(v) + 1) // 2 for v in line.strip().split(',')] for line in f]
        train_tags = csr_matrix(np.array(y_train))

    with open(data_dir / 'TestTruth.csv') as f:
        y_test = [[(int(v) + 1) // 2 for v in line.strip().split(',')] for line in f]
        test_tags = csr_matrix(np.array(y_test))

    return train_docs, train_tags, test_docs, test_tags


def load_rcv1(num_labels: int = 40, num_vocabs: int = 15000):
    """加载 RCV1 数据集 (多标签, 已经是 TF-IDF 格式)"""
    from sklearn.datasets import fetch_rcv1

    print("加载 RCV1 数据集...")
    rcv1 = fetch_rcv1()

    # 选择最频繁的标签
    feature_indices = np.argsort(-rcv1.target.sum(axis=0), axis=1)[0, :num_labels]
    feature_indices = np.asarray(feature_indices).squeeze()
    targets = rcv1.target[:, feature_indices]

    # 选择最频繁的词
    word_indices = np.argsort(-rcv1.data.sum(axis=0), axis=1)[0, :num_vocabs]
    word_indices = np.asarray(word_indices).squeeze()
    documents = rcv1.data[:, word_indices]

    return documents, targets, rcv1.sample_id


# ============================================================================
# 数据处理函数
# ============================================================================

def create_dataframe(doc_bow, doc_targets) -> pd.DataFrame:
    """创建 DataFrame"""
    docs = []
    for i, bow in enumerate(doc_bow):
        d = {'doc_id': i, 'bow': bow, 'label': doc_targets[i]}
        docs.append(d)
    df = pd.DataFrame.from_dict(docs)
    df.set_index('doc_id', inplace=True)
    return df


def get_doc_length(doc_bow) -> int:
    """获取文档长度"""
    return doc_bow.sum()


def get_num_labels(label_bow) -> int:
    """获取标签数量"""
    return label_bow.nonzero()[1].shape[0]


def filter_documents(
    df: pd.DataFrame,
    min_length: int = 10,
    max_length: int = 500,
    remove_empty_labels: bool = False,
    is_multi_label: bool = False,
) -> pd.DataFrame:
    """过滤文档"""
    original_len = len(df)

    # 移除空文档
    df = df[df.bow.apply(get_doc_length) > 0]
    print(f"移除空文档后: {len(df)} 条 (移除 {original_len - len(df)} 条)")

    # 移除过短文档
    before = len(df)
    df = df[df.bow.apply(get_doc_length) >= min_length]
    print(f"移除过短文档 (<{min_length}): {len(df)} 条 (移除 {before - len(df)} 条)")

    # 移除过长文档
    before = len(df)
    df = df[df.bow.apply(get_doc_length) < max_length]
    print(f"移除过长文档 (>={max_length}): {len(df)} 条 (移除 {before - len(df)} 条)")

    # 移除无标签文档 (多标签数据集)
    if remove_empty_labels and is_multi_label:
        before = len(df)
        df = df[df.label.apply(get_num_labels) > 0]
        print(f"移除无标签文档: {len(df)} 条 (移除 {before - len(df)} 条)")

    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_ratio: float = 0.5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """划分数据集为 train/val/test"""
    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=val_ratio, random_state=random_state)

    print(f"数据集划分: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def convert_to_tfidf(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """将 TF 格式转换为 TF-IDF 格式"""
    print("转换为 TF-IDF 格式...")

    train_tf = vstack(list(train_df.bow))
    val_tf = vstack(list(val_df.bow))
    test_tf = vstack(list(test_df.bow))

    transformer = TfidfTransformer(sublinear_tf=True)
    train_tfidf = transformer.fit_transform(train_tf)
    val_tfidf = transformer.transform(val_tf)
    test_tfidf = transformer.transform(test_tf)

    def create_tfidf_df(source_df, tfidf_data):
        return pd.DataFrame({
            'doc_id': list(source_df.index),
            'bow': [bow for bow in tfidf_data],
            'label': list(source_df.label)
        }).set_index('doc_id')

    return (
        create_tfidf_df(train_df, train_tfidf),
        create_tfidf_df(val_df, val_tfidf),
        create_tfidf_df(test_df, test_tfidf),
    )


def convert_to_binary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """将 TF 格式转换为 Binary 格式"""
    print("转换为 Binary 格式...")

    def create_bin_matrix(df):
        doc_bin = []
        for _, row in df.iterrows():
            bow = (row.bow.toarray().squeeze() > 0).astype(np.float64)
            doc_bin.append(csr_matrix(bow))
        return vstack(doc_bin)

    train_bin = create_bin_matrix(train_df)
    val_bin = create_bin_matrix(val_df)
    test_bin = create_bin_matrix(test_df)

    return (
        create_dataframe(train_bin, list(train_df.label)),
        create_dataframe(val_bin, list(val_df.label)),
        create_dataframe(test_bin, list(test_df.label)),
    )


def save_dataset(
    save_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    format_name: str,
    vocab: Optional[dict] = None,
):
    """保存数据集"""
    save_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_pickle(save_dir / f'train.{format_name}.df.pkl')
    val_df.to_pickle(save_dir / f'cv.{format_name}.df.pkl')
    test_df.to_pickle(save_dir / f'test.{format_name}.df.pkl')

    if vocab is not None:
        with open(save_dir / 'vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"数据已保存到 {save_dir}/")


# ============================================================================
# 主处理函数
# ============================================================================

def preprocess_single_label_dataset(
    dataset_name: str,
    vocab_size: int = 10000,
    max_df: float = 0.8,
    min_df: int = 3,
    min_doc_length: int = 10,
    max_doc_length: int = 500,
):
    """处理单标签数据集"""
    data_dir = TEXTDATA_DIR / dataset_name

    # 加载数据
    if dataset_name == 'ng20':
        train_docs, train_tags, test_docs, test_tags = load_ng20()
    elif dataset_name == 'agnews':
        train_docs, train_tags, test_docs, test_tags = load_agnews(data_dir)
    elif dataset_name == 'dbpedia':
        train_docs, train_tags, test_docs, test_tags = load_dbpedia(data_dir)
    elif dataset_name == 'yahooanswer':
        train_docs, train_tags, test_docs, test_tags = load_yahooanswer(data_dir)
    else:
        raise ValueError(f"不支持的单标签数据集: {dataset_name}")

    print(f"原始数据: train={len(train_docs)}, test={len(test_docs)}")

    # 合并数据用于统一词表
    all_docs = list(train_docs) + list(test_docs)
    all_tags = np.concatenate([train_tags, test_tags])

    # 文本向量化
    count_vect = CountVectorizer(
        stop_words='english',
        max_features=vocab_size,
        max_df=max_df,
        min_df=min_df
    )
    all_tf = count_vect.fit_transform(all_docs)

    # 创建 DataFrame
    df = create_dataframe(all_tf, all_tags)

    # 过滤文档
    df = filter_documents(
        df,
        min_length=min_doc_length,
        max_length=max_doc_length,
        is_multi_label=False,
    )

    # 划分数据集
    train_df, val_df, test_df = split_dataset(df)

    # 保存 TF 格式
    save_dataset(data_dir, train_df, val_df, test_df, 'tf', count_vect.vocabulary_)

    # 转换并保存 TF-IDF 格式
    train_tfidf, val_tfidf, test_tfidf = convert_to_tfidf(train_df, val_df, test_df)
    save_dataset(data_dir, train_tfidf, val_tfidf, test_tfidf, 'tfidf')

    # 转换并保存 Binary 格式
    train_bin, val_bin, test_bin = convert_to_binary(train_df, val_df, test_df)
    save_dataset(data_dir, train_bin, val_bin, test_bin, 'bin')

    print("处理完成!")


def preprocess_multi_label_dataset(
    dataset_name: str,
    vocab_size: int = 10000,
    max_df: float = 0.8,
    min_df: int = 3,
    min_doc_length: int = 10,
    max_doc_length: int = 500,
):
    """处理多标签数据集"""
    data_dir = TEXTDATA_DIR / dataset_name

    # 加载数据
    if dataset_name == 'reuters':
        train_docs, train_tags, test_docs, test_tags = load_reuters()
    elif dataset_name == 'tmc':
        train_docs, train_tags, test_docs, test_tags = load_tmc(data_dir)
    else:
        raise ValueError(f"不支持的多标签数据集: {dataset_name}")

    print(f"原始数据: train={len(train_docs)}, test={len(test_docs)}")

    # 文本向量化
    count_vect = CountVectorizer(
        stop_words='english',
        max_features=vocab_size,
        max_df=max_df,
        min_df=min_df
    )
    train_tf = count_vect.fit_transform(train_docs)
    test_tf = count_vect.transform(test_docs)

    # 创建 DataFrame
    train_df = create_dataframe(train_tf, [t for t in train_tags])
    test_df = create_dataframe(test_tf, [t for t in test_tags])

    # 合并后过滤
    df = pd.concat([train_df, test_df], axis=0)
    df = filter_documents(
        df,
        min_length=min_doc_length,
        max_length=max_doc_length,
        remove_empty_labels=True,
        is_multi_label=True,
    )

    # 划分数据集
    train_df, val_df, test_df = split_dataset(df)

    # 保存 TF 格式
    save_dataset(data_dir, train_df, val_df, test_df, 'tf', count_vect.vocabulary_)

    # 转换并保存 TF-IDF 格式
    train_tfidf, val_tfidf, test_tfidf = convert_to_tfidf(train_df, val_df, test_df)
    save_dataset(data_dir, train_tfidf, val_tfidf, test_tfidf, 'tfidf')

    # 转换并保存 Binary 格式
    train_bin, val_bin, test_bin = convert_to_binary(train_df, val_df, test_df)
    save_dataset(data_dir, train_bin, val_bin, test_bin, 'bin')

    print("处理完成!")


def preprocess_rcv1(
    num_labels: int = 40,
    num_vocabs: int = 15000,
    num_train: int = 100000,
    num_test: int = 20000,
    min_doc_length: int = 5,
    max_doc_length: int = 500,
):
    """处理 RCV1 数据集 (特殊处理，因为 sklearn 已提供 TF-IDF)"""
    data_dir = TEXTDATA_DIR / 'rcv1'
    data_dir.mkdir(parents=True, exist_ok=True)

    documents, targets, sample_ids = load_rcv1(num_labels, num_vocabs)

    # 创建 DataFrame
    df = pd.DataFrame({
        'doc_id': sample_ids.tolist(),
        'bow': [d for d in documents],
        'label': [t for t in targets]
    })
    df.set_index('doc_id', inplace=True)

    # 过滤
    def count_num_tags(target):
        return target.sum()

    def get_num_word(bow):
        return bow.count_nonzero()

    df = df[df.label.apply(count_num_tags) > 0]
    df = df[df.bow.apply(get_num_word) > 0]

    if min_doc_length > 0:
        df = df[df.bow.apply(get_num_word) > min_doc_length]
    if max_doc_length > 0:
        df = df[df.bow.apply(get_num_word) <= max_doc_length]

    print(f"过滤后: {len(df)} 条")

    # 随机打乱并采样
    df = df.reindex(np.random.permutation(df.index))
    sampled_df = df.sample(min(num_train + num_test, len(df)))

    train_df = sampled_df.iloc[:num_train]
    temp_df = sampled_df.iloc[num_train:]
    val_df = temp_df[:len(temp_df) // 2]
    test_df = temp_df[len(temp_df) // 2:]

    print(f"数据集划分: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # RCV1 已经是 TF-IDF 格式
    save_dataset(data_dir, train_df, val_df, test_df, 'tfidf')

    print("处理完成!")


def main():
    parser = argparse.ArgumentParser(description='数据预处理工具')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='数据集名称: ng20, agnews, dbpedia, yahooanswer, reuters, tmc, rcv1')
    parser.add_argument('-v', '--vocab_size', type=int, default=10000,
                        help='词表大小 (default: 10000)')
    parser.add_argument('--max_df', type=float, default=0.8,
                        help='词频上限 (default: 0.8)')
    parser.add_argument('--min_df', type=int, default=3,
                        help='词频下限 (default: 3)')
    parser.add_argument('--min_doc_length', type=int, default=10,
                        help='最小文档长度 (default: 10)')
    parser.add_argument('--max_doc_length', type=int, default=500,
                        help='最大文档长度 (default: 500)')

    args = parser.parse_args()

    if args.dataset in SINGLE_LABEL_DATASETS:
        preprocess_single_label_dataset(
            args.dataset,
            vocab_size=args.vocab_size,
            max_df=args.max_df,
            min_df=args.min_df,
            min_doc_length=args.min_doc_length,
            max_doc_length=args.max_doc_length,
        )
    elif args.dataset in MULTI_LABEL_DATASETS:
        if args.dataset == 'rcv1':
            preprocess_rcv1()
        else:
            preprocess_multi_label_dataset(
                args.dataset,
                vocab_size=args.vocab_size,
                max_df=args.max_df,
                min_df=args.min_df,
                min_doc_length=args.min_doc_length,
                max_doc_length=args.max_doc_length,
            )
    else:
        print(f"不支持的数据集: {args.dataset}")
        print(f"支持的单标签数据集: {SINGLE_LABEL_DATASETS}")
        print(f"支持的多标签数据集: {MULTI_LABEL_DATASETS}")


if __name__ == '__main__':
    main()
