"""
PyTorch Dataset 类
用于加载预处理后的文本哈希数据集
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SingleLabelTextDataset(Dataset):
    """
    单标签文本数据集
    适用于: 20Newsgroups, AGNews, DBpedia, YahooAnswer

    Args:
        data_dir: 数据目录路径
        subset: 数据子集 ('train', 'test', 'cv')
        bow_format: 词袋表示格式 ('tf', 'tfidf', 'bin')

    Example:
        >>> from textdata import SingleLabelTextDataset
        >>> dataset = SingleLabelTextDataset('datasets/ng20', subset='train', bow_format='tfidf')
        >>> bow, label = dataset[0]
        >>> print(bow.shape, label)
    """

    def __init__(
        self,
        data_dir: str,
        subset: str = 'train',
        bow_format: str = 'tf',
    ):
        self.data_dir = Path(data_dir)
        self.subset = subset
        self.bow_format = bow_format

        # 加载数据
        df_file = self.data_dir / f'{subset}.{bow_format}.df.pkl'
        if not df_file.exists():
            raise FileNotFoundError(
                f"数据文件不存在: {df_file}\n"
                f"请先运行 preprocess.py 进行数据预处理"
            )
        self.df = pd.read_pickle(df_file)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        doc_bow = torch.from_numpy(
            row.bow.toarray().squeeze().astype(np.float32)
        )
        label = row.label
        return doc_bow, label

    def num_classes(self) -> int:
        """获取类别数量"""
        return len(set(self.df.label))

    def num_features(self) -> int:
        """获取特征维度"""
        return self.df.bow.iloc[0].shape[1]


class MultiLabelTextDataset(SingleLabelTextDataset):
    """
    多标签文本数据集
    适用于: Reuters, TMC, RCV1

    Args:
        data_dir: 数据目录路径
        subset: 数据子集 ('train', 'test', 'cv')
        bow_format: 词袋表示格式 ('tf', 'tfidf', 'bin')

    Example:
        >>> from textdata import MultiLabelTextDataset
        >>> dataset = MultiLabelTextDataset('datasets/tmc', subset='train', bow_format='tfidf')
        >>> bow, labels = dataset[0]
        >>> print(bow.shape, labels.shape)
    """

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        doc_bow = torch.from_numpy(
            row.bow.toarray().squeeze().astype(np.float32)
        )
        label_bow = torch.from_numpy(
            row.label.toarray().squeeze().astype(np.float32)
        )
        return doc_bow, label_bow

    def num_classes(self) -> int:
        """获取标签数量"""
        return self.df.iloc[0].label.shape[1]


class SingleLabelTextDatasetDocID(Dataset):
    """
    带文档ID的单标签文本数据集
    适用于需要文档ID的场景（如邻居正则化）

    Args:
        data_dir: 数据目录路径
        subset: 数据子集 ('train', 'test', 'cv')
        bow_format: 词袋表示格式 ('tf', 'tfidf', 'bin')

    Returns:
        Tuple[doc_bow, doc_id, label]
    """

    def __init__(
        self,
        data_dir: str,
        subset: str = 'train',
        bow_format: str = 'tf',
    ):
        self.data_dir = Path(data_dir)
        self.subset = subset
        self.bow_format = bow_format

        df_file = self.data_dir / f'{subset}.{bow_format}.df.pkl'
        if not df_file.exists():
            raise FileNotFoundError(
                f"数据文件不存在: {df_file}\n"
                f"请先运行 preprocess.py 进行数据预处理"
            )
        self.df = pd.read_pickle(df_file)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        row = self.df.iloc[idx]
        doc_bow = torch.from_numpy(
            row.bow.toarray().squeeze().astype(np.float32)
        )
        label = row.label
        return doc_bow, idx, label

    def num_classes(self) -> int:
        return len(set(self.df.label))

    def num_features(self) -> int:
        return self.df.bow.iloc[0].shape[1]


class MultiLabelTextDatasetDocID(SingleLabelTextDatasetDocID):
    """
    带文档ID的多标签文本数据集
    适用于需要文档ID的场景（如邻居正则化）

    Args:
        data_dir: 数据目录路径
        subset: 数据子集 ('train', 'test', 'cv')
        bow_format: 词袋表示格式 ('tf', 'tfidf', 'bin')

    Returns:
        Tuple[doc_bow, doc_id, label_bow]
    """

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        row = self.df.iloc[idx]
        doc_bow = torch.from_numpy(
            row.bow.toarray().squeeze().astype(np.float32)
        )
        label_bow = torch.from_numpy(
            row.label.toarray().squeeze().astype(np.float32)
        )
        return doc_bow, idx, label_bow

    def num_classes(self) -> int:
        return self.df.iloc[0].label.shape[1]


def get_dataset(
    dataset_name: str,
    subset: str = 'train',
    bow_format: str = 'tfidf',
    with_doc_id: bool = False,
) -> Dataset:
    """
    便捷函数：根据数据集名称获取对应的 Dataset 对象

    Args:
        dataset_name: 数据集名称 (ng20, agnews, dbpedia, yahooanswer, reuters, tmc, rcv1)
        subset: 数据子集 ('train', 'test', 'cv')
        bow_format: 词袋表示格式 ('tf', 'tfidf', 'bin')
        with_doc_id: 是否返回文档ID

    Returns:
        Dataset 对象

    Example:
        >>> from textdata.dataset import get_dataset
        >>> train_data = get_dataset('ng20', subset='train', bow_format='tfidf')
        >>> test_data = get_dataset('ng20', subset='test', bow_format='tfidf')
    """
    # 数据集类型映射
    single_label_datasets = ['ng20', 'agnews', 'dbpedia', 'yahooanswer']
    multi_label_datasets = ['reuters', 'tmc', 'rcv1']

    # 获取数据目录
    datasets_dir = Path(__file__).parent
    data_dir = datasets_dir / dataset_name

    if not data_dir.exists():
        raise FileNotFoundError(
            f"数据集目录不存在: {data_dir}\n"
            f"请先运行 download.py 和 preprocess.py"
        )

    # 选择合适的 Dataset 类
    if dataset_name in single_label_datasets:
        if with_doc_id:
            return SingleLabelTextDatasetDocID(str(data_dir), subset, bow_format)
        return SingleLabelTextDataset(str(data_dir), subset, bow_format)
    elif dataset_name in multi_label_datasets:
        if with_doc_id:
            return MultiLabelTextDatasetDocID(str(data_dir), subset, bow_format)
        return MultiLabelTextDataset(str(data_dir), subset, bow_format)
    else:
        raise ValueError(
            f"不支持的数据集: {dataset_name}\n"
            f"支持的单标签数据集: {single_label_datasets}\n"
            f"支持的多标签数据集: {multi_label_datasets}"
        )


def get_data_loader(
    dataset_name: str,
    subset: str = 'train',
    bow_format: str = 'tfidf',
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    with_doc_id: bool = False,
):
    """
    便捷函数：获取 DataLoader

    Args:
        dataset_name: 数据集名称
        subset: 数据子集
        bow_format: 词袋表示格式
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        with_doc_id: 是否返回文档ID

    Returns:
        DataLoader 对象
    """
    from torch.utils.data import DataLoader

    dataset = get_dataset(dataset_name, subset, bow_format, with_doc_id)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
