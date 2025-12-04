# textdata module for Deep Text Hashing
# 提供数据下载、预处理和 PyTorch Dataset 类

from .dataset import (
    SingleLabelTextDataset,
    MultiLabelTextDataset,
    SingleLabelTextDatasetDocID,
    MultiLabelTextDatasetDocID,
    get_dataset,
    get_data_loader,
)

__all__ = [
    'SingleLabelTextDataset',
    'MultiLabelTextDataset',
    'SingleLabelTextDatasetDocID',
    'MultiLabelTextDatasetDocID',
    'get_dataset',
    'get_data_loader',
]
