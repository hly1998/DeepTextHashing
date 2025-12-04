# 数据模块 (textdata)

本模块提供了深度文本哈希研究中常用数据集的下载、预处理和加载功能。

## 支持的数据集

| 数据集 | 类型 | 样本数 | 类别数 | 下载方式 |
|--------|------|--------|--------|----------|
| 20Newsgroups (ng20) | 单标签 | 18,846 | 20 | 自动 (sklearn) |
| AG News (agnews) | 单标签 | 127,600 | 4 | 手动 |
| Reuters | 多标签 | 10,788 | 20 | 自动 (nltk) |
| DBpedia | 单标签 | 60,000 | 14 | 手动 |
| RCV1 | 多标签 | 804,414 | 103 | 自动 (sklearn) |
| TMC | 多标签 | 28,596 | 22 | 已包含 |
| Yahoo Answer | 单标签 | 1,460,000 | 10 | 手动 |

## 快速开始

### 1. 下载数据集

```bash
# 查看所有支持的数据集
python textdata/download.py --list

# 下载可自动获取的数据集
python textdata/download.py -d ng20      # 20 Newsgroups
python textdata/download.py -d reuters   # Reuters
python textdata/download.py -d rcv1      # RCV1

# 下载所有可自动下载的数据集
python textdata/download.py --all
```

对于需要手动下载的数据集（agnews, dbpedia, yahooanswer），请从对应链接下载后放置到 `textdata/{dataset_name}/` 目录。

### 2. 预处理数据

```bash
# 预处理单标签数据集
python textdata/preprocess.py -d ng20 -v 10000

# 预处理多标签数据集  
python textdata/preprocess.py -d tmc -v 10000

# 自定义参数
python textdata/preprocess.py -d ng20 \
    --vocab_size 10000 \
    --max_df 0.8 \
    --min_df 3 \
    --min_doc_length 10 \
    --max_doc_length 500
```

预处理后会在数据集目录下生成以下文件：
- `train.tf.df.pkl` / `cv.tf.df.pkl` / `test.tf.df.pkl` - 词频 (TF) 格式
- `train.tfidf.df.pkl` / `cv.tfidf.df.pkl` / `test.tfidf.df.pkl` - TF-IDF 格式
- `train.bin.df.pkl` / `cv.bin.df.pkl` / `test.bin.df.pkl` - 二值格式
- `vocab.pkl` - 词表

### 3. 在代码中使用

```python
from textdata import SingleLabelTextDataset, MultiLabelTextDataset
from textdata.dataset import get_dataset, get_data_loader

# 方式1: 直接使用 Dataset 类
train_data = SingleLabelTextDataset('textdata/ng20', subset='train', bow_format='tfidf')
test_data = SingleLabelTextDataset('textdata/ng20', subset='test', bow_format='tfidf')

# 方式2: 使用便捷函数
train_data = get_dataset('ng20', subset='train', bow_format='tfidf')

# 方式3: 直接获取 DataLoader
train_loader = get_data_loader('ng20', subset='train', batch_size=64, shuffle=True)

# 获取数据
for batch_bow, batch_labels in train_loader:
    # batch_bow: [batch_size, vocab_size]
    # batch_labels: [batch_size] (单标签) 或 [batch_size, num_classes] (多标签)
    pass

# 获取数据集信息
print(f"样本数: {len(train_data)}")
print(f"特征维度: {train_data.num_features()}")
print(f"类别数: {train_data.num_classes()}")
```

## 数据格式

### 词袋表示格式

- **tf (Term Frequency)**: 词频统计
- **tfidf**: TF-IDF 加权
- **bin (Binary)**: 二值表示 (词是否出现)

### DataFrame 结构

每个 `.df.pkl` 文件包含一个 pandas DataFrame，结构如下：

| 列名 | 类型 | 说明 |
|------|------|------|
| doc_id (index) | int | 文档ID |
| bow | scipy.sparse.csr_matrix | 词袋向量 |
| label | int / scipy.sparse.csr_matrix | 标签 (单标签为int，多标签为稀疏矩阵) |

## 目录结构

```
textdata/
├── __init__.py          # 模块入口
├── download.py          # 数据下载脚本
├── preprocess.py        # 数据预处理脚本
├── dataset.py           # PyTorch Dataset 类
├── README.md            # 本文档
└── tmc/                 # TMC 数据集 (已包含)
    ├── TrainingData.txt
    ├── TestData.txt
    ├── TrainCategoryMatrix.csv
    ├── TestTruth.csv
    ├── train.tf.df.pkl
    ├── train.tfidf.df.pkl
    ├── train.bin.df.pkl
    ├── cv.tf.df.pkl
    ├── cv.tfidf.df.pkl
    ├── cv.bin.df.pkl
    ├── test.tf.df.pkl
    ├── test.tfidf.df.pkl
    ├── test.bin.df.pkl
    └── vocab.pkl
```

## 数据集详细信息

### 20 Newsgroups (ng20)
- **来源**: sklearn.datasets
- **描述**: 新闻组文档分类数据集
- **链接**: https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html

### AG News (agnews)
- **来源**: 手动下载
- **描述**: 新闻分类数据集
- **链接**: http://groups.di.unipi.it/gulli/AG_corpus_of_news_articles.html
- **文件**: train.csv, test.csv

### Reuters
- **来源**: nltk.corpus
- **描述**: 路透社新闻多标签分类数据集
- **链接**: https://www.nltk.org/book/ch02.html

### DBpedia
- **来源**: 手动下载
- **描述**: DBpedia 本体分类数据集
- **链接**: https://www.csie.ntu.edu.tw/cjlin/libsvmtools/datasets/multilabel.html
- **文件**: train.csv, test.csv

### RCV1
- **来源**: sklearn.datasets
- **描述**: 路透社新闻语料库
- **链接**: https://catalog.data.gov/dataset/siam-2007-text-mining-competition-dataset

### TMC (SIAM 2007)
- **来源**: 已包含在仓库中
- **描述**: 航空安全报告多标签分类数据集
- **链接**: https://catalog.data.gov/dataset/siam-2007-text-mining-competition-dataset

### Yahoo Answer
- **来源**: 手动下载
- **描述**: 问答分类数据集
- **链接**: https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset
- **文件**: train.csv, test.csv
