"""
数据集下载脚本
支持的数据集:
- 20Newsgroups (ng20): 通过 sklearn 自动下载
- Agnews: 需手动下载
- Reuters: 通过 nltk 自动下载
- DBpedia: 需手动下载
- RCV1: 通过 sklearn 自动下载
- TMC: 需手动下载 (已包含在本仓库)
- Yahooanswer: 需手动下载
"""

import argparse
from pathlib import Path

# 数据集信息
DATASET_INFO = {
    'ng20': {
        'name': '20 Newsgroups',
        'type': 'single-label',
        'instances': 18846,
        'categories': 20,
        'download': 'auto',  # sklearn 自动下载
        'link': 'https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html',
    },
    'agnews': {
        'name': 'AG News',
        'type': 'single-label',
        'instances': 127600,
        'categories': 4,
        'download': 'manual',
        'link': 'http://groups.di.unipi.it/gulli/AG_corpus_of_news_articles.html',
        'files': ['train.csv', 'test.csv'],
    },
    'reuters': {
        'name': 'Reuters',
        'type': 'multi-label',
        'instances': 10788,
        'categories': 20,
        'download': 'auto',  # nltk 自动下载
        'link': 'https://www.nltk.org/book/ch02.html',
    },
    'dbpedia': {
        'name': 'DBpedia',
        'type': 'single-label',
        'instances': 60000,
        'categories': 14,
        'download': 'manual',
        'link': 'https://www.csie.ntu.edu.tw/cjlin/libsvmtools/datasets/multilabel.html',
        'files': ['train.csv', 'test.csv'],
    },
    'rcv1': {
        'name': 'RCV1',
        'type': 'multi-label',
        'instances': 804414,
        'categories': 103,
        'download': 'auto',  # sklearn 自动下载
        'link': 'https://catalog.data.gov/dataset/siam-2007-text-mining-competition-dataset',
    },
    'tmc': {
        'name': 'TMC',
        'type': 'multi-label',
        'instances': 28596,
        'categories': 22,
        'download': 'included',  # 已包含在仓库中
        'link': 'https://catalog.data.gov/dataset/siam-2007-text-mining-competition-dataset',
        'files': ['TrainingData.txt', 'TestData.txt', 'TrainCategoryMatrix.csv', 'TestTruth.csv'],
    },
    'yahooanswer': {
        'name': 'Yahoo Answer',
        'type': 'single-label',
        'instances': 1460000,
        'categories': 10,
        'download': 'manual',
        'link': 'https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset',
        'files': ['train.csv', 'test.csv'],
    },
}


def get_data_dir(dataset_name: str) -> Path:
    """获取数据集目录路径"""
    base_dir = Path(__file__).parent
    return base_dir / dataset_name


def download_ng20(data_dir: Path):
    """下载 20 Newsgroups 数据集"""
    from sklearn.datasets import fetch_20newsgroups

    print("正在下载 20 Newsgroups 数据集...")
    # sklearn 会自动缓存数据
    train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    print(f"下载完成! 训练集: {len(train.data)} 条, 测试集: {len(test.data)} 条")
    return train, test


def download_reuters(data_dir: Path):
    """下载 Reuters 数据集"""
    import nltk

    print("正在下载 Reuters 数据集...")
    try:
        nltk.download('reuters', quiet=True)
        nltk.download('punkt', quiet=True)
        from nltk.corpus import reuters
        print(f"下载完成! 共 {len(reuters.fileids())} 个文档")
        return reuters
    except Exception as e:
        print(f"下载失败: {e}")
        return None


def download_rcv1(data_dir: Path):
    """下载 RCV1 数据集"""
    from sklearn.datasets import fetch_rcv1

    print("正在下载 RCV1 数据集 (这可能需要一些时间)...")
    rcv1 = fetch_rcv1()
    print(f"下载完成! 共 {rcv1.data.shape[0]} 个文档, {rcv1.data.shape[1]} 个特征")
    return rcv1


def check_manual_dataset(dataset_name: str, data_dir: Path) -> bool:
    """检查手动下载的数据集是否存在"""
    info = DATASET_INFO[dataset_name]
    if 'files' not in info:
        return False

    missing_files = []
    for f in info['files']:
        if not (data_dir / f).exists():
            missing_files.append(f)

    if missing_files:
        print(f"缺少文件: {missing_files}")
        print(f"请从 {info['link']} 下载并放置到 {data_dir}/")
        return False
    return True


def download_dataset(dataset_name: str, force: bool = False):
    """下载指定数据集"""
    if dataset_name not in DATASET_INFO:
        print(f"不支持的数据集: {dataset_name}")
        print(f"支持的数据集: {list(DATASET_INFO.keys())}")
        return False

    info = DATASET_INFO[dataset_name]
    data_dir = get_data_dir(dataset_name)

    # 创建数据目录
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"数据集: {info['name']}")
    print(f"类型: {info['type']}")
    print(f"样本数: {info['instances']}")
    print(f"类别数: {info['categories']}")
    print(f"{'='*60}\n")

    if info['download'] == 'auto':
        if dataset_name == 'ng20':
            download_ng20(data_dir)
        elif dataset_name == 'reuters':
            download_reuters(data_dir)
        elif dataset_name == 'rcv1':
            download_rcv1(data_dir)
        return True

    elif info['download'] == 'included':
        if check_manual_dataset(dataset_name, data_dir):
            print(f"数据集 {dataset_name} 已存在")
            return True
        return False

    elif info['download'] == 'manual':
        print(f"数据集 {dataset_name} 需要手动下载")
        print(f"下载地址: {info['link']}")
        print(f"下载后请将文件放置到: {data_dir}/")
        if 'files' in info:
            print(f"需要的文件: {info['files']}")
        return check_manual_dataset(dataset_name, data_dir)


def list_datasets():
    """列出所有支持的数据集"""
    print("\n支持的数据集:")
    print("-" * 80)
    print(f"{'名称':<15} {'类型':<15} {'样本数':<12} {'类别数':<10} {'下载方式':<10}")
    print("-" * 80)
    for name, info in DATASET_INFO.items():
        print(f"{name:<15} {info['type']:<15} {info['instances']:<12} {info['categories']:<10} {info['download']:<10}")
    print("-" * 80)
    print("\n下载方式说明:")
    print("  auto: 自动下载 (通过 sklearn 或 nltk)")
    print("  manual: 需要手动下载")
    print("  included: 已包含在仓库中")


def main():
    parser = argparse.ArgumentParser(description='数据集下载工具')
    parser.add_argument('-d', '--dataset', type=str, help='要下载的数据集名称')
    parser.add_argument('-l', '--list', action='store_true', help='列出所有支持的数据集')
    parser.add_argument('-a', '--all', action='store_true', help='下载所有可自动下载的数据集')
    parser.add_argument('-f', '--force', action='store_true', help='强制重新下载')

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if args.all:
        for name, info in DATASET_INFO.items():
            if info['download'] == 'auto':
                download_dataset(name, force=args.force)
        return

    if args.dataset:
        download_dataset(args.dataset, force=args.force)
        return

    # 默认显示帮助
    parser.print_help()
    print("\n")
    list_datasets()


if __name__ == '__main__':
    main()
