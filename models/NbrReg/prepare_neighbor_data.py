"""
预处理脚本: 生成文档邻居索引
使用余弦相似度计算文档间的相似性，并保存Top-K邻居索引

用法:
    python prepare_neighbor_data.py --dataset ng20 --use_train
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# 获取当前文件所在目录
CURRENT_DIR = Path(__file__).parent
NEIGHBOR_DATA_DIR = CURRENT_DIR / 'neighbor_data'

# 创建目录
NEIGHBOR_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_argparser():
    parser = argparse.ArgumentParser(description='生成文档邻居索引')
    parser.add_argument('--dataset', type=str, default='ng20',
                        help='数据集名称 (ng20, agnews, dbpedia, reuters, tmc, rcv1)')
    parser.add_argument('--use_train', action='store_true', default=True,
                        help='使用训练集作为查询语料')
    parser.add_argument('--top_k', type=int, default=101,
                        help='保存的邻居数量')
    parser.add_argument('--query_batch_size', type=int, default=500,
                        help='查询批大小')
    parser.add_argument('--doc_batch_size', type=int, default=100,
                        help='文档批大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备: cuda 或 cpu')
    parser.add_argument('--gpu', type=str, default='0',
                        help='指定 GPU 设备号')
    return parser


def get_device(config):
    """获取运行设备"""
    if config['device'] == 'cuda' and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
        device = torch.device('cuda')
        print(f"使用 GPU: {config['gpu']}")
    else:
        device = torch.device('cpu')
        print("使用 CPU")
    return device


def load_dataset(dataset_name):
    """
    加载数据集

    Args:
        dataset_name: 数据集名称

    Returns:
        包含训练集、测试集和验证集的数据对象
    """
    data_path = Path(__file__).parent.parent.parent / 'textdata'

    train_df = pd.read_pickle(f"{data_path}/{dataset_name}/train.tfidf.df.pkl")
    test_df = pd.read_pickle(f"{data_path}/{dataset_name}/test.tfidf.df.pkl")
    cv_df = pd.read_pickle(f"{data_path}/{dataset_name}/cv.tfidf.df.pkl")

    # 转换为numpy数组
    train = np.array([csr_data.toarray()[0] for csr_data in train_df.bow.tolist()])
    test = np.array([csr_data.toarray()[0] for csr_data in test_df.bow.tolist()])
    cv = np.array([csr_data.toarray()[0] for csr_data in cv_df.bow.tolist()])

    # 过滤空文档
    train_len = np.sum(train > 0, axis=1)
    nz_indices = np.where(train_len > 0)[0]
    train = train[nz_indices, :]

    class DataContainer:
        pass

    data = DataContainer()
    data.train = train
    data.test = test
    data.cv = cv
    data.n_features = train.shape[1]

    return data


def compute_topk_cosine_similarity(
    out_fn,
    queries,
    documents,
    top_k,
    device,
    query_batch_size=500,
    doc_batch_size=100
):
    """
    使用余弦相似度计算Top-K邻居并保存到文件

    Args:
        out_fn: 输出文件路径
        queries: 查询文档矩阵
        documents: 候选文档矩阵
        top_k: 返回的邻居数量
        device: 计算设备
        query_batch_size: 查询批大小
        doc_batch_size: 文档批大小
    """
    n_docs = documents.shape[0]
    n_queries = queries.shape[0]
    query_row = 0

    with open(out_fn, 'w') as out_file:
        for q_idx in tqdm(range(0, n_queries, query_batch_size), desc='处理查询', ncols=0):
            query_batch_s_idx = q_idx
            query_batch_e_idx = min(query_batch_s_idx + query_batch_size, n_queries)

            # 加载查询批次
            query_mats = torch.FloatTensor(queries[query_batch_s_idx:query_batch_e_idx]).to(device)
            query_norm = torch.norm(query_mats, 2, dim=1, keepdim=True)
            query_mats_3d = query_mats.unsqueeze(2)

            score_list = []
            indices_list = []

            for idx in tqdm(range(0, n_docs, doc_batch_size), desc='文档', leave=False, ncols=0):
                batch_s_idx = idx
                batch_e_idx = min(batch_s_idx + doc_batch_size, n_docs)
                n_doc_in_batch = batch_e_idx - batch_s_idx

                # 加载候选文档批次
                candidate_mats = torch.FloatTensor(documents[batch_s_idx:batch_e_idx]).to(device)
                candidate_norm = torch.norm(candidate_mats, 2, dim=1, keepdim=True)

                # 准备用于批量矩阵乘法
                candidate_mats_3d = candidate_mats.unsqueeze(2).permute(2, 1, 0)

                # 扩展维度以进行批量计算
                query_expanded = query_mats_3d.expand(
                    query_mats_3d.size(0),
                    query_mats_3d.size(1),
                    candidate_mats_3d.size(2)
                )
                candidate_expanded = candidate_mats_3d.expand_as(query_expanded)

                # 计算余弦相似度
                cos_sim = torch.sum(query_expanded * candidate_expanded, dim=1) / (
                    query_norm * candidate_norm.T + 1e-8
                )

                # 获取当前批次的Top-K
                k = min(top_k, n_doc_in_batch)
                scores, indices = torch.topk(cos_sim, k, dim=1, largest=True)

                del cos_sim
                del query_expanded
                del candidate_expanded
                del candidate_norm

                score_list.append(scores)
                indices_list.append(indices + batch_s_idx)

            # 合并所有批次的结果
            all_scores = torch.cat(score_list, dim=1)
            all_indices = torch.cat(indices_list, dim=1)
            _, sort_indices = torch.topk(all_scores, top_k, dim=1, largest=True)

            topk_indices = torch.gather(all_indices, 1, sort_indices)

            del query_mats
            del query_norm
            del score_list
            del indices_list

            # 写入文件
            topk_indices = topk_indices.cpu().numpy()
            for row in topk_indices:
                out_file.write(f"{query_row}:")
                out_text = ','.join([str(col) for col in row])
                out_file.write(out_text)
                out_file.write('\n')
                query_row += 1

            torch.cuda.empty_cache()


def main():
    argparser = get_argparser()
    args = argparser.parse_args()
    config = vars(args)

    device = get_device(config)

    dataset = config["dataset"]
    use_train = config["use_train"]
    top_k = config["top_k"]

    print(f"加载数据集: {dataset}")
    data = load_dataset(dataset)
    print(f"训练集大小: {data.train.shape}")
    print(f"特征维度: {data.n_features}")

    if use_train:
        print("使用训练集作为查询语料")
        query_corpus = data.train
        out_fn = NEIGHBOR_DATA_DIR / f"{dataset}_train_top{top_k}.txt"
    else:
        print("使用测试集作为查询语料")
        query_corpus = data.test
        out_fn = NEIGHBOR_DATA_DIR / f"{dataset}_test_top{top_k}.txt"

    print(f"结果保存到: {out_fn}")

    compute_topk_cosine_similarity(
        str(out_fn),
        query_corpus,
        data.train,
        top_k,
        device,
        query_batch_size=config["query_batch_size"],
        doc_batch_size=config["doc_batch_size"]
    )

    print("完成!")


if __name__ == "__main__":
    main()
