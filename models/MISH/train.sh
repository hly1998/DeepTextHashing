#!/bin/bash

# MISH 训练脚本
# 注意：bit 需要能被 block_size 整除

# 单标签数据集 (32 bits, 4 blocks of 8 bits each)
python MISH.py --dataset ng20.tfidf --bit 32 --block_size 8 --memory_size 5000 --problem_pair_weight 1.0
# python MISH.py --dataset ng20.tfidf --bit 64 --block_size 8 --memory_size 5000 --problem_pair_weight 1.0
# python MISH.py --dataset ng20.tfidf --bit 64 --block_size 16 --memory_size 5000 --problem_pair_weight 1.0

# 多标签数据集
# python MISH.py --dataset reuters.tfidf --bit 32 --block_size 8 --memory_size 5000
# python MISH.py --dataset reuters.tfidf --bit 64 --block_size 8 --memory_size 5000
# python MISH.py --dataset tmc.tfidf --bit 32 --block_size 8 --memory_size 5000
# python MISH.py --dataset tmc.tfidf --bit 64 --block_size 8 --memory_size 5000
