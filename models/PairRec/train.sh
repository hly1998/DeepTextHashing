#!/bin/bash

# PairRec 训练脚本

# 单标签数据集
python PairRec.py --dataset ng20.tfidf --bit 8
python PairRec.py --dataset ng20.tfidf --bit 16
python PairRec.py --dataset ng20.tfidf --bit 32
python PairRec.py --dataset ng20.tfidf --bit 64

# 多标签数据集
# python PairRec.py --dataset reuters.tfidf --bit 16
# python PairRec.py --dataset reuters.tfidf --bit 32
# python PairRec.py --dataset reuters.tfidf --bit 64
# python PairRec.py --dataset tmc.tfidf --bit 16
# python PairRec.py --dataset tmc.tfidf --bit 32
# python PairRec.py --dataset tmc.tfidf --bit 64

