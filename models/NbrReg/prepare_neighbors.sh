#!/bin/bash

# 生成邻居索引脚本

# 单标签数据集
python prepare_neighbor_data.py --dataset ng20

# 多标签数据集
# python prepare_neighbor_data.py --dataset reuters
# python prepare_neighbor_data.py --dataset tmc

