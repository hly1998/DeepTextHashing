# 参数选项
lsc_weights=(1 5 10)
bb_weights=(0.01 0.005 0.001)
bd_weights=(0.01 0.005 0.001)

# 生成组合并写入.sh文件
for lsc_weight in "${lsc_weights[@]}"; do
    for bb_weight in "${bb_weights[@]}"; do
        for bd_weight in "${bd_weights[@]}"; do
            echo "python SMASH.py --lsc_weight $lsc_weight --bb_weight $bb_weight --bd_weight $bd_weight"
        done
    done
done