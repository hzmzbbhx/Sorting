#!/usr/bin/env bash

device=0

# 配置参数
depth=(9)
n_ctx=(12)
t_n_ctx=(4)

for i in "${!depth[@]}"; do
    for j in "${!n_ctx[@]}"; do
        # mvtec 测试
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset mvtec \
            --data_path /remote-home/iot_zhouqihang/data/mvdataset --save_path ./results/${base_dir}/zero_shot \
            --checkpoint_path ${save_dir}epoch_15.pth \
            --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
        wait
    done
done

# SDD 测试
for i in "${!depth[@]}"; do
    for j in "${!n_ctx[@]}"; do
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset SDD \
            --data_path "/mnt/d/test/AnomalyCLIP/SDD" \
            --save_path ./results/${base_dir}/zero_shot \
            --checkpoint_path ${save_dir}epoch_15.pth \
            --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
        wait
    done
done