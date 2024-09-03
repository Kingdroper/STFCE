#!/bin/bash

### 该作业的作业名
#SBATCH --job-name=run_test

### 任务输出打印至test.out，默认是slurm-任务编号.out
#SBATCH --output train.out

### 该作业需要1个节点
#SBATCH --nodes=1

### 指定rtx分区的gpu20，申请2块卡
#SBATCH -p titan --gres=gpu:titan:1
### 作业最大的运行时间，超过时间后作业资源会被SLURM回收
#SBATCH --time=100:00:00

### 运行
CUDA_VISIBLE_DEVICES=0,1 python feature_extractor/extract_tfrecords_main.py  --input_videos_csv vids_1027/train_data.csv \
    --output_tfrecords_file train.tfrecord