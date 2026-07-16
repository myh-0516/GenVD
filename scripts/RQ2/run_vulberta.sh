#!/bin/bash

DATASET="$1"
ROOT="../.."

python $ROOT/src/RQ2/vulberta/run.py \
    --mode train \
    --dataset=$DATASET \
    --output_dir=$ROOT/results/RQ2/vulberta/${DATASET} \
    --batch_size=32 \
    --epochs=3 \
    --learning_rate=3e-5 \
    --seed=42 \
    --fp16

python $ROOT/src/RQ2/vulberta/run.py \
    --mode evaluate \
    --dataset=$DATASET \
    --output_dir=$ROOT/results/RQ2/vulberta/${DATASET} \
    --metrics_file=metrics.txt
