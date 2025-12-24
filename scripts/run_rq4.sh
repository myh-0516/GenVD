#!/bin/bash

ROOT="../.."
MODEL="$1"
DATASET="top10cwe"

python $ROOT/src/RQ4/run.py \
    --dataset="$DATASET" \
    --data_dir="$ROOT/datasets" \
    --output_dir="$ROOT/results/RQ4_test/${MODEL}/generate/${DATASET}" \
    --pretrainedmodel_path="$ROOT/pretrained_models/$MODEL-base" \
    --seed=42 \
    --batch_size=32 \
    --max_seq_length=512 \
    --learning_rate=6e-5 \
    --num_epochs=20 \
    --early_stop_threshold=5 \
    --weight_decay=0.01 \
    --max_code_words=512 \
    --do_train \
    --do_eval \
    --do_test \
    --fp16