#!/bin/bash

ROOT="../.."
MODEL="codebert"   # codebert, graphcodebert, unixcoder
DATASET="diversevul_cwe" # top10cwe diversevul_cwe

python $ROOT/src/RQ4/classify/run.py \
    --model_name_or_path="$ROOT/pretrained_models/${MODEL}-base" \
    --train_data_file="$ROOT/datasets/${DATASET}/train.jsonl" \
    --eval_data_file="$ROOT/datasets/${DATASET}/valid.jsonl" \
    --test_data_file="$ROOT/datasets/${DATASET}/test.jsonl" \
    --output_dir="$ROOT/results/RQ4/${MODEL}/classify/${DATASET}" \
    --do_train \
    --do_test \
    --epoch 20 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --dropout_probability 0.1 \
    --early_stopping_patience 3 \
    --seed 42 \
    --fp16