#!/bin/bash

DATASET="$1"
ROOT="../.."

python $ROOT/src/RQ2/graphcodebert/run.py \
    --output_dir=$ROOT/results/RQ2/graphcodebert/${DATASET} \
    --model_type=roberta \
    --tokenizer_name=$ROOT/pretrained_models/graphcodebert-base \
    --model_name_or_path=$ROOT/pretrained_models/graphcodebert-base \
    --do_train \
    --do_test \
    --do_eval \
    --train_data_file=$ROOT/datasets/${DATASET}/train.jsonl \
    --eval_data_file=$ROOT/datasets/${DATASET}/valid.jsonl \
    --test_data_file=$ROOT/datasets/${DATASET}/test.jsonl \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --fp16 \
    --seed 42