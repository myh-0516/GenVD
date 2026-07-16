#!/bin/bash

MODEL="$1"
DATASET="$2" 
ARCHITECTURE="$3" # encoder encoder-decoder decoder
ROOT="../.."

python $ROOT/src/RQ3/run.py \
    --dataset="$DATASET" \
    --data_dir="$ROOT/datasets" \
    --output_dir="$ROOT/results/RQ3_test/$MODEL/$DATASET" \
    --pretrainedmodel_path="$ROOT/pretrained_models/$MODEL" \
    --architecture="$ARCHITECTURE" \
    --seed=42 \
    --batch_size=32 \
    --max_seq_length=512 \
    --learning_rate=6e-5 \
    --num_epochs=15 \
    --early_stop_threshold=4 \
    --weight_decay=0.01 \
    --do_train \
    --do_eval \
    --do_test \
    --fp16 \
    --gradient_checkpointing