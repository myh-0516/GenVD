#!/bin/bash
export OMP_NUM_THREADS=1

MODEL="${1:-unixcoder}"
ROOT="../.."

DATASETS=("devign" "reveal" "bigvul")

for DATASET in "${DATASETS[@]}"; do
    echo "========================================================="
    echo "Running RQ2: Dataset=$DATASET | Model=$MODEL"
    echo "========================================================="

    python $ROOT/src/RQ1/run.py \
        --train_data_file="$ROOT/datasets/$DATASET/train.jsonl" \
        --eval_data_file="$ROOT/datasets/$DATASET/valid.jsonl" \
        --test_data_file="$ROOT/datasets/$DATASET/test.jsonl" \
        --output_dir="$ROOT/results/RQ2/$MODEL/generate/$DATASET/normal" \
        --pretrainedmodel_path="$ROOT/pretrained_models/$MODEL-base" \
        --model_name="$MODEL" \
        --seed=42 \
        --batch_size=32 \
        --max_seq_length=512 \
        --learning_rate=6e-5 \
        --num_epochs=20 \
        --early_stop_threshold=3 \
        --weight_decay=0.01 \
        --do_train \
        --do_eval \
        --do_test \
        --fp16 \
        --imbalance_mode="normal"
done
