#!/bin/bash

MODEL="codebert"
ROOT="../.."

DATASETS=("cvefixes/c" "cvefixes/java" "cvefixes/php" "cvefixes/python" "cvefixes/mixed")
# DATASETS=("diversevul")

for DATASET in "${DATASETS[@]}"; do
  echo "===== RQ6: Generate on $DATASET ====="

  python $ROOT/src/RQ6/run.py \
      --train_data_file="$ROOT/datasets/$DATASET/train.jsonl" \
      --eval_data_file="$ROOT/datasets/$DATASET/valid.jsonl" \
      --test_data_file="$ROOT/datasets/$DATASET/test.jsonl" \
      --output_dir="$ROOT/results/RQ6/$MODEL/generate/$DATASET" \
      --pretrainedmodel_path="$ROOT/pretrained_models/$MODEL-base" \
      --model_name="$MODEL" \
      --seed=42 \
      --batch_size=32 \
      --max_seq_length=512 \
      --learning_rate=2e-5 \
      --num_epochs=15 \
      --early_stop_threshold=2 \ 
      --weight_decay=0.01 \
      --max_code_words=450 \
      --do_train \
      --do_eval \
      --do_test \
      --verbalizer_type multi_manual \
      --fp16

done

for DATASET in "${DATASETS[@]}"; do
  echo "===== RQ6: Classify on $DATASET ====="

  python $ROOT/src/RQ1/classify/codebert/run.py \
      --output_dir=$ROOT/results/RQ6/$MODEL/classify/${DATASET} \
      --model_type=roberta \
      --model_name=$MODEL \
      --tokenizer_name=$ROOT/pretrained_models/$MODEL-base \
      --model_name_or_path=$ROOT/pretrained_models/$MODEL-base \
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
      --learning_rate 1e-5 \
      --max_grad_norm 1.0 \
      --evaluate_during_training \
      --fp16 \
      --seed 42

done