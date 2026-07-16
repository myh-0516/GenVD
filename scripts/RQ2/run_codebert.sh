#!/bin/bash

# DATASET="$1"
# ROOT="../.."

# python $ROOT/src/RQ2/codebert/run.py \
#     --output_dir=$ROOT/results/RQ2/codebert/${DATASET} \
#     --model_type=roberta \
#     --tokenizer_name=$ROOT/pretrained_models/codebert-base \
#     --model_name_or_path=$ROOT/pretrained_models/codebert-base \
#     --do_train \
#     --do_test \
#     --do_eval \
#     --train_data_file=$ROOT/datasets/${DATASET}/train.jsonl \
#     --eval_data_file=$ROOT/datasets/${DATASET}/valid.jsonl \
#     --test_data_file=$ROOT/datasets/${DATASET}/test.jsonl \
#     --epoch 5 \
#     --block_size 512 \
#     --train_batch_size 32 \
#     --eval_batch_size 64 \
#     --learning_rate 2e-5 \
#     --max_grad_norm 1.0 \
#     --evaluate_during_training \
#     --fp16 \
#     --seed 42


#!/bin/bash
export OMP_NUM_THREADS=1

MODEL_NAME="${1:-codebert}"
ROOT="../.."
DATASETS=("devign") # "devign" "reveal" "bigvul"
IMBALANCE_MODES=("normal") #"weighted_ce" "focal" "oversample" "undersample"
TUNING_MODES=("lora") #"full" "lora" "adapter"

for DATASET in "${DATASETS[@]}"; do
  if [ "$DATASET" == "bigvul" ]; then
        WEIGHT="16.2"
        ALPHA="0.94"
    elif [ "$DATASET" == "reveal" ]; then
        WEIGHT="9.1"
        ALPHA="0.90"
     elif [ "$DATASET" == "devign" ]; then
        WEIGHT="1.2" 
        ALPHA="0.55"
    else
        WEIGHT="1.0"
        ALPHA="0.50"
    fi

  for IMBALANCE_MODE in "${IMBALANCE_MODES[@]}"; do
    for TUNING in "${TUNING_MODES[@]}"; do
        if [ "$TUNING" == "full" ]; then
            LR="2e-5"
        else
            LR="2e-4"
        fi
        echo "========================================================="
        echo "Running: Dataset=$DATASET | Mode=$IMBALANCE_MODE | Tuning=$TUNING | Model=$MODEL_NAME"
        echo "========================================================="
        
        python $ROOT/src/RQ1/classify/codebert/run.py \
          --output_dir=$ROOT/results/RQ1/$MODEL_NAME/classify/${DATASET}/${IMBALANCE_MODE}_${TUNING} \
          --model_type=roberta \
          --tokenizer_name=$ROOT/pretrained_models/$MODEL_NAME-base \
          --model_name_or_path=$ROOT/pretrained_models/$MODEL_NAME-base \
          --do_train \
          --do_test \
          --do_eval \
          --train_data_file=$ROOT/datasets/${DATASET}/train.jsonl \
          --eval_data_file=$ROOT/datasets/${DATASET}/valid.jsonl \
          --test_data_file=$ROOT/datasets/${DATASET}/test.jsonl \
          --epoch 10 \
          --block_size 512 \
          --train_batch_size 32 \
          --eval_batch_size 64 \
          --learning_rate $LR \
          --max_grad_norm 1.0 \
          --evaluate_during_training \
          --fp16 \
          --seed 42 \
          --imbalance_mode $IMBALANCE_MODE \
          --class_weights 1.0 $WEIGHT \
          --focal_alpha $ALPHA \
          --focal_gamma 2.0 \
          --tuning_mode $TUNING \
          --lora_r 16 \
          --lora_alpha 32
    done
  done
done