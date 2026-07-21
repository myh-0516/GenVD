#!/bin/bash
export OMP_NUM_THREADS=1

MODEL="${1:-codebert}" 
ROOT="../.."

DATASETS=("bigvul") #"bigvul"
MODES=("normal") #"normal" "weighted_ce" "focal" "undersample" "oversample"

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

    for MODE in "${MODES[@]}"; do
        echo "========================================================="
        echo "Running: Dataset=$DATASET | Mode=$MODE | Model=$MODEL"
        echo "========================================================="
        
        python $ROOT/src/RQ1/run.py \
            --train_data_file="$ROOT/datasets/$DATASET/train.jsonl" \
            --eval_data_file="$ROOT/datasets/$DATASET/valid.jsonl" \
            --test_data_file="$ROOT/datasets/$DATASET/test.jsonl" \
            --output_dir="$ROOT/results/RQ1/$MODEL/generate/$DATASET/$MODE" \
            --pretrainedmodel_path="$ROOT/pretrained_models/$MODEL-base" \
            --model_name="$MODEL" \
            --seed=42 \
            --batch_size=32 \
            --max_seq_length=512 \
            --learning_rate=6e-5 \
            --num_epochs=15 \
            --early_stop_threshold=3 \
            --weight_decay=0.01 \
            --do_train \
            --do_eval \
            --do_test \
            --fp16 \
            --imbalance_mode="$MODE" \
            --class_weights 1.0 $WEIGHT \
            --focal_alpha="$ALPHA" \
            --focal_gamma=2.0
    done
done
