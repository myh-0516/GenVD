#!/bin/bash

# MODEL="codebert"
# ROOT="../.."
# SEEDS=(42 43 44)
# DATASETS=("reveal")
# MODES=("null") #("hard" "soft" "mixed" "null")

# echo "=== Template Ablation with Multiple Seeds and Datasets ==="

# for DATASET in "${DATASETS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for MODE in "${MODES[@]}"; do
#             echo "Running: Dataset=$DATASET, Template=$MODE, Seed=$SEED"
            
#             OUTPUT_DIR="$ROOT/results/RQ5/template/$MODEL/$DATASET/$MODE/seed_$SEED"
            
#             python "$ROOT/src/RQ5/run.py" \
#                 --dataset="$DATASET" \
#                 --data_dir="$ROOT/datasets" \
#                 --output_dir="$OUTPUT_DIR" \
#                 --pretrainedmodel_path="$ROOT/pretrained_models/$MODEL-base" \
#                 --seed="$SEED" \
#                 --batch_size=32 \
#                 --max_seq_length=512 \
#                 --learning_rate=2e-5 \
#                 --num_epochs=15 \
#                 --early_stop_threshold=3 \
#                 --weight_decay=0.01 \
#                 --template_type="$MODE" \
#                 --verbalizer_type="manual" \
#                 --do_train \
#                 --do_eval \
#                 --do_test \
#                 --fp16
            
#             echo "Completed: $DATASET, $MODE, Seed=$SEED"
#         done
#     done
# done



#!/bin/bash

MODEL="codebert"
ROOT="../.."
SEEDS=(45)
DATASETS=("devign")   #"reveal"
MODES=("multi_manual") #("auto" "soft" "multi_manual")

echo "=== Verbalizer Ablation ==="

for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for MODE in "${MODES[@]}"; do
            echo "Running: Dataset=$DATASET, Verbalizer=$MODE, Seed=$SEED"
            
            OUTPUT_DIR="$ROOT/results/RQ5/verbalizer/$MODEL/$DATASET/$MODE/seed_$SEED"
            
            python "$ROOT/src/RQ5/run.py" \
                --dataset="$DATASET" \
                --data_dir="$ROOT/datasets" \
                --output_dir="$OUTPUT_DIR" \
                --pretrainedmodel_path="$ROOT/pretrained_models/$MODEL-base" \
                --seed="$SEED" \
                --batch_size=32 \
                --max_seq_length=512 \
                --learning_rate=6e-5 \
                --num_epochs=15 \
                --early_stop_threshold=3 \
                --weight_decay=0.01 \
                --template_type="mixed" \
                --verbalizer_type="$MODE" \
                --do_train \
                --do_eval \
                --do_test \
                --fp16
            
            echo "Completed: $DATASET, $MODE, Seed=$SEED"
        done
    done
done

echo "All experiments completed!"