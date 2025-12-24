#!/bin/bash

MODEL="codebert"
DATASET="devign"
ROOT="../.."

# Template ablation study
# echo "=== Template Ablation ==="
# for MODE in "hard" "soft" "mixed"; do
#     echo "Running: $MODE"
    
#     python $ROOT/src/RQ5/run.py \
#         --dataset="$DATASET" \
#         --data_dir="$ROOT/datasets" \
#         --output_dir="$ROOT/results/RQ5/template/$DATASET/$MODE" \
#         --pretrainedmodel_path="$ROOT/pretrained_models/$MODEL-base" \
#         --seed=42 \
#         --batch_size=32 \
#         --max_seq_length=512 \
#         --learning_rate=6e-5 \
#         --num_epochs=15 \
#         --early_stop_threshold=3 \
#         --weight_decay=0.01 \
#         --template_type="$MODE" \
#         --verbalizer_type="manual" \
#         --do_train \
#         --do_eval \
#         --do_test \
#         --fp16
        
#     echo "Completed: $MODE"
# done

# Verbalizer ablation study  (auto, manual, soft, multi_manual)
echo "=== Verbalizer Ablation ==="
for MODE in "multi_manual"; do
    echo "Running: $MODE"
    
    python $ROOT/src/RQ5/run.py \
        --dataset="$DATASET" \
        --data_dir="$ROOT/datasets" \
        --output_dir="$ROOT/results/RQ5/verbalizer/$DATASET/$MODE" \
        --pretrainedmodel_path="$ROOT/pretrained_models/$MODEL-base" \
        --seed=42 \
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
        --do_test
        
    echo "Completed: $MODE"
done

echo "All experiments completed!"