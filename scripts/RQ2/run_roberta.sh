export TOKENIZERS_PARALLELISM=false

DATASET="$1"
ROOT="../.."

python $ROOT/src/RQ2/unixcoder/run.py \
    --output_dir=$ROOT/results/RQ2/roberta/${DATASET} \
    --model_type=roberta \
    --tokenizer_name=$ROOT/pretrained_models/roberta-base \
    --model_name_or_path=$ROOT/pretrained_models/roberta-base \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=$ROOT/datasets/${DATASET}/train.jsonl \
    --eval_data_file=$ROOT/datasets/${DATASET}/valid.jsonl \
    --test_data_file=$ROOT/datasets/${DATASET}/test.jsonl \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --dropout_probability 0.1 \
    --evaluate_during_training \
    --early_stopping_patience 3 \
    --fp16 \
    --seed 42 

