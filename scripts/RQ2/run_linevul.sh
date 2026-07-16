DATASET="$1"
ROOT="../.."

python $ROOT/src/RQ2/linevul/linevul_main.py \
  --output_dir=$ROOT/results/RQ2/linevul/${DATASET} \
  --model_type=roberta \
  --tokenizer_name=$ROOT/pretrained_models/codebert-base \
  --model_name_or_path=$ROOT/pretrained_models/codebert-base \
  --do_train \
  --do_test \
  --train_data_file=$ROOT/datasets/${DATASET}/train.jsonl \
  --eval_data_file=$ROOT/datasets/${DATASET}/valid.jsonl \
  --test_data_file=$ROOT/datasets/${DATASET}/test.jsonl \
  --epochs 10 \
  --block_size 512 \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 42 \
  --fp16