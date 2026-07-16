@echo off
setlocal enabledelayedexpansion

set OMP_NUM_THREADS=1

set "ROOT=..\.."

set MODELS=("graphcodebert" "unixcoder")
set DATASETS=("devign" "reveal" "bigvul")
set MODES=("normal")

for %%A in %MODELS% do (
    set "MODEL=%%~A"
    for %%D in %DATASETS% do (
        set "DATASET=%%~D"
        if "!DATASET!"=="bigvul" (
            set "WEIGHT=16.2"
            set "ALPHA=0.94"
        ) else if "!DATASET!"=="reveal" (
            set "WEIGHT=9.1"
            set "ALPHA=0.90"
        ) else if "!DATASET!"=="devign" (
            set "WEIGHT=1.2"
            set "ALPHA=0.55"
        ) else (
            set "WEIGHT=1.0"
            set "ALPHA=0.50"
        )

        for %%M in %MODES% do (
            set "MODE=%%~M"
            echo =========================================================
            echo Running: Model=!MODEL! ^| Dataset=!DATASET! ^| Mode=!MODE!
            echo =========================================================
            python %ROOT%\src\RQ1\run_balence_threshold.py ^
                --train_data_file="%ROOT%\datasets\!DATASET!\train.jsonl" ^
                --eval_data_file="%ROOT%\datasets\!DATASET!\valid.jsonl" ^
                --test_data_file="%ROOT%\datasets\!DATASET!\test.jsonl" ^
                --output_dir="%ROOT%\results\RQ1\!MODEL!\generate\!DATASET!" ^
                --pretrainedmodel_path="%ROOT%\pretrained_models\!MODEL!-base" ^
                --model_name="!MODEL!" ^
                --seed=42 ^
                --batch_size=32 ^
                --max_seq_length=512 ^
                --learning_rate=6e-5 ^
                --num_epochs=15 ^
                --early_stop_threshold=3 ^
                --weight_decay=0.01 ^
                --max_code_words=450 ^
                --do_eval ^
                --do_test ^
                --fp16 ^
                --imbalance_mode="!MODE!" ^
                --class_weights 1.0 !WEIGHT! ^
                --focal_alpha="!ALPHA!" ^
                --focal_gamma=2.0
        )
    )
)

endlocal