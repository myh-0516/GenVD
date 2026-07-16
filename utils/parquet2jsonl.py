import os
import pandas as pd
from pathlib import Path

BASE_DIR = r"datasets\\crossvul"
PARQUET_DIR = os.path.join(BASE_DIR, "data")
JSONL_OUTPUT_DIR = os.path.join(BASE_DIR)

Path(JSONL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

train_dfs = []
val_df = None
test_df = None

for file in os.listdir(PARQUET_DIR):
    if not file.endswith(".parquet"):
        continue
    path = os.path.join(PARQUET_DIR, file)
    df = pd.read_parquet(path)
    if "train" in file:
        train_dfs.append(df)
    elif "validation" in file:
        val_df = df
    elif "test" in file:
        test_df = df

train_combined = pd.concat(train_dfs, ignore_index=True)

train_combined.to_json(os.path.join(JSONL_OUTPUT_DIR, "train.jsonl"), orient="records", lines=True, force_ascii=False)
val_df.to_json(os.path.join(JSONL_OUTPUT_DIR, "valid.jsonl"), orient="records", lines=True, force_ascii=False)
test_df.to_json(os.path.join(JSONL_OUTPUT_DIR, "test.jsonl"), orient="records", lines=True, force_ascii=False)