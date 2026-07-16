import os
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

FILE_PATHS = {
    "java": r"datasets\cvefixes1\java.csv",
    "python": r"datasets\cvefixes1\python.csv"
}
BASE_OUTPUT_DIR = r"datasets\cvefixes1"
RANDOM_SEED = 42

def process_and_split():
    global_counter = 0
    mixed_splits = {"train": [], "valid": [], "test": []}

    for lang, path in FILE_PATHS.items():
        if not os.path.exists(path):
            continue
        
        df = pd.read_csv(path)
        pairs = []
        
        for _, row in df.iterrows():
            f_before = row["func_before"] if pd.notna(row.get("func_before")) else row.get("code_before", "")
            f_after = row["func_after"] if pd.notna(row.get("func_after")) else row.get("code_after", "")
            
            if not f_before or not f_after:
                continue

            vul_item = {
                "idx": str(global_counter),
                "func": f_before,
                "target": 1
            }
            global_counter += 1
            
            fix_item = {
                "idx": str(global_counter),
                "func": f_after,
                "target": 0
            }
            global_counter += 1
            
            pairs.append([vul_item, fix_item])

        if len(pairs) < 3:
            continue

        train_pairs, temp_pairs = train_test_split(pairs, test_size=0.2, random_state=RANDOM_SEED)
        val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.5, random_state=RANDOM_SEED)

        lang_dir = os.path.join(BASE_OUTPUT_DIR, lang)
        os.makedirs(lang_dir, exist_ok=True)

        for name, p_list in {"train": train_pairs, "valid": val_pairs, "test": test_pairs}.items():
            flattened = [item for pair in p_list for item in pair]
            df_split = pd.DataFrame(flattened)
            mixed_splits[name].append(df_split)
            
            output_path = os.path.join(lang_dir, f"{name}.jsonl")
            Dataset.from_pandas(df_split[["idx", "func", "target"]], preserve_index=False).to_json(output_path, force_ascii=False)

    mixed_dir = os.path.join(BASE_OUTPUT_DIR, "mixed")
    os.makedirs(mixed_dir, exist_ok=True)

    for split_name, df_list in mixed_splits.items():
        if not df_list:
            continue
        df_mixed = pd.concat(df_list, ignore_index=True)
        df_mixed = df_mixed.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
        output_path = os.path.join(mixed_dir, f"{split_name}.jsonl")
        Dataset.from_pandas(df_mixed[["idx", "func", "target"]], preserve_index=False).to_json(output_path, force_ascii=False)

if __name__ == "__main__":
    process_and_split()