import os
import logging
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

INPUT_FILE = r"datasets\CVEfixes\CVEFixes.csv"  
BASE_OUTPUT_DIR = r"datasets\cvefixes"

TARGET_LANG_MAP = {
    "java": "java",
    "py": "python",
}

RANDOM_SEED = 42
COL_CODE = "code"
COL_LANG = "language"
COL_LABEL = "safety"
LABEL_MAP = {"vulnerable": 1, "safe": 0}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def process_and_split():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    raw_dataset = load_dataset('csv', data_files=INPUT_FILE, split='train')
    df_all = raw_dataset.to_pandas()

    df_all = df_all[df_all[COL_LANG].isin(TARGET_LANG_MAP.keys())].copy()
    df_all = df_all.dropna(subset=[COL_CODE, COL_LABEL])
    df_all = df_all[df_all[COL_LABEL].isin(LABEL_MAP.keys())]
    
    df_all = df_all.reset_index(drop=True)


    df_all['idx'] = df_all.index
    
    df_all[COL_LABEL] = df_all[COL_LABEL].map(LABEL_MAP)
    df_all = df_all.rename(columns={COL_CODE: "func", COL_LABEL: "target"})
    df_all = df_all[df_all["func"].str.len() > 20]

    garbage_strings = ["404: Not Found", "page not found", "timeout"]
    df_all = df_all[~df_all["func"].str.contains('|'.join(garbage_strings), case=False, na=False)]

    mixed_splits = {"train": [], "valid": [], "test": []}

    for raw_lang, folder_name in TARGET_LANG_MAP.items():
        df_lang = df_all[df_all[COL_LANG] == raw_lang].copy()
        if df_lang.empty:
            continue
        
        try:
            train_df, temp_df = train_test_split(
                df_lang, test_size=0.2, random_state=RANDOM_SEED, stratify=df_lang["target"]
            )
            val_df, test_df = train_test_split(
                temp_df, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_df["target"]
            )
        except ValueError:
            continue

        mixed_splits["train"].append(train_df)
        mixed_splits["valid"].append(val_df)
        mixed_splits["test"].append(test_df)

        lang_dir = os.path.join(BASE_OUTPUT_DIR, folder_name)
        os.makedirs(lang_dir, exist_ok=True)
        
        splits_dict = {"train": train_df, "valid": val_df, "test": test_df}
        for name, data in splits_dict.items():
            output_path = os.path.join(lang_dir, f"{name}.jsonl")
            final_data = data[["idx", "func", "target", COL_LANG]]
            Dataset.from_pandas(final_data, preserve_index=False).to_json(output_path, force_ascii=False)

        vul_count = int(df_lang["target"].sum())
        logger.info(f"Report [{folder_name.upper()}]: Total={len(df_lang)}, Vul={vul_count}({vul_count/len(df_lang):.1%}), Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")


    mixed_dir = os.path.join(BASE_OUTPUT_DIR, "mixed")
    os.makedirs(mixed_dir, exist_ok=True)
    
    logger.info("-" * 30)
    for split_name in ["train", "valid", "test"]:
        if not mixed_splits[split_name]: continue
        df_mixed_split = pd.concat(mixed_splits[split_name], ignore_index=True)
        df_mixed_split = df_mixed_split.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
        output_path = os.path.join(mixed_dir, f"{split_name}.jsonl")
        final_mixed_data = df_mixed_split[["idx", "func", "target", COL_LANG]]
        Dataset.from_pandas(final_mixed_data, preserve_index=False).to_json(output_path, force_ascii=False)
        
        if split_name == "train":
            total_m = sum(len(df) for df in mixed_splits["train"]) + sum(len(df) for df in mixed_splits["valid"]) + sum(len(df) for df in mixed_splits["test"])
            total_v = sum(df["target"].sum() for df in mixed_splits["train"]) + sum(df["target"].sum() for df in mixed_splits["valid"]) + sum(df["target"].sum() for df in mixed_splits["test"])
            logger.info(f"Report [MIXED]: Total={total_m}, Vul={int(total_v)}({total_v/total_m:.1%})")
        logger.info(f"  - Mixed {split_name} saved: {len(df_mixed_split)}")

if __name__ == "__main__":
    process_and_split()