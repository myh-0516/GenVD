import os
import json
import logging
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

INPUT_FILE = r"datasets\crossvul\train.jsonl"
BASE_OUTPUT_DIR = r"datasets\crossvul"

    # "c": "c",
    # "php": "php",
    # "javascript": "javascript",
TARGET_LANG_MAP = {
    "python": "python",
    "java": "java"
}

RANDOM_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def process_and_split():
    if not os.path.exists(INPUT_FILE):
        return

    lang_groups = {lang: [] for lang in TARGET_LANG_MAP}
    global_counter = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            
            lang = str(data.get("language", "")).lower()
            if lang not in TARGET_LANG_MAP:
                continue

            pair = []
            if "vulnerable_code" in data and "fixed_code" in data:
                pair.append({"func": data["vulnerable_code"], "target": 1, "language": lang, "idx": str(global_counter)})
                global_counter += 1
                pair.append({"func": data["fixed_code"], "target": 0, "language": lang, "idx": str(global_counter)})
                global_counter += 1
            elif "func" in data and "target" in data:
                pair.append({"func": data["func"], "target": int(data["target"]), "language": lang, "idx": str(global_counter)})
                global_counter += 1
            
            if pair:
                lang_groups[lang].append(pair)

    mixed_splits = {"train": [], "valid": [], "test": []}

    for raw_lang, folder_name in TARGET_LANG_MAP.items():
        groups = lang_groups[raw_lang]
        if len(groups) < 3:
            continue
        
        try:
            train_groups, temp_groups = train_test_split(
                groups, test_size=0.2, random_state=RANDOM_SEED
            )
            val_groups, test_groups = train_test_split(
                temp_groups, test_size=0.5, random_state=RANDOM_SEED
            )
        except ValueError:
            continue

        lang_res = {}
        for name, g_list in {"train": train_groups, "valid": val_groups, "test": test_groups}.items():
            unpacked = [item for sublist in g_list for item in sublist]
            df = pd.DataFrame(unpacked)
            lang_res[name] = df
            mixed_splits[name].append(df)

        lang_dir = os.path.join(BASE_OUTPUT_DIR, folder_name)
        os.makedirs(lang_dir, exist_ok=True)
        
        for name, df in lang_res.items():
            output_path = os.path.join(lang_dir, f"{name}.jsonl")
            Dataset.from_pandas(df[["idx", "func", "target"]], preserve_index=False).to_json(output_path, force_ascii=False)

        total_len = sum(len(df) for df in lang_res.values())
        vul_count = sum(df["target"].sum() for df in lang_res.values())
        logger.info(f"Report [{folder_name.upper()}]: Total={total_len}, Vul={int(vul_count)}({vul_count/total_len:.1%}), Train={len(lang_res['train'])}, Val={len(lang_res['valid'])}, Test={len(lang_res['test'])}")

    mixed_dir = os.path.join(BASE_OUTPUT_DIR, "mixed")
    os.makedirs(mixed_dir, exist_ok=True)
    
    logger.info("-" * 30)
    for split_name in ["train", "valid", "test"]:
        if not mixed_splits[split_name]: 
            continue
        df_mixed_split = pd.concat(mixed_splits[split_name], ignore_index=True)
        df_mixed_split = df_mixed_split.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
        output_path = os.path.join(mixed_dir, f"{split_name}.jsonl")
        Dataset.from_pandas(df_mixed_split[["idx", "func", "target"]], preserve_index=False).to_json(output_path, force_ascii=False)
        
        if split_name == "train":
            total_m = sum(len(pd.concat(mixed_splits[s])) for s in ["train", "valid", "test"] if mixed_splits[s])
            total_v = sum(pd.concat(mixed_splits[s])["target"].sum() for s in ["train", "valid", "test"] if mixed_splits[s])
            logger.info(f"Report [MIXED]: Total={int(total_m)}, Vul={int(total_v)}({total_v/total_m:.1%})")

if __name__ == "__main__":
    process_and_split()