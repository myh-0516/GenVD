import os
import pandas as pd
from datasets import Dataset

CROSSVUL_DIR = r"datasets\crossvul"
CVEFIXES_DIR = r"datasets\cvefixes"
OUTPUT_DIR = r"datasets\merge"
LANGS = ["java", "python"]
SPLITS = ["train", "valid", "test"]

def merge_and_report():
    global_idx = 0
    
    for lang in LANGS:
        lang_dir = os.path.join(OUTPUT_DIR, lang)
        os.makedirs(lang_dir, exist_ok=True)
        lang_stats = []

        for split in SPLITS:
            p1 = os.path.join(CROSSVUL_DIR, lang, f"{split}.jsonl")
            p2 = os.path.join(CVEFIXES_DIR, lang, f"{split}.jsonl")
            
            dfs = []
            if os.path.exists(p1): dfs.append(pd.read_json(p1, lines=True))
            if os.path.exists(p2): dfs.append(pd.read_json(p2, lines=True))
            
            if not dfs: continue
            
            df = pd.concat(dfs, ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            df["idx"] = [str(i) for i in range(global_idx, global_idx + len(df))]
            global_idx += len(df)
            
            df = df[["idx", "func", "target"]]
            Dataset.from_pandas(df, preserve_index=False).to_json(os.path.join(lang_dir, f"{split}.jsonl"), force_ascii=False)
            
            lang_stats.append((split, len(df), int(df["target"].sum())))

        print(f"\nREPORT [{lang.upper()}]")
        for s_name, total, vuls in lang_stats:
            print(f"  {s_name:<6}: Total={total:<6} Vul={vuls:<5} ({vuls/total:.1%})")

    mixed_dir = os.path.join(OUTPUT_DIR, "mixed")
    os.makedirs(mixed_dir, exist_ok=True)
    print("\nREPORT [MIXED SPLITS]")
    for split in SPLITS:
        dfs = [pd.read_json(os.path.join(OUTPUT_DIR, l, f"{split}.jsonl"), lines=True) for l in LANGS if os.path.exists(os.path.join(OUTPUT_DIR, l, f"{split}.jsonl"))]
        if not dfs: continue
        df_m = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        Dataset.from_pandas(df_m, preserve_index=False).to_json(os.path.join(mixed_dir, f"{split}.jsonl"), force_ascii=False)
        m_vuls = int(df_m["target"].sum())
        print(f"  {split:<6}: Total={len(df_m):<6} Vul={m_vuls:<5} ({m_vuls/len(df_m):.1%})")

if __name__ == "__main__":
    merge_and_report()