# GenVD

PyTorch implementation of *Rethinking Code Vulnerability Detection: A Generation-Based Approach*.

## Dataset
Pre-split train/val/test data: [Google Drive Download](https://drive.google.com/file/d/1Bjs7GDz7GkVX1CQp6AHPs_ONedFfqQkY/view?usp=drive_link)  
Extract to `datasets/<dataset_name>/` with `train.jsonl` / `valid.jsonl` / `test.jsonl`.

## Requirements
`pip install -r requirements.txt`

## Usage
- Python scripts: run from the repository root
- Shell scripts: run from the corresponding `a/RQ*/` directory

| RQ  | Entry | Task |
|-----|-------|------|
| RQ1 | `scripts/RQ1/run.sh` | Generative vs. discriminative vulnerability detection |
| RQ2 | `scripts/RQ2/run.sh` | Comparison with vulnerability detection baselines |
| RQ3 | `scripts/RQ3/run.sh` | Cross-architecture experiments |
| RQ4 | `scripts/RQ4/run.sh` | Prompt, verbalizer, and imbalance-handling analysis |
| RQ5 | `scripts/RQ5/run.sh` | Representation, prediction, and efficiency analysis |

## Pretrained Models
Place Hugging Face weights under `pretrained_models/`  
(e.g., `pretrained_models/codebert-base`).

## Output
Results are saved to `results/`.
