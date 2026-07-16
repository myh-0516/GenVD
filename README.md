```markdown
# GenVD

PyTorch implementation of *Rethinking Code Vulnerability Detection: A Generation-Based Approach*.

## Dataset
Pre-split train/val/test data: [Google Drive Download](https://drive.google.com/file/d/1Bjs7GDz7GkVX1CQp6AHPs_ONedFfqQkY/view?usp=drive_link)  
Extract to `datasets/<dataset_name>/` with `train.jsonl` / `valid.jsonl` / `test.jsonl`.

## Requirements
```bash
pip install -r requirements.txt
```

## Usage
- Python scripts: run from repository root
- Shell scripts: run from corresponding `scripts/RQ*/` directory

| RQ  | Entry | Task |
|-----|-------|------|
| RQ1 | `src/RQ1/run.py` | Binary detection (generative vs. discriminative) |
| RQ3 | `src/RQ3/run.py` | Cross-architecture experiments |
| RQ4 | `src/RQ4/run.py` | CWE multi-class classification |
| RQ5 | `src/RQ5/run.py` | Prompt & verbalizer analysis |
| RQ6 | `src/RQ6/cka.py`<br>`src/RQ6/PR_Calibration_curve.py`<br>`src/RQ6/cost_radar.js` | Analysis & visualization |

## Pretrained Models
Place Hugging Face weights under `pretrained_models/` (e.g. `pretrained_models/codebert-base`).

## Output
Results saved to `results/`.

## License
See `LICENSE`.
```
