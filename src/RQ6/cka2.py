import os
import json
import torch
import gc
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaConfig, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

PRETRAINED_DIR = r"pretrained_models/codebert-base"
OUTPUT_BASE = r"results/RQ7"
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 512

datasets = [
    {"name": "devign", "test_path": "datasets/devign/test.jsonl", "abbr": "Devign"},
    {"name": "reveal", "test_path": "datasets/reveal/test.jsonl", "abbr": "Reveal"},
    {"name": "bigvul", "test_path": "datasets/bigvul/test.jsonl", "abbr": "BigVul"},
]

# disc_colors = {
#     "devign": "#1f77b4",
#     "reveal": "#f38638",
#     "bigvul": "#549c3d",
# }

# gen_colors = {
#     "devign": "#e8aa00",
#     "reveal": "#8e71b6",
#     "bigvul": "#c43d35",
# }

disc_colors = {
    "devign": "#f7ca02",
    "reveal": "#549c3d",  
    "bigvul": "#f46d43",
}
gen_colors = {
    "devign": "#0ca8df",
    "reveal": "#5e4fa2", 
    "bigvul": '#9e0142',
}

def linear_cka(X, Y):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    C = torch.mm(X.t(), Y)
    return (torch.norm(C) ** 2 / (torch.norm(torch.mm(X.t(), X)) * torch.norm(torch.mm(Y.t(), Y)))).item()

def load_encoder(model_weights_path, pretrained_dir):
    config = RobertaConfig.from_pretrained(pretrained_dir)
    model = RobertaModel(config)
    state_dict = torch.load(model_weights_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    enc_dict = {k.replace("roberta.", ""): v for k, v in state_dict.items() if "roberta" in k}
    model.load_state_dict(enc_dict if enc_dict else state_dict, strict=False)
    return model.eval()

def load_datasets(filepath, tokenizer):
    texts_pure, texts_prompt = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line.strip())
            code = data.get('func', data.get('functionSource', ''))
            if code:
                texts_pure.append(code)
                texts_prompt.append(f"Question: Is this code vulnerable? Code: {code} Answer: {tokenizer.mask_token}")
    return texts_pure, texts_prompt

class CodeDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    def __len__(self): return len(self.encodings["input_ids"])
    def __getitem__(self, idx): return {k: v[idx] for k, v in self.encodings.items()}

def extract_cls_layerwise(model, dataloader, device, desc="CLS"):
    model.to(device)
    layer_feats = None
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            if layer_feats is None:
                layer_feats = [[] for _ in range(len(hidden_states))]
            for i, h in enumerate(hidden_states):
                layer_feats[i].append(h[:, 0, :].cpu())
    return [torch.cat(f, dim=0) for f in layer_feats]

def extract_mask_layerwise(model, dataloader, device, tokenizer, desc="MASK"):
    model.to(device)
    layer_feats = None
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            if layer_feats is None:
                layer_feats = [[] for _ in range(len(hidden_states))]
            for i, h in enumerate(hidden_states):
                batch_feats = []
                for j in range(input_ids.size(0)):
                    mask_idx = (input_ids[j] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                    idx = mask_idx[0] if len(mask_idx) > 0 else -1
                    batch_feats.append(h[j, idx, :])
                layer_feats[i].append(torch.stack(batch_feats).cpu())
    return [torch.cat(f, dim=0) for f in layer_feats]

def process_dataset(ds_cfg, tokenizer, device):
    name = ds_cfg["name"]
    print(f"\n===== Processing {name} =====")

    cls_path = f"results/RQ1/codebert/classify/{name}/best-f1-model.bin"
    gen_path = f"results/RQ1/codebert/generate/{name}/best-f1-model.bin"

    texts_pure, texts_prompt = load_datasets(ds_cfg["test_path"], tokenizer)
    ds_pure = CodeDataset(texts_pure, tokenizer, MAX_SEQ_LENGTH)
    ds_prompt = CodeDataset(texts_prompt, tokenizer, MAX_SEQ_LENGTH)
    dl_pure = DataLoader(ds_pure, batch_size=BATCH_SIZE, shuffle=False)
    dl_prompt = DataLoader(ds_prompt, batch_size=BATCH_SIZE, shuffle=False)

    model_pre = load_encoder(os.path.join(PRETRAINED_DIR, "pytorch_model.bin"), PRETRAINED_DIR)
    print(">> Pre-trained CLS (pure code)")
    pre_cls = extract_cls_layerwise(model_pre, dl_pure, device, desc="Pre-trained CLS")
    print(">> Pre-trained MASK (prompt)")
    pre_mask = extract_mask_layerwise(model_pre, dl_prompt, device, tokenizer, desc="Pre-trained MASK")
    del model_pre; gc.collect(); torch.cuda.empty_cache()

    model_cls = load_encoder(cls_path, PRETRAINED_DIR)
    print(">> Discriminative CLS")
    cls_cls = extract_cls_layerwise(model_cls, dl_pure, device, desc="Discriminative CLS")
    del model_cls; gc.collect(); torch.cuda.empty_cache()

    model_gen = load_encoder(gen_path, PRETRAINED_DIR)
    print(">> Generative MASK")
    gen_mask = extract_mask_layerwise(model_gen, dl_prompt, device, tokenizer, desc="Generative MASK")
    del model_gen; gc.collect(); torch.cuda.empty_cache()

    cka_cls = [linear_cka(cls_cls[i], pre_cls[i]) for i in range(len(pre_cls))]
    cka_gen = [linear_cka(gen_mask[i], pre_mask[i]) for i in range(len(pre_mask))]

    return name, cka_cls, cka_gen

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_DIR)

    all_results = {}
    for ds in datasets:
        name, cka_cls, cka_gen = process_dataset(ds, tokenizer, device)
        all_results[name] = {"cls": cka_cls, "gen": cka_gen}

    num_layers = len(all_results[datasets[0]["name"]]["cls"])
    layers = list(range(num_layers))

    plt.figure(figsize=(7, 5))

    for ds_cfg in datasets:
        name = ds_cfg["name"]
        abbr = ds_cfg["abbr"]
        cls_cka = all_results[name]["cls"].copy()
        gen_cka = all_results[name]["gen"].copy()
        cls_cka[0] = 1.0
        gen_cka[0] = 1.0
        plt.plot(layers, cls_cka, linestyle='-', color=disc_colors[name], linewidth=1.5, label=f'PT vs Disc on {abbr}')
        plt.plot(layers, gen_cka, linestyle='-', color=gen_colors[name], linewidth=1.5, label=f'PT vs Gen on {abbr}')
        

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))

    ax.spines['bottom'].set_bounds(layers[0], layers[-1])
    ax.spines['left'].set_bounds(0, 1)   

    plt.xlabel("Layer", fontsize=11)
    plt.ylabel("Linear CKA", fontsize=11)
    plt.ylim(0, 1.05)
    # layer_labels = ["0 (Embed)"] + [str(i) for i in range(1, num_layers)]
    plt.xticks(layers)
    plt.tick_params(axis='x') 
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=3, frameon=False, fontsize=10)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    out_png = os.path.join(OUTPUT_BASE, "cka_all_datasets_comparison.png")
    out_pdf = os.path.join(OUTPUT_BASE, "cka_all_datasets_comparison.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.show()
    print(f"\nPlots saved to {out_png} and {out_pdf}")

if __name__ == "__main__":
    main()