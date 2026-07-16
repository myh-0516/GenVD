import os
import json
import torch
import gc
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaConfig, AutoTokenizer
from tqdm import tqdm

DATASET_PATH = r"datasets\devign\test.jsonl"
PRETRAINED_DIR = r"pretrained_models\codebert-base"
CLS_MODEL_PATH = r"results\RQ1\codebert\classify\devign\best-f1-model.bin"
GEN_MODEL_PATH = r"results\RQ1\codebert\generate\devign\best-f1-model.bin"
OUTPUT_PATH = r"results\RQ7\cka_task_token_results.txt"
BATCH_SIZE = 4
MAX_SEQ_LENGTH = 512

def linear_cka(X, Y):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    C = torch.mm(X.t(), Y)
    cka_score = torch.norm(C)**2 / (torch.norm(torch.mm(X.t(), X)) * torch.norm(torch.mm(Y.t(), Y)))
    return cka_score.item()

def load_encoder(model_weights_path, pretrained_dir):
    config = RobertaConfig.from_pretrained(pretrained_dir)
    model = RobertaModel(config)
    try:
        state_dict = torch.load(model_weights_path, map_location="cpu", weights_only=True)
    except Exception:
        state_dict = torch.load(model_weights_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    enc_dict = {k.replace("roberta.", ""): v for k, v in state_dict.items() if "roberta" in k}
    model.load_state_dict(enc_dict if enc_dict else state_dict, strict=False)
    return model.eval()

def load_datasets(filepath, tokenizer):
    texts_pure = []
    texts_prompt = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            code = data.get('func', data.get('functionSource', ''))
            if code:
                texts_pure.append(code)
                texts_prompt.append(f"Question: Is this code vulnerable? Code: {code} Answer: {tokenizer.mask_token}")
    return texts_pure, texts_prompt

class CodeDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding="max_length", 
            max_length=max_len, 
            return_tensors="pt"
        )
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

def extract_cls(model, dataloader, device, desc="Ext"):
    model.to(device)
    feats = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            feats.append(outputs.last_hidden_state[:, 0, :].cpu())
    return torch.cat(feats, dim=0)

def extract_mask(model, dataloader, device, tokenizer, desc="Ext"):
    model.to(device)
    feats = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_feats = []
            for i in range(input_ids.size(0)):
                mask_idx = (input_ids[i] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                if len(mask_idx) > 0:
                    batch_feats.append(outputs.last_hidden_state[i, mask_idx[0], :])
                else:
                    batch_feats.append(outputs.last_hidden_state[i, -1, :])
            feats.append(torch.stack(batch_feats).cpu())
    return torch.cat(feats, dim=0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_DIR)
    
    texts_pure, texts_prompt = load_datasets(DATASET_PATH, tokenizer)
    ds_pure = CodeDataset(texts_pure, tokenizer, max_len=MAX_SEQ_LENGTH)
    ds_prompt = CodeDataset(texts_prompt, tokenizer, max_len=MAX_SEQ_LENGTH)
    
    dl_pure = DataLoader(ds_pure, batch_size=BATCH_SIZE, shuffle=False)
    dl_prompt = DataLoader(ds_prompt, batch_size=BATCH_SIZE, shuffle=False)

    model_pre = load_encoder(os.path.join(PRETRAINED_DIR, "pytorch_model.bin"), PRETRAINED_DIR)
    pre_cls = extract_cls(model_pre, dl_pure, device, desc="Pre-trained [CLS] (Pure Code)")
    pre_mask = extract_mask(model_pre, dl_prompt, device, tokenizer, desc="Pre-trained [MASK] (Prompted)")
    del model_pre
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    model_cls = load_encoder(CLS_MODEL_PATH, PRETRAINED_DIR)
    cls_cls = extract_cls(model_cls, dl_pure, device, desc="Discriminative [CLS] (Pure Code)")
    del model_cls
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    model_gen = load_encoder(GEN_MODEL_PATH, PRETRAINED_DIR)
    gen_mask = extract_mask(model_gen, dl_prompt, device, tokenizer, desc="Generative [MASK] (Prompted)")
    del model_gen
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    cka_cls_vs_pre = linear_cka(cls_cls, pre_cls)
    cka_gen_vs_pre = linear_cka(gen_mask, pre_mask)

    result_text = (
        f"================ CKA Target Token Results ================\n"
        f"Test samples: {len(texts_pure)}\n"
        f"CKA [CLS] (Discriminative vs Pre-trained) : {cka_cls_vs_pre:.4f}\n"
        f"CKA [MASK] (Generative vs Pre-trained)    : {cka_gen_vs_pre:.4f}\n"
        f"==========================================================\n"
    )
    
    print("\n" + result_text)
    
    dir_name = os.path.dirname(OUTPUT_PATH)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
        f.write(result_text)

if __name__ == "__main__":
    main()