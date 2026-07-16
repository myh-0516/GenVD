#!/usr/bin/env python3

import os
import torch
import json
import numpy as np
import random
import argparse
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from tqdm.auto import tqdm
from transformers import RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import transformers
from torch.utils.data import Dataset, DataLoader

# VulBERTa tokenizer
import sys
sys.path.append('src/RQ1/classify/vulberta')
from RQ2.vulberta.run import create_tokenizer, MyTokenizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class VulBERTaTokenizerWrapper:
    def __init__(self, max_seq_length=512):
        self.max_seq_length = max_seq_length
        self.tokenizer = create_tokenizer(max_seq_length)
        
        # 基本属性
        self.vocab_size = 50265
        self.pad_token_id = 1
        self.unk_token_id = 3
        self.mask_token_id = 50264
    
    def tokenize(self, text):
        encoding = self.tokenizer.encode(text)
        return encoding.tokens
    
    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            encoding = self.tokenizer.encode(tokens)
            return encoding.ids[0] if encoding.ids else self.unk_token_id
        return [self.convert_tokens_to_ids(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.tokenizer.decode([ids])
        return [self.tokenizer.decode([id_]) for id_ in ids]

class VulBERTaEfficientMLM(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        classifier = RobertaForSequenceClassification.from_pretrained(model_path)
        self.roberta = classifier.roberta
        self.config = self.roberta.config
        
        # 完整的MLM head以保持生成式特性
        self.lm_head = torch.nn.Linear(self.config.hidden_size, 50265, bias=False)
        self.lm_head.weight = torch.nn.Parameter(
            self.roberta.embeddings.word_embeddings.weight.clone()
        )
        
        # target token IDs: "no"=3738, "yes"=14037
        self.target_token_ids = [3738, 14037]
    
    def get_input_embeddings(self):
        return self.roberta.embeddings.word_embeddings
    
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            outputs = self.roberta(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        elif input_ids is not None:
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        # 高效MLM计算：只为target tokens计算logits
        batch_size, seq_len, hidden_size = outputs.last_hidden_state.shape
        last_hidden = outputs.last_hidden_state[:, -1, :]  # 使用最后位置作为mask位置
        
        # 计算target token logits
        mask_logits = torch.zeros(batch_size, len(self.target_token_ids), device=outputs.last_hidden_state.device)
        for i, token_id in enumerate(self.target_token_ids):
            token_weight = self.lm_head.weight[token_id]
            token_logit = torch.matmul(last_hidden, token_weight)
            mask_logits[:, i] = token_logit
        
        # 创建稀疏MLM logits tensor
        mlm_logits = torch.full((batch_size, seq_len, 50265), -65504.0, 
                               device=outputs.last_hidden_state.device, dtype=torch.float16)
        
        # 只在最后位置设置target token logits
        for i, token_id in enumerate(self.target_token_ids):
            mlm_logits[:, -1, token_id] = mask_logits[:, i]
        
        return type('Output', (), {
            'logits': mlm_logits,
            'last_hidden_state': outputs.last_hidden_state,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        })()

def cleaner(code):
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat, '', code)
    code = re.sub('\n', '', code)
    code = re.sub('\t', '', code)
    return code

class SimpleExample:
    def __init__(self, guid, text_a, label):
        self.guid = guid
        self.text_a = text_a
        self.label = label

def load_data(file_path, max_code_words=450, enable_cleaning=False):
    examples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                
                if 'func' in data:
                    code, label = data['func'], data['target']
                elif 'functionSource' in data:
                    code, label = data['functionSource'], data['label']
                else:
                    continue
                
                if enable_cleaning:
                    code = cleaner(code)

                code_words = code.split(' ')
                if len(code_words) > max_code_words:
                    code = ' '.join(code_words[:max_code_words])
                
                examples.append(SimpleExample(
                    guid=idx,
                    text_a=code,
                    label=label
                ))
                
            except Exception as e:
                continue
    
    return examples

def create_prompt_model(model_path, max_seq_length=512):
    """创建VulBERTa生成式模型"""
    tokenizer_wrapper = VulBERTaTokenizerWrapper(max_seq_length)
    plm = VulBERTaEfficientMLM(model_path)
    return plm, tokenizer_wrapper

def create_simple_dataloader(examples, tokenizer_wrapper, batch_size=4, shuffle=False):
    """创建简单的DataLoader"""
    
    class SimpleDataset(Dataset):
        def __init__(self, examples, tokenizer_wrapper):
            self.examples = examples
            self.tokenizer = tokenizer_wrapper
            
        def __len__(self):
            return len(self.examples)
            
        def __getitem__(self, idx):
            example = self.examples[idx]
            # 直接处理文本，添加简单的prompt
            text = f"Question: Is this code vulnerable? Code: {example.text_a} Answer:"
            
            encoding = self.tokenizer.tokenizer.encode(text)
            input_ids = encoding.ids[:self.tokenizer.max_seq_length]
            attention_mask = encoding.attention_mask[:self.tokenizer.max_seq_length]
            
            # Padding
            if len(input_ids) < self.tokenizer.max_seq_length:
                pad_length = self.tokenizer.max_seq_length - len(input_ids)
                input_ids.extend([1] * pad_length)  # pad_token_id = 1
                attention_mask.extend([0] * pad_length)
            
            return {
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask),
                'label': torch.tensor(example.label),
                'guid': example.guid
            }
    
    dataset = SimpleDataset(examples, tokenizer_wrapper)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if torch.cuda.is_available():
                batch['input_ids'] = batch['input_ids'].cuda()
                batch['attention_mask'] = batch['attention_mask'].cuda()
                batch['label'] = batch['label'].cuda()
            
            # 直接调用PLM
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            
            # 提取mask位置的"yes"/"no"logits
            if logits.shape[-1] == 50265:  # MLM logits
                mask_pos = -1  # 最后一个位置
                no_logits = logits[:, mask_pos, 3738]  # "no" token
                yes_logits = logits[:, mask_pos, 14037]  # "yes" token  
                logits = torch.stack([no_logits, yes_logits], dim=1)
            
            preds = torch.argmax(logits, dim=-1)
            labels = [int(label.cpu()) if torch.is_tensor(label) else int(label) for label in batch['label']]
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels)
    
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    return acc, precision, recall, f1, mcc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../pretrained_models/vulberta/")
    parser.add_argument("--dataset", type=str, default="devign")
    parser.add_argument("--data_dir", type=str, default="../datasets")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_code_words", type=int, default=450)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--early_stop_threshold", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")
    
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--enable_cleaning", action="store_true")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    prompt_model, tokenizer_wrapper = create_prompt_model(args.model_path, args.max_seq_length)
    
    if torch.cuda.is_available():
        prompt_model = prompt_model.cuda()
    
    if args.do_train:
        train_examples = load_data(
            os.path.join(args.data_dir, args.dataset, "train.jsonl"),
            args.max_code_words,
            args.enable_cleaning
        )
        val_examples = load_data(
            os.path.join(args.data_dir, args.dataset, "valid.jsonl"),
            args.max_code_words,
            args.enable_cleaning
        )
        
        
        # 使用简单的DataLoader
        train_dataloader = create_simple_dataloader(train_examples, tokenizer_wrapper, args.batch_size, shuffle=True)
        val_dataloader = create_simple_dataloader(val_examples, tokenizer_wrapper, args.batch_size, shuffle=False)
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        
        num_training_steps = args.num_epochs * len(train_dataloader)
        num_warmup_steps = int(args.warmup_ratio * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        scaler = torch.cuda.amp.GradScaler() if args.fp16 and torch.cuda.is_available() else None
        
        best_f1 = 0
        best_epoch = 0
        early_stop_count = 0
        loss_fn = torch.nn.CrossEntropyLoss()
        
        for epoch in range(args.num_epochs):
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
            
            prompt_model.train()
            total_loss = 0
            
            for batch in tqdm(train_dataloader, desc="Training"):
                if torch.cuda.is_available():
                    batch['input_ids'] = batch['input_ids'].cuda()
                    batch['attention_mask'] = batch['attention_mask'].cuda()
                    batch['label'] = batch['label'].cuda()
                
                labels = batch['label']
                
                with torch.cuda.amp.autocast(enabled=args.fp16 and torch.cuda.is_available()):
                    # 直接调用PLM
                    input_ids = batch['input_ids'] 
                    attention_mask = batch['attention_mask']
                    logits = prompt_model(input_ids=input_ids, attention_mask=attention_mask).logits
                    
                    # 提取mask位置的"yes"/"no"logits
                    if logits.shape[-1] == 50265:  # MLM logits
                        mask_pos = -1
                        no_logits = logits[:, mask_pos, 3738]  # "no" token
                        yes_logits = logits[:, mask_pos, 14037]  # "yes" token  
                        logits = torch.stack([no_logits, yes_logits], dim=1)
                    
                    loss = loss_fn(logits, labels)
                
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), args.max_grad_norm)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Average Loss: {avg_loss:.4f}")
            
            acc, precision, recall, f1, mcc = evaluate(prompt_model, val_dataloader, device)
            print(f"Validation - ACC: {acc:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                torch.save(prompt_model.state_dict(), os.path.join(args.output_dir, "best_model.bin"))
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count >= args.early_stop_threshold:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Training completed. Best epoch: {best_epoch+1}, Best F1: {best_f1:.4f}")
    
    if args.do_eval:
        model_path = os.path.join(args.output_dir, "best_model.bin")
        if os.path.exists(model_path):
            try:
                prompt_model.load_state_dict(torch.load(model_path))
                print("Loaded best model for evaluation")
            except Exception as e:
                print(f"Warning: Could not load saved model ({e}), using current model")
        
        val_examples = load_data(
            os.path.join(args.data_dir, args.dataset, "valid.jsonl"),
            args.max_code_words,
            args.enable_cleaning
        )
        
        # 使用全部验证样本
        print(f"Using {len(val_examples)} validation samples")
        
        val_dataloader = create_simple_dataloader(val_examples, tokenizer_wrapper, args.batch_size, shuffle=False)
        
        acc, precision, recall, f1, mcc = evaluate(prompt_model, val_dataloader, device)
        print(f"Validation F1: {f1:.4f}")
    
    if args.do_test:
        model_path = os.path.join(args.output_dir, "best_model.bin")
        if os.path.exists(model_path):
            try:
                prompt_model.load_state_dict(torch.load(model_path))
                print("Loaded best model for testing")
            except Exception as e:
                print(f"Warning: Could not load saved model ({e}), using current model")
        
        test_examples = load_data(
            os.path.join(args.data_dir, args.dataset, "test.jsonl"),
            args.max_code_words,
            args.enable_cleaning
        )
        
        
        test_dataloader = create_simple_dataloader(test_examples, tokenizer_wrapper, args.batch_size, shuffle=False)
        
        acc, precision, recall, f1, mcc = evaluate(prompt_model, test_dataloader, device)
        
        print(f"\nTest Results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"MCC: {mcc:.4f}")
        
        with open(os.path.join(args.output_dir, "test_results.txt"), "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1: {f1:.4f}\n")
            f.write(f"MCC: {mcc:.4f}\n")

if __name__ == "__main__":
    main()