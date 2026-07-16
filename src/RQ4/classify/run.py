import os
import json
import random
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix, classification_report
from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig, AutoModel, AutoTokenizer
from model import Model

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

class InputFeatures(object):
    def __init__(self, input_ids, idx, label):
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label

def get_dynamic_classes(dataset_path):
    unique_classes = set()
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                label = data.get('cwe_label') or data.get('cwe_id') or data.get('label')
                if label:
                    unique_classes.add(label)
            except:
                continue
    return sorted(list(unique_classes))

def convert_examples_to_features(js, tokenizer, args, classes, fallback_idx):
    code = js.get('func') or js.get('code') or js.get('source_code') or ''
    code = ' '.join(code.split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    
    idx = js.get('hash') or js.get('idx') or js.get('id') or str(fallback_idx)
    cwe_label = js.get('cwe_label') or js.get('cwe_id') or js.get('label')
    label = classes.index(cwe_label) if cwe_label in classes else 0
    
    return InputFeatures(source_ids, idx, label)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, classes):
        self.examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    js = json.loads(line.strip())
                    self.examples.append(convert_examples_to_features(js, tokenizer, args, classes, i))
                except:
                    continue

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), self.examples[i].idx

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_topk_accuracy(y_true, y_prob, k):
    correct = sum(1 for true_label, probs in zip(y_true, y_prob)
                  if true_label in np.argsort(probs)[-k:])
    return correct / len(y_true)

def train(args, train_dataset, eval_dataset, model):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)
    
    num_train_steps = args.epoch * len(train_dataloader)
    num_warmup_steps = int(0.1 * num_train_steps)
    
    model.to(args.device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
    scaler = GradScaler() if args.fp16 else None
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        
    best_f1 = 0.0
    early_stop_count = 0
    
    for epoch in range(args.epoch):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epoch}")
        
        for step, batch in enumerate(progress_bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            
            if args.fp16:
                with autocast():
                    loss, logits = model(inputs, labels)
            else:
                loss, logits = model(inputs, labels)
            
            if args.n_gpu > 1:
                loss = loss.mean()
            
            if args.fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            
            optimizer.zero_grad()
            scheduler.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        results = evaluate(args, eval_dataset, model)
        print(f"  Validation Accuracy: {results['eval_acc']:.4f}, Macro F1: {results['eval_f1']:.4f}")
        
        if results['eval_f1'] > best_f1:
            best_f1 = results['eval_f1']
            early_stop_count = 0
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, 'checkpoint-best-f1.bin'))
        else:
            early_stop_count += 1
            if args.early_stopping_patience and early_stop_count >= args.early_stopping_patience:
                print("Early stopping triggered")
                break

def evaluate(args, eval_dataset, model):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for batch in eval_dataloader:
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            if args.fp16:
                with autocast():
                    _, logits = model(inputs, labels)
            else:
                _, logits = model(inputs, labels)
            logits_list.append(logits.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            
    logits = np.concatenate(logits_list, 0)
    labels = np.concatenate(labels_list, 0)
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    return {'eval_acc': acc, 'eval_f1': f1}

def test(args, test_dataset, model, classes):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)
    
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
        
    model.eval()
    logits_list, labels_list, indices_list = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            hashes = batch[2]
            
            if args.fp16:
                with autocast():
                    logits = model(inputs)
            else:
                logits = model(inputs)
                
            logits_list.append(logits.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            indices_list.extend(hashes)
            
    logits = np.concatenate(logits_list, 0)
    labels = np.concatenate(labels_list, 0)
    preds = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
    
    acc = accuracy_score(labels, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds, average='micro', zero_division=0)
    precision_wei, recall_wei, f1_wei, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    
    top1_acc = calculate_topk_accuracy(labels, probs, 1)
    top2_acc = calculate_topk_accuracy(labels, probs, 2)
    top3_acc = calculate_topk_accuracy(labels, probs, 3)
    top4_acc = calculate_topk_accuracy(labels, probs, 4)
    top5_acc = calculate_topk_accuracy(labels, probs, 5)
    
    print(f"\n=== Test Results ===")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(f"  Macro Precision: {precision_macro:.4f}")
    print(f"  Macro Recall: {recall_macro:.4f}")
    print(f"  MCC: {mcc:.4f}")
    
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write("=== Overall Metrics ===\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}\n")
        f.write(f"Macro Precision: {precision_macro:.4f}\n")
        f.write(f"Macro Recall: {recall_macro:.4f}\n")
        f.write(f"Macro F1: {f1_macro:.4f}\n")
        f.write(f"Micro Precision: {precision_micro:.4f}\n")
        f.write(f"Micro Recall: {recall_micro:.4f}\n")
        f.write(f"Micro F1: {f1_micro:.4f}\n")
        f.write(f"Weighted Precision: {precision_wei:.4f}\n")
        f.write(f"Weighted Recall: {recall_wei:.4f}\n")
        f.write(f"Weighted F1: {f1_wei:.4f}\n\n")
        
        f.write("=== Top-K Accuracy ===\n")
        f.write(f"Top-1 Accuracy: {top1_acc:.4f}\n")
        f.write(f"Top-2 Accuracy: {top2_acc:.4f}\n")
        f.write(f"Top-3 Accuracy: {top3_acc:.4f}\n")
        f.write(f"Top-4 Accuracy: {top4_acc:.4f}\n")
        f.write(f"Top-5 Accuracy: {top5_acc:.4f}\n\n")
        
        f.write("=== Per-Class Metrics ===\n")
        f.write(f"{'CWE':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Count':<10}\n")
        f.write("-" * 80 + "\n")
        
        cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))
        report = classification_report(labels, preds, target_names=classes, labels=list(range(len(classes))), output_dict=True, zero_division=0)
        total_samples = np.sum(cm)
        
        for i, cwe in enumerate(classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = total_samples - tp - fp - fn
            class_acc = (tp + tn) / total_samples if total_samples > 0 else 0
            
            if cwe in report:
                metrics_cwe = report[cwe]
                f.write(f"{cwe:<15} | {class_acc:<10.4f} | {metrics_cwe['precision']:<10.4f} | {metrics_cwe['recall']:<10.4f} | {metrics_cwe['f1-score']:<10.4f} | {int(metrics_cwe['support']):<10}\n")
        f.write("\n")
        
        f.write("=== Confusion Matrix ===\n")
        f.write(f"Classes: {classes}\n")
        f.write(f"{cm}\n")

    with open(os.path.join(args.output_dir, "predictions.csv"), "w") as f:
        f.write("idx,probability,prediction_label,true_label\n")
        for idx, pred, prob_arr, true_label in zip(indices_list, preds, probs, labels):
            max_prob = np.max(prob_arr)
            f.write(f"{idx},{max_prob:.6f},{classes[pred]},{classes[true_label]}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_file", type=str, required=True)
    parser.add_argument("--eval_data_file", type=str, required=True)
    parser.add_argument("--test_data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--dropout_probability", type=float, default=0.0)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--early_stopping_patience", type=int, default=4)
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action='store_true')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size // max(1, args.n_gpu)
    args.per_gpu_eval_batch_size = args.eval_batch_size // max(1, args.n_gpu)
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    classes = get_dynamic_classes(args.train_data_file)
    num_labels = len(classes)
    print(f"Discovered {num_labels} unique classes: {classes}")
    
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    encoder = AutoModel.from_pretrained(args.model_name_or_path, config=config, trust_remote_code=True)
    model = Model(encoder, config, tokenizer, args, num_labels)
    
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, args.train_data_file, classes)
        eval_dataset = TextDataset(tokenizer, args, args.eval_data_file, classes)
        train(args, train_dataset, eval_dataset, model)
        
    if args.do_test:
        test_dataset = TextDataset(tokenizer, args, args.test_data_file, classes)
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint-best-f1.bin')
        model.load_state_dict(torch.load(checkpoint_path))
        model.to(args.device)
        test(args, test_dataset, model, classes)

if __name__ == "__main__":
    main()