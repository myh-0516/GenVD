import os
import torch
import json
import transformers
import numpy as np
import random
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, classification_report, confusion_matrix
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from openprompt.data_utils import InputExample
from openprompt.plms import MLMTokenizerWrapper
from openprompt.prompts import MixedTemplate, ManualVerbalizer, AutomaticVerbalizer, SoftVerbalizer
from openprompt import PromptDataLoader, PromptForClassification

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrainedmodel_path", type=str, default="pretrained_models/codebert-base")
    parser.add_argument("--model_name", type=str, default="roberta")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--early_stop_threshold", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--template_type", type=str, default="mixed", choices=["hard", "soft", "mixed", "null"])
    parser.add_argument("--verbalizer_type", type=str, default="auto", choices=["manual", "auto", "soft", "multi_manual"])
    parser.add_argument("--dataset", type=str, default="reveal")
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--max_code_words", type=int, default=400)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()

args = parse_args()
set_seed(args.seed)

batch_size = args.batch_size
max_seq_l = args.max_seq_length
lr = args.learning_rate
num_epochs = args.num_epochs
use_cuda = torch.cuda.is_available()
pretrainedmodel_path = args.pretrainedmodel_path
early_stop_threshold = args.early_stop_threshold

global_guid_counter = 0
global_guid_map = {}

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

train_file_path = os.path.join(args.data_dir, args.dataset, "train.jsonl")
classes = get_dynamic_classes(train_file_path)
num_class = len(classes)

plm = AutoModelForMaskedLM.from_pretrained(pretrainedmodel_path)
tokenizer = AutoTokenizer.from_pretrained(pretrainedmodel_path, use_fast=False)
WrapperClass = MLMTokenizerWrapper

TEMPLATES = {
    "null": '{"placeholder":"text_a"} {"mask"}',
    "hard": 'Question: What type of vulnerability is this? Code: {"placeholder":"text_a"} Answer: {"mask"}',
    "soft": '{"placeholder":"text_a"} {"soft":"Answer:"} {"mask"}',
    "mixed": 'Question: What type of vulnerability is this? Code: {"placeholder":"text_a"} {"soft":"Answer:"} {"mask"}'
}

template_text = TEMPLATES[args.template_type]
if args.template_type in ["soft", "mixed"]:
    mytemplate = MixedTemplate(tokenizer=tokenizer, text=template_text, model=plm)
else:
    from openprompt.prompts import ManualTemplate
    mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

def create_dataloader(dataset_path, split_name, shuffle_flag=True):
    global global_guid_counter
    examples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                code_text = data.get('func') or data.get('code') or data.get('source_code')
                label_text = data.get('cwe_label') or data.get('cwe_id') or data.get('label')
                real_hash = data.get('hash') or data.get('idx') or data.get('id') or str(idx)
                
                if code_text is None or label_text is None or label_text not in classes:
                    continue
                
                global_guid_map[global_guid_counter] = real_hash
                safe_guid = global_guid_counter
                global_guid_counter += 1
                
                simple_code = ' '.join(code_text.split(' ')[:args.max_code_words])
                label_idx = classes.index(label_text)
                examples.append(
                    InputExample(
                        guid=safe_guid,
                        text_a=simple_code,
                        tgt_text=label_text,
                        label=label_idx
                    )
                )
            except (json.JSONDecodeError, KeyError):
                continue
                
    dataloader = PromptDataLoader(
        dataset=examples,
        template=mytemplate, 
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, 
        max_seq_length=max_seq_l,
        batch_size=batch_size, 
        shuffle=shuffle_flag,
        teacher_forcing=False, 
        predict_eos_token=False, 
        truncate_method="head"
    )
    return dataloader

dataset_path = os.path.join(args.data_dir, args.dataset)
train_dataloader = create_dataloader(os.path.join(dataset_path, "train.jsonl"), "train", True)
validation_dataloader = create_dataloader(os.path.join(dataset_path, "valid.jsonl"), "valid", True)  
test_dataloader = create_dataloader(os.path.join(dataset_path, "test.jsonl"), "test", False)

def create_verbalizer(verbalizer_type, tokenizer, classes, plm=None):
    dynamic_label_words = {}
    for cwe in classes:
        if '-' in cwe:
            dynamic_label_words[cwe] = [cwe.split('-')[1]]
        else:
            dynamic_label_words[cwe] = [cwe]
            
    if verbalizer_type == "manual":
        return ManualVerbalizer(tokenizer=tokenizer, classes=classes, label_words=dynamic_label_words)
    elif verbalizer_type == "auto":
        return AutomaticVerbalizer(
            tokenizer=tokenizer,
            classes=classes,
            num_candidates=1000,
            label_word_num_per_class=1,
            score_fct='llr',
            balance=True
        )
    elif verbalizer_type == "soft":
        return SoftVerbalizer(tokenizer=tokenizer, classes=classes, model=plm)
    else:
        return ManualVerbalizer(tokenizer=tokenizer, classes=classes, label_words=dynamic_label_words)

myverbalizer = create_verbalizer(args.verbalizer_type, tokenizer, classes, plm)

prompt_model = PromptForClassification(
    plm=plm, 
    template=mytemplate, 
    verbalizer=myverbalizer, 
    freeze_plm=False
)

if use_cuda:
    prompt_model = prompt_model.cuda()

loss_func = nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': args.weight_decay},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 
     'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

num_training_steps = num_epochs * len(train_dataloader)
num_warmup_steps = int(args.warmup_ratio * num_training_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

scaler = torch.cuda.amp.GradScaler() if args.fp16 and use_cuda else None

def test(prompt_model, test_dataloader, name, save_results=True):
    prompt_model.eval()
    torch.cuda.empty_cache()
    
    alllabels, alllogits = [], []
    sample_indices = []  
    
    with torch.no_grad():
        progress_bar = tqdm(test_dataloader, desc=f"{name.capitalize()} inference")
        for inputs in progress_bar:
            if use_cuda:
                inputs = inputs.cuda()
            
            with torch.cuda.amp.autocast(enabled=args.fp16 and use_cuda):
                logits = prompt_model(inputs)
                
            if hasattr(prompt_model.verbalizer, 'probs_buffer'):
                prompt_model.verbalizer.probs_buffer = None
            
            string_labels = inputs['tgt_text']
            alllabels.extend(string_labels)
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            alllogits.extend(probs)
            
            if 'guid' in inputs:
                if isinstance(inputs['guid'], torch.Tensor):
                    sample_indices.extend(inputs['guid'].cpu().tolist())
                else:
                    sample_indices.extend(inputs['guid'])
            else:
                batch_size_cur = len(string_labels)
                sample_indices.extend(list(range(len(sample_indices), len(sample_indices) + batch_size_cur)))
    
    allpreds = [classes[np.argmax(prob)] for prob in alllogits]
    
    acc = accuracy_score(alllabels, allpreds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(alllabels, allpreds, average='macro', zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(alllabels, allpreds, average='micro', zero_division=0)
    precision_wei, recall_wei, f1_wei, _ = precision_recall_fscore_support(alllabels, allpreds, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(alllabels, allpreds)
    
    def calculate_topk_accuracy(y_true, y_prob, classes, k):
        correct = sum(1 for true_label, probs in zip(y_true, y_prob)
                     if true_label in [classes[i] for i in np.argsort(probs)[-k:]])
        return correct / len(y_true)
    
    top1_acc = calculate_topk_accuracy(alllabels, alllogits, classes, 1)
    top2_acc = calculate_topk_accuracy(alllabels, alllogits, classes, 2)
    top3_acc = calculate_topk_accuracy(alllabels, alllogits, classes, 3)
    top4_acc = calculate_topk_accuracy(alllabels, alllogits, classes, 4)
    top5_acc = calculate_topk_accuracy(alllabels, alllogits, classes, 5)

    print(f"\n=== {name.capitalize()} Results ===")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(f"  Macro Precision: {precision_macro:.4f}")
    print(f"  Macro Recall: {recall_macro:.4f}")
    print(f"  MCC: {mcc:.4f}")
    
    if save_results and name.startswith('test'):
        os.makedirs(args.output_dir, exist_ok=True)
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
            
            cm = confusion_matrix(alllabels, allpreds, labels=classes)
            report = classification_report(alllabels, allpreds, labels=classes, output_dict=True, zero_division=0)
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
            for idx, pred, probs, true_label in zip(sample_indices, allpreds, alllogits, alllabels):
                max_prob = max(probs)
                real_hash = global_guid_map.get(idx, idx)
                f.write(f"{real_hash},{max_prob:.6f},{pred},{true_label}\n")
                
    return acc, precision_macro, recall_macro, f1_wei, f1_macro

os.makedirs(args.output_dir, exist_ok=True)

if args.do_train:
    bestmetric, bestepoch, early_stop_count = 0, 0, 0

    for epoch in range(num_epochs):
        prompt_model.train()
        tot_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Training")
        
        for step, inputs in enumerate(progress_bar):
            if use_cuda:
                inputs = inputs.cuda()
            
            string_labels = inputs['tgt_text']
            labels = torch.tensor([classes.index(label) for label in string_labels])
            if use_cuda:
                labels = labels.cuda()
            
            with torch.cuda.amp.autocast(enabled=args.fp16 and use_cuda):
                logits = prompt_model(inputs)
                loss = loss_func(logits, labels)
            
            INIT_STEPS = min(200, len(train_dataloader) - 1)
            if epoch == 0 and args.verbalizer_type == "auto" and step == INIT_STEPS:
                myverbalizer.optimize_to_initialize()
                if hasattr(myverbalizer, 'probs_buffer'):
                    myverbalizer.probs_buffer = None
                torch.cuda.empty_cache()
                
            is_auto_init_phase = (epoch == 0 and args.verbalizer_type == "auto" and step <= INIT_STEPS)
            
            if not is_auto_init_phase:
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
                
                optimizer.zero_grad()
                scheduler.step()
            else:
                optimizer.zero_grad()
            
            tot_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if not is_auto_init_phase and step > 0 and step % 100 == 0:
                if hasattr(myverbalizer, 'probs_buffer'):
                    myverbalizer.probs_buffer = None
                torch.cuda.empty_cache()
        
        avg_loss = tot_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        torch.cuda.empty_cache()
        acc, precision, recall, f1wei, f1_macro = test(prompt_model, validation_dataloader, "valid", save_results=False)
        
        if f1_macro > bestmetric:
            bestmetric = f1_macro
            bestepoch = epoch
            torch.save(prompt_model.state_dict(), os.path.join(args.output_dir, "best-f1-model.bin"))
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= early_stop_threshold:
                break

if args.do_eval:
    model_path = os.path.join(args.output_dir, "best-f1-model.bin")
    if os.path.exists(model_path):
        prompt_model.load_state_dict(torch.load(model_path))
    test(prompt_model, validation_dataloader, "valid", save_results=False)

if args.do_test:
    model_path = os.path.join(args.output_dir, "best-f1-model.bin")
    if os.path.exists(model_path):
        prompt_model.load_state_dict(torch.load(model_path))
    
    test(prompt_model, test_dataloader, "test", save_results=True)