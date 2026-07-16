import os
import torch
import json
import transformers
import numpy as np
import random
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, matthews_corrcoef
import sklearn.metrics
from tqdm.auto import tqdm
from openprompt.data_utils import InputExample
from transformers import AutoModelForMaskedLM, AutoTokenizer
from openprompt.plms import MLMTokenizerWrapper
from openprompt.prompts import ManualTemplate, MixedTemplate
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, AutomaticVerbalizer, SoftVerbalizer
from openprompt import PromptForClassification
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

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
    parser = argparse.ArgumentParser(description="CodeBERT Generative Vulnerability Detection")
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
    parser.add_argument("--template_type", type=str, default="mixed", choices=["hard", "soft", "mixed","null"])
    parser.add_argument("--verbalizer_type", type=str, default="manual", choices=["manual", "auto", "soft", "multi_manual"])
    parser.add_argument("--dataset", type=str, default="reveal")
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()

args = parse_args()

seed = args.seed
set_seed(seed)
batch_size = args.batch_size
num_class = 2
max_seq_l = args.max_seq_length
lr = args.learning_rate
num_epochs = args.num_epochs
use_cuda = torch.cuda.is_available()
pretrainedmodel_path = args.pretrainedmodel_path
early_stop_threshold = args.early_stop_threshold

classes = ["NON_VULNERABLE", "VULNERABLE"]

plm = AutoModelForMaskedLM.from_pretrained(pretrainedmodel_path)
tokenizer = AutoTokenizer.from_pretrained(pretrainedmodel_path, use_fast=False)
WrapperClass = MLMTokenizerWrapper
model_config = plm.config

TEMPLATES = {
    "null": '{"placeholder":"text_a"} {"mask"}',
    "hard": 'Question: Is this code vulnerable? Code: {"placeholder":"text_a"} Answer: {"mask"}',
    "soft": '{"placeholder":"text_a"} {"soft":"Answer:"} {"mask"}',
    "mixed": 'Question: Is this code vulnerable? Code: {"placeholder":"text_a"} {"soft":"Answer:"} {"mask"}'
}

template_text = TEMPLATES[args.template_type]
if args.template_type in ["soft", "mixed"]:
    mytemplate = MixedTemplate(tokenizer=tokenizer, text=template_text, model=plm)
else:
    mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

print(f"Using {args.template_type} template: {template_text}")

DATASET_CONFIG = {
    'devign': {'code_field': 'func', 'label_field': 'target'},
    'bigvul': {'code_field': 'func', 'label_field': 'target'},
    'reveal': {'code_field': 'functionSource', 'label_field': 'label'}
}

def create_dataloader(dataset_path, split_name, shuffle_flag=True):
    print(f"Processing data from {dataset_path}")
    examples = []
    config = DATASET_CONFIG.get(args.dataset)
    if not config:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                code_text = data[config['code_field']]
                label = int(data[config['label_field']])
                label_text = classes[label]
                simple_code = code_text
                
                examples.append(
                    InputExample(
                        guid=idx,
                        text_a=simple_code,
                        tgt_text=label_text,
                        label=label
                    )
                )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing line {idx}: {e}")
                continue
    
    print(f"Creating dataloader with {len(examples)} examples")
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

def create_verbalizer(verbalizer_type, tokenizer, classes, plm=None):
    if verbalizer_type == "manual":
        return ManualVerbalizer(
            tokenizer=tokenizer,
            classes=classes,
            label_words={"NON_VULNERABLE": ["no"], "VULNERABLE": ["yes"]}
        )
    elif verbalizer_type == "multi_manual":
        return ManualVerbalizer(
            tokenizer=tokenizer,
            classes=classes,
            label_words={"NON_VULNERABLE": ["no", "good", "correct"], 
                        "VULNERABLE": ["yes", "bad", "wrong"]}
        )
    elif verbalizer_type == "auto":
        verbalizer = AutomaticVerbalizer(
            tokenizer=tokenizer,
            classes=classes,
            num_candidates=1000,
            label_word_num_per_class=1,
            score_fct='llr',
            balance=True
        )
        return verbalizer
    elif verbalizer_type == "soft":
        return SoftVerbalizer(
            tokenizer=tokenizer,
            classes=classes,
            model=plm, 
            label_words={"NON_VULNERABLE": ["no"], "VULNERABLE": ["yes"]}
        )
    else:
        raise ValueError(f"Unknown verbalizer type: {verbalizer_type}")

myverbalizer = create_verbalizer(args.verbalizer_type, tokenizer, classes, plm)
print(f"Using {args.verbalizer_type} verbalizer")

prompt_model = PromptForClassification(
    plm=plm, 
    template=mytemplate, 
    verbalizer=myverbalizer, 
    freeze_plm=False 
)

if use_cuda:
    prompt_model = prompt_model.cuda()

def test(prompt_model, test_dataloader, name, threshold=0.5, save_results=True):
    prompt_model.eval()
    alllabels, alllogits = [], []
    sample_indices = []  
    token_map = {0: "no", 1: "yes"}
    sample_counter = 0  
    
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
            labels = [classes.index(label) for label in string_labels]
            
            alllabels.extend(labels)
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            alllogits.extend(probs)
            
            batch_size = len(labels)
            batch_indices = list(range(sample_counter, sample_counter + batch_size))
            sample_indices.extend(batch_indices)
            sample_counter += batch_size
            
    torch.cuda.empty_cache()
    
    allpreds = [1 if prob[1] >= threshold else 0 for prob in alllogits]
    allpred_tokens = [token_map[pred] for pred in allpreds]
    
    acc = accuracy_score(alllabels, allpreds)
    precisionwei, recallwei, f1wei, _ = precision_recall_fscore_support(alllabels, allpreds, average='weighted', zero_division=0)
    precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(alllabels, allpreds, average='binary', zero_division=0)
    mcc = matthews_corrcoef(alllabels, allpreds)
    
    metrics = {
        'accuracy': acc, 'precision': precision_binary, 'recall': recall_binary,
        'precision_weighted': precisionwei, 'recall_weighted': recallwei,
        'f1_binary': f1_binary, 'f1_weighted': f1wei, 'mcc': mcc
    }
    
    print(f"{name.capitalize()} Results (threshold={threshold:.3f}):")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    if save_results and name == 'test':
        os.makedirs(args.output_dir, exist_ok=True)
        
        with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
            f.write(f"Accuracy: {acc}\n")
            f.write(f"Binary Precision: {precision_binary}\n") 
            f.write(f"Binary Recall: {recall_binary}\n")
            f.write(f"Binary F1: {f1_binary}\n")
            f.write(f"Weighted F1: {f1wei}\n")
            f.write(f"MCC: {mcc}\n")
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=alllabels, y_pred=allpreds).ravel()
            f.write(f"TP: {tp}\n")
            f.write(f"TN: {tn}\n") 
            f.write(f"FP: {fp}\n")
            f.write(f"FN: {fn}\n")
        
        with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
            for idx, pred, prob in zip(sample_indices, allpreds, [prob[1] for prob in alllogits]):
                f.write(f"{idx}, {prob:.6f}, {pred}\n")
                
    probs_pos = [prob[1] for prob in alllogits]
    return acc, precision_binary, recall_binary, f1wei, f1_binary, alllabels, probs_pos

os.makedirs(args.output_dir, exist_ok=True)

if args.do_train:
    print("=== Starting Training ===")
    train_dataloader = create_dataloader(os.path.join(dataset_path, "train.jsonl"), "train", True)
    validation_dataloader = create_dataloader(os.path.join(dataset_path, "valid.jsonl"), "valid", True)
    
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
        
        acc, precision, recall, f1wei, f1_binary, _, _ = test(prompt_model, validation_dataloader, "valid", threshold=0.5, save_results=False)
        
        if f1_binary > bestmetric:
            bestmetric = f1_binary
            bestepoch = epoch
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(prompt_model.state_dict(), os.path.join(args.output_dir, "best-f1-model.bin"))
            print(f"New best F1-binary: {f1_binary:.4f}")
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= early_stop_threshold:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"Training completed. Best epoch: {bestepoch+1}, Best F1-binary: {bestmetric:.4f}")


if args.do_eval or args.do_test:
    from sklearn.metrics import precision_recall_curve

    print("\n=== Starting Post-Training Evaluation ===")

    model_path = os.path.join(args.output_dir, "best-f1-model.bin")
    if os.path.exists(model_path):
        print(f"Loading best model from: {model_path}")
        prompt_model.load_state_dict(torch.load(model_path), strict=False)
        print("Model loaded successfully")
    else:
        print(f"Warning: Model file {model_path} not found. Using current model weights.")

    print("Finding Best Threshold on Validation Set...")

    validation_dataloader = create_dataloader(os.path.join(dataset_path, "valid.jsonl"), "valid", False)
    _, _, _, _, _, val_labels, val_probs = test(prompt_model, validation_dataloader, "valid", save_results=False)
    
    precisions, recalls, thresholds = precision_recall_curve(val_labels, val_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8) 
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"Optimal Threshold (Validation): {best_threshold:.4f} (F1: {f1_scores[best_idx]:.4f})")


    if args.do_test:
        print("\n=== Testing with Fixed 0.5 and Best Thresholds ===")
        test_dataloader = create_dataloader(os.path.join(dataset_path, "test.jsonl"), "test", False)
        _, _, _, _, _, test_labels, test_probs = test(prompt_model, test_dataloader, "test", save_results=False)
        
        def calc_metrics(labels, probs, th):
            preds = [1 if p >= th else 0 for p in probs]
            acc = accuracy_score(labels, preds)
            p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
            f1_w = f1_score(labels, preds, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(labels, preds)
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(labels, preds).ravel()
            
            return (f"Threshold: {th:.4f}\nAccuracy: {acc:.4f}\n"
                    f"Binary Precision: {p:.4f}\nBinary Recall: {r:.4f}\n"
                    f"Binary F1: {f1:.4f}\nWeighted F1: {f1_w:.4f}\n"
                    f"MCC: {mcc:.4f}\n"
                    f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}\n")

      
        metrics_05 = calc_metrics(test_labels, test_probs, 0.5)
        metrics_best = calc_metrics(test_labels, test_probs, best_threshold)
        

        with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
            f.write("=== Results (Fixed Threshold = 0.5) ===\n")
            f.write(metrics_05)
            f.write(f"\n=== Results (Best Threshold = {best_threshold:.4f}) ===\n")
            f.write(metrics_best)
            
  
        best_preds = [1 if p >= best_threshold else 0 for p in test_probs]
        with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
            for idx, (pred, prob) in enumerate(zip(best_preds, test_probs)):
                f.write(f"{idx}, {prob:.6f}, {pred}\n")
                
        print(f"Testing completed. Full metrics saved to {args.output_dir}/metrics.txt")