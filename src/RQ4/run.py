import os
import torch
import json
import transformers
import numpy as np
# pandas removed - no longer needed for CSV export
import random
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, matthews_corrcoef
import sklearn.metrics
from tqdm.auto import tqdm

# Set random seeds for reproducibility
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
    
    # Model parameters
    parser.add_argument("--pretrainedmodel_path", type=str, default="pretrained_models/codebert-base", 
                        help="Path to pretrained CodeBERT model")
    parser.add_argument("--model_name", type=str, default="roberta", help="Model name")
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--early_stop_threshold", type=int, default=4, help="Early stopping threshold")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    
    
    # Data parameters
    parser.add_argument("--dataset", type=str, default="reveal", help="Dataset name (devign/bigvul/reveal)")
    parser.add_argument("--data_dir", type=str, default="datasets", help="Directory containing datasets")
    parser.add_argument("--max_code_words", type=int, default=400, help="Maximum number of words in code")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    # Control parameters
    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation on validation set")
    parser.add_argument("--do_test", action="store_true", help="Whether to run testing")
    
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Configuration from arguments
seed = args.seed
set_seed(seed)
batch_size = args.batch_size
num_class = 10
max_seq_l = args.max_seq_length
lr = args.learning_rate
num_epochs = args.num_epochs
use_cuda = torch.cuda.is_available()
pretrainedmodel_path = args.pretrainedmodel_path
early_stop_threshold = args.early_stop_threshold


from openprompt.data_utils import InputExample

# 10 CWE classes based on top10cwe dataset
classes = ["CWE-119", "CWE-125", "CWE-189", "CWE-190", "CWE-20", 
           "CWE-200", "CWE-264", "CWE-362", "CWE-399", "CWE-416"]

# Load model as MaskedLM for generative detection
from transformers import AutoModelForMaskedLM, AutoTokenizer
from openprompt.plms import MLMTokenizerWrapper

# Load model for masked language modeling
plm = AutoModelForMaskedLM.from_pretrained(pretrainedmodel_path)
tokenizer = AutoTokenizer.from_pretrained(pretrainedmodel_path, use_fast=False)
WrapperClass = MLMTokenizerWrapper
model_config = plm.config


# Construct template for generative detection
from openprompt.prompts import MixedTemplate

# Template for CWE classification
template_text = 'Question: What type of vulnerability is this? Code: {"placeholder":"text_a"} {"soft":"Answer:"} {"mask"}'

mytemplate = MixedTemplate(tokenizer=tokenizer, text=template_text, model=plm)

from openprompt import PromptDataLoader

def create_dataloader(dataset_path, split_name, shuffle_flag=True):
    print(f"Processing data from {dataset_path}")
    examples = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                

                code_text = data['func']
                label_text = data['cwe_id']
                
                # Truncate code by words (more suitable for CodeBERT)
                simple_code = ' '.join(code_text.split(' ')[:args.max_code_words])
                
                examples.append(
                    InputExample(
                        guid=idx,
                        text_a=simple_code,
                        tgt_text=label_text,  # 字符串标签
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

# Load data
dataset_path = os.path.join(args.data_dir, args.dataset)
train_dataloader = create_dataloader(os.path.join(dataset_path, "train.jsonl"), "train", True)
validation_dataloader = create_dataloader(os.path.join(dataset_path, "valid.jsonl"), "valid", True)  
test_dataloader = create_dataloader(os.path.join(dataset_path, "test.jsonl"), "test", False)

# Define ManualVerbalizer for CWE classification
from openprompt.prompts import ManualVerbalizer

# Use ManualVerbalizer - manually specify label words for each CWE class
myverbalizer = ManualVerbalizer(
    tokenizer=tokenizer,
    classes=classes,
    label_words={
        "CWE-119": ["119"],          # Buffer Overflow
        "CWE-125": ["125"],          # Out-of-bounds Read
        "CWE-189": ["189"],          # Numeric Errors
        "CWE-190": ["190"],          # Integer Overflow or Wraparound
        "CWE-20": ["20"],            # Improper Input Validation
        "CWE-200": ["200"],          # Information Exposure
        "CWE-264": ["264"],          # Permissions, Privileges, and Access Controls
        "CWE-362": ["362"],          # Race Condition
        "CWE-399": ["399"],          # Resource Management Errors
        "CWE-416": ["416"]           # Use After Free
    }
)

# Create prompt model for generative classification
from openprompt import PromptForClassification

prompt_model = PromptForClassification(
    plm=plm, 
    template=mytemplate, 
    verbalizer=myverbalizer, 
    freeze_plm=False  # Fine-tune
)

if use_cuda:
    prompt_model = prompt_model.cuda()

# Optimizer and scheduler setup
from transformers import AdamW, get_linear_schedule_with_warmup

# Standard Cross Entropy Loss
import torch.nn as nn

loss_func = nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']

# Optimizer for different parameter groups
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

# Memory optimization
torch.cuda.empty_cache()

# Mixed precision setup
scaler = torch.cuda.amp.GradScaler() if args.fp16 and use_cuda else None

# Using fixed threshold 0.5 - no threshold tuning needed

def test(prompt_model, test_dataloader, name, threshold=0.5, save_results=True):
    prompt_model.eval()
    torch.cuda.empty_cache()
    
    alllabels, alllogits = [], []
    sample_indices = []  
    sample_counter = 0  
    
    with torch.no_grad():
        progress_bar = tqdm(test_dataloader, desc=f"{name.capitalize()} inference")
        for inputs in progress_bar:
            if use_cuda:
                inputs = inputs.cuda()
            
            # Mixed precision inference
            with torch.cuda.amp.autocast(enabled=args.fp16 and use_cuda):
                logits = prompt_model(inputs)
            
            string_labels = inputs['tgt_text']
            # Use string labels directly
            alllabels.extend(string_labels)
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            alllogits.extend(probs)
            
            # 记录当前batch的样本索引
            batch_size = len(string_labels)
            batch_indices = list(range(sample_counter, sample_counter + batch_size))
            sample_indices.extend(batch_indices)
            sample_counter += batch_size
            
    
    # Use argmax for multi-class prediction  
    allpreds = [classes[np.argmax(prob)] for prob in alllogits]
    
    # Calculate multi-class metrics
    acc = accuracy_score(alllabels, allpreds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(alllabels, allpreds, average='macro', zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(alllabels, allpreds, average='micro', zero_division=0)
    precision_wei, recall_wei, f1_wei, _ = precision_recall_fscore_support(alllabels, allpreds, average='weighted', zero_division=0)
    
    # Calculate Matthews Correlation Coefficient
    mcc = matthews_corrcoef(alllabels, allpreds)
    
    # Calculate Top-K Accuracy
    def calculate_topk_accuracy(y_true, y_prob, classes, k):
        correct = sum(1 for true_label, probs in zip(y_true, y_prob)
                     if true_label in [classes[i] for i in np.argsort(probs)[-k:]])
        return correct / len(y_true)
    
    top1_acc = calculate_topk_accuracy(alllabels, alllogits, classes, 1)
    top2_acc = calculate_topk_accuracy(alllabels, alllogits, classes, 2)
    top3_acc = calculate_topk_accuracy(alllabels, alllogits, classes, 3)
    top4_acc = calculate_topk_accuracy(alllabels, alllogits, classes, 4)
    top5_acc = calculate_topk_accuracy(alllabels, alllogits, classes, 5)
    
    metrics = {
        'accuracy': acc, 
        'precision_macro': precision_macro, 'recall_macro': recall_macro, 'f1_macro': f1_macro,
        'precision_micro': precision_micro, 'recall_micro': recall_micro, 'f1_micro': f1_micro,
        'precision_weighted': precision_wei, 'recall_weighted': recall_wei, 'f1_weighted': f1_wei,
        'mcc': mcc,
        'top1_accuracy': top1_acc,
        'top2_accuracy': top2_acc,
        'top3_accuracy': top3_acc,
        'top4_accuracy': top4_acc,
        'top5_accuracy': top5_acc
    }
    
    print(f"\n=== {name.capitalize()} Results ===")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(f"  Macro Precision: {precision_macro:.4f}")
    print(f"  Macro Recall: {recall_macro:.4f}")
    print(f"  MCC: {mcc:.4f}")
    
    # Detailed metrics only saved to file, not printed to console
    
    if save_results and name == 'test':
        os.makedirs(args.output_dir, exist_ok=True)
        
        with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
            # Overall metrics (all summary metrics for the multi-class classification task)
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
            
            # Top-K Accuracy
            f.write("=== Top-K Accuracy ===\n")
            f.write(f"Top-1 Accuracy: {top1_acc:.4f}\n")
            f.write(f"Top-2 Accuracy: {top2_acc:.4f}\n")
            f.write(f"Top-3 Accuracy: {top3_acc:.4f}\n")
            f.write(f"Top-4 Accuracy: {top4_acc:.4f}\n")
            f.write(f"Top-5 Accuracy: {top5_acc:.4f}\n\n")
            
            # Confusion matrix
            f.write("=== Confusion Matrix ===\n")
            cm = sklearn.metrics.confusion_matrix(y_true=alllabels, y_pred=allpreds)
            f.write(f"Classes: {classes}\n")
            f.write(f"{cm}\n")
        
        with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
            for idx, pred, probs in zip(sample_indices, allpreds, alllogits):
                max_prob = max(probs)
                f.write(f"{idx}, {max_prob:.6f}, {pred}\n")
        
         
 
    
    return acc, precision_macro, recall_macro, f1_wei, f1_macro

# Main execution
os.makedirs(args.output_dir, exist_ok=True)

if args.do_train:
    print("=== Starting Training ===")
    bestmetric, bestepoch, early_stop_count = 0, 0, 0

    for epoch in range(num_epochs):

        prompt_model.train()
        tot_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Training")
        
        for step, inputs in enumerate(progress_bar):
            if use_cuda:
                inputs = inputs.cuda()
            
            # Convert string labels to integers for loss calculation
            string_labels = inputs['tgt_text']
            labels = torch.tensor([classes.index(label) for label in string_labels])
            if use_cuda:
                labels = labels.cuda()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=args.fp16 and use_cuda):
                logits = prompt_model(inputs)
                loss = loss_func(logits, labels)
            
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
            tot_loss += loss.item()
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = tot_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        torch.cuda.empty_cache()
        
        acc, precision, recall, f1wei, f1_macro = test(prompt_model, validation_dataloader, "valid", save_results=False)
        
        # Save best model based on macro F1
        if f1_macro > bestmetric:
            bestmetric = f1_macro
            bestepoch = epoch
            torch.save(prompt_model.state_dict(), os.path.join(args.output_dir, "best-f1-model.bin"))
            print(f"New best F1-macro: {f1_macro:.4f}")
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= early_stop_threshold:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"Training completed. Best epoch: {bestepoch+1}, Best F1-macro: {bestmetric:.4f}")

if args.do_eval:
    print("=== Starting Validation ===")
    
    # Load best model for evaluation
    model_path = os.path.join(args.output_dir, "best-f1-model.bin")
    if os.path.exists(model_path):
        print(f"Loading best model from: {model_path}")
        prompt_model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully")
    else:
        print(f"Warning: Model file {model_path} not found. Using current model weights.")
    
    # Run validation
    acc, precision, recall, f1wei, f1_macro = test(prompt_model, validation_dataloader, "valid", save_results=False)
    print(f"Validation completed. F1-macro: {f1_macro:.4f}")

if args.do_test:
    print("=== Starting Testing ===")
    
    # Load best model for testing
    model_path = os.path.join(args.output_dir, "best-f1-model.bin")
    if os.path.exists(model_path):
        print(f"Loading best model from: {model_path}")
        prompt_model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully")
    else:
        print(f"Warning: Model file {model_path} not found. Using current model weights.")
    
    # Run test
    acc, precision, recall, f1wei, f1_macro = test(prompt_model, test_dataloader, "test", save_results=True)
    print(f"Test completed. F1-macro: {f1_macro:.4f}")

