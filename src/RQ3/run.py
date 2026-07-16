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
    parser = argparse.ArgumentParser(description="Generative Vulnerability Detection")
    
    # Model parameters
    parser.add_argument("--pretrainedmodel_path", type=str, default="pretrained_models/codebert-base", 
                        help="Path to pretrained CodeBERT model")
    parser.add_argument("--model_name", type=str, default="roberta", help="Model name")
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=6e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--early_stop_threshold", type=int, default=4, help="Early stopping threshold")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
    
    
    # Data parameters
    parser.add_argument("--dataset", type=str, default="reveal", help="Dataset name (devign/bigvul/reveal)")
    parser.add_argument("--data_dir", type=str, default="datasets", help="Directory containing datasets")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    # Control parameters
    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation on validation set")
    parser.add_argument("--do_test", action="store_true", help="Whether to run testing")
    
    # Strategy parameters
    parser.add_argument("--architecture", type=str, required=True,
                        choices=["encoder", "encoder-decoder", "decoder"],
                        help="Model architecture")
    
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Configuration from arguments
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


from openprompt.data_utils import InputExample

# Architecture detection and model loading
from transformers import (AutoConfig, AutoModelForMaskedLM, AutoModelForCausalLM, 
                         AutoModelForSeq2SeqLM, AutoTokenizer)
from openprompt.plms import MLMTokenizerWrapper, LMTokenizerWrapper, T5TokenizerWrapper

def get_architecture_config(architecture):
    """Get model class and wrapper for specified architecture"""
    configs = {
        'encoder': {
            'model_class': AutoModelForMaskedLM,
            'wrapper_class': MLMTokenizerWrapper
        },
        'encoder-decoder': {
            'model_class': AutoModelForSeq2SeqLM,
            'wrapper_class': T5TokenizerWrapper
        },
        'decoder': {
            'model_class': AutoModelForCausalLM,
            'wrapper_class': LMTokenizerWrapper
        }
    }
    return configs.get(architecture)

# Load model based on architecture
architecture = args.architecture
config = get_architecture_config(architecture)
if not config:
    raise ValueError(f"Unknown architecture: {architecture}")
plm = config['model_class'].from_pretrained(pretrainedmodel_path)
WrapperClass = config['wrapper_class']

# Enable gradient checkpointing if specified
if args.gradient_checkpointing:
    if hasattr(plm, 'gradient_checkpointing_enable'):
        plm.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    else:
        print("Warning: Model does not support gradient checkpointing")

tokenizer = AutoTokenizer.from_pretrained(pretrainedmodel_path, use_fast=False)

# Ensure tokenizer has pad_token (required for decoder models in batch processing)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    else:
        # Add new pad token
        num_added = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if num_added > 0:
            # Resize model embeddings to match new vocab size
            plm.resize_token_embeddings(len(tokenizer))
        print("Added new pad_token: [PAD]")

# Unified template for all architectures
template_text =  'Question: Is this code vulnerable? Code: {"placeholder":"text_a"} {"soft":"Answer:"} {"mask"}'

# Construct template 
from openprompt.prompts import MixedTemplate

mytemplate = MixedTemplate(tokenizer=tokenizer, text=template_text, model=plm)

from openprompt import PromptDataLoader

# Dataset configuration
DATASET_CONFIG = {
    'devign': {'code_field': 'func', 'label_field': 'target'},
    'bigvul': {'code_field': 'func', 'label_field': 'target'},
    'reveal': {'code_field': 'functionSource', 'label_field': 'label'}
}

def create_dataloader(dataset_path, split_name, shuffle_flag=True):
    print(f"Processing data from {dataset_path}")
    examples = []
    
    # Get configuration based on dataset name
    config = DATASET_CONFIG.get(args.dataset)
    if not config:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                
                code_text = data[config['code_field']]
                label = int(data[config['label_field']])
                
                # Unified binary labels: 0->no, 1->yes (vulnerable)
                label_text = "yes" if label == 1 else "no"
                
                simple_code = code_text
                
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
    # Strategy-specific dataloader parameters
    dataloader_kwargs = {
        'dataset': examples,
        'template': mytemplate, 
        'tokenizer': tokenizer,
        'tokenizer_wrapper_class': WrapperClass, 
        'max_seq_length': max_seq_l,
        'batch_size': batch_size, 
        'shuffle': shuffle_flag,
        'truncate_method': "head"
    }
    
    # Add architecture-specific parameters
    if architecture == 'encoder-decoder':
        dataloader_kwargs.update({
            'teacher_forcing': False,
            'predict_eos_token': False,
            'decoder_max_length': 10
        })
    elif architecture == 'decoder':
        dataloader_kwargs.update({
            'teacher_forcing': False,
            'predict_eos_token': False,
            'decoder_max_length': 10
        })
    else:  # encoder
        dataloader_kwargs.update({
            'teacher_forcing': False,
            'predict_eos_token': False
        })
    
    dataloader = PromptDataLoader(**dataloader_kwargs)
    
    return dataloader

# Load data - auto-detect dataset format
dataset_path = os.path.join(args.data_dir, args.dataset)
train_dataloader = create_dataloader(os.path.join(dataset_path, "train.jsonl"), "train", True)
validation_dataloader = create_dataloader(os.path.join(dataset_path, "valid.jsonl"), "valid", True)  
test_dataloader = create_dataloader(os.path.join(dataset_path, "test.jsonl"), "test", False)

# Unified verbalizer for binary vulnerability classification
from openprompt.prompts import ManualVerbalizer

# All strategies use unified yes/no verbalizer
verbalizer_classes = ["no", "yes"]  # 0->no (not vulnerable), 1->yes (vulnerable)
label_words = {
    "no": ["no"],        # Not vulnerable  
    "yes": ["yes"]       # Vulnerable
}

myverbalizer = ManualVerbalizer(
    tokenizer, 
    classes=verbalizer_classes,
    label_words=label_words
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
    token_map = {0: "no", 1: "yes"}
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
            # Convert unified string labels back to binary integers
            labels = [1 if label == "yes" else 0 for label in string_labels]
            
            alllabels.extend(labels)
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            alllogits.extend(probs)
            
            # 记录当前batch的样本索引
            batch_size = len(labels)
            batch_indices = list(range(sample_counter, sample_counter + batch_size))
            sample_indices.extend(batch_indices)
            sample_counter += batch_size
            
    
    # Use threshold for prediction
    allpreds = [1 if prob[1] >= threshold else 0 for prob in alllogits]
    allpred_tokens = [token_map[pred] for pred in allpreds]
    
    # Calculate metrics
    acc = accuracy_score(alllabels, allpreds)
    precisionwei, recallwei, f1wei, _ = precision_recall_fscore_support(alllabels, allpreds, average='weighted', zero_division=0)
    # Use binary average for F1 to match classify script
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
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Binary Precision: {precision_binary:.4f}\n") 
            f.write(f"Binary Recall: {recall_binary:.4f}\n")
            f.write(f"Binary F1: {f1_binary:.4f}\n")
            f.write(f"Weighted F1: {f1wei:.4f}\n")
            f.write(f"MCC: {mcc:.4f}\n")
            # 计算混淆矩阵
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=alllabels, y_pred=allpreds).ravel()
            f.write(f"TP: {tp}\n")
            f.write(f"TN: {tn}\n") 
            f.write(f"FP: {fp}\n")
            f.write(f"FN: {fn}\n")
        
        
        with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
            for idx, pred, prob in zip(sample_indices, allpreds, [prob[1] for prob in alllogits]):
                f.write(f"{idx}, {prob:.6f}, {pred}\n")
        
         
 
    
    return acc, precision_binary, recall_binary, f1wei, f1_binary

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
            
            # Convert unified string labels back to integers for loss calculation
            string_labels = inputs['tgt_text']
            labels = torch.tensor([1 if label == "yes" else 0 for label in string_labels])
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
        
        acc, precision, recall, f1wei, f1_binary = test(prompt_model, validation_dataloader, "valid", threshold=0.5, save_results=False)
        
        # Save best model
        if f1_binary > bestmetric:
            bestmetric = f1_binary
            bestepoch = epoch
            torch.save(prompt_model.state_dict(), os.path.join(args.output_dir, "best-f1-model.bin"))
            print(f"New best F1-binary: {f1_binary:.4f}")
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= early_stop_threshold:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"Training completed. Best epoch: {bestepoch+1}, Best F1-binary: {bestmetric:.4f}")

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
    acc, precision, recall, f1wei, f1_binary = test(prompt_model, validation_dataloader, "valid", threshold=0.5, save_results=False)
    print(f"Validation completed. F1-binary: {f1_binary:.4f}")

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
    acc, precision, recall, f1wei, f1_binary = test(prompt_model, test_dataloader, "test", threshold=0.5, save_results=True)
    print(f"Test completed. F1-binary: {f1_binary:.4f}")

