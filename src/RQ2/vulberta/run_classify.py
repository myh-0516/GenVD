#!/usr/bin/env python3

import numpy as np
import re
import torch
import sklearn
import os
import random
import argparse
import logging
import json
import warnings
import pickle
import hashlib
import math
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')


# 自定义模块导入
# import custom
# import models
import clang
from clang import *
from clang import cindex

# Tokenizer相关导入
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import NormalizedString, PreTokenizedString
from tokenizers import Tokenizer
from tokenizers import normalizers, decoders
from tokenizers.normalizers import StripAccents, unicode_normalizer_from_str, Replace
from tokenizers.processors import TemplateProcessing
from tokenizers import processors, pre_tokenizers
from tokenizers.models import BPE
from typing import List

# PyTorch和Transformers相关导入
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM, RobertaForSequenceClassification
from transformers import RobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
from transformers.modeling_outputs import SequenceClassifierOutput


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment(seed=42):
    """Setup environment"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Set random seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Disable wandb and transformers logging
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['WANDB_MODE'] = 'dryrun'
    
    # Set transformers logging to error only
    import transformers
    transformers.logging.set_verbosity_error()
    
    # Also suppress training info logs
    import logging
    logging.getLogger("transformers.trainer").setLevel(logging.ERROR)
    logging.getLogger("transformers.training_args").setLevel(logging.ERROR)
    logging.getLogger("transformers.trainer_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.trainer_callback").setLevel(logging.ERROR)
    
    # 完全禁用transformers的进度条和日志
    import transformers
    transformers.logging.disable_progress_bar()
    transformers.logging.set_verbosity_error()
    
    return device


class MyTokenizer:
    """Custom tokenizer using Clang for C/C++ code"""
    
    cidx = cindex.Index.create()
    
    def clang_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        """Split code using Clang"""
        tok = []
        tu = self.cidx.parse('tmp.c',
                       args=[''],  
                       unsaved_files=[('tmp.c', str(normalized_string.original))],  
                       options=0)
        for t in tu.get_tokens(extent=tu.cursor.extent):
            spelling = t.spelling.strip()
            
            if spelling == '':
                continue
                
            tok.append(NormalizedString(spelling))

        return tok
    
    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.clang_split)


def create_tokenizer():
    """Create custom tokenizer"""
    logger.info("Loading tokenizer...")
    
    # Load pretrained tokenizer
    vocab, merges = BPE.read_file(vocab="../pretrained_models/vulberta/drapgh-vocab.json", 
                                  merges="../pretrained_models/vulberta/drapgh-merges.txt")
    my_tokenizer = Tokenizer(BPE(vocab, merges, unk_token="<unk>"))

    my_tokenizer.normalizer = normalizers.Sequence([StripAccents(), Replace(" ", "Ä")])
    my_tokenizer.pre_tokenizer = PreTokenizer.custom(MyTokenizer())
    my_tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    my_tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
        ("<s>", 0),
        ("<pad>", 1),
        ("</s>", 2),
        ("<unk>", 3),
        ("<mask>", 4)
        ]
    )
    
    # Set truncation and padding 
    my_tokenizer.enable_truncation(max_length=512)  
    my_tokenizer.enable_padding(direction='right', pad_id=1, pad_type_id=0, 
                               pad_token='<pad>', length=None, pad_to_multiple_of=None)
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    logger.info("Tokenizer loaded")
    return my_tokenizer


def process_encodings(encodings):
    """Process tokenization results"""
    input_ids = []
    attention_mask = []
    for enc in encodings:
        input_ids.append(enc.ids)
        attention_mask.append(enc.attention_mask)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}


def get_cache_path(file_path, tokenizer_info="vulberta"):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    cache_filename = f"{base_name}.pkl"
    
    cache_dir = os.path.join(os.path.dirname(file_path), "tokenizer_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, cache_filename)

def load_cache(cache_path):
    if os.path.exists(cache_path):
        try:
            logger.info(f"Loading from cache: {os.path.basename(cache_path)}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    return None

def save_cache(data, cache_path):
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved tokenization cache: {os.path.basename(cache_path)}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

def cleaner(code):
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat, '', code)
    code = re.sub('\n', '', code)
    code = re.sub('\t', '', code)
    return code


class MyCustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        assert len(self.encodings['input_ids']) == len(self.encodings['attention_mask']) == len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def map_dataset_fields(data_item):
    code = data_item.get('func') or data_item.get('functionSource', '')
    label = data_item.get('target', data_item.get('label', 0))
    idx = data_item.get('idx', data_item.get('hash', 0))
    
    return {'code': code, 'label': label, 'idx': idx}


def load_dataset_jsonl(file_path, tokenizer):
    cache_path = get_cache_path(file_path)
    cached_dataset = load_cache(cache_path)
    if cached_dataset is not None:
        return cached_dataset
    
    logger.info(f"Loading {os.path.basename(file_path)}...")
    
    data = []
    labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            mapped_item = map_dataset_fields(item)
            clean_code = cleaner(mapped_item['code'])
            data.append(clean_code)
            labels.append(mapped_item['label'])
    
    logger.info(f"Read {len(data)} samples, starting tokenization...")
    
    batch_size = 1000 
    all_encodings = []
    
    total_batches = (len(data) + batch_size - 1) // batch_size
    with tqdm(total=total_batches, desc="Tokenizing", unit="batch") as pbar:
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            batch_encodings = tokenizer.encode_batch(batch_data)
            all_encodings.extend(batch_encodings)
            pbar.update(1)
    
    encodings = process_encodings(all_encodings)
    logger.info(f"Tokenization complete: {len(labels)} samples")
    
    dataset = MyCustomDataset(encodings, labels)
    save_cache(dataset, cache_path)
    
    return dataset


def load_data(dataset_name, mode, tokenizer):
    logger.info(f"Loading dataset: {dataset_name}, mode: {mode}")
    
    base_path = f"../datasets/{dataset_name}"
    
    if mode == 'train':
        train_path = f"{base_path}/train.jsonl"
        val_path = f"{base_path}/valid.jsonl"
        
        train_dataset = load_dataset_jsonl(train_path, tokenizer)
        val_dataset = load_dataset_jsonl(val_path, tokenizer)
        
        return train_dataset, val_dataset
        
    elif mode == 'evaluate':
        test_path = f"{base_path}/test.jsonl"
        return load_dataset_jsonl(test_path, tokenizer)
        
    else:
        raise ValueError(f"Unsupported mode: {mode}")




# Global epoch tracking
CURRENT_EPOCH = 1
TOTAL_EPOCHS = 3

def compute_metrics_func(eval_pred):
    global CURRENT_EPOCH
    
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = predictions.argmax(-1)
    
    precision = sklearn.metrics.precision_score(labels, preds, average='binary')
    recall = sklearn.metrics.recall_score(labels, preds, average='binary')
    f1 = sklearn.metrics.f1_score(labels, preds, average='binary')
    accuracy = sklearn.metrics.accuracy_score(labels, preds)
    
    print(f"\nEpoch {CURRENT_EPOCH}: ACC={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}\n", flush=True)
    
    CURRENT_EPOCH += 1
    
    return {
        'eval_accuracy': accuracy,
        'eval_precision': precision,
        'eval_recall': recall,
        'eval_f1': f1,
    }


class MyTrainer(Trainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_f1 = 0.0
        self.best_model_state = None
    
    def get_train_dataloader(self):
        dataloader = super().get_train_dataloader()
        
        class TrainDataLoader:
            def __init__(self, dataloader):
                self.dataloader = dataloader
            
            def __iter__(self):
                global CURRENT_EPOCH, TOTAL_EPOCHS
                pbar = tqdm(self.dataloader, 
                           desc=f"Epoch {CURRENT_EPOCH}/{TOTAL_EPOCHS} - Training", 
                           unit="batch", leave=True)
                for batch in pbar:
                    yield batch
                pbar.close()
            
            def __len__(self):
                return len(self.dataloader)
        
        return TrainDataLoader(dataloader)
    
    def get_eval_dataloader(self, eval_dataset=None):
        dataloader = super().get_eval_dataloader(eval_dataset)
        
        class EvalDataLoader:
            def __init__(self, dataloader):
                self.dataloader = dataloader
            
            def __iter__(self):
                pbar = tqdm(self.dataloader, desc="Evaluating", unit="batch", leave=True)
                for batch in pbar:
                    yield batch
                pbar.close()
            
            def __len__(self):
                return len(self.dataloader)
        
        return EvalDataLoader(dataloader)
    
    def log(self, logs):
        pass
    
    def get_best_model_state(self):
        return self.best_model_state


def train_model(dataset_name, train_dataset, val_dataset, output_dir, batch_size=4, epochs=10, learning_rate=3e-5, seed=42, fp16=False):
    global CURRENT_EPOCH, TOTAL_EPOCHS
    
    CURRENT_EPOCH = 1
    TOTAL_EPOCHS = epochs
    
    logger.info(f"Training {dataset_name} - {len(train_dataset)} train, {len(val_dataset)} val")
    logger.info(f"Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}, FP16: {'ON' if fp16 else 'OFF'}")
    
    model = RobertaForSequenceClassification.from_pretrained('../pretrained_models/vulberta/')
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,  
        num_train_epochs=epochs,
        evaluation_strategy='epoch',
        save_strategy='no',
        learning_rate=learning_rate,
        fp16=fp16,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        load_best_model_at_end=False,
        logging_steps=100000,
        report_to=[],
        disable_tqdm=True,
        logging_first_step=False,
        log_level='error',
        logging_dir=None,
        logging_strategy='no',
        push_to_hub=False,
        prediction_loss_only=False,
        include_inputs_for_metrics=True,
        greater_is_better=True,
        metric_for_best_model="eval_f1",
    )
    
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_func
    )
    
    print("Training started...")
    trainer.train()
    
    eval_results = {'status': 'completed'}
    
    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("=== Training Completed ===\n")
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Training metrics saved to: {os.path.basename(metrics_path)}")
    
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "checkpoint-best-f1.bin")
    
    best_model_state = trainer.get_best_model_state()
    if best_model_state is not None:
        torch.save(best_model_state, best_model_path)
        print(f"Best F1 model weights (F1={trainer.best_f1:.4f}) saved to: {os.path.basename(best_model_path)}")
    else:
        torch.save(trainer.model.state_dict(), best_model_path)
        print(f"Current model weights saved to: {os.path.basename(best_model_path)}")
    
    return trainer


def softmax_accuracy(probs, all_labels):
    all_labels = all_labels.tolist()
    probs_list = probs.tolist()
    all_predicted = [x.index(max(x)) for x in probs_list]
    
    correct = sum(1 for pred, label in zip(all_predicted, all_labels) if pred == label)
    acc = correct / len(all_labels)
    
    return acc, all_predicted


def evaluate_model(dataset_name, test_dataset, device, output_dir=None, metrics_file="metrics.txt"):
    """Evaluate model"""
    logger.info(f"Evaluating model - dataset: {dataset_name}")
    
    # Load model with best F1 weights
    best_model_path = os.path.join(output_dir, "checkpoint-best-f1.bin")
    
    if os.path.exists(best_model_path):
        # Load pretrained model first, then load fine-tuned weights
        model = RobertaForSequenceClassification.from_pretrained('../pretrained_models/vulberta/')
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best F1 model weights from: {os.path.basename(best_model_path)}")
    else:
        # Fallback to loading from pretrained models directory
        model = RobertaForSequenceClassification.from_pretrained('../pretrained_models/vulberta/')
        print(f"Warning: No fine-tuned weights found, using base pretrained model")
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    # Start inference with progress bar
    test_pbar = tqdm(test_loader, desc="Testing", unit="batch")
    
    all_pred = []
    all_labels = []
    all_probs = []
    model.eval()
    
    with torch.no_grad():
        for batch in test_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            probs = torch.nn.functional.softmax(outputs[1], dim=1)
            acc_val, pred = softmax_accuracy(probs, labels)
            all_pred += pred
            all_labels += labels.tolist()
            all_probs += probs.tolist()
    
    test_pbar.close()
    
    # Calculate metrics
    confusion = sklearn.metrics.confusion_matrix(y_true=all_labels, y_pred=all_pred)
    tn, fp, fn, tp = confusion.ravel()

    probs2 = [x[1] for x in all_probs]

    # Performance metrics
    accuracy = sklearn.metrics.accuracy_score(y_true=all_labels, y_pred=all_pred)
    precision = sklearn.metrics.precision_score(y_true=all_labels, y_pred=all_pred, average='binary')
    recall = sklearn.metrics.recall_score(y_true=all_labels, y_pred=all_pred, average='binary')
    f1 = sklearn.metrics.f1_score(y_true=all_labels, y_pred=all_pred, average='binary')
    pr_auc = sklearn.metrics.average_precision_score(y_true=all_labels, y_score=probs2)
    auc = sklearn.metrics.roc_auc_score(y_true=all_labels, y_score=probs2)
    mcc = sklearn.metrics.matthews_corrcoef(y_true=all_labels, y_pred=all_pred)
    
    print(f'Test Results: ACC={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}')
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': pr_auc,
        'auc': auc,
        'mcc': mcc,
        'confusion_matrix': confusion.tolist()
    }
    
    # Save metrics
    metrics_path = os.path.join(output_dir, metrics_file)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"Confusion Matrix: {confusion.tolist()}\n")
        f.write(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n")
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='VulBERTa training and evaluation')
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True,
                       help='Mode: train or evaluate')
    parser.add_argument('--dataset', required=True,
                       help='Dataset name (e.g., devign, reveal, bigvul)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Training hyperparameters
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Model output directory')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                       help='Learning rate (default: 3e-5)')
    parser.add_argument('--fp16', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--metrics_file', type=str, default='metrics.txt',
                       help='Metrics output file (default: metrics.txt)')
    
    args = parser.parse_args()
    
    # Setup environment
    device = setup_environment(args.seed)
    
    # Create tokenizer
    tokenizer = create_tokenizer()
    
    if args.mode == 'train':
        # Load training data
        train_dataset, val_dataset = load_data(args.dataset, 'train', tokenizer)
        
        # Train model
        train_model(args.dataset, train_dataset, val_dataset, 
                   args.output_dir, args.batch_size, args.epochs, 
                   args.learning_rate, args.seed, args.fp16)
        
    elif args.mode == 'evaluate':
        # Load test data
        test_dataset = load_data(args.dataset, 'evaluate', tokenizer)
        
        # Evaluate model
        evaluate_model(args.dataset, test_dataset, device, args.output_dir, args.metrics_file)


if __name__ == "__main__":
    main()
