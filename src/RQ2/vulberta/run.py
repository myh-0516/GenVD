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
from transformers.modeling_outputs import SequenceClassifierOutput, MaskedLMOutput
import torch.nn as nn
import torch.nn.functional as F


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


def create_tokenizer(max_seq_length=512):
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
    
    # Set truncation and padding with configurable max_length
    my_tokenizer.enable_truncation(max_length=max_seq_length)  
    my_tokenizer.enable_padding(direction='right', pad_id=1, pad_type_id=0, 
                               pad_token='<pad>', length=None, pad_to_multiple_of=None)
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    logger.info("Tokenizer loaded")
    return my_tokenizer


class RobertaMLMAdapter(nn.Module):
    """Adapter to convert RobertaForSequenceClassification to MLM model with soft prompt support"""
    
    def __init__(self, base_model_path, vocab_size=50265, soft_prompt_length=1):
        super().__init__()
        # Load the classification model and extract RoBERTa backbone
        classifier = RobertaForSequenceClassification.from_pretrained(base_model_path)
        self.roberta = classifier.roberta
        
        # Add MLM head
        self.lm_head = nn.Linear(self.roberta.config.hidden_size, vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(self.roberta.embeddings.word_embeddings.weight.clone())
        
        # Soft prompt support - corresponds to {"soft":"Answer:"} in generic script
        self.soft_prompt_length = soft_prompt_length
        if soft_prompt_length > 0:
            self.soft_prompt = nn.Parameter(
                torch.randn(soft_prompt_length, self.roberta.config.hidden_size) * 0.01
            )
        
        # Verbalizer mapping: 0->no, 1->yes (CORRECTED IDs with space prefix)
        # Space prefix becomes Ä after normalization: " no" -> "Äno", " yes" -> "Äyes"
        self.verbalizer = {"no": 7352, "yes": 44515}  # Correct Token IDs for VulBERTa vocab with space prefix
        
    def forward(self, input_ids, attention_mask=None, labels=None, mask_positions=None):
        batch_size = input_ids.size(0)
        
        # Soft prompt processing - insert before mask position like generic script
        if self.soft_prompt_length > 0 and mask_positions is not None:
            # Get word embeddings
            word_embeddings = self.roberta.embeddings.word_embeddings(input_ids)
            
            # Insert soft prompt before mask position (like "Answer:" in generic script)
            new_embeddings = []
            new_attention_masks = []
            adjusted_mask_positions = []
            
            for i in range(batch_size):
                orig_embeddings = word_embeddings[i]  # [seq_len, hidden_size]
                orig_attention = attention_mask[i]    # [seq_len]
                mask_pos = mask_positions[i].item()
                
                # Split at mask position: [before_mask, mask_token, after_mask]
                before_mask = orig_embeddings[:mask_pos]
                mask_and_after = orig_embeddings[mask_pos:]
                
                before_attention = orig_attention[:mask_pos]
                mask_and_after_attention = orig_attention[mask_pos:]
                
                # Insert soft prompt before mask
                soft_prompt = self.soft_prompt  # [soft_prompt_length, hidden_size]
                soft_attention = torch.ones(self.soft_prompt_length, device=input_ids.device)
                
                # Concatenate: [before_mask, soft_prompt, mask_and_after]
                new_embedding = torch.cat([before_mask, soft_prompt, mask_and_after], dim=0)
                new_attention = torch.cat([before_attention, soft_attention, mask_and_after_attention], dim=0)
                
                new_embeddings.append(new_embedding)
                new_attention_masks.append(new_attention)
                
                # Adjust mask position: mask_pos + soft_prompt_length
                adjusted_mask_positions.append(mask_pos + self.soft_prompt_length)
            
            # Pad to same length
            max_len = max(emb.size(0) for emb in new_embeddings)
            padded_embeddings = []
            padded_attention = []
            
            for i in range(batch_size):
                emb = new_embeddings[i]
                att = new_attention_masks[i]
                pad_len = max_len - emb.size(0)
                
                if pad_len > 0:
                    pad_emb = torch.zeros(pad_len, emb.size(1), device=emb.device)
                    pad_att = torch.zeros(pad_len, device=att.device)
                    emb = torch.cat([emb, pad_emb], dim=0)
                    att = torch.cat([att, pad_att], dim=0)
                
                padded_embeddings.append(emb)
                padded_attention.append(att)
            
            inputs_embeds = torch.stack(padded_embeddings, dim=0)
            extended_attention_mask = torch.stack(padded_attention, dim=0)
            adjusted_mask_positions = torch.tensor(adjusted_mask_positions, device=input_ids.device)
            
            # Soft prompt insertion completed
            outputs = self.roberta(inputs_embeds=inputs_embeds, attention_mask=extended_attention_mask)
        else:
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            adjusted_mask_positions = mask_positions if mask_positions is not None else None
        
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)
        
        if labels is not None and mask_positions is not None and adjusted_mask_positions is not None:
            # Extract predictions at mask positions (with soft prompt adjustment)
            max_seq_len = prediction_scores.size(1)
            valid_mask_positions = torch.clamp(adjusted_mask_positions, 0, max_seq_len - 1)
            
            mask_predictions = prediction_scores[torch.arange(prediction_scores.size(0)), valid_mask_positions]
            # Only consider verbalizer tokens
            verbalizer_logits = torch.stack([
                mask_predictions[:, self.verbalizer["no"]],
                mask_predictions[:, self.verbalizer["yes"]]
            ], dim=1)
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(verbalizer_logits, labels)
            return MaskedLMOutput(loss=loss, logits=verbalizer_logits)
        else:
            # For inference, also extract mask positions and return verbalizer logits
            if mask_positions is not None and adjusted_mask_positions is not None:
                max_seq_len = prediction_scores.size(1)
                valid_mask_positions = torch.clamp(adjusted_mask_positions, 0, max_seq_len - 1)
                
                mask_predictions = prediction_scores[torch.arange(prediction_scores.size(0)), valid_mask_positions]
                verbalizer_logits = torch.stack([
                    mask_predictions[:, self.verbalizer["no"]],
                    mask_predictions[:, self.verbalizer["yes"]]
                ], dim=1)
                return MaskedLMOutput(logits=verbalizer_logits)
            else:
                # Fallback: return full prediction scores (should be avoided)
                return MaskedLMOutput(logits=prediction_scores)


def process_encodings(encodings):
    """Process tokenization results"""
    input_ids = []
    attention_mask = []
    for enc in encodings:
        input_ids.append(enc.ids)
        attention_mask.append(enc.attention_mask)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}


def get_cache_path(file_path, generative=False, tokenizer_info="vulberta"):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    mode_prefix = "generate" if generative else "classify"
    cache_filename = f"{mode_prefix}_{base_name}.pkl"
    
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
    def __init__(self, encodings, labels, mask_positions=None):
        self.encodings = encodings
        self.labels = labels
        self.mask_positions = mask_positions
        assert len(self.encodings['input_ids']) == len(self.encodings['attention_mask']) == len(self.labels)
        if mask_positions is not None:
            assert len(self.mask_positions) == len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        if self.mask_positions is not None:
            item['mask_positions'] = torch.tensor(self.mask_positions[idx])
        return item

    def __len__(self):
        return len(self.labels)


def map_dataset_fields(data_item):
    code = data_item.get('func') or data_item.get('functionSource', '')
    label = data_item.get('target', data_item.get('label', 0))
    idx = data_item.get('idx', data_item.get('hash', 0))
    
    return {'code': code, 'label': label, 'idx': idx}


def load_dataset_jsonl(file_path, tokenizer, generative=False, max_code_words=450):
    cache_path = get_cache_path(file_path, generative)
    cached_dataset = load_cache(cache_path)
    if cached_dataset is not None:
        return cached_dataset
    
    logger.info(f"Loading {os.path.basename(file_path)}...")
    logger.info(f"Mode: {'Generative' if generative else 'Classification'}")
    logger.info(f"Max code words: {max_code_words}")
    
    data = []
    labels = []
    mask_positions = [] if generative else None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            mapped_item = map_dataset_fields(item)
            
            # Use the same preprocessing as generic script
            if generative and max_code_words is not None:
                # Pre-truncate by words EXACTLY like generic script
                code_words = mapped_item['code'].split(' ')
                if len(code_words) > max_code_words:
                    truncated_code = ' '.join(code_words[:max_code_words])
                else:
                    truncated_code = mapped_item['code']
                clean_code = truncated_code  # Keep original format, no cleaning
            else:
                # For classification mode, keep original behavior (clean comments)
                clean_code = cleaner(mapped_item['code'])
            
            if generative:
                # Use the same template as generic script: "Question: Is this code vulnerable? Code: {code} Answer: MASK"
                prompt_template = f"Question: Is this code vulnerable? Code: {clean_code} Answer: MASK"
                data.append(prompt_template)
            else:
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
    
    # Find mask positions for generative mode
    if generative:
        for enc in all_encodings:
            # Find MASK token (ID: 5523) - now at the end of sequence
            mask_pos = None
            for i in range(len(enc.ids) - 1, -1, -1):  # Search from end
                if enc.ids[i] == 5523:
                    mask_pos = i
                    break
            if mask_pos is None:
                mask_pos = len(enc.ids) - 1  # Fallback to last position
            mask_positions.append(mask_pos)
    
    encodings = process_encodings(all_encodings)
    logger.info(f"Tokenization complete: {len(labels)} samples")
    
    dataset = MyCustomDataset(encodings, labels, mask_positions)
    save_cache(dataset, cache_path)
    
    return dataset


def load_data(dataset_name, mode, tokenizer, generative=False, max_code_words=450):
    logger.info(f"Loading dataset: {dataset_name}, mode: {mode}")
    
    base_path = f"../datasets/{dataset_name}"
    
    if mode == 'train':
        train_path = f"{base_path}/train.jsonl"
        val_path = f"{base_path}/valid.jsonl"
        
        train_dataset = load_dataset_jsonl(train_path, tokenizer, generative, max_code_words)
        val_dataset = load_dataset_jsonl(val_path, tokenizer, generative, max_code_words)
        
        return train_dataset, val_dataset
        
    elif mode == 'evaluate':
        test_path = f"{base_path}/test.jsonl"
        return load_dataset_jsonl(test_path, tokenizer, generative, max_code_words)
        
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
    
    print(f"Epoch {CURRENT_EPOCH}: ACC={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}\n", flush=True)
    
    CURRENT_EPOCH += 1
    
    return {
        'eval_accuracy': accuracy,
        'eval_precision': precision,
        'eval_recall': recall,
        'eval_f1': f1,
    }


class MyTrainer(Trainer):
    
    def __init__(self, *args, early_stop_threshold=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_f1 = 0.0
        self.best_model_state = None
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_counter = 0
        self.should_stop = False
    
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
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to implement early stopping"""
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        current_f1 = eval_results.get('eval_f1', 0.0)
        
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.best_model_state = self.model.state_dict().copy()
            self.early_stop_counter = 0
            print(f"New best F1: {current_f1:.4f} (saved best model)")
        else:
            self.early_stop_counter += 1
            print(f"F1 did not improve. Early stop counter: {self.early_stop_counter}/{self.early_stop_threshold}")
            
            if self.early_stop_counter >= self.early_stop_threshold:
                self.should_stop = True
                print(f"Early stopping triggered after {self.early_stop_counter} epochs without improvement")
                # Stop training by setting current step to max steps
                self.state.global_step = self.state.max_steps
        
        return eval_results
    
    def get_best_model_state(self):
        return self.best_model_state


def train_model(dataset_name, train_dataset, val_dataset, output_dir, batch_size=4, epochs=10, learning_rate=3e-5, seed=42, fp16=False, generative=False, early_stop_threshold=3, soft_prompt_length=1):
    global CURRENT_EPOCH, TOTAL_EPOCHS
    
    CURRENT_EPOCH = 1
    TOTAL_EPOCHS = epochs
    
    logger.info(f"Training {dataset_name} - {len(train_dataset)} train, {len(val_dataset)} val")
    logger.info(f"Mode: {'Generative' if generative else 'Classification'}")
    logger.info(f"Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}, FP16: {'ON' if fp16 else 'OFF'}")
    logger.info(f"Early stopping threshold: {early_stop_threshold}")
    if generative:
        logger.info(f"Soft prompt length: {soft_prompt_length} ({'enabled' if soft_prompt_length > 0 else 'disabled'})")

    if generative:
        model = RobertaMLMAdapter('../pretrained_models/vulberta/', soft_prompt_length=soft_prompt_length)
    else:
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
        compute_metrics=compute_metrics_func,
        early_stop_threshold=early_stop_threshold
    )
    
    print("Training started...")
    
    # Use standard trainer.train() which will call evaluate() automatically
    trainer.train()
    
    eval_results = {'status': 'completed', 'best_f1': trainer.best_f1}
    
    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("=== Training Completed ===\n")
        f.write(f"Best F1: {trainer.best_f1:.4f}\n")
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


def evaluate_model(dataset_name, test_dataset, device, output_dir=None, metrics_file="metrics.txt", generative=False, eval_batch_size=32, soft_prompt_length=1):
    """Evaluate model"""
    logger.info(f"Evaluating model - dataset: {dataset_name}")
    logger.info(f"Mode: {'Generative' if generative else 'Classification'}")
    
    # Clear any previous GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Load model with best F1 weights
    best_model_path = os.path.join(output_dir, "checkpoint-best-f1.bin")
    
    if os.path.exists(best_model_path):
        if generative:
            model = RobertaMLMAdapter('../pretrained_models/vulberta/', soft_prompt_length=soft_prompt_length)
        else:
            model = RobertaForSequenceClassification.from_pretrained('../pretrained_models/vulberta/')
        
        # Load state dict with explicit device mapping
        checkpoint = torch.load(best_model_path, map_location=device)
        
        # Handle soft_prompt parameter mismatch (when switching between different soft_prompt_length values)
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint.keys())
        
        # Check for soft_prompt parameter mismatch
        if 'soft_prompt' in checkpoint_keys and 'soft_prompt' not in model_keys:
            logger.info("Removing 'soft_prompt' from checkpoint (current model has soft_prompt_length=0)")
            checkpoint = {k: v for k, v in checkpoint.items() if k != 'soft_prompt'}
        elif 'soft_prompt' not in checkpoint_keys and 'soft_prompt' in model_keys:
            logger.info("Warning: Model expects 'soft_prompt' but checkpoint doesn't contain it (will use random initialization)")
        
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded best F1 model weights from: {os.path.basename(best_model_path)}")
    else:
        if generative:
            model = RobertaMLMAdapter('../pretrained_models/vulberta/', soft_prompt_length=soft_prompt_length)
        else:
            model = RobertaForSequenceClassification.from_pretrained('../pretrained_models/vulberta/')
        print(f"Warning: No fine-tuned weights found, using base pretrained model")
    
    # Move to device and set eval mode before DataParallel
    model.to(device)
    model.eval()  # Set to evaluation mode to disable dropout and batch norm
    
    # Multi-GPU support (after moving to device)
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for inference")
        model = torch.nn.DataParallel(model)
    
    # Clear cache again after model loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create data loader with no pin_memory to avoid issues
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, pin_memory=False, num_workers=0)
    
    logger.info(f"Using evaluation batch size: {eval_batch_size}")
    logger.info(f"Test dataset size: {len(test_dataset)} samples")
    
    # Start inference with progress bar
    test_pbar = tqdm(test_loader, desc="Testing", unit="batch")
    
    all_pred = []
    all_labels = []
    all_probs = []
    model.eval()
    
    with torch.no_grad():
        for batch in test_pbar:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            if generative:
                mask_positions = batch.get('mask_positions', None)
                if mask_positions is not None:
                    mask_positions = mask_positions.to(device, non_blocking=True)
                outputs = model(input_ids, attention_mask=attention_mask, mask_positions=mask_positions)
            else:
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            if generative:
                logits = outputs.logits  # Shape: [batch_size, 2] for verbalizer
            else:
                logits = outputs[1]  # Shape: [batch_size, num_classes]
            
            probs = torch.nn.functional.softmax(logits, dim=1)
            acc_val, pred = softmax_accuracy(probs, labels)
            all_pred += pred
            all_labels += labels.tolist()
            all_probs += probs.tolist()
            
            # Clear batch data from GPU
            del input_ids, attention_mask, labels, logits, probs
            if generative and 'mask_positions' in locals():
                del mask_positions
    
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
        f.write(f"Mode: {'Generative' if generative else 'Classification'}\n")
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
    parser.add_argument('--generative', action='store_true',
                       help='Use generative mode (MLM-based) instead of classification')
    parser.add_argument('--soft_prompt_length', type=int, default=2,
                       help='Length of soft prompt tokens (default: 2, 0 to disable)')
    parser.add_argument('--max_seq_length', type=int, default=512,
                       help='Maximum sequence length for tokenization (default: 512)')
    parser.add_argument('--max_code_words', type=int, default=450,
                       help='Maximum number of words in code before tokenization (default: 450)')
    
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
    parser.add_argument('--early_stop_threshold', type=int, default=3,
                       help='Early stopping threshold (default: 3)')
    parser.add_argument('--metrics_file', type=str, default='metrics.txt',
                       help='Metrics output file (default: metrics.txt)')
    
    args = parser.parse_args()
    
    # Setup environment
    device = setup_environment(args.seed)
    
    # Create tokenizer
    tokenizer = create_tokenizer(args.max_seq_length)
    
    if args.mode == 'train':
        # Load training data
        train_dataset, val_dataset = load_data(args.dataset, 'train', tokenizer, args.generative, args.max_code_words)
        
        # Train model
        train_model(args.dataset, train_dataset, val_dataset, 
                   args.output_dir, args.batch_size, args.epochs, 
                   args.learning_rate, args.seed, args.fp16, args.generative,
                   args.early_stop_threshold, args.soft_prompt_length)
        
    elif args.mode == 'evaluate':
        # Load test data
        test_dataset = load_data(args.dataset, 'evaluate', tokenizer, args.generative, args.max_code_words)
        
        # Evaluate model
        evaluate_model(args.dataset, test_dataset, device, args.output_dir, 
                      args.metrics_file, args.generative, args.batch_size, args.soft_prompt_length)


if __name__ == "__main__":
    main()
