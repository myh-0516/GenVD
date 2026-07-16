from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
os.environ["OMP_NUM_THREADS"] = "1" 
import pickle
import random
import re
import shutil
import json
import numpy as np
import torch

import torch.utils._pytree as pytree
if not hasattr(pytree, 'register_pytree_node'):
    pytree.register_pytree_node = lambda *args, **kwargs: None

import transformers.utils.generic
if not hasattr(transformers.utils.generic, '_CAN_RECORD_REGISTRY'):
    transformers.utils.generic._CAN_RECORD_REGISTRY = False
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import Model
import transformers
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

import transformers.modeling_utils
if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None
cpu_cont = multiprocessing.cpu_count()
from torch.optim import AdamW

from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer, BertForSequenceClassification,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertForSequenceClassification, DistilBertTokenizer)

from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}

class InputFeatures(object):
    def __init__(self, input_tokens, input_ids, idx, label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label

def convert_examples_to_features(js, tokenizer, args):
    code = js.get('func', js.get('functionSource', ''))
    code = ' '.join(code.split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    idx = str(js.get('hash', js.get('idx', '0')))
    label = int(js.get('target', js.get('label', 0)))
    
    return InputFeatures(input_tokens=source_tokens, input_ids=source_ids, idx=idx, label=label)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    js = json.loads(line)
                    self.examples.append(convert_examples_to_features(js, tokenizer, args))
                except Exception as e:
                    logger.warning(f"Skipping invalid line: {line[:100]}... Error: {e}")
                    continue

        if file_path and 'train' in file_path and getattr(args, 'imbalance_mode', '') in ['oversample', 'undersample']:
            labels = [ex.label for ex in self.examples]
            classes = list(set(labels))
            if len(classes) == 2:
                c1_ex = [ex for ex in self.examples if ex.label == classes[0]]
                c2_ex = [ex for ex in self.examples if ex.label == classes[1]]
                maj_ex, min_ex = (c1_ex, c2_ex) if len(c1_ex) > len(c2_ex) else (c2_ex, c1_ex)
                
                if args.imbalance_mode == 'oversample':
                    min_ex.extend(random.choices(min_ex, k=len(maj_ex) - len(min_ex)))
                    self.examples = maj_ex + min_ex
                    random.shuffle(self.examples)
                elif args.imbalance_mode == 'undersample':
                    maj_ex = random.sample(maj_ex, len(min_ex))
                    self.examples = maj_ex + min_ex
                    random.shuffle(self.examples)

        logger.info(f"Parsed {len(self.examples)} examples from {file_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if isinstance(self.examples[i].label, str):
            self.examples[i].label = int(self.examples[i].label)       
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)

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

def train(args, train_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0, pin_memory=True)
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)

    if getattr(args, 'imbalance_mode', '') == 'weighted_ce':
        args.class_weights = torch.tensor(args.class_weights, dtype=torch.float).to(args.device)

    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1, num_training_steps=args.max_steps)
    scaler = GradScaler() if args.fp16 else None

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0.0 
    early_stopping_counter = 0
    best_loss = None

    model.zero_grad()
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)        
            labels = batch[1].to(args.device) 
            model.train()
        
            def compute_loss(base_loss, probs, labels):
                probs = probs.view(-1)
                labels = labels.float()

                probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
                
                if args.imbalance_mode == 'weighted_ce':
                    w0, w1 = args.class_weights[0], args.class_weights[1]
                    loss = - (w1 * labels * torch.log(probs) + w0 * (1.0 - labels) * torch.log(1.0 - probs))
                    return loss.mean()
                    
                elif args.imbalance_mode == 'focal':
                    bce_loss = - (labels * torch.log(probs) + (1.0 - labels) * torch.log(1.0 - probs))
                    pt = torch.where(labels == 1.0, probs, 1.0 - probs)
                    alpha_t = torch.where(labels == 1.0, torch.tensor(args.focal_alpha).to(args.device), torch.tensor(1.0 - args.focal_alpha).to(args.device))
                    return (alpha_t * ((1.0 - pt) ** args.focal_gamma) * bce_loss).mean()
                    
                elif args.imbalance_mode == 'normal':
                    return - (labels * torch.log(probs) + (1.0 - labels) * torch.log(1.0 - probs)).mean()
                    
                return base_loss

            if args.fp16:
                with autocast():
                    base_loss, logits = model(inputs, labels)
                    loss = compute_loss(base_loss, logits, labels)
            else:
                base_loss, logits = model(inputs, labels)
                loss = compute_loss(base_loss, logits, labels)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)) if global_step > tr_nb else 0, 4)
                
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, eval_when_training=True)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value, 4))                    
                        if results['eval_f1'] > best_f1:
                            best_f1 = results['eval_f1']
                            logger.info("  " + "*" * 20)  
                            logger.info("  Best f1:%s", round(best_f1, 4))
                            logger.info("  " + "*" * 20)                          
                            model_to_save = model.module if hasattr(model, 'module') else model
                            model_path = os.path.join(args.output_dir, 'best-f1-model.bin')
                            torch.save(model_to_save.state_dict(), model_path)
                            logger.info("Saving model checkpoint to %s", model_path)

        avg_loss = train_loss / tr_num
        if args.early_stopping_patience is not None:
            if best_loss is None or avg_loss < best_loss - args.min_loss_delta:
                best_loss = avg_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.early_stopping_patience:
                    logger.info("Early stopping")
                    break 

def evaluate(args, model, tokenizer, eval_when_training=False):
    eval_output_dir = args.output_dir
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0, pin_memory=True)

    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = [] 
    labels = []
    
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)        
        label = batch[1].to(args.device) 
        with torch.no_grad():
            lm_loss, logit = model(inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
        
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    probs = logits[:, 0] 
    preds = (probs > 0.5).astype(int)

    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
            
    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
        "eval_precision": round(precision, 4),
        "eval_recall": round(recall, 4),
        "eval_f1": round(f1, 4),
    }
    return result

def test(args, model, tokenizer):
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            outputs = model(inputs)
            logit = outputs[0] if isinstance(outputs, tuple) else outputs
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    probs = logits[:, 0] 
    preds = (probs > 0.5).astype(int)
    
    acc = np.mean(labels == preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    print(f'\nAccuracy: {acc:.4f}')
    print(f'Binary Precision: {precision:.4f}')
    print(f'Binary Recall: {recall:.4f}')
    print(f'Binary F1: {f1:.4f}')
    print(f'Weighted F1: {weighted_f1:.4f}')
    print(f'MCC: {mcc:.4f}')
    print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
    
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Binary Precision: {precision:.4f}\n")
        f.write(f"Binary Recall: {recall:.4f}\n")
        f.write(f"Binary F1: {f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"TP: {tp}\nTN: {tn}\nFP: {fp}\nFN: {fn}\n")
    
    with open(os.path.join(args.output_dir, "predictions.csv"), 'w') as f:
        f.write("idx,probability,prediction,true_label\n")
        for example, pred, prob, true_label in zip(eval_dataset.examples, preds, probs, labels):
            f.write(f'{example.idx},{prob:.4f},{int(pred)},{int(true_label)}\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_file", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--eval_data_file", default=None, type=str)
    parser.add_argument("--test_data_file", default=None, type=str)
    parser.add_argument("--model_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--mlm", action='store_true')
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--block_size", default=-1, type=int)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_test", action='store_true')    
    parser.add_argument("--evaluate_during_training", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--eval_batch_size", default=4, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=1.0, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--save_steps', type=int, default=50)
    parser.add_argument('--save_total_limit', type=int, default=None)
    parser.add_argument("--eval_all_checkpoints", action='store_true')
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--overwrite_output_dir', action='store_true')
    parser.add_argument('--overwrite_cache', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--server_ip', type=str, default='')
    parser.add_argument('--server_port', type=str, default='')
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--min_loss_delta", type=float, default=0.001)
    parser.add_argument('--dropout_probability', type=float, default=0)
    
    parser.add_argument("--imbalance_mode", type=str, default="normal",choices=["normal","weighted_ce","focal","oversample","undersample"])
    parser.add_argument("--class_weights", type=float, nargs='+', default=[1.0, 1.0])
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument("--tuning_mode", type=str, default="full", choices=["full", "lora", "adapter"])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16,)


    args = parser.parse_args()

    if args.server_ip and args.server_port:
        import ptvsd
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size // max(1, args.n_gpu)
    args.per_gpu_eval_batch_size = args.eval_batch_size // max(1, args.n_gpu)
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    set_seed(args.seed)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.start_epoch = 0
    args.start_step = 0

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case, cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)

    if args.tuning_mode != "full":
        import adapters
        adapters.init(model) 

        if args.tuning_mode == "lora":
            adapter_config = adapters.LoRAConfig(r=args.lora_r, alpha=args.lora_alpha)
            model.add_adapter("lora_tuning", config=adapter_config)
            model.set_active_adapters("lora_tuning")
            model.train_adapter("lora_tuning")
            
        elif args.tuning_mode == "adapter":
            adapter_config = adapters.AdapterConfig.load("houlsby", reduction_factor=16)
            model.add_adapter("bottleneck_adapter", config=adapter_config)
            model.set_active_adapters("bottleneck_adapter")
            model.train_adapter("bottleneck_adapter")
            
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        logger.info(f"[{args.tuning_mode}] trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.4f}")
    
    model = Model(model, config, tokenizer, args)
    if args.local_rank == 0:
        torch.distributed.barrier()

    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier() 
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        if args.local_rank == 0:
            torch.distributed.barrier()
        train(args, train_dataset, model, tokenizer)

    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        model_path = os.path.join(args.output_dir, 'best-f1-model.bin')
        model.load_state_dict(torch.load(model_path))      
        model.to(args.device)
        result = evaluate(args, model, tokenizer)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))
            
    if args.do_test and args.local_rank in [-1, 0]:
        model_path = os.path.join(args.output_dir, 'best-f1-model.bin')
        model.load_state_dict(torch.load(model_path), strict=False)                  
        model.to(args.device)
        test(args, model, tokenizer)

    return results

if __name__ == "__main__":
    main()