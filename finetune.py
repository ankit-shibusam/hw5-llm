import numpy as np
import tiktoken
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torchtext.data.metrics import bleu_score
from ignite.metrics import Rouge
from transformers import AdamW

from datetime import datetime
import os
import random

from model import GPTConfig, GPT
from data.cnn_dailymail.prepare import enc
from finetune_config import *

import gc
import wandb
import warnings

import math

torch.cuda.empty_cache()
gc.collect()

# Ignore all warnings
warnings.filterwarnings("ignore")

## Dataset Class
class MergedDataset(Dataset):
    def __init__(self, summary_root, squad_root, file, length=None):
        # Merge the datasets for the summarizer and QA tasks
        self.summarise_data = np.load(os.path.join(summary_root, file+'.npy'), mmap_mode='r')[:length]
        self.summarise_lens = np.load(os.path.join(summary_root, file+'_lens.npy'), mmap_mode='r')[:length]

        self.qa_data = np.load(os.path.join(squad_root, file+'.npy'), mmap_mode='r')[:length]
        self.qa_lens = np.load(os.path.join(squad_root, file+'_lens.npy'), mmap_mode='r')[:length]

        self.data = np.concatenate([self.summarise_data, self.qa_data])
        self.data_lens = np.concatenate([self.summarise_lens, self.qa_lens])

        self.length = self.data.shape[0]

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        d = self.data[idx]
        l = self.data_lens[idx]

        return d, l, idx<len(self.summarise_data)


train_dataset = MergedDataset(SUMMARY_ROOT, SQUAD_ROOT, 'train', length=75000)
val_dataset = MergedDataset(SUMMARY_ROOT, SQUAD_ROOT,'validation', length=300)

## Intiialize dataloader
train_dataloader =  DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4)

val_dataloader =  DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4)


## Load pretrained model ##
checkpoint = torch.load(CHK_PT_PATH, map_location=DEVICE)
checkpoint_model_args = checkpoint['model_args']
checkpoint_model_args['dropout'] = dropout

gptconf = GPTConfig(**checkpoint_model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']

unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
iter_num = checkpoint['iter_num']
best_val_loss = checkpoint['best_val_loss']

# Setup training functions
loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
val_loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

## Init wandb
wandb.login(key="42926c014f7382a7434aad694030d03d52f66025")
wandb_config = {
    'BATCH_SIZE': BATCH_SIZE,
    'learning_rate': learning_rate,
    'gradient_accumulation_steps': gradient_accumulation_steps
}
wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config)

model = model.to(DEVICE)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def train(model):
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(1337)
    for _ in np.arange(EPOCHS):
        for step, (data, article_len, is_summariser) in enumerate(train_dataloader):
            inputs, labels = torch.tensor(data), torch.tensor(data)
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            model.train()
            logits = model(inputs, retuan_all_logits=True)[0]
            
            # only consider loss on reference summary just like seq2seq models
            shift_logits = []
            shift_labels = []
            for batch_idx in range(logits.shape[0]):
                idx = article_len[batch_idx].item() # index of separator token
                shift_logits.append(logits[batch_idx, idx:-1, :])
                shift_labels.append(labels[batch_idx, idx+1:])
            shift_logits = torch.cat(shift_logits, dim=0)
            shift_labels = torch.cat(shift_labels, dim=0)

            loss = loss_fct(shift_logits, shift_labels)
            loss = loss/gradient_accumulation_steps
            scaler.scale(loss).backward()

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                lr = get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()
               
                model.zero_grad()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                logging_loss = tr_loss
                print("loss:", loss.item(), end='\n\n')
                
                if (step + 1)/gradient_accumulation_steps == 1.0:
                    print('After 1st update: ', end='\n\n')
                    generate_sample(0)
                    generate_sample(-5)
                
                
            if (step + 1) % (20*gradient_accumulation_steps) == 0:
                results = evaluate(model, global_step, lr, loss.item())
                print('After', global_step+1,'updates: ', end='\n\n')
                generate_sample(0)
                generate_sample(-5)
                
            
            del inputs, labels
            torch.cuda.empty_cache()
            gc.collect()

def generate_sample(index):
    data_sample, art_len_sample, _ = val_dataset[index]
    data_sample = torch.tensor(data_sample[None,:]).to(DEVICE)
    idx = art_len_sample.item()

    logits = model(data_sample, retuan_all_logits=True)[0]
    preds = logits[0, idx:-1, :].argmax(dim=-1).tolist()

    labels = data_sample[0, idx+1:].tolist()

    if index == 0:
        print("Pred Summary:\n %s \n" % enc.decode(preds))
        print("True Summary:\n %s \n\n" % enc.decode(labels))
    else:
        print("Pred Answer:\n %s \n" % enc.decode(preds))
        print("True Answer:\n %s \n\n" % enc.decode(labels))



def evaluate(model, global_step=None, lr=None, tr_loss=None):
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    eval_output_dir = OUTPUT_DIR

    results = {}

    eval_loss = 0.0
    eval_bleu_scores = 0.0
    eval_rouge_scores = 0.0
    nb_eval_steps = 0
    model.eval()

    for (data, article_len, is_summariser) in val_dataloader:
        inputs, labels = torch.tensor(data).to(DEVICE), torch.tensor(data).to(DEVICE)
        with torch.no_grad():
            logits = model(inputs, retuan_all_logits=True)[0]
            shift_logits = []
            shift_labels = []
            avg_eval_bleu = 0.0
            avg_rouge_score = 0.0
            m = Rouge(variants=["L",1,2], multiref="best")

            for batch_idx in range(logits.shape[0]):
                idx = article_len[batch_idx].item() # index of separator token
                
                shift_logits.append(logits[batch_idx, idx:-1, :])
                shift_labels.append(labels[batch_idx, idx+1:])
                
                greedy_labels = labels[batch_idx, idx+1:].tolist()
                index = greedy_labels.index(enc.eot_token)
                greedy_labels  = greedy_labels[:index]
                references = [[enc.decode(greedy_labels).split()]]

                greedy_preds = logits[batch_idx, idx:-1, :].argmax(dim=-1).tolist()
                greedy_preds = greedy_preds[:index]
                hypotheses = [enc.decode(greedy_preds).split()]

                if is_summariser[batch_idx].item():
                    bleu4 = bleu_score(hypotheses, references, max_n=2, weights=[0.5, 0.5])
                    avg_eval_bleu += bleu4
                else:
                    m.update((hypotheses, references))
                    rouge = m.compute()
                    avg_rouge_score += max(rouge.values())

            
            shift_logits = torch.cat(shift_logits, dim=0)
            shift_labels = torch.cat(shift_labels, dim=0)
            
            lm_loss = loss_fct(shift_logits, shift_labels)
            eval_loss += lm_loss.mean().item()

            eval_bleu_scores += avg_eval_bleu/logits.shape[0]
            eval_rouge_scores += avg_rouge_score/logits.shape[0]
        
        del inputs, labels
        torch.cuda.empty_cache()
        gc.collect()


        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_bleu_scores = 2*eval_bleu_scores / nb_eval_steps
    eval_rouge_scores = 2*eval_rouge_scores / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity,
        'eval_bleu_scores': eval_bleu_scores,
        'eval_rouge_scores': eval_rouge_scores
    }
    print("perplexity:", perplexity.item())
    print('eval_bleu_scores: ', eval_bleu_scores)
    print('eval_rouge_scores: ', eval_rouge_scores)

    global best_bleu_score
    global best_rouge_score
    if global_step:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as f:
            for key in sorted(result.keys()):
                f.write('\n\n')
                f.write("time = %s, %s = %s, step = %s\n" % (datetime.now().strftime("%d/%m/%Y %H:%M:%S"), key, str(result[key]), str(global_step)))

        wandb.log({
                "iter": global_step,
                "train/loss": tr_loss,
                "val/loss": eval_loss,
                'eval_bleu_scores': eval_bleu_scores,
                'eval_rouge_scores': eval_rouge_scores,
                "lr": lr,
            })

        if eval_bleu_scores >= best_bleu_score:
            best_bleu_score = eval_bleu_scores
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': checkpoint_model_args,
                'iter_num': global_step,
                'best_val_loss': min(best_val_loss, eval_loss),
                'best_bleu_score': best_bleu_score,
                'best_rouge_score': best_rouge_score,
                'config': gptconf,
            }
            print(f"saving checkpoint to {eval_output_dir}")
            torch.save(checkpoint, os.path.join(eval_output_dir, 'bleu_ckpt.pt'))
        
        if eval_rouge_scores >= best_rouge_score:
            best_rouge_score = eval_rouge_scores
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': checkpoint_model_args,
                'iter_num': global_step,
                'best_val_loss': min(best_val_loss, eval_loss),
                'best_bleu_score': best_bleu_score,
                'best_rouge_score': best_rouge_score,
                'config': gptconf,
            }
            print(f"saving checkpoint to {eval_output_dir}")
            torch.save(checkpoint, os.path.join(eval_output_dir, 'rouge_ckpt.pt'))

    return result 

best_bleu_score = 0.21
best_rouge_score = 0.43

train(model)
