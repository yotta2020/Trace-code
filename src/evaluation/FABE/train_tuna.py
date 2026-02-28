import torch
from torch.utils.data import Dataset, DataLoader
from unsloth import FastLanguageModel
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
import os
os.environ['UNSLOTH_OFFLINE'] = '1'
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
from typing import Dict, List
import argparse

class RankingDataset(Dataset):
    """
    Dataset for ranking-based training with JSONL format.
    保持原有逻辑不变
    """
    def __init__(self, data_path: str, tokenizer, max_length: int = 4096):
        self.data = []
        print(f"Loading data from: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        self.data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                        continue
        
        print(f"Loaded {len(self.data)} samples")
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item['instruction']
        input_text = item['input']
        prompt = f"{instruction}\n\n{input_text}"
        
        candidates = item['output']  
        scores = item['score']       
        
        assert len(candidates) == 4, f"Expected 4 candidates, got {len(candidates)}"
        assert len(scores) == 4, f"Expected 4 scores, got {len(scores)}"
        
        all_input_ids = []
        all_attention_masks = []
        
        for cand in candidates:
            full_text = prompt + cand
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            all_input_ids.append(encoding['input_ids'].squeeze(0))
            all_attention_masks.append(encoding['attention_mask'].squeeze(0))
        
        all_input_ids = torch.stack(all_input_ids)
        all_attention_masks = torch.stack(all_attention_masks)
        all_scores = torch.tensor(scores, dtype=torch.float32)
        
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        prompt_len = prompt_encoding['attention_mask'].sum().item()
        
        return {
            'all_input_ids': all_input_ids,
            'all_attention_masks': all_attention_masks,
            'all_scores': all_scores,
            'prompt_len': prompt_len
        }

def compute_ranking_loss_vectorized(all_losses: torch.Tensor, margin_scale: float = 0.1):
    """
    保持原有 Ranking Loss 向量化计算逻辑不变
    """
    losses_i = all_losses.unsqueeze(1)
    losses_j = all_losses.unsqueeze(0)
    diff_matrix = losses_i - losses_j
    
    indices = torch.arange(4, device=all_losses.device)
    i_matrix = indices.unsqueeze(1)
    j_matrix = indices.unsqueeze(0)
    margin_matrix = (j_matrix - i_matrix) * margin_scale
    
    loss_matrix = torch.relu(diff_matrix + margin_matrix)
    mask = torch.triu(torch.ones(4, 4, device=all_losses.device), diagonal=1).bool()
    ranking_losses = loss_matrix * mask
    ranking_loss = ranking_losses.sum()
    
    return ranking_loss

def train():
    parser = argparse.ArgumentParser(description='Train ranking model with Unsloth and LoRA')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-Coder-7B-Instruct')
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--eval_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum_steps', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--gen_weight', type=float, default=1.0)
    parser.add_argument('--rank_weight', type=float, default=1.0)
    parser.add_argument('--margin_scale', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=4096)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--gpu_id', type=int, default=0)
    
    # --- LoRA 相关新增参数 ---
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--load_in_4bit', action='store_true')
    # -----------------------
    
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu_id}')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 核心修改：使用 Unsloth 加载并配置 LoRA ---
    print(f"Loading model with Unsloth: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_length,
        dtype = torch.bfloat16,
        load_in_4bit = args.load_in_4bit,
        device_map = {"": args.gpu_id},
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = args.lora_alpha,
        lora_dropout = 0, # Unsloth 推荐设置为 0
        bias = "none",    
        use_gradient_checkpointing = "unsloth", # 节省显存
        random_state = 3407,
    )
    # ---------------------------------------------
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    # Datasets
    train_dataset = RankingDataset(args.train_data, tokenizer, args.max_length)
    eval_dataset = RankingDataset(args.eval_data, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    total_steps = len(train_loader) * args.num_epochs // args.grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    scaler = GradScaler()
    
    # Training loop 保持原有逻辑
    global_step = 0
    best_eval_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses, epoch_gen_losses, epoch_rank_losses = [], [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for step, batch in enumerate(pbar):
            all_input_ids = batch['all_input_ids'].to(device)
            all_attention_masks = batch['all_attention_masks'].to(device)
            prompt_len = batch['prompt_len']
            bs = all_input_ids.size(0)
            
            input_ids_flat = all_input_ids.view(-1, all_input_ids.size(-1))
            attention_mask_flat = all_attention_masks.view(-1, all_attention_masks.size(-1))
            
            with autocast(dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids_flat,
                    attention_mask=attention_mask_flat,
                    labels=input_ids_flat
                )
                
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids_flat[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                per_token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                ).view(bs * 4, -1)
                
                all_losses_list = []
                for i in range(bs):
                    for j in range(4):
                        idx = i * 4 + j
                        plen = prompt_len[i].item()
                        response_mask = attention_mask_flat[idx, plen:].float()
                        response_loss = (per_token_loss[idx, plen-1:] * response_mask).sum() / (response_mask.sum() + 1e-8)
                        all_losses_list.append(response_loss)
                
                all_losses = torch.stack(all_losses_list).view(bs, 4)
                generation_loss = all_losses[:, 0].mean()
                
                ranking_loss = sum(compute_ranking_loss_vectorized(all_losses[i], args.margin_scale) for i in range(bs)) / bs
                total_loss = (args.gen_weight * generation_loss + args.rank_weight * ranking_loss) / args.grad_accum_steps
            
            scaler.scale(total_loss).backward()
            
            if (step + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # 记录详细 Loss
                epoch_losses.append(total_loss.item() * args.grad_accum_steps)
                epoch_gen_losses.append(generation_loss.item())
                epoch_rank_losses.append(ranking_loss.item())
                
                # 在进度条显示拆解后的指标
                pbar.set_postfix({
                    'total': f"{epoch_losses[-1]:.4f}",
                    'gen': f"{epoch_gen_losses[-1]:.4f}",
                    'rank': f"{epoch_rank_losses[-1]:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                if global_step % args.eval_steps == 0:
                    eval_loss = evaluate(model, eval_loader, device, args)
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_path = os.path.join(args.output_dir, 'best_model')
                        model.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                    model.train()

                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f'checkpoint-{global_step}')
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
    
    model.save_pretrained(os.path.join(args.output_dir, 'final_model'))
    tokenizer.save_pretrained(os.path.join(args.output_dir, 'final_model'))

def evaluate(model, eval_loader, device, args):
    """保持原有评估逻辑不变"""
    model.eval()
    total_loss, num_batches = 0, 0
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            all_input_ids = batch['all_input_ids'].to(device)
            all_attention_masks = batch['all_attention_masks'].to(device)
            prompt_len = batch['prompt_len']
            bs = all_input_ids.size(0)
            input_ids_flat = all_input_ids.view(-1, all_input_ids.size(-1))
            attention_mask_flat = all_attention_masks.view(-1, all_attention_masks.size(-1))
            with autocast(dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids_flat, attention_mask=attention_mask_flat, labels=input_ids_flat)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids_flat[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(bs * 4, -1)
                all_losses_list = []
                for i in range(bs):
                    for j in range(4):
                        idx = i * 4 + j
                        plen = prompt_len[i].item()
                        response_mask = attention_mask_flat[idx, plen:].float()
                        response_loss = (per_token_loss[idx, plen-1:] * response_mask).sum() / (response_mask.sum() + 1e-8)
                        all_losses_list.append(response_loss)
                all_losses = torch.stack(all_losses_list).view(bs, 4)
                generation_loss = all_losses[:, 0].mean()
                ranking_loss = sum(compute_ranking_loss_vectorized(all_losses[i], args.margin_scale) for i in range(bs)) / bs
                total_loss += (args.gen_weight * generation_loss + args.rank_weight * ranking_loss).item()
                num_batches += 1
    avg_loss = total_loss / num_batches
    print(f"Eval Loss: {avg_loss:.4f}")
    return avg_loss

if __name__ == '__main__':
    train()