
import argparse
import json
import os
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.cuda.amp import GradScaler, autocast
import time
from typing import Optional
from config import get_kaggle_config
from model import build_transformer, ModelConfig
from utils import wrap_model_with_fsdp
from contextlib import nullcontext

def benchmark_memory_fsdp(config: Optional[dict] = None):
    # Setup distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if config is None:
        config = get_kaggle_config()
    
    # Ensure config has necessary keys or defaults
    config.setdefault('use_amp', True)
    config.setdefault('amp_dtype', 'fp16')
    
    # Use a fixed large vocab size or from config
    # Loading tokenizer just for vocab size might be slow/failed if path invalid, so use config or default
    vocab_size = config.get('vocab_size', 24000) 
    
    if rank == 0:
        print(f"Starting FSDP Memory Benchmark on Rank {rank}")
        print(f"World Size: {world_size}")
        print(f"Vocab Size: {vocab_size}")
        print("-" * 50)

    model_config = ModelConfig(vocab_size=vocab_size)

    # Initialize model
    model = build_transformer(config=model_config).to(local_rank)
    model = wrap_model_with_fsdp(model, config)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=config.get('use_amp', True))
    
    batch_sizes = [128]
    seq_len = 160
    steps = 20
    
    # AMP settings
    use_amp = config.get('use_amp', True)
    amp_dtype = torch.float16 if config.get('amp_dtype', 'fp16') == 'fp16' else torch.bfloat16

    for bs in batch_sizes:
        if rank == 0:
            print(f"Testing Batch Size: {bs}, Sequence Length: {seq_len}")
        
        # Create dummy batch
        # Ensure we don't exceed vocab size
        src_input_ids = torch.randint(0, vocab_size, (bs, seq_len)).to(local_rank)
        tgt_input_ids = torch.randint(0, vocab_size, (bs, seq_len)).to(local_rank)
        src_attention_mask = torch.ones((bs, seq_len)).to(local_rank)
        tgt_attention_mask = torch.ones((bs, seq_len)).to(local_rank)
        labels = torch.randint(0, vocab_size, (bs, seq_len)).to(local_rank)

        torch.cuda.reset_peak_memory_stats()
        model.train()
        
        # Warmup
        if rank == 0:
            print("  Warming up...")
        for _ in range(3):
            optimizer.zero_grad()
            with autocast(dtype=amp_dtype, enabled=use_amp):
                 _, loss, _, _ = model(src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        
        for step in range(steps):
            optimizer.zero_grad()
            
            with autocast(dtype=amp_dtype, enabled=use_amp):
                logits, loss_lm, enc_aux_loss, dec_aux_loss = model(
                    src_input_ids, 
                    src_attention_mask, 
                    tgt_input_ids, 
                    tgt_attention_mask, 
                    labels
                )
                total_loss = loss_lm + enc_aux_loss + dec_aux_loss
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        torch.cuda.synchronize()
        end_time = time.time()
        
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) # GB
        avg_time = (end_time - start_time) / steps
        
        if rank == 0:
            print(f"  Batch Size {bs} Result:")
            print(f"    Max Memory Allocated: {max_mem:.2f} GB")
            print(f"    Avg Time per Step: {avg_time:.4f} s")
            print("-" * 50)
            
        # Clean up
        del logits, total_loss
        torch.cuda.empty_cache()

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
            
    benchmark_memory_fsdp(config)