import argparse
import json
import os
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import wandb
from tqdm import tqdm
from typing import Optional
from config import get_kaggle_config
from model import build_transformer, ModelConfig
from dataset import BidirectionalDataset, Collator
from transformers import AutoTokenizer
from utils import wrap_model_with_fsdp, create_cosine_scheduler
from checkpoint_utils import save_checkpoint_simple, save_checkpoint
from contextlib import nullcontext

def train_fsdp(config: Optional[dict] = None):
    # Setup distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if config is None:
        config = get_kaggle_config()
    
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
    train_dataset = BidirectionalDataset(
        dataset_path=config['train_hf_dataset_path'],
        tokenizer=tokenizer
    )
    val_dataset = BidirectionalDataset(
        dataset_path=config['val_flores_hf_dataset_path'],
        tokenizer=tokenizer
    )
    collate_fn = Collator(tokenizer=tokenizer)
    
    # Use DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    num_workers = config.get('num_workers', os.cpu_count() or 1)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['train_batch_size'],
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['val_batch_size'],
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )

    model_config = ModelConfig(vocab_size=tokenizer.vocab_size)
    # Update model_config with values from config if they exist
    for key, value in config.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)

    model = build_transformer(config=model_config).to(local_rank)
    
    model = wrap_model_with_fsdp(model, config)
    fsdp_model = model

    if rank == 0:
        print(f"Model wrapped with FSDP: \n{model}")
        
        def check_fsdp_unit(name, module):
            is_fsdp = isinstance(module, FSDP)
            unit_id = id(module)
            weight_id = id(module.weight) if hasattr(module, "weight") else None
            print(f"[{name}] Is FSDP: {is_fsdp}, Unit ID: {unit_id}, Weight ID: {weight_id}")
            return unit_id

        print("--- Checking FSDP Wrapping & Shared Embeddings ---")
        shared_unit = check_fsdp_unit("Shared Embedding", model.shared)
        
        if hasattr(model.encoder, "embedding"):
             enc_unit = check_fsdp_unit("Encoder Embedding", model.encoder.embedding)
             if shared_unit != enc_unit:
                 print(f"Warning: Shared and Encoder Embedding are different units/objects!")
             else:
                 print("Shared and Encoder Embedding are the SAME unit/object.")

        if hasattr(model.decoder, "embedding"):
             dec_unit = check_fsdp_unit("Decoder Embedding", model.decoder.embedding)
             if shared_unit != dec_unit:
                 print(f"Warning: Shared and Decoder Embedding are different units/objects!")
             else:
                 print("Shared and Decoder Embedding are the SAME unit/object.")

        print(f"LM Head weight pointer: {id(model.lm_head.weight)}")
        print("------------------------------------------------")


    if config['use_torch_compile']:
        model = torch.compile(model)
        print("Model compiled with torch.compile")
    else:
        print("Model not compiled with torch.compile")

    # Calculate training steps
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    num_training_steps = len(train_dataloader) * config['num_epochs'] // gradient_accumulation_steps

    optimizer = optim.AdamW(model.parameters(), **config['optimizer']['params'])
    scheduler = create_cosine_scheduler(optimizer, config, num_training_steps)
    
    # Initialize AMP GradScaler
    use_amp = config.get('use_amp', True)
    amp_dtype = torch.float16 if config.get('amp_dtype', 'fp16') == 'fp16' else torch.bfloat16
    scaler = GradScaler(enabled=use_amp)
    
    # Gradient clipping config
    max_grad_norm = config.get('max_grad_norm', 1.0)
    
    # Checkpoint config
    save_steps = config.get('save_steps', 500)
    save_total_limit = config.get('save_total_limit', 3)
    
    if rank == 0:
        wandb_config = config.get("wandb", {})
        if wandb_config.get("enabled", True):
            print(f"Initializing wandb project: {wandb_config.get('project', 'Translate-Vi-En')}")
            wandb.init(
                project=wandb_config.get("project", "Translate-Vi-En"),
                name=wandb_config.get("name", "fsdp_run"),
                config=config
            )
        print(f"==================================================")
        print(f"FSDP Training Started. World Size: {world_size}")
        print(f"AMP Enabled: {use_amp}, Dtype: {amp_dtype}")
        print(f"Gradient Clipping: max_norm={max_grad_norm}")
        print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
        print(f"Total Training Steps: {num_training_steps}")
        print(f"Save Steps: {save_steps}, Save Total Limit: {save_total_limit}")
        print(f"==================================================")
        if wandb.run is not None:
             wandb.config.update({
                 "fsdp": True,
                 "use_amp": use_amp,
                 "amp_dtype": str(amp_dtype),
                 "max_grad_norm": max_grad_norm,
                 "gradient_accumulation_steps": gradient_accumulation_steps,
                 "save_steps": save_steps,
                 "save_total_limit": save_total_limit,
                 "config": config
             }, allow_val_change=True)

    global_step = 0
    for epoch in range(config['num_epochs']):
        train_sampler.set_epoch(epoch)
        model.train()
        
        # Create tqdm progress bar for training
        train_pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch} [Train]",
            disable=rank != 0
        )
        
        optimizer.zero_grad()
        accumulated_loss = 0.0
        
        for step, batch in enumerate(train_pbar):
            batch = {k: v.to(local_rank) for k, v in batch.items()}
            
            # Determine if we are accumulating gradients (no sync)
            is_accumulating = (step + 1) % gradient_accumulation_steps != 0
            
            with fsdp_model.no_sync() if is_accumulating else nullcontext():
                # Forward pass with AMP autocast
                with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    logits, loss_lm, enc_aux_loss, dec_aux_loss = model(
                        batch['src_input_ids'], 
                        batch['src_attention_mask'], 
                        batch['tgt_input_ids'], 
                        batch['tgt_attention_mask'], 
                        batch['labels']
                    )
                    total_loss = loss_lm + enc_aux_loss + dec_aux_loss
                    # Scale loss for gradient accumulation
                    scaled_loss = total_loss / gradient_accumulation_steps
                
                # Backward pass with GradScaler
                scaler.scale(scaled_loss).backward()
            
            accumulated_loss += total_loss
            
            # Perform optimizer step after accumulation
            if not is_accumulating:
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
                
                # Step the scheduler
                scheduler.step()
                
                optimizer.zero_grad()
                global_step += 1
                
                # Update tqdm progress bar
                avg_loss = accumulated_loss / gradient_accumulation_steps
                train_pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                    'grad_norm': f'{grad_norm:.2f}'
                })
                
                if global_step % 10 == 0 and rank == 0:
                    if wandb.run is not None:
                        wandb.log({
                            "loss_lm": loss_lm.item(),
                            "enc_aux_loss": enc_aux_loss.item(),
                            "dec_aux_loss": dec_aux_loss.item(),
                            "total_loss": avg_loss.item(),
                            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                            "lr": scheduler.get_last_lr()[0],
                            "global_step": global_step,
                            "epoch": epoch,
                            "scaler_scale": scaler.get_scale()
                        })
                
                # Save checkpoint at regular intervals
                # if global_step > 0 and global_step % save_steps == 0:
                #     save_checkpoint(
                #         model=model,
                #         optimizer=optimizer,
                #         scheduler=scheduler,
                #         scaler=scaler,
                #         epoch=epoch,
                #         step=global_step,
                #         global_step=global_step,
                #         config=config,
                #         rank=rank
                #     )
                #     # Sync all ranks after checkpoint
                #     if dist.is_available() and dist.is_initialized():
                #         dist.barrier()
                
                accumulated_loss = 0.0
        
        model.eval()
        
        # Create tqdm progress bar for validation
        val_pbar = tqdm(
            val_dataloader,
            desc=f"Epoch {epoch} [Val]",
            disable=rank != 0
        )
        
        val_losses = []
        with torch.no_grad():
            for step, batch in enumerate(val_pbar):
                batch = {k: v.to(local_rank) for k, v in batch.items()}
                
                # Use autocast for validation too
                with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    logits, loss_lm, enc_aux_loss, dec_aux_loss = model(
                        batch['src_input_ids'], 
                        batch['src_attention_mask'], 
                        batch['tgt_input_ids'], 
                        batch['tgt_attention_mask'], 
                        batch['labels']
                    )
                    total_loss = loss_lm + enc_aux_loss + dec_aux_loss
                
                val_losses.append(total_loss.item())
                
                # Update tqdm progress bar
                val_pbar.set_postfix({'eval_loss': f'{total_loss.item():.4f}'})
                
                if rank == 0 and step % 10 == 0:
                    if wandb.run is not None:
                        wandb.log({
                            "eval_loss": total_loss.item(),
                            "eval_loss_lm": loss_lm.item(),
                            "eval_enc_aux_loss": enc_aux_loss.item(),
                            "eval_dec_aux_loss": dec_aux_loss.item(),
                            "eval_step": step,
                            "eval_epoch": epoch
                        })
        
        # Log average validation loss
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        if rank == 0:
            print(f"Epoch {epoch} | Avg Validation Loss: {avg_val_loss:.4f}")
            if wandb.run is not None:
                wandb.log({
                    "avg_eval_loss": avg_val_loss,
                    "epoch": epoch
                })
        
        # Save checkpoint at end of epoch
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            step=global_step,
            global_step=global_step,
            config=config,
            rank=rank,
            tag=f"epoch-{epoch}"
        )
        # Sync all ranks after checkpoint
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    train_fsdp(config=config)

