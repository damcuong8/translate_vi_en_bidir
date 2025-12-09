import argparse
import json
import os
import torch
import deepspeed
import torch.distributed as dist
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from typing import Optional
from config import get_deepspeed_config, get_kaggle_config
from model import build_transformer, ModelConfig
from dataset import BidirectionalDataset, Collator
from transformers import AutoTokenizer

def train_deepspeed(config: Optional[dict] = None, ds_config: Optional[dict] = None):

    if ds_config is None:
        ds_config = get_deepspeed_config()
    if config is None:
        config = get_kaggle_config()
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
    train_dataset = BidirectionalDataset(
        dataset_path=config['train_hf_dataset_path'],
        tokenizer=tokenizer
    )
    val_dataset = BidirectionalDataset(
        dataset_path=config['val_hf_dataset_path'],
        tokenizer=tokenizer
    )
    collate_fn = Collator(tokenizer=tokenizer)

    num_workers = config.get('num_workers', os.cpu_count() or 1)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=ds_config['train_batch_size'],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )

    # --- Auto-calculate Total Steps for Scheduler ---
    if 'scheduler' in ds_config and 'params' in ds_config['scheduler']:
        # Get world size (number of GPUs)
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Calculate steps per epoch
        # len(train_dataloader) is total batches. In distributed training, work is split.
        steps_per_epoch = len(train_dataloader) // world_size
        
        gradient_accumulation_steps = ds_config.get('gradient_accumulation_steps', 1)
        
        # Effective update steps per epoch
        update_steps_per_epoch = steps_per_epoch // gradient_accumulation_steps
        
        # Total steps for training
        total_num_steps = update_steps_per_epoch * config['num_epochs']
        
        # Update config
        ds_config['scheduler']['params']['total_num_steps'] = total_num_steps
        
        # Optional: Set warmup to 5% of total steps
        ds_config['scheduler']['params']['warmup_num_steps'] = int(total_num_steps * 0.03)
        
        print(f"Auto-configured Scheduler: World Size={world_size}, Total Steps={total_num_steps}, Warmup={ds_config['scheduler']['params']['warmup_num_steps']}")
    # ------------------------------------------------

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=ds_config['train_batch_size'],
        shuffle=False,
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

    model = build_transformer(
        config=model_config
    )
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        config=ds_config,
        collate_fn=collate_fn
    )

    if dist.get_rank() == 0:
        if wandb.run is None and config.get("wandb", {}).get("enabled", False):
            print(f"Initializing wandb project: {config['wandb'].get('project', 'Translate-Vi-En')}")
            wandb.init(
                project=config["wandb"].get("project", "Translate-Vi-En"),
                name=config["wandb"].get("name", "deepspeed_run"),
                config=config
            )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"==================================================")
        print(f"Model Summary:")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print(f"==================================================")
        
        if wandb.run is not None:
            wandb.config.update({
                "total_params": total_params,
                "trainable_params": trainable_params,
                "full_config": config,
                "ds_config": ds_config
            }, allow_val_change=True)
            wandb.run.summary["total_params"] = total_params
            wandb.run.summary["trainable_params"] = trainable_params

    for epoch in range(config['num_epochs']):
        model_engine.train()
        
        # Create tqdm progress bar for training
        train_pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch} [Train]",
            disable=dist.get_rank() != 0
        )
        
        for step, batch in enumerate(train_pbar):
            batch = {k: v.to(model_engine.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            logits, loss_lm, enc_aux_loss, dec_aux_loss = model_engine(batch['src_input_ids'], batch['src_attention_mask'], batch['tgt_input_ids'], batch['tgt_attention_mask'], batch['labels'])
            
            total_loss = loss_lm + enc_aux_loss + dec_aux_loss

            model_engine.backward(total_loss)

            model_engine.step()

            # Update tqdm progress bar
            train_pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

            if step % 10 == 0:
                if dist.get_rank() == 0:
                    wandb.log({
                        "loss_lm": loss_lm.item(),
                        "enc_aux_loss": enc_aux_loss.item(),
                        "dec_aux_loss": dec_aux_loss.item(),
                        "total_loss": total_loss.item(),
                        "lr": optimizer.param_groups[0]['lr'],
                        "step": step,
                        "epoch": epoch
                    })
        
        model_engine.eval()
        
        # Create tqdm progress bar for validation
        val_pbar = tqdm(
            val_dataloader,
            desc=f"Epoch {epoch} [Val]",
            disable=dist.get_rank() != 0
        )
        
        with torch.no_grad():
            for step, batch in enumerate(val_pbar):
                batch = {k: v.to(model_engine.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                logits, loss_lm, enc_aux_loss, dec_aux_loss = model_engine(batch['src_input_ids'], batch['src_attention_mask'], batch['tgt_input_ids'], batch['tgt_attention_mask'], batch['labels'])
                total_loss = loss_lm + enc_aux_loss + dec_aux_loss
                
                # Update tqdm progress bar
                val_pbar.set_postfix({'eval_loss': f'{total_loss.item():.4f}'})
                
                if dist.get_rank() == 0:
                    wandb.log({
                        "eval_loss": total_loss.item(),
                        "eval_loss_lm": loss_lm.item(),
                        "eval_enc_aux_loss": enc_aux_loss.item(),
                        "eval_dec_aux_loss": dec_aux_loss.item(),
                        "eval_step": step,
                        "eval_epoch": epoch
                    })
        
        model_engine.save_checkpoint(save_dir="./checkpoints", tag=f"epoch_{epoch}", client_config=ds_config)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ds_config", type=str, required=False)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    ds_config = None
    if args.ds_config:
        with open(args.ds_config) as f:
            ds_config = json.load(f)

    train_deepspeed(config=config, ds_config=ds_config)

