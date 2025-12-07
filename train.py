import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn import functional as F
import wandb
from typing import Optional

from config import get_deepspeed_config, get_kaggle_config
from model import build_transformer, ModelConfig
from dataset import BidirectionalDataset, Collator
from transformers import AutoTokenizer


def train_main(config: Optional[dict] = None, ds_config: Optional[dict] = None):

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
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=ds_config['train_batch_size'],
        shuffle=True,
        num_workers=0,  # Reduced from 2 to 0 to save memory and avoid multiprocessing overhead
        collate_fn=collate_fn,
        pin_memory=False, # Disabled pin_memory to save RAM
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
        num_workers=0, # Reduced from 2 to 0
        collate_fn=collate_fn,
        pin_memory=False, # Disabled pin_memory
    )
    model = build_transformer(
        config=ModelConfig(
            vocab_size=tokenizer.vocab_size,
        )
    )
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        config=ds_config,
        collate_fn=collate_fn
    )

    if dist.get_rank() == 0:
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
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            logits, loss_lm, enc_aux_loss, dec_aux_loss = model_engine(batch['src_input_ids'], batch['src_attention_mask'], batch['tgt_input_ids'], batch['tgt_attention_mask'], batch['labels'])
            
            total_loss = loss_lm + enc_aux_loss + dec_aux_loss

            model_engine.backward(total_loss)

            model_engine.step()

            if step % 10 == 0:
                if dist.get_rank() == 0:
                    print(f"Epoch {epoch} | Step {step} | Loss: {total_loss.item()}")
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
        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                batch = {k: v.to(model_engine.device) for k, v in batch.items()}
                logits, loss_lm, enc_aux_loss, dec_aux_loss = model_engine(batch['src_input_ids'], batch['src_attention_mask'], batch['tgt_input_ids'], batch['tgt_attention_mask'], batch['labels'])
                total_loss = loss_lm + enc_aux_loss + dec_aux_loss
                if dist.get_rank() == 0:
                    print(f"Evaluation Epoch {epoch} | Step {step} | Loss: {total_loss.item()}")
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
    parser.add_argument("--deepspeed_config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=0,
                        help="local GPU rank supplied by DeepSpeed launcher")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    with open(args.deepspeed_config) as f:
        ds_config = json.load(f)

    train_main(config=config, ds_config=ds_config)