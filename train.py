import argparse
import json
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
from tokenizer import EnViTokenizer


def train_main(config: Optional[dict] = None, ds_config: Optional[dict] = None):

    if ds_config is None:
        ds_config = get_deepspeed_config()
    if config is None:
        config = get_kaggle_config()
    tokenizer = EnViTokenizer(lang_token_map=config['lang_token_map'])
    train_dataset = BidirectionalDataset(
        tokenizer=tokenizer,
        dataset_path=config['train_hf_dataset_path']
    )
    val_dataset = BidirectionalDataset(
        tokenizer=tokenizer,
        dataset_path=config['val_hf_dataset_path']
    )
    collate_fn = Collator(
        tokenizer=tokenizer
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=ds_config['train_batch_size'],
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=ds_config['train_batch_size'],
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    model = build_transformer(
        config=ModelConfig(
            vocab_size=tokenizer.get_vocab_size(),
        )
    )
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        config=ds_config,
        collate_fn=collate_fn
    )

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