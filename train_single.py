import argparse
import json
import os
import sys
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import wandb
from tqdm import tqdm
from typing import Optional
from config import get_kaggle_config
from model import build_transformer, ModelConfig
from dataset import BidirectionalDataset, Collator
from transformers import AutoTokenizer
from utils import create_cosine_scheduler
from checkpoint_utils import load_checkpoint
from contextlib import nullcontext
from evaluate import calculate_metrics

# Setup logging
logger = logging.getLogger(__name__)

def setup_logging():
    """
    Setup logging configuration for single GPU training
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("train_single.log", mode="a", encoding="utf-8")
        ]
    )

def save_checkpoint_single(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    step: int,
    config: dict,
    global_step: Optional[int] = None,
    tag: Optional[str] = None,
):
    """
    Save checkpoint in standard PyTorch format for single GPU training.
    Compatible with load_checkpoint() function.
    """
    output_dir = config.get("output_dir", "./output")
    base_checkpoint_dir = config.get("checkpoint_path") or config.get("checkpoint_dir") or os.path.join(output_dir, "checkpoints")
    os.makedirs(base_checkpoint_dir, exist_ok=True)
    
    checkpoint_step = global_step if global_step is not None else step
    checkpoint_dir = os.path.join(base_checkpoint_dir, f"checkpoint-{checkpoint_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
    
    # Prepare checkpoint dictionary
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if config.get("save_optimizer_state", True) else None,
        "scheduler": scheduler.state_dict() if config.get("save_optimizer_state", True) else None,
        "scaler": scaler.state_dict() if (scaler is not None and config.get("save_optimizer_state", True)) else None,
        "epoch": epoch,
        "step": step,
        "global_step": global_step if global_step is not None else step,
    }
    
    if tag is not None:
        checkpoint["stage"] = tag
    
    # Save checkpoint
    logger.info(f"Saving checkpoint to {checkpoint_file}")
    torch.save(checkpoint, checkpoint_file)
    logger.info(f"✓ Checkpoint saved successfully to {checkpoint_dir}")
    
    # Cleanup old checkpoints
    save_total_limit = config.get("save_total_limit", 3)
    if save_total_limit > 0:
        _cleanup_old_checkpoints(base_checkpoint_dir, checkpoint_step, save_total_limit)

def _cleanup_old_checkpoints(base_checkpoint_dir: str, current_step: int, keep_latest_n: int):
    """Remove old checkpoints, keeping only the latest N checkpoints."""
    try:
        checkpoints = []
        for item in os.listdir(base_checkpoint_dir):
            checkpoint_path = os.path.join(base_checkpoint_dir, item)
            if os.path.isdir(checkpoint_path) and item.startswith("checkpoint-"):
                try:
                    step = int(item.split("-")[1])
                    checkpoints.append((step, checkpoint_path))
                except ValueError:
                    continue
        
        # Sort by step (descending)
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # Remove old checkpoints
        for step, checkpoint_path in checkpoints[keep_latest_n:]:
            if step != current_step:  # Don't remove current checkpoint
                try:
                    import shutil
                    shutil.rmtree(checkpoint_path)
                    logger.info(f"Removed old checkpoint: {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove old checkpoint {checkpoint_path}: {e}")
    except Exception as e:
        logger.warning(f"Failed to cleanup old checkpoints: {e}")

def train_single(config: Optional[dict] = None):
    """
    Single GPU training function.
    No distributed setup required.
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Enable fast math on modern GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # Setup logging
    setup_logging()
    
    if config is None:
        config = get_kaggle_config()
    
    logger.info(f"Starting single GPU training with config: {json.dumps(config, indent=2, default=str)}")
    
    # Load tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
    max_seq_len = config.get("max_seq_len", 149)
    train_dataset = BidirectionalDataset(
        dataset_path=config['train_hf_dataset_path'],
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )
    val_dataset = BidirectionalDataset(
        dataset_path=config['val_flores_hf_dataset_path'],
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )
    collate_fn = Collator(tokenizer=tokenizer, max_seq_len=max_seq_len)
    
    # Create DataLoaders (no DistributedSampler needed)
    num_workers = config.get('num_workers', os.cpu_count() or 4)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,  # Simple shuffle, no distributed sampler
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=num_workers if num_workers > 0 else None,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['val_batch_size'],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=num_workers if num_workers > 0 else None
    )
    
    # Build model
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size)
    # Update model_config with values from config if they exist
    for key, value in config.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
    
    model = build_transformer(config=model_config).to(device)
    
    logger.info(f"Model created: \n{model}")
    
    # Optional: torch.compile
    if config.get('use_torch_compile', False):
        compile_mode = config.get('torch_compile_mode', 'default')
        compile_dynamic = config.get('torch_compile_dynamic', True)
        compile_backend = config.get('torch_compile_backend', 'inductor')
        compile_fullgraph = config.get('torch_compile_fullgraph', False)

        # Enable capturing scalar outputs to reduce graph breaks from .item() calls
        try:
            import torch._dynamo.config as dynamo_config
            dynamo_config.capture_scalar_outputs = True
        except Exception:
            pass  # Ignore if config not available

        logger.info(
            f"Compiling model with mode={compile_mode}, dynamic={compile_dynamic}, "
            f"backend={compile_backend}, fullgraph={compile_fullgraph}"
        )

        model = torch.compile(
            model,
            mode=compile_mode,
            backend=compile_backend,
            fullgraph=compile_fullgraph,
            dynamic=compile_dynamic,
        )
        logger.info("Model compiled with torch.compile")
    else:
        logger.info("Model not compiled with torch.compile")
    
    # Calculate training steps
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    num_training_steps = len(train_dataloader) * config['num_epochs'] // gradient_accumulation_steps
    
    # Setup optimizer, scheduler, scaler
    optimizer = optim.AdamW(model.parameters(), **config['optimizer']['params'])
    scheduler = create_cosine_scheduler(optimizer, config, num_training_steps)
    
    # Initialize AMP GradScaler (standard, not sharded)
    use_amp = config.get('use_amp', True)
    amp_dtype = torch.float16 if config.get('amp_dtype', 'fp16') == 'fp16' else torch.bfloat16
    scaler = GradScaler(enabled=use_amp)
    
    # Gradient clipping config
    max_grad_norm = config.get('max_grad_norm', 1.0)
    
    # Checkpoint config
    save_strategy = config.get('save_strategy', 'epoch')
    save_steps = config.get('save_steps', None) if save_strategy == 'steps' else None
    save_total_limit = config.get('save_total_limit', 3)
    
    # WandB initialization
    wandb_config = config.get("wandb", {})
    if wandb_config.get("enabled", True):
        logger.info(f"Initializing wandb project: {wandb_config.get('project', 'Translate-Vi-En')}")
        wandb.init(
            project=wandb_config.get("project", "Translate-Vi-En"),
            name=wandb_config.get("name", "single_gpu_run"),
            config=config
        )
    
    logger.info(f"==================================================")
    logger.info(f"Single GPU Training Started")
    logger.info(f"Device: {device}")
    logger.info(f"AMP Enabled: {use_amp}, Dtype: {amp_dtype}")
    logger.info(f"Gradient Clipping: max_norm={max_grad_norm}")
    logger.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
    logger.info(f"Total Training Steps: {num_training_steps}")
    logger.info(f"Save Steps: {save_steps}, Save Total Limit: {save_total_limit}")
    logger.info(f"==================================================")
    
    if wandb.run is not None:
        wandb.config.update({
            "single_gpu": True,
            "use_amp": use_amp,
            "amp_dtype": str(amp_dtype),
            "max_grad_norm": max_grad_norm,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "save_steps": save_steps,
            "save_total_limit": save_total_limit,
            "save_strategy": save_strategy,
            "config": config
        }, allow_val_change=True)
    
    # Evaluation config
    eval_steps = config.get('eval_steps', 5000)
    
    # Resume logic
    start_epoch = 0
    start_global_step = 0
    resume_path = config.get('resume_from_checkpoint')
    
    if resume_path:
        logger.info(f"Resuming training from checkpoint: {resume_path}")
        
        try:
            metadata = load_checkpoint(
                checkpoint_path=resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config
            )
            checkpoint_epoch = metadata.get('epoch', 0)
            start_global_step = metadata.get('global_step', 0)
            
            # Recalculate start_epoch based on global_step (more accurate than saved epoch)
            # global_step represents number of effective batches completed
            # Each effective batch = gradient_accumulation_steps micro-batches
            if len(train_dataloader) > 0:
                batches_per_epoch = len(train_dataloader)
                total_micro_batches_done = start_global_step * gradient_accumulation_steps
                
                # Calculate which epoch we're in based on micro-batches processed
                calculated_epoch = total_micro_batches_done // batches_per_epoch
                batches_in_current_epoch = total_micro_batches_done % batches_per_epoch
                
                # If we're in the middle of an epoch, continue from that epoch
                # If we've completed the epoch (batches_in_current_epoch == 0), start from next epoch
                if batches_in_current_epoch == 0:
                    # Completed this epoch, start from next epoch
                    start_epoch = calculated_epoch
                    completed_epoch = calculated_epoch - 1
                else:
                    # Still in the middle of this epoch, continue from this epoch
                    start_epoch = calculated_epoch
                    completed_epoch = calculated_epoch
                
                if start_epoch != checkpoint_epoch:
                    logger.info(f"Recalculating start epoch: checkpoint epoch={checkpoint_epoch}, calculated start_epoch={start_epoch} (from global_step={start_global_step}, batches_in_current_epoch={batches_in_current_epoch})")
            else:
                # Fallback: start from checkpoint_epoch + 1 (assuming checkpoint was saved at end of epoch)
                start_epoch = checkpoint_epoch + 1
                completed_epoch = checkpoint_epoch
            
            logger.info(f"Resumed state: Completed Epoch {completed_epoch}, Starting Epoch {start_epoch}, Global Step {start_global_step}")
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            raise e
    
    def run_validation(curr_epoch, curr_step, is_end_of_epoch=False):
        """Run validation without distributed gathering."""
        model.eval()
        
        # Create tqdm progress bar for validation
        desc = f"Epoch {curr_epoch} [Val]" if is_end_of_epoch else f"Step {curr_step} [Val]"
        val_pbar = tqdm(val_dataloader, desc=desc)
        
        predictions = []
        references = []
        
        with torch.no_grad():
            for step, batch in enumerate(val_pbar):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Validation with Generation (No Teacher Forcing)
                src_input_ids = batch['src_input_ids']
                src_attention_mask = batch['src_attention_mask']
                
                # Prepare start tokens: [BOS, Lang_Token]
                tgt_start_ids = batch['tgt_input_ids'][:, :2]
                
                bs = src_input_ids.size(0)
                max_gen_len = 152
                
                with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    # Encode
                    encoder_output, _ = model.encoder(src_input_ids, mask=src_attention_mask)
                    
                    decoder_input = tgt_start_ids
                    
                    # Generation loop
                    for _ in range(max_gen_len):
                        decoder_output, _ = model.decoder(
                            decoder_input,
                            encoder_output,
                            tgt_mask=None,
                            src_mask=src_attention_mask
                        )
                        
                        # Project to vocab
                        logits = model.lm_head(model.norm(decoder_output[:, -1]))
                        next_tokens = torch.argmax(logits, dim=-1)
                        
                        decoder_input = torch.cat([decoder_input, next_tokens.unsqueeze(1)], dim=1)
                
                generated_ids = decoder_input
                
                # Collect predictions and references
                for i in range(bs):
                    # Prediction
                    pred_tokens = generated_ids[i, 2:].tolist()  # Skip start tokens
                    try:
                        eos_idx = pred_tokens.index(tokenizer.eos_token_id)
                        pred_tokens = pred_tokens[:eos_idx]
                    except ValueError:
                        pass
                    pred_text = tokenizer.decode(pred_tokens)
                    predictions.append(pred_text)
                    
                    # Reference
                    ref_text = batch['tgt_text'][i]
                    references.append(ref_text)
                
                # Logging samples
                if step % 100 == 0:
                    logger.info(f"\nStep {step} | Src: {batch['src_text'][0]}")
                    logger.info(f"Ref: {batch['tgt_text'][0]}")
                    logger.info(f"Pred: {predictions[-bs]}")  # Last batch first item
                    
                    if wandb.run is not None:
                        wandb.log({
                            "validation_samples": wandb.Table(
                                columns=["Step", "Source", "Reference", "Prediction"],
                                data=[[curr_step, batch['src_text'][:50], batch['tgt_text'][:50], predictions[-bs]]]
                            )
                        }, commit=False)
        
        # Calculate metrics directly (no gathering needed for single GPU)
        metrics = calculate_metrics(predictions, references)
        logger.info(f"{desc} | BLEU: {metrics['bleu']['score']:.4f} | chrF++: {metrics['chrf_plus']['score']:.4f}")
        
        if wandb.run is not None:
            wandb.log({
                "eval_bleu": metrics['bleu']['score'],
                "eval_chrf_plus": metrics['chrf_plus']['score'],
                "epoch": curr_epoch,
                "global_step": curr_step
            })
        
        model.train()
    
    # Training loop
    global_step = start_global_step
    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        
        # Skip batches logic temporarily removed - batch size may have changed
        # TODO: Re-implement skip logic if needed with correct batch size calculation
        
        # Create tqdm progress bar for training
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} [Train]")
        
        optimizer.zero_grad()
        accumulated_loss = 0.0
        
        for step, batch in enumerate(train_pbar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Determine if we are accumulating gradients
            is_accumulating = (step + 1) % gradient_accumulation_steps != 0
            
            # Use nullcontext instead of fsdp_model.no_sync() for single GPU
            with nullcontext():
                # Forward pass with AMP autocast
                with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    # Mark step boundary for CUDA graphs when using torch.compile
                    if torch.cuda.is_available():
                        try:
                            import importlib
                            torch_compiler = importlib.import_module("torch.compiler")
                            torch_compiler.cudagraph_mark_step_begin()
                        except Exception:
                            pass

                    # Determine if this is the final batch in effective batch
                    is_final = not is_accumulating
                    logits, loss_lm, enc_aux_loss, dec_aux_loss = model(
                        batch['src_input_ids'], 
                        batch['src_attention_mask'], 
                        batch['tgt_input_ids'], 
                        batch['tgt_attention_mask'], 
                        batch['labels'],
                        is_final=is_final
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
                accumulated_loss = 0.0  # Reset after optimizer step
                train_pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                    'grad_norm': f'{grad_norm:.2f}'
                })
                
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
                if save_steps is not None and global_step > 0 and global_step % save_steps == 0:
                    save_checkpoint_single(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        step=global_step,
                        global_step=global_step,
                        config=config
                    )
                
                # Evaluate at regular intervals
                if global_step > 0 and global_step % eval_steps == 0:
                    run_validation(epoch, global_step)
        
        # Handle remaining accumulated gradients at end of epoch
        if accumulated_loss > 0:
            # Perform final optimizer step with remaining gradients
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
        
        run_validation(epoch, global_step, is_end_of_epoch=True)
        
        # Save checkpoint at end of epoch
        save_checkpoint_single(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            step=global_step,
            global_step=global_step,
            config=config,
            tag=f"epoch-{epoch}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    train_single(config=config)

