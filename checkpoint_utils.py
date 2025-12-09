"""
Checkpoint utilities for distributed training.
Supports DCP (Distributed Checkpoint) for FSDP and standard torch.save for single GPU/DDP.
"""

import os
import shutil
import time
import logging
import glob
import re
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions
import wandb
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WANDB_AVAILABLE = True

class AppState:
    """Stateful wrapper for DCP checkpoint saving/loading."""
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        state_dict_options=None,
        meta: Optional[Dict[str, Any]] = None,
        dataloader_state: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.state_dict_options = state_dict_options
        self.meta = meta or {}
        self.dataloader_state = dataloader_state or {}

    def state_dict(self):
        """Return state dict for checkpoint."""
        state = {}
        
        # Get model state dict
        if self.model is not None:
            if self.state_dict_options is not None:
                model_state, _ = get_state_dict(
                    self.model, 
                    [] if self.optimizer is None else [self.optimizer],
                    options=self.state_dict_options
                )
                state["model"] = model_state.get("model", model_state)
                if self.optimizer is not None and "optimizer" in model_state:
                    state["optimizer"] = model_state["optimizer"]
            else:
                state["model"] = self.model.state_dict()
        
        # Get optimizer state dict (if not already captured)
        if self.optimizer is not None and "optimizer" not in state:
            try:
                state["optimizer"] = self.optimizer.state_dict()
            except Exception as e:
                logger.warning(f"Could not get optimizer state_dict: {e}")
        
        # Get scheduler state dict
        if self.scheduler is not None:
            try:
                state["scheduler"] = self.scheduler.state_dict()
            except Exception as e:
                logger.warning(f"Could not get scheduler state_dict: {e}")
        
        # Get scaler state dict
        if self.scaler is not None:
            try:
                state["scaler"] = self.scaler.state_dict()
            except Exception as e:
                logger.warning(f"Could not get scaler state_dict: {e}")
        
        # Add metadata
        state.update(self.meta)
        
        # Add dataloader state
        if self.dataloader_state:
            state["dataloader_state"] = self.dataloader_state
        
        return state

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        # Load metadata
        for k, v in state_dict.items():
            if k not in ["model", "optimizer", "scheduler", "scaler", "dataloader_state"]:
                self.meta[k] = v
        
        if "model" in state_dict and self.model is not None:
            self.model.load_state_dict(state_dict["model"])
        
        if "optimizer" in state_dict and self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        
        if "scheduler" in state_dict and self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        
        if "scaler" in state_dict and self.scaler is not None:
            self.scaler.load_state_dict(state_dict["scaler"])
            
        if "dataloader_state" in state_dict:
            self.dataloader_state.update(state_dict["dataloader_state"])


def _cleanup_old_checkpoints(
    base_checkpoint_dir: str, 
    keep_latest_n: int = 3
):
    """Remove old checkpoints, keeping only the latest N."""
    if keep_latest_n <= 0:
        return
    
    if not os.path.exists(base_checkpoint_dir):
        return

    # Find all checkpoint directories/files
    # Pattern: matches checkpoint-100, checkpoint-100.pt, checkpoint-epoch-1 etc.
    checkpoint_pattern = os.path.join(base_checkpoint_dir, "checkpoint-*")
    all_paths = glob.glob(checkpoint_pattern)
    
    if len(all_paths) <= keep_latest_n:
        return
    
    # Extract step numbers and sort
    def get_step(path):
        # Match number at the end of the filename/dirname, ignoring extension
        base = os.path.basename(path)
        # Remove extension if any
        base = os.path.splitext(base)[0]
        match = re.search(r'(\d+)$', base)
        return int(match.group(1)) if match else 0
    
    all_paths.sort(key=get_step, reverse=True)
    
    # Remove old checkpoints
    for old_path in all_paths[keep_latest_n:]:
        try:
            if os.path.isdir(old_path):
                shutil.rmtree(old_path)
            else:
                os.remove(old_path)
            logger.info(f"Removed old checkpoint: {old_path}")
        except Exception as e:
            logger.warning(f"Failed to remove old checkpoint {old_path}: {e}")


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    config: dict,
    rank: int,
    scaler: Optional[object] = None,
    global_step: Optional[int] = None,
    tag: Optional[str] = None,
):
    """
    Save checkpoint using DCP (Distributed Checkpoint) for FSDP or standard torch.save.
    """
    # 1. Resolve checkpoint directory
    # Priority: config['checkpoint_path'] -> config['checkpoint_dir'] -> config['output_dir']/checkpoints -> ./checkpoints
    if 'checkpoint_path' in config:
        base_checkpoint_dir = config['checkpoint_path']
    elif 'checkpoint_dir' in config:
        base_checkpoint_dir = config['checkpoint_dir']
    else:
        output_dir = config.get("output_dir", "./output")
        base_checkpoint_dir = os.path.join(output_dir, "checkpoints")
    
    if rank == 0:
        os.makedirs(base_checkpoint_dir, exist_ok=True)
    
    # Wait for directory creation
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Determine checkpoint name
    checkpoint_step = global_step if global_step is not None else step
    if tag:
        checkpoint_name = f"checkpoint-{tag}"
    else:
        checkpoint_name = f"checkpoint-{checkpoint_step}"
    
    checkpoint_path = os.path.join(base_checkpoint_dir, checkpoint_name)
    
    # Check if we should use DCP (FSDP enabled) or Simple Save
    use_fsdp = config.get("use_fsdp", False)
    
    # --- FSDP / DCP Save ---
    if use_fsdp:
        # Build StateDictOptions
        save_options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        )

        meta = {
            "epoch": epoch, 
            "step": step, 
            "global_step": checkpoint_step,
            "config": config
        }
        
        save_optimizer_state = config.get("save_optimizer_state", True)
        
        app_state = AppState(
            model=model, 
            optimizer=optimizer if save_optimizer_state else None,
            scheduler=scheduler if save_optimizer_state else None,
            scaler=scaler if save_optimizer_state else None,
            state_dict_options=save_options,
            meta=meta
        )

        state_to_save = {"app": app_state}

        if rank == 0:
            logger.info(f"Saving FSDP checkpoint to {checkpoint_path}...")
            # Remove existing dir if it exists (DCP requires fresh dir)
            if os.path.exists(checkpoint_path):
                shutil.rmtree(checkpoint_path, ignore_errors=True)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        try:
            dcp.save(state_to_save, checkpoint_id=checkpoint_path)
            if rank == 0:
                logger.info(f"✓ FSDP Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            if rank == 0:
                logger.error(f"Failed to save FSDP checkpoint: {e}")
            raise e

    # --- Simple Save (Single GPU / DDP) ---
    else:
        if rank == 0:
            # Add extension for simple save
            if not checkpoint_path.endswith(".pt"):
                checkpoint_path += ".pt"
                
            logger.info(f"Saving checkpoint to {checkpoint_path}...")
            
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer else None,
                "scheduler": scheduler.state_dict() if scheduler else None,
                "scaler": scaler.state_dict() if scaler else None,
                "epoch": epoch,
                "global_step": checkpoint_step,
                "config": config,
            }
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"✓ Checkpoint saved to {checkpoint_path}")

    # Cleanup old checkpoints
    if rank == 0:
        save_total_limit = config.get("save_total_limit", 3)
        if save_total_limit:
            _cleanup_old_checkpoints(base_checkpoint_dir, keep_latest_n=save_total_limit)
    
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device='cuda',
    config: Optional[dict] = None
) -> Dict[str, Any]:
    """
    Load checkpoint from file (standard) or directory (DCP).
    Returns the loaded metadata dict.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # metadata to return
    meta = {}

    # Check if it's a directory (DCP)
    if os.path.isdir(checkpoint_path):
        # DCP Load
        # We need to wrap objects in AppState to load into them
        # DCP loads in-place into the state_dict of the object
        
        # Note: For DCP load to work with FSDP model, the model should already be FSDP wrapped
        # and on the correct device.
        
        load_optimizer = optimizer is not None
        
        app_state = AppState(
            model=model,
            optimizer=optimizer if load_optimizer else None,
            scheduler=scheduler if load_optimizer else None,
            scaler=scaler if load_optimizer else None,
            meta=meta
        )
        
        state_to_load = {"app": app_state}
        
        dcp.load(
            state_dict=state_to_load,
            checkpoint_id=checkpoint_path,
        )
        
        # meta is populated by AppState.load_state_dict
        return app_state.meta
        
    else:
        # Standard torch.load
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint["model"])
        
        if optimizer and "optimizer" in checkpoint and checkpoint["optimizer"]:
            optimizer.load_state_dict(checkpoint["optimizer"])
        
        if scheduler and "scheduler" in checkpoint and checkpoint["scheduler"]:
            scheduler.load_state_dict(checkpoint["scheduler"])
        
        if scaler and "scaler" in checkpoint and checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
            
        # Return metadata
        meta = {k: v for k, v in checkpoint.items() if k not in ["model", "optimizer", "scheduler", "scaler"]}
        return meta


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory (supports both .pt files and DCP directories).
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Find all candidates
    candidates = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if not candidates:
        return None
        
    def get_step(path):
        base = os.path.basename(path)
        # Remove extension if any
        base = os.path.splitext(base)[0]
        # Match number at the end
        match = re.search(r'(\d+)$', base)
        return int(match.group(1)) if match else -1
    
    # Filter only those that have a step number
    candidates = [p for p in candidates if get_step(p) >= 0]
    
    if not candidates:
        return None
        
    candidates.sort(key=get_step, reverse=True)
    return candidates[0]


def resume_from_checkpoint(
    checkpoint_dir: str,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device='cuda',
    config: Optional[dict] = None
) -> Optional[dict]:
    """
    Resume training from the latest checkpoint in a directory.
    Returns the checkpoint metadata if found, None otherwise.
    """
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint is None:
        logger.info(f"No checkpoint found in {checkpoint_dir}, starting from scratch")
        return None
    
    logger.info(f"Resuming from latest checkpoint: {latest_checkpoint}")
    return load_checkpoint(latest_checkpoint, model, optimizer, scheduler, scaler, device, config)
