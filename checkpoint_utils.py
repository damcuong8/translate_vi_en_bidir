"""
Checkpoint utilities for distributed training.
Supports DCP (Distributed Checkpoint), HuggingFace Hub, and Wandb.
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
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check optional dependencies
try:
    from huggingface_hub import HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

WANDB_AVAILABLE = True  # Assume wandb is available since it's imported


class AppState:
    """Stateful wrapper for DCP checkpoint saving."""
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
        if "model" in state_dict and self.model is not None:
            self.model.load_state_dict(state_dict["model"])
        
        if "optimizer" in state_dict and self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        
        if "scheduler" in state_dict and self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        
        if "scaler" in state_dict and self.scaler is not None:
            self.scaler.load_state_dict(state_dict["scaler"])


def _capture_dataloader_state(dataloader) -> Optional[Dict[str, Any]]:
    """Capture dataloader state for resumable training."""
    if dataloader is None:
        return None
    
    state = {}
    
    # Try to capture sampler state
    if hasattr(dataloader, 'sampler'):
        sampler = dataloader.sampler
        if hasattr(sampler, 'epoch'):
            state['sampler_epoch'] = sampler.epoch
        if hasattr(sampler, 'state_dict'):
            state['sampler_state'] = sampler.state_dict()
    
    return state if state else None


def _cleanup_old_checkpoints(
    config: dict, 
    base_checkpoint_dir: str, 
    keep_latest_n: Optional[int] = None
):
    """Remove old checkpoints, keeping only the latest N."""
    if keep_latest_n is None:
        keep_latest_n = config.get("save_total_limit", 3)
    
    if keep_latest_n is None or keep_latest_n <= 0:
        return
    
    # Find all checkpoint directories
    checkpoint_pattern = os.path.join(base_checkpoint_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    if len(checkpoint_dirs) <= keep_latest_n:
        return
    
    # Extract step numbers and sort
    def get_step(path):
        match = re.search(r'checkpoint-(?:epoch-)?(\d+)', os.path.basename(path))
        return int(match.group(1)) if match else 0
    
    checkpoint_dirs.sort(key=get_step, reverse=True)
    
    # Remove old checkpoints
    for old_dir in checkpoint_dirs[keep_latest_n:]:
        try:
            if os.path.isdir(old_dir):
                shutil.rmtree(old_dir)
            else:
                os.remove(old_dir)
            logger.info(f"Removed old checkpoint: {old_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove old checkpoint {old_dir}: {e}")


def _upload_checkpoint_to_hf_hub(
    checkpoint_dict: dict, 
    step: int, 
    stage: Optional[str], 
    config: dict
) -> bool:
    """Upload checkpoint to Hugging Face Hub."""
    if not HF_HUB_AVAILABLE:
        logger.warning("huggingface_hub not available, skipping HF Hub upload")
        return False
    
    hf_hub_repo_id = config.get("hf_hub_repo_id")
    if not hf_hub_repo_id:
        logger.warning("hf_hub_repo_id not configured, skipping HF Hub upload")
        return False
    
    try:
        api = HfApi(token=config.get("hf_hub_token"))
        
        # Create temporary file for upload
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(checkpoint_dict, f.name)
            temp_path = f.name
        
        # Upload to HF Hub
        filename = f"checkpoint-{step}.pt" if stage is None else f"checkpoint-{stage}-{step}.pt"
        api.upload_file(
            path_or_fileobj=temp_path,
            path_in_repo=f"checkpoints/{filename}",
            repo_id=hf_hub_repo_id,
            commit_message=f"Add checkpoint at step {step}"
        )
        
        # Clean up temp file
        os.remove(temp_path)
        
        logger.info(f"✓ Uploaded checkpoint to HF Hub: {hf_hub_repo_id}/checkpoints/{filename}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to upload to HF Hub: {e}")
        return False


def _upload_checkpoint_to_wandb(
    archive_path: str, 
    step: int, 
    stage: Optional[str], 
    config: dict
):
    """Upload checkpoint archive to wandb."""
    if wandb.run is None:
        logger.warning("wandb run not active, skipping wandb upload")
        return
    
    try:
        artifact_name = f"checkpoint-{step}" if stage is None else f"checkpoint-{stage}-{step}"
        artifact = wandb.Artifact(artifact_name, type="model")
        artifact.add_file(archive_path)
        wandb.log_artifact(artifact)
        logger.info(f"✓ Uploaded checkpoint artifact to wandb: {artifact_name}")
    except Exception as e:
        logger.error(f"Failed to upload to wandb: {e}")


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    config: dict,
    rank: int,
    scaler: Optional[object] = None,
    stage: Optional[str] = None,
    dataloader=None,
    global_step: Optional[int] = None,
    tag: Optional[str] = None,
):
    """
    DCP + Stateful-aware checkpoint saver.
    - Uses dcp.save(...) to write sharded checkpoint files (collective).
    - Optionally gathers full in-memory checkpoint on rank0 for HF Hub upload.
    - Archives directory for wandb upload if requested.
    - No parameter pruning.
    """
    is_rank0 = (rank == 0)

    def _finalize_and_barrier(state_ref=None, checkpoint_dir=None, uploaded=False):
        if state_ref is not None:
            del state_ref
        # free caches
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        # barrier to ensure all ranks synced after save/upload
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if uploaded and checkpoint_dir and is_rank0:
            try:
                shutil.rmtree(checkpoint_dir)
                logger.info(f"Deleted local checkpoint dir: {checkpoint_dir}")
            except Exception:
                logger.warning(f"Could not delete local checkpoint dir: {checkpoint_dir}")
        try:
            model.train()
        except Exception:
            pass
        try:
            optimizer.zero_grad(set_to_none=True)
        except Exception:
            pass
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    # prepare base dir and disk check
    output_dir = config.get("output_dir", "./output")
    base_checkpoint_dir = config.get("checkpoint_dir") or os.path.join(output_dir, "checkpoints")
    stat = shutil.disk_usage(base_checkpoint_dir if os.path.exists(base_checkpoint_dir) else output_dir)
    free_gb = stat.free / (1024 ** 3)
    if free_gb < 2.0:
        logger.warning(f"⚠️ Low disk space: {free_gb:.2f}GB free — attempting cloud-only save")

    # Build StateDictOptions used for dcp.save (we choose sharded/save-per-rank mode for efficiency)
    save_options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
    )

    # Build the Stateful wrapper
    meta = {"epoch": epoch, "step": step}
    if global_step is not None:
        meta["global_step"] = global_step
    if stage is not None:
        meta["stage"] = stage

    save_optimizer_state = config.get("save_optimizer_state", True)
    app_state = AppState(
        model=model, 
        optimizer=optimizer if save_optimizer_state else None,
        scheduler=scheduler if save_optimizer_state else None,
        scaler=scaler if save_optimizer_state else None,
        state_dict_options=save_options,
        meta=meta, 
        dataloader_state=_capture_dataloader_state(dataloader)
    )

    # create a namespace dict (DCP expects mapping of stateful objects)
    state_to_save = {"app": app_state}

    # checkpoint directory for this step
    # Use global_step if available, otherwise use step (step_in_stage)
    if tag:
        checkpoint_dirname = f"checkpoint-{tag}"
    else:
        checkpoint_step = global_step if global_step is not None else step
        checkpoint_dirname = f"checkpoint-{checkpoint_step}"
    
    checkpoint_dir = os.path.join(base_checkpoint_dir, checkpoint_dirname)
    os.makedirs(base_checkpoint_dir, exist_ok=True)

    # Cleanup existing checkpoint directory if it exists (may be corrupted from previous failed save)
    if os.path.exists(checkpoint_dir):
        if is_rank0:
            logger.warning(f"⚠️ Checkpoint directory already exists: {checkpoint_dir}. Removing to prevent corruption...")
        try:
            # Only rank 0 removes to avoid race conditions
            if is_rank0:
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
            # Sync before proceeding
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception as e:
            if is_rank0:
                logger.warning(f"Failed to remove existing checkpoint dir: {e}")

    # Re-check disk space before save
    try:
        stat = shutil.disk_usage(base_checkpoint_dir if os.path.exists(base_checkpoint_dir) else output_dir)
        free_gb = stat.free / (1024 ** 3)
        if free_gb < 1.0:  # Less than 1GB free
            raise RuntimeError(f"Insufficient disk space: {free_gb:.2f}GB free. Need at least 1GB for checkpoint.")
        if is_rank0:
            logger.info(f"Disk space check: {free_gb:.2f}GB free")
    except Exception as e:
        if is_rank0:
            logger.error(f"Disk space check failed: {e}")
        raise

    # CALL DCP SAVE (collective) -- this writes sharded checkpoint files to checkpoint_dir
    # Add retry logic for transient errors (all ranks must participate in retries)
    max_retries = 3
    retry_delay = 2.0  # seconds
    save_success = False
    
    for attempt in range(max_retries):
        try:
            if is_rank0 and attempt > 0:
                logger.info(f"Attempting checkpoint save (attempt {attempt + 1}/{max_retries})...")
            
            # All ranks participate in dcp.save (collective operation)
            dcp.save(state_to_save, checkpoint_id=checkpoint_dir)
            
            # Success - break out of retry loop
            save_success = True
            if is_rank0:
                logger.info(f"✓ Checkpoint saved successfully to {checkpoint_dir}")
            break
            
        except RuntimeError as e:
            error_msg = str(e)
            # Check if it's a file corruption/IO error
            if "unexpected pos" in error_msg or "inline_container" in error_msg:
                if attempt < max_retries - 1:
                    # All ranks need to sync before cleanup/retry
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    
                    # Only rank 0 does cleanup and logging
                    if is_rank0:
                        logger.warning(f"Checkpoint save failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
                        logger.warning(f"Cleaning up and retrying in {retry_delay * (attempt + 1):.1f}s...")
                        try:
                            if os.path.exists(checkpoint_dir):
                                shutil.rmtree(checkpoint_dir, ignore_errors=True)
                        except Exception as cleanup_err:
                            logger.warning(f"Cleanup warning: {cleanup_err}")
                    
                    # All ranks wait before retry (exponential backoff)
                    time.sleep(retry_delay * (attempt + 1))
                    
                    # Sync again before retry
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    continue
                else:
                    # Final attempt failed - all ranks raise
                    if is_rank0:
                        logger.error(f"Checkpoint save failed after {max_retries} attempts: {error_msg}")
                    raise
            else:
                # Different error - don't retry, all ranks raise
                if is_rank0:
                    logger.error(f"Checkpoint save failed with non-retryable error: {error_msg}")
                raise
        except Exception as e:
            # Other errors - log and re-raise (all ranks)
            if is_rank0:
                logger.error(f"Checkpoint save failed with unexpected error: {e}")
            raise
    
    if not save_success:
        if is_rank0:
            logger.error("Checkpoint save failed - no successful save after all retries")
        raise RuntimeError("Failed to save checkpoint after all retry attempts")

    # After dcp.save returns, all ranks have participated and files are on disk.
    # Non-rank0 can finalize and return (we still barrier inside finalize).
    if not is_rank0:
        _finalize_and_barrier(state_ref=None)
        return

    save_optimizer_state = config.get("save_optimizer_state", True)
    full_checkpoint_cache = None
    dcp_state_cache = None

    def _ensure_full_checkpoint_in_memory():
        nonlocal full_checkpoint_cache, dcp_state_cache
        if full_checkpoint_cache is not None:
            return full_checkpoint_cache, dcp_state_cache
        gather_options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        )
        dcp_state_cache = get_state_dict(
            model, 
            [optimizer] if (optimizer is not None and save_optimizer_state) else [], 
            options=gather_options
        )
        if not isinstance(dcp_state_cache, dict) or ("model" not in dcp_state_cache):
            raise RuntimeError("DCP get_state_dict returned unexpected result while materializing checkpoint")
        checkpoint_dict = {"model": dcp_state_cache.get("model")}
        if save_optimizer_state:
            optim_state = dcp_state_cache.get("optimizer") or dcp_state_cache.get("optim")
            if optim_state is not None:
                checkpoint_dict["optimizer"] = optim_state
            if scheduler is not None:
                try:
                    checkpoint_dict["scheduler"] = scheduler.state_dict()
                except Exception:
                    logger.warning("Could not collect scheduler.state_dict() while materializing checkpoint")
            if scaler is not None:
                try:
                    checkpoint_dict["scaler"] = scaler.state_dict()
                except Exception as exc:
                    logger.warning(f"Could not collect GradScaler state while materializing checkpoint: {exc}")
        checkpoint_dict.update(meta)
        full_checkpoint_cache = checkpoint_dict
        return full_checkpoint_cache, dcp_state_cache

    # Rank0: Optionally upload to HF Hub (we need a full in-memory state for single-file upload).
    uploaded_to_hf = False
    if config.get("use_hf_hub", False) and HF_HUB_AVAILABLE:
        checkpoint_in_memory, dcp_res_for_upload = _ensure_full_checkpoint_in_memory()
        uploaded_to_hf = _upload_checkpoint_to_hf_hub(checkpoint_in_memory, step, stage, config)
        if uploaded_to_hf:
            logger.info(f"✓ Checkpoint at step {step} uploaded to HF Hub (no local disk used)")
            if config.get("save_total_limit") is not None:
                _cleanup_old_checkpoints(config, base_checkpoint_dir, keep_latest_n=0)
            # finalize and return (will do barrier)
            _finalize_and_barrier(state_ref=dcp_res_for_upload.get("model"), checkpoint_dir=None, uploaded=True)
            return

    # If not uploaded to HF, prepare local artifact for wandb if requested:
    checkpoint_uploaded = False
    if config.get("use_wandb", False) and config.get("wandb_save_checkpoints", False) and WANDB_AVAILABLE:
        # create a tar.gz archive of the checkpoint dir (rank0 only)
        archive_base = os.path.join(base_checkpoint_dir, f"checkpoint-{step}")
        archive_path = shutil.make_archive(archive_base, 'gztar', root_dir=checkpoint_dir)
        # upload archive file via your helper
        _upload_checkpoint_to_wandb(archive_path, step, stage, config)
        checkpoint_uploaded = True
        # delete archive and optionally the directory
        try:
            os.remove(archive_path)
        except Exception:
            pass
        try:
            shutil.rmtree(checkpoint_dir)
        except Exception:
            pass
        logger.info("✓ Checkpoint uploaded to wandb and local copy cleaned up")

    # Cleanup old checkpoints if save_total_limit is set (only for local checkpoints)
    if not uploaded_to_hf and not checkpoint_uploaded and config.get("save_total_limit") is not None:
        _cleanup_old_checkpoints(config, base_checkpoint_dir)

    state_ref = (dcp_state_cache or {}).get("model") if uploaded_to_hf else None
    _finalize_and_barrier(state_ref=state_ref, checkpoint_dir=checkpoint_dir, uploaded=checkpoint_uploaded)


def save_checkpoint_simple(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    config: dict,
    rank: int,
    avg_loss: Optional[float] = None,
    tag: Optional[str] = None,
):
    """
    Simple checkpoint saver for non-DCP scenarios (single GPU or basic distributed).
    Saves model, optimizer, scheduler, scaler states along with training metadata.
    """
    if rank != 0:
        return
    
    checkpoint_dir = config.get('checkpoint_path', './checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if tag:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{tag}.pt")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}.pt")
    
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
    }
    
    if avg_loss is not None:
        checkpoint["avg_loss"] = avg_loss
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"✓ Checkpoint saved to {checkpoint_path}")
    
    # Cleanup old checkpoints
    save_total_limit = config.get('save_total_limit', 3)
    if save_total_limit:
        checkpoint_files = sorted(
            glob.glob(os.path.join(checkpoint_dir, "checkpoint-*.pt")),
            key=lambda x: int(re.search(r'checkpoint-(?:epoch-)?(\d+)', x).group(1)) if re.search(r'checkpoint-(?:epoch-)?(\d+)', x) else 0,
            reverse=True
        )
        for old_ckpt in checkpoint_files[save_total_limit:]:
            try:
                os.remove(old_ckpt)
                logger.info(f"Removed old checkpoint: {old_ckpt}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {old_ckpt}: {e}")


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device='cuda',
):
    """
    Load checkpoint from file.
    Returns the loaded checkpoint dict for accessing metadata like epoch and global_step.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model"])
    
    if optimizer and "optimizer" in checkpoint and checkpoint["optimizer"]:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    if scheduler and "scheduler" in checkpoint and checkpoint["scheduler"]:
        scheduler.load_state_dict(checkpoint["scheduler"])
    
    if scaler and "scaler" in checkpoint and checkpoint["scaler"]:
        scaler.load_state_dict(checkpoint["scaler"])
    
    logger.info(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}, step {checkpoint.get('global_step', 'N/A')}")
    
    return checkpoint


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.
    Returns the path to the latest checkpoint file, or None if no checkpoints found.
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*.pt"))
    
    if not checkpoint_files:
        return None
    
    # Sort by step number (descending)
    def get_step(path):
        match = re.search(r'checkpoint-(?:epoch-)?(\d+)', os.path.basename(path))
        return int(match.group(1)) if match else 0
    
    checkpoint_files.sort(key=get_step, reverse=True)
    
    return checkpoint_files[0]


def resume_from_checkpoint(
    checkpoint_dir: str,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    device='cuda',
) -> Optional[dict]:
    """
    Resume training from the latest checkpoint in a directory.
    Returns the checkpoint dict if found, None otherwise.
    """
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint is None:
        logger.info(f"No checkpoint found in {checkpoint_dir}, starting from scratch")
        return None
    
    logger.info(f"Resuming from latest checkpoint: {latest_checkpoint}")
    return load_checkpoint(latest_checkpoint, model, optimizer, scheduler, scaler, device)
