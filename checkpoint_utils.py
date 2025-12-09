"""
Checkpoint management utilities for distributed training.
"""

import os
import copy
import logging
import shutil
import io
import time
from typing import Optional, Dict, Any, Union
from collections.abc import Mapping

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

logger = logging.getLogger(__name__)

# Check wandb availability
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Check Hugging Face Hub availability
try:
    from huggingface_hub import HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("huggingface_hub not available, checkpoint upload to HF Hub will be disabled")


def _upload_checkpoint_to_hf_hub(
    checkpoint: dict,
    step: int,
    stage: Optional[str],
    config: dict
) -> bool:
    """
    Upload checkpoint directly to Hugging Face Hub via BytesIO buffer (no disk write).
    """
    if not HF_HUB_AVAILABLE:
        logger.warning("huggingface_hub not available, skipping HF Hub upload")
        return False
    
    use_hf_hub = config.get("use_hf_hub", False)
    repo_id = config.get("hf_hub_repo_id")
    
    if not use_hf_hub or not repo_id:
        return False
    
    try:
        # Get HF token
        hf_token = config.get("hf_hub_token") or os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning("HF token not found (set hf_hub_token or HF_TOKEN env var)")
            return False
        
        logger.info(f"Uploading checkpoint to HF Hub: {repo_id}")
        start_time = time.time()
        
        # Initialize HF API
        api = HfApi()
        
        # Create repo if it doesn't exist
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=config.get("hf_hub_private", True),
                exist_ok=True,
                token=hf_token
            )
            logger.info(f"✓ Repository ready: {repo_id}")
        except Exception as e:
            logger.warning(f"Note: create_repo returned: {e} (may already exist)")
        
        # Save checkpoint to BytesIO buffer (in-memory, no disk write)
        buf = io.BytesIO()
        torch.save(checkpoint, buf)
        buf.seek(0)
        
        # Calculate size for logging
        checkpoint_size_mb = buf.getbuffer().nbytes / (1024 * 1024)
        logger.info(f"Checkpoint size: {checkpoint_size_mb:.2f} MB")
        
        # Upload directly from buffer
        stage_suffix = f"-stage-{stage}" if stage else ""
        path_in_repo = f"checkpoints/checkpoint-{step}{stage_suffix}/pytorch_model.bin"
        
        api.upload_file(
            path_or_fileobj=buf,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
            commit_message=f"Upload checkpoint at step {step}" + (f" (Stage {stage})" if stage else "")
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Successfully uploaded checkpoint to HF Hub (took {elapsed:.1f}s)")
        logger.info(f"  URL: https://huggingface.co/{repo_id}/tree/main/{path_in_repo}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to upload checkpoint to HF Hub: {e}")
        return False


def _upload_checkpoint_to_wandb(
    checkpoint_path: str,
    step: int,
    stage: Optional[str],
    config: dict
):
    """
    Upload checkpoint to wandb as an artifact.
    """
    if not WANDB_AVAILABLE:
        logger.warning("wandb not available, skipping checkpoint upload")
        return
    
    try:
        # Get file size for logging
        file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        logger.info(f"Uploading checkpoint to wandb (size: {file_size_mb:.2f} MB)...")
        
        start_time = time.time()
        
        # Create artifact name
        stage_suffix = f"-stage-{stage}" if stage else ""
        artifact_name = f"model-checkpoint-step-{step}{stage_suffix}"
        
        # Create wandb artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=f"Model checkpoint at step {step}" + (f" (Stage {stage})" if stage else ""),
            metadata={
                "step": step,
                "stage": stage or "unknown",
                "config": config # Save full config
            }
        )
        
        # Add checkpoint file to artifact
        artifact.add_file(checkpoint_path, name="pytorch_model.bin")
        
        # Log artifact to wandb
        wandb.log_artifact(artifact)
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Successfully uploaded checkpoint to wandb: {artifact_name} (took {elapsed:.1f}s)")
        
        # Also log as a simple file for quick access (only if requested)
        if config.get("wandb_save_local", False):
            try:
                wandb.save(checkpoint_path, base_path=os.path.dirname(checkpoint_path))
            except Exception as e:
                logger.debug(f"Note: wandb.save() failed: {e} (artifact upload succeeded)")
            
    except Exception as e:
        logger.error(f"Failed to upload checkpoint to wandb: {e}")
        raise


def _capture_dataloader_state(dataloader) -> Optional[Mapping]:
    """
    Best-effort serialization of a stateful dataloader.
    """
    if dataloader is None:
        return None
    if not hasattr(dataloader, "state_dict"):
        return None
    try:
        state = dataloader.state_dict()
        if state is None:
            return None
        if not isinstance(state, Mapping):
            # Some implementations may return custom objects; allow them as-is
            return state
        return state
    except Exception as exc:
        logger.warning(f"Failed to capture dataloader state_dict: {exc}")
        return None


def _resolve_checkpoint_dir(path: str) -> str:
    """
    Resolve a user-provided checkpoint path to the actual directory that contains
    the torch.distributed.checkpoint shards.
    """
    if not path:
        raise FileNotFoundError("Empty checkpoint path provided")
    
    if os.path.isfile(path):
        candidate = os.path.dirname(path)
        if os.path.basename(candidate).startswith("checkpoint-"):
            return candidate
        raise FileNotFoundError(f"Checkpoint directory not found for file path: {path}")
    
    if os.path.isdir(path):
        if os.path.basename(path).startswith("checkpoint-"):
            return path
        # look for checkpoint-* subdirectories
        subdirs = []
        for entry in os.listdir(path):
            full_entry = os.path.join(path, entry)
            if os.path.isdir(full_entry) and entry.startswith("checkpoint-"):
                try:
                    step_val = int(entry.split("-")[-1])
                except Exception:
                    step_val = -1
                subdirs.append((step_val, full_entry))
        if not subdirs:
            # Maybe it IS the checkpoint dir (e.g. if passed exact path)
            if _has_dcp_artifacts(path):
                return path
            raise FileNotFoundError(f"No checkpoint-* directories found under {path}")
        subdirs.sort(key=lambda x: x[0])
        return subdirs[-1][1]
    
    raise FileNotFoundError(f"Checkpoint path does not exist: {path}")


def _has_dcp_artifacts(checkpoint_dir: str) -> bool:
    """Check whether a checkpoint directory contains DCP shard artifacts."""
    if not os.path.exists(checkpoint_dir):
        return False
    metadata_path = os.path.join(checkpoint_dir, ".metadata")
    if os.path.isfile(metadata_path):
        return True
    try:
        for entry in os.listdir(checkpoint_dir):
            if entry.startswith("__") and entry.endswith(".distcp"):
                return True
    except Exception:
        pass
    return False


def _maybe_init_process_group_for_checkpoint() -> bool:
    """Ensure torch.distributed process group exists for single-rank DCP usage."""
    if not dist.is_available():
        return False
    if dist.is_initialized():
        return False
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    try:
        dist.init_process_group(backend=backend, rank=0, world_size=1)
        return True
    except Exception as e:
        logger.warning(f"Failed to init process group: {e}")
        return False


class AppState(Stateful):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        state_dict_options: Optional[StateDictOptions] = None,
        meta: Optional[dict] = None,
        dataloader_state: Optional[dict] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.state_dict_options = state_dict_options
        self.meta = meta or {}
        self.dataloader_state = dataloader_state

    def state_dict(self):
        # Return serializable state for DCP.
        model_state, optim_state = get_state_dict(self.model, [self.optimizer] if self.optimizer is not None else [], options=self.state_dict_options)
        out = {"model": model_state}
        if optim_state is not None:
            out["optim"] = optim_state
        # Save meta info (epoch/step/stage) here as well
        out["meta"] = dict(self.meta)
        
        try:
            if self.scheduler is not None:
                out["scheduler"] = self.scheduler.state_dict()
        except Exception:
            pass
        try:
            if self.scaler is not None:
                out["scaler"] = self.scaler.state_dict()
        except Exception:
            pass
        if self.dataloader_state is not None:
            out["dataloader"] = self.dataloader_state
        return out

    def load_state_dict(self, state_dict):
        # Use set_state_dict to restore model+optimizer shards
        set_state_dict(self.model, self.optimizer, model_state_dict=state_dict.get("model"), optim_state_dict=state_dict.get("optim"), options=self.state_dict_options)
        # restore scheduler/scaler/meta if present
        if "scheduler" in state_dict and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(state_dict["scheduler"])
            except Exception:
                logger.warning("Failed to load scheduler state dict during restore")
        if "scaler" in state_dict and self.scaler is not None:
            try:
                self.scaler.load_state_dict(state_dict["scaler"])
            except Exception:
                logger.warning("Failed to load scaler state dict during restore")
        if "meta" in state_dict:
            self.meta.update(state_dict["meta"])
        if "dataloader" in state_dict:
            self.dataloader_state = state_dict["dataloader"]


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
    tag: Optional[str] = None
):
    """
    DCP + Stateful-aware checkpoint saver.
    """
    is_rank0 = (rank == 0)

    def _finalize_and_barrier(state_ref=None, checkpoint_dir=None, uploaded=False):
        if state_ref is not None:
            del state_ref
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
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
            # We don't want to zero grad generally if we are just saving, but okay to be safe if at end of step
            pass 
        except Exception:
            pass
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    output_dir = config.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)
    base_checkpoint_dir = config.get("checkpoint_dir") or os.path.join(output_dir, "checkpoints")
    
    try:
        stat = shutil.disk_usage(base_checkpoint_dir if os.path.exists(base_checkpoint_dir) else output_dir)
        free_gb = stat.free / (1024 ** 3)
        if free_gb < 2.0:
            logger.warning(f"⚠️ Low disk space: {free_gb:.2f}GB free — attempting cloud-only save")
    except Exception:
        pass

    save_options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        ignore_frozen_params=False,
        keep_submodule_prefixes=True,
        strict=True,
        broadcast_from_rank0=True,
        flatten_optimizer_state_dict=False
    )

    meta = {"epoch": epoch, "step": step}
    if global_step is not None:
        meta["global_step"] = global_step
    if stage is not None:
        meta["stage"] = stage

    save_optimizer_state = config.get("save_optimizer_state", True)
    
    app_state = AppState(
        model=model, 
        optimizer=optimizer if save_optimizer_state else None,
        scheduler=(scheduler if save_optimizer_state else None),
        scaler=scaler if save_optimizer_state else None,
        state_dict_options=save_options,
        meta=meta, 
        dataloader_state=_capture_dataloader_state(dataloader)
    )

    state_to_save = {"app": app_state}

    checkpoint_step = global_step if global_step is not None else step
    if tag:
        checkpoint_dirname = f"checkpoint-{tag}"
    else:
        checkpoint_dirname = f"checkpoint-{checkpoint_step}"
        
    checkpoint_dir = os.path.join(base_checkpoint_dir, checkpoint_dirname)
    os.makedirs(base_checkpoint_dir, exist_ok=True)

    if os.path.exists(checkpoint_dir):
        if is_rank0:
            logger.warning(f"⚠️ Checkpoint directory already exists: {checkpoint_dir}. Removing...")
        try:
            if is_rank0:
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception as e:
            if is_rank0:
                logger.warning(f"Failed to remove existing checkpoint dir: {e}")

    # Re-check disk space
    try:
        stat = shutil.disk_usage(base_checkpoint_dir if os.path.exists(base_checkpoint_dir) else output_dir)
        free_gb = stat.free / (1024 ** 3)
        if free_gb < 1.0:
            raise RuntimeError(f"Insufficient disk space: {free_gb:.2f}GB free. Need at least 1GB.")
    except Exception:
        pass

    # Save logic with retry
    max_retries = 3
    retry_delay = 2.0
    save_success = False
    
    for attempt in range(max_retries):
        try:
            if is_rank0 and attempt > 0:
                logger.info(f"Attempting checkpoint save (attempt {attempt + 1}/{max_retries})...")
            
            dcp.save(state_to_save, checkpoint_id=checkpoint_dir)
            
            save_success = True
            if is_rank0:
                logger.info(f"✓ Checkpoint saved successfully to {checkpoint_dir}")
            break
            
        except RuntimeError as e:
            error_msg = str(e)
            if "unexpected pos" in error_msg or "inline_container" in error_msg:
                if attempt < max_retries - 1:
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    if is_rank0:
                        logger.warning(f"Checkpoint save failed (retryable): {error_msg}")
                        try:
                            if os.path.exists(checkpoint_dir):
                                shutil.rmtree(checkpoint_dir, ignore_errors=True)
                        except Exception:
                            pass
                    time.sleep(retry_delay * (attempt + 1))
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    continue
                else:
                    raise
            else:
                if is_rank0:
                    logger.error(f"Checkpoint save failed with non-retryable error: {error_msg}")
                raise
        except Exception as e:
            if is_rank0:
                logger.error(f"Checkpoint save failed with unexpected error: {e}")
            raise
    
    if not save_success:
        if is_rank0:
            logger.error("Checkpoint save failed - no successful save after all retries")
        raise RuntimeError("Failed to save checkpoint after all retry attempts")

    if not is_rank0:
        _finalize_and_barrier(state_ref=None)
        return

    # Post-save actions (rank 0 only)
    full_checkpoint_cache = None
    dcp_state_cache = None

    def _ensure_full_checkpoint_in_memory():
        nonlocal full_checkpoint_cache, dcp_state_cache
        if full_checkpoint_cache is not None:
            return full_checkpoint_cache, dcp_state_cache
        gather_options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            ignore_frozen_params=False,
            keep_submodule_prefixes=True,
            strict=True,
            broadcast_from_rank0=True,
            flatten_optimizer_state_dict=False,
        )
        dcp_state_cache = get_state_dict(model, [optimizer] if (optimizer is not None and save_optimizer_state) else [], options=gather_options)
        checkpoint_dict = {"model": dcp_state_cache.get("model")}
        if save_optimizer_state:
            optim_state = dcp_state_cache.get("optimizer") or dcp_state_cache.get("optim")
            if optim_state is not None:
                checkpoint_dict["optimizer"] = optim_state
            if scheduler is not None:
                try:
                    checkpoint_dict["scheduler"] = scheduler.state_dict()
                except Exception:
                    pass
            if scaler is not None:
                try:
                    checkpoint_dict["scaler"] = scaler.state_dict()
                except Exception:
                    pass
        checkpoint_dict.update(meta)
        full_checkpoint_cache = checkpoint_dict
        return full_checkpoint_cache, dcp_state_cache

    uploaded_to_hf = False
    if config.get("use_hf_hub", False) and HF_HUB_AVAILABLE:
        checkpoint_in_memory, dcp_res_for_upload = _ensure_full_checkpoint_in_memory()
        uploaded_to_hf = _upload_checkpoint_to_hf_hub(checkpoint_in_memory, step, stage, config)
        if uploaded_to_hf:
            logger.info(f"✓ Checkpoint at step {step} uploaded to HF Hub")
            if config.get("save_total_limit") is not None:
                _cleanup_old_checkpoints(config, base_checkpoint_dir, keep_latest_n=0)
            _finalize_and_barrier(state_ref=dcp_res_for_upload.get("model"), checkpoint_dir=None, uploaded=True)
            return

    checkpoint_uploaded = False
    if config.get("use_wandb", False) and config.get("wandb_save_checkpoints", False) and WANDB_AVAILABLE:
        archive_base = os.path.join(base_checkpoint_dir, f"checkpoint-{step}")
        archive_path = shutil.make_archive(archive_base, 'gztar', root_dir=checkpoint_dir)
        _upload_checkpoint_to_wandb(archive_path, step, stage, config)
        checkpoint_uploaded = True
        try:
            os.remove(archive_path)
        except Exception:
            pass
        
    if not uploaded_to_hf and not checkpoint_uploaded and config.get("save_total_limit") is not None:
        _cleanup_old_checkpoints(config, base_checkpoint_dir)

    _finalize_and_barrier(state_ref=None, checkpoint_dir=checkpoint_dir, uploaded=checkpoint_uploaded)


def resume_training_state(
    checkpoint_path: str,
    model,
    optimizer=None,
    scheduler=None,
    config: dict = None,
    dataloader = None,
    scaler: Optional[object] = None,
) -> dict:
    """
    Restore training state from a DCP checkpoint.
    """
    if config is None:
        config = {} # Safe default
    
    checkpoint_dir = _resolve_checkpoint_dir(checkpoint_path)
    load_options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        ignore_frozen_params=False,
        keep_submodule_prefixes=True,
        strict=True,
        broadcast_from_rank0=True,
        flatten_optimizer_state_dict=False
    )
    
    save_opt = config.get("save_optimizer_state", True)
    load_optimizer_state = optimizer is not None and save_opt
    load_scheduler_state = scheduler is not None and save_opt
    load_scaler_state = scaler is not None and save_opt
    
    app_state = AppState(
        model=model,
        optimizer=optimizer if load_optimizer_state else None,
        scheduler=scheduler if load_scheduler_state else None,
        scaler=scaler if load_scaler_state else None,
        state_dict_options=load_options,
    )
    state_map = {"app": app_state}
    
    logger.info(f"Loading training state from checkpoint directory: {checkpoint_dir}")
    dcp.load(state_map, checkpoint_id=checkpoint_dir)
    
    if dataloader is not None and app_state.dataloader_state is not None:
        if hasattr(dataloader, "load_state_dict"):
            try:
                dataloader.load_state_dict(app_state.dataloader_state)
                logger.info("✓ Restored dataloader state from checkpoint")
            except Exception as exc:
                logger.warning(f"Failed to load dataloader state_dict: {exc}")
        else:
            logger.warning("Provided dataloader does not implement load_state_dict; skipping dataloader resume")
    
    meta = dict(app_state.meta)
    
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    
    logger.info(f"✓ Training state loaded: {meta}")
    return meta


def _cleanup_old_checkpoints(config: dict, checkpoint_dir: Optional[str] = None, keep_latest_n: Optional[int] = None):
    """Remove old LOCAL checkpoints exceeding save_total_limit"""
    if checkpoint_dir is None:
        checkpoint_dir = config.get("checkpoint_dir") or os.path.join(config.get("output_dir", "./output"), "checkpoints")
    
    if not os.path.exists(checkpoint_dir):
        return
    
    def _try_parse_step(name):
        try:
            return int(name.split("-")[-1])
        except Exception:
            return None

    checkpoints = []
    try:
        for d in os.listdir(checkpoint_dir):
            if d.startswith("checkpoint-"):
                step_num = _try_parse_step(d)
                if step_num is not None:
                    checkpoints.append((step_num, d))
    except Exception as e:
        logger.warning(f"Failed to list checkpoints in {checkpoint_dir}: {e}")
        return

    checkpoints = sorted(checkpoints, key=lambda x: x[0])

    limit = keep_latest_n if keep_latest_n is not None else config.get("save_total_limit")
    if limit is not None and len(checkpoints) > limit:
        to_remove = checkpoints[:-limit] if limit > 0 else checkpoints
        for (step_num, old_checkpoint) in to_remove:
            old_path = os.path.join(checkpoint_dir, old_checkpoint)
            try:
                logger.info(f"Removing old checkpoint: {old_path}")
                shutil.rmtree(old_path)
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {old_path}: {e}")


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory (supports both .pt files and DCP directories).
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Find all candidates (dirs starting with checkpoint-)
    candidates = []
    for d in os.listdir(checkpoint_dir):
        if d.startswith("checkpoint-"):
            candidates.append(os.path.join(checkpoint_dir, d))
            
    if not candidates:
        return None
        
    def get_step(path):
        base = os.path.basename(path)
        # Remove extension if any
        base = os.path.splitext(base)[0]
        try:
            return int(base.split("-")[-1])
        except:
            return -1
    
    candidates.sort(key=get_step, reverse=True)
    if candidates:
        return candidates[0]
    return None

def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    strict: bool = True,
    dataloader = None,
    config: Optional[dict] = None,
    use_dcp: Optional[bool] = None,
) -> dict:
    """
    Load checkpoint from disk or DCP shards.
    """
    original_path = checkpoint_path
    checkpoint_path = os.path.abspath(checkpoint_path)
    
    checkpoint_file = checkpoint_path
    if os.path.isdir(checkpoint_path):
        checkpoint_file = os.path.join(checkpoint_path, "pytorch_model.bin")
    file_checkpoint_exists = os.path.isfile(checkpoint_file)
    
    dcp_checkpoint_dir = None
    try:
        dcp_checkpoint_dir = _resolve_checkpoint_dir(checkpoint_path)
    except FileNotFoundError:
        pass
    
    has_dcp = dcp_checkpoint_dir is not None and _has_dcp_artifacts(dcp_checkpoint_dir)
    
    if use_dcp is None:
        use_dcp = has_dcp and not file_checkpoint_exists
    
    if use_dcp:
        if not has_dcp:
            raise FileNotFoundError(f"DCP checkpoint not found at: {original_path}")
        config = config or {}
        
        # If no distributed, temporarily init for loading
        initialized_pg = False
        try:
            initialized_pg = _maybe_init_process_group_for_checkpoint()
            metadata = resume_training_state(
                checkpoint_path=dcp_checkpoint_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                dataloader=dataloader,
                scaler=scaler,
            )
        finally:
            if initialized_pg and dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass
        return metadata
    
    # Standard torch.load logic (fallback for non-DCP)
    if not file_checkpoint_exists:
        raise FileNotFoundError(f"Checkpoint not found: {original_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    
    # Handle DCP-like dictionary if saved via simple save using AppState structure
    if "app" in checkpoint and isinstance(checkpoint["app"], dict):
        app_state = checkpoint["app"]
        if "model" in app_state:
            checkpoint_model = app_state["model"]
        else:
            checkpoint_model = None
            
        if "meta" in app_state:
            meta = app_state["meta"]
        else:
            meta = {}
            
        new_checkpoint = {"model": checkpoint_model} if checkpoint_model else {}
        if "optim" in app_state: new_checkpoint["optimizer"] = app_state["optim"]
        if "scheduler" in app_state: new_checkpoint["scheduler"] = app_state["scheduler"]
        if "scaler" in app_state: new_checkpoint["scaler"] = app_state["scaler"]
        
        new_checkpoint.update(meta)
        checkpoint = new_checkpoint

    if "model" in checkpoint:
        try:
            model.load_state_dict(checkpoint["model"], strict=strict)
        except RuntimeError as e:
            logger.error(f"Failed to load model state dict: {e}")
            if strict:
                 logger.info("Retrying with strict=False...")
                 model.load_state_dict(checkpoint["model"], strict=False)
    
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
        
    metadata = {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "stage": checkpoint.get("stage", None),
        "global_step": checkpoint.get("global_step", 0)
    }
    return metadata