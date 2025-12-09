"""
Checkpoint management utilities for SeamlessM4T v2 training
Extracted from train_kaggle.py for better modularity
"""

import os
import copy
import logging
import shutil
import io
import threading
from typing import Optional, Union, Dict, Any
from pathlib import Path
import time
from collections.abc import Mapping

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torchdata.stateful_dataloader import StatefulDataLoader

import torch

logger = logging.getLogger(__name__)


class ConfigWrapper:
    """Helper to wrap dict config and access as attributes"""
    def __init__(self, config: Union[Dict[str, Any], object]):
        self._config = config

    def __getattr__(self, name):
        # If it's a dict, try to get key
        if isinstance(self._config, dict):
            # Check for direct key
            if name in self._config:
                return self._config[name]
            # Handle nested wandb config
            if name.startswith("wandb_") and "wandb" in self._config:
                 sub_key = name.replace("wandb_", "")
                 if isinstance(self._config["wandb"], dict) and sub_key in self._config["wandb"]:
                     return self._config["wandb"][sub_key]
            
            # Map some common mismatches
            if name == "checkpoint_dir":
                return self._config.get("checkpoint_path", self._config.get("output_dir", "./output/checkpoints"))
            if name == "output_dir":
                return self._config.get("output_dir", "./output")
            if name == "use_hf_hub":
                 return self._config.get("push_to_hub", False)
            
            # Return None for missing keys instead of raising AttributeError, 
            # effectively acting as "default=None"
            return self._config.get(name, None)
        
        # If it's an object, just getattr
        return getattr(self._config, name)

    def get(self, key, default=None):
        val = getattr(self, key)
        return val if val is not None else default


def _load_env_file(env_file: str = "key.env"):
    """Load environment variables from .env file if exists"""
    env_path = Path(env_file)
    
    if not env_path.exists():
        return False
    
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value
        
        return True
    except Exception:
        return False


# Try to load .env file on module import
_load_env_file("key.env") or _load_env_file(".env")

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
    config: Union[Dict, object]
) -> bool:
    """
    Upload checkpoint directly to Hugging Face Hub via BytesIO buffer (no disk write).
    
    Args:
        checkpoint: Checkpoint dictionary to upload
        step: Training step number
        stage: Current training stage (A/B/C)
        config: Training configuration
        
    Returns:
        True if upload succeeded, False otherwise
    """
    if not isinstance(config, ConfigWrapper):
        config = ConfigWrapper(config)

    if not HF_HUB_AVAILABLE:
        logger.warning("huggingface_hub not available, skipping HF Hub upload")
        return False
    
    if not config.use_hf_hub or not config.hf_hub_repo_id:
        return False
    
    try:
        # Get HF token
        hf_token = config.hf_hub_token or os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning("HF token not found (set hf_hub_token or HF_TOKEN env var)")
            return False
        
        logger.info(f"Uploading checkpoint to HF Hub: {config.hf_hub_repo_id}")
        start_time = time.time()
        
        # Initialize HF API
        api = HfApi()
        
        # Create repo if it doesn't exist
        try:
            api.create_repo(
                repo_id=config.hf_hub_repo_id,
                repo_type="model",
                private=config.hf_hub_private,
                exist_ok=True,
                token=hf_token
            )
            logger.info(f"✓ Repository ready: {config.hf_hub_repo_id}")
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
            repo_id=config.hf_hub_repo_id,
            repo_type="model",
            token=hf_token,
            commit_message=f"Upload checkpoint at step {step}" + (f" (Stage {stage})" if stage else "")
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Successfully uploaded checkpoint to HF Hub (took {elapsed:.1f}s)")
        logger.info(f"  URL: https://huggingface.co/{config.hf_hub_repo_id}/tree/main/{path_in_repo}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to upload checkpoint to HF Hub: {e}")
        return False


def _upload_checkpoint_to_wandb(
    checkpoint_path: str,
    step: int,
    stage: Optional[str],
    config: Union[Dict, object]
):
    """
    Upload checkpoint to wandb as an artifact.
    
    Args:
        checkpoint_path: Path to the saved checkpoint file
        step: Training step number
        stage: Current training stage (A/B/C)
        config: Training configuration
    """
    if not isinstance(config, ConfigWrapper):
        config = ConfigWrapper(config)

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
                "config": {
                    "encoder_lr": config.encoder_lr,
                    "is_pretrained": config.is_pretrained,
                    "enable_curriculum": config.enable_curriculum,
                    "unfreeze_top_k": config.unfreeze_top_k,
                }
            }
        )
        
        # Add checkpoint file to artifact
        artifact.add_file(checkpoint_path, name="pytorch_model.bin")
        
        # Log artifact to wandb
        wandb.log_artifact(artifact)
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Successfully uploaded checkpoint to wandb: {artifact_name} (took {elapsed:.1f}s)")
        
        # Also log as a simple file for quick access
        try:
            wandb.save(checkpoint_path, base_path=os.path.dirname(checkpoint_path))
        except Exception as e:
            logger.debug(f"Note: wandb.save() failed: {e} (artifact upload succeeded)")
            
    except Exception as e:
        logger.error(f"Failed to upload checkpoint to wandb: {e}")
        raise


def _capture_dataloader_state(dataloader: Optional[StatefulDataLoader]) -> Optional[Mapping]:
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


def _log_rank_value_discrepancy(name: str, value, rank: int):
    """Gather a value across ranks and log discrepancies for debugging."""
    if not dist.is_available() or not dist.is_initialized():
        return
    try:
        world_size = dist.get_world_size()
    except Exception:
        return
    gathered = [None for _ in range(world_size)]
    try:
        dist.all_gather_object(gathered, value)
    except Exception as exc:
        if rank == 0:
            logger.debug(f"[Checkpoint Debug] Failed to gather {name}: {exc}")
        return
    unique_values: dict = {}
    for idx, val in enumerate(gathered):
        unique_values.setdefault(val, []).append(idx)
    if len(unique_values) > 1:
        if rank == 0:
            logger.error(f"[Checkpoint Debug] {name} mismatch across ranks: {unique_values}")
    else:
        if rank == 0:
            logger.debug(f"[Checkpoint Debug] {name} consistent across ranks: {value}")


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
            raise FileNotFoundError(f"No checkpoint-* directories found under {path}")
        subdirs.sort(key=lambda x: x[0])
        return subdirs[-1][1]
    
    raise FileNotFoundError(f"Checkpoint path does not exist: {path}")


def _has_dcp_artifacts(checkpoint_dir: str) -> bool:
    """Check whether a checkpoint directory contains DCP shard artifacts."""
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
    dist.init_process_group(backend=backend, rank=0, world_size=1)
    return True


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
        # Return serializable state for DCP. Use get_state_dict to collect model+optimizer in DCP-friendly form.
        # We intentionally do not force full_state_dict here (dcp.save will be used for sharded save).
        model_state, optim_state = get_state_dict(self.model, [self.optimizer] if self.optimizer is not None else [], options=self.state_dict_options)
        out = {"model": model_state}
        if optim_state is not None:
            out["optim"] = optim_state
        # Save meta info (epoch/step/stage) here as well
        out["meta"] = dict(self.meta)
        # Scheduler/scaler are often non-sharded; include their states if present (not DCP-managed)
        try:
            if self.scheduler is not None:
                out["scheduler"] = self.scheduler.state_dict()
        except Exception:
            # best-effort
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
    config: Union[Dict, object],
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
    # Wrap config
    if not isinstance(config, ConfigWrapper):
        config = ConfigWrapper(config)
    
    # Use tag as stage if stage is not provided
    if stage is None and tag is not None:
        stage = tag

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
    os.makedirs(output_dir, exist_ok=True)
    base_checkpoint_dir = config.get("checkpoint_dir") or os.path.join(output_dir, "checkpoints")
    stat = shutil.disk_usage(base_checkpoint_dir if os.path.exists(base_checkpoint_dir) else output_dir)
    free_gb = stat.free / (1024 ** 3)
    if free_gb < 2.0:
        logger.warning(f"⚠️ Low disk space: {free_gb:.2f}GB free — attempting cloud-only save")

    # Build StateDictOptions used for dcp.save (we choose sharded/save-per-rank mode for efficiency)
    save_options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        ignore_frozen_params=False,
        keep_submodule_prefixes=True,
        strict=True,
        broadcast_from_rank0=True,
        flatten_optimizer_state_dict=False
    )

    # Build the Stateful wrapper
    meta = {"epoch": epoch, "step": step}
    if global_step is not None:
        meta["global_step"] = global_step
    if stage is not None:
        meta["stage"] = stage

    save_optimizer_state = config.get("save_optimizer_state", True)
    app_state = AppState(model=model, optimizer=optimizer if save_optimizer_state else None,
                         scheduler=(scheduler if save_optimizer_state else None),
                         scaler=scaler if save_optimizer_state else None,
                         state_dict_options=save_options,
                         meta=meta, dataloader_state=_capture_dataloader_state(dataloader))

    # create a namespace dict (DCP expects mapping of stateful objects)
    state_to_save = {"app": app_state}

    # checkpoint directory for this step
    # Use global_step if available, otherwise use step (step_in_stage)
    checkpoint_step = global_step if global_step is not None else step
    checkpoint_dir = os.path.join(base_checkpoint_dir, f"checkpoint-{checkpoint_step}")
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
        stat = shutil.disk_usage(base_checkpoint_dir if os.path.exists(base_checkpoint_dir) else config.output_dir)
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
            ignore_frozen_params=False,
            keep_submodule_prefixes=True,
            strict=True,
            broadcast_from_rank0=True,
            flatten_optimizer_state_dict=False,
        )
        dcp_state_cache = get_state_dict(model, [optimizer] if (optimizer is not None and save_optimizer_state) else [], options=gather_options)
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
            if config.get("save_total_limit", None) is not None:
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
    if not uploaded_to_hf and not checkpoint_uploaded and config.get("save_total_limit", None) is not None:
        _cleanup_old_checkpoints(config, base_checkpoint_dir)

    state_ref = (dcp_state_cache or {}).get("model") if uploaded_to_hf else None
    _finalize_and_barrier(state_ref=state_ref, checkpoint_dir=checkpoint_dir, uploaded=checkpoint_uploaded)


def resume_training_state(
    checkpoint_path: str,
    model,
    optimizer=None,
    scheduler=None,
    config: Union[Dict, object] = None,
    dataloader: Optional[StatefulDataLoader] = None,
    scaler: Optional[object] = None,
) -> dict:
    """
    Restore training state (model/optimizer/scheduler/scaler + dataloader) from a DCP checkpoint.
    
    Args:
        checkpoint_path: Path to a checkpoint directory or the parent "checkpoints" directory.
        model: Model instance to load weights into.
        optimizer: Optional optimizer instance.
        scheduler: Optional LR scheduler.
        config: Training configuration (used for logging + options).
        dataloader: Optional StatefulDataLoader whose state should be restored.
        scaler: Optional GradScaler instance.
    
    Returns:
        Metadata dictionary stored in the checkpoint (epoch, step, stage, etc.).
    """
    if config is None:
        raise ValueError("TrainingConfig must be provided when resuming from checkpoint")
    
    if not isinstance(config, ConfigWrapper):
        config = ConfigWrapper(config)
    
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
    
    load_optimizer_state = optimizer is not None and config.get("save_optimizer_state", True)
    load_scheduler_state = scheduler is not None and config.get("save_optimizer_state", True)
    load_scaler_state = scaler is not None and config.get("save_optimizer_state", True)
    
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
    if "stage" not in meta:
        meta["stage"] = None
    if "step" not in meta:
        meta["step"] = 0
    if "global_step" not in meta:
        meta["global_step"] = 0
    
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    
    logger.info(
        f"✓ Training state loaded: stage={meta.get('stage')}, step={meta.get('step')}, "
        f"global_step={meta.get('global_step')}"
    )
    return meta


def _cleanup_old_checkpoints(config: Union[Dict, object], checkpoint_dir: Optional[str] = None, keep_latest_n: Optional[int] = None):
    """Remove old LOCAL checkpoints exceeding save_total_limit
    
    Note: This only removes local checkpoint directories. Checkpoints uploaded to wandb/HF Hub
    are stored in the cloud and are NOT affected by this cleanup.
    
    Args:
        config: Training configuration
        checkpoint_dir: Directory containing checkpoints (if None, uses config.checkpoint_dir or output_dir/checkpoints)
        keep_latest_n: If specified, keep only this many latest checkpoints (otherwise use config.save_total_limit)
    """
    if not isinstance(config, ConfigWrapper):
        config = ConfigWrapper(config)

    # Determine checkpoint directory
    if checkpoint_dir is None:
        if config.get("checkpoint_dir", None) is not None:
            checkpoint_dir = config.checkpoint_dir
        else:
            checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
    
    # Check if directory exists
    if not os.path.exists(checkpoint_dir):
        return
    
    # find checkpoint-* dirs and parse step numbers robustly
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

    checkpoints = sorted(checkpoints, key=lambda x: x[0])  # sort by step number

    limit = keep_latest_n if keep_latest_n is not None else config.save_total_limit
    if len(checkpoints) > limit:
        to_remove = checkpoints[:-limit] if limit > 0 else checkpoints
        for (step_num, old_checkpoint) in to_remove:
            old_path = os.path.join(checkpoint_dir, old_checkpoint)
            try:
                logger.info(f"Removing old checkpoint: {old_path}")
                shutil.rmtree(old_path)
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {old_path}: {e}")


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    strict: bool = True,
    dataloader: Optional[StatefulDataLoader] = None,
    config: Optional[Union[Dict, object]] = None,
    use_dcp: Optional[bool] = None,
) -> dict:
    """
    Load checkpoint from disk or DCP shards.
    
    Args:
        checkpoint_path: Path to checkpoint file or directory
        model: Model to load state dict into
        optimizer: Optional optimizer to load state dict into
        scheduler: Optional scheduler to load state dict into
        scaler: Optional GradScaler to load state dict into
        strict: Whether to strictly enforce state dict keys match
        dataloader: Optional StatefulDataLoader to restore (DCP only)
        config: Training configuration (required for DCP restore; auto-created if None)
        use_dcp: Force using DCP loader (True/False). If None, auto-detect.
        
    Returns:
        Dictionary containing checkpoint metadata (epoch, step, stage, etc.)
    """
    if config is not None and not isinstance(config, ConfigWrapper):
        config = ConfigWrapper(config)

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
        config = config or ConfigWrapper({})
        if optimizer is None and config.get("save_optimizer_state", True):
            # We can't deepcopy ConfigWrapper easily if it wraps a dict, 
            # but we can create a new one with modified dict
            if isinstance(config, ConfigWrapper) and isinstance(config._config, dict):
                 new_conf_dict = copy.deepcopy(config._config)
                 new_conf_dict["save_optimizer_state"] = False
                 config = ConfigWrapper(new_conf_dict)
            else:
                config = copy.deepcopy(config)
                try:
                    config.save_optimizer_state = False
                except Exception:
                    pass
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
    
    if not file_checkpoint_exists:
        raise FileNotFoundError(f"Checkpoint not found: {original_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    
    # Handle DCP format: {"app": {"model": ..., "meta": {...}, ...}}
    # Convert to regular format: {"model": ..., "epoch": ..., "step": ..., "stage": ...}
    if "app" in checkpoint and isinstance(checkpoint["app"], dict):
        app_state = checkpoint["app"]
        logger.info("Detected DCP format checkpoint (with 'app' wrapper), converting to regular format...")
        # Extract model state dict
        if "model" in app_state:
            checkpoint_model = app_state["model"]
        else:
            # Fallback: check if it's already unwrapped
            checkpoint_model = None
        
        # Extract metadata
        if "meta" in app_state:
            meta = app_state["meta"]
            checkpoint_epoch = meta.get("epoch", 0)
            checkpoint_step = meta.get("step", 0)
            checkpoint_stage = meta.get("stage", None)
        else:
            checkpoint_epoch = 0
            checkpoint_step = 0
            checkpoint_stage = None
        
        # Reconstruct checkpoint in regular format
        new_checkpoint = {"model": checkpoint_model} if checkpoint_model else {}
        
        # Copy optimizer/scheduler/scaler if present
        if "optim" in app_state or "optimizer" in app_state:
            new_checkpoint["optimizer"] = app_state.get("optim") or app_state.get("optimizer")
        if "scheduler" in app_state:
            new_checkpoint["scheduler"] = app_state["scheduler"]
        if "scaler" in app_state:
            new_checkpoint["scaler"] = app_state["scaler"]
        
        # Add metadata at top level
        new_checkpoint["epoch"] = checkpoint_epoch
        new_checkpoint["step"] = checkpoint_step
        new_checkpoint["stage"] = checkpoint_stage
        
        checkpoint = new_checkpoint
        logger.info(f"Converted DCP format: epoch={checkpoint_epoch}, step={checkpoint_step}, stage={checkpoint_stage}")
    
    # Load model state dict
    if "model" in checkpoint:
        # Note: For SeamlessM4Tv2ForSpeechToText: no text_encoder exists, so no keys to skip
        # For SeamlessM4Tv2ForTextToText: text_encoder may be missing if it was excluded
        # The model class has _keys_to_ignore_on_load_missing = ["text_encoder", "t2u_model", "vocoder"]
        # so it will automatically skip missing text_encoder keys if present
        
        try:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=strict)
            
            # Log missing keys (excluding those in _keys_to_ignore_on_load_missing)
            if missing_keys:
                ignored_keys = getattr(model, "_keys_to_ignore_on_load_missing", [])
                actual_missing = [k for k in missing_keys if not any(k.startswith(prefix) for prefix in ignored_keys)]
                
                if actual_missing:
                    logger.warning(f"⚠️  {len(actual_missing)} keys were not initialized from checkpoint:")
                    for key in actual_missing[:10]:
                        logger.warning(f"      - {key}")
                    if len(actual_missing) > 10:
                        logger.warning(f"      ... and {len(actual_missing) - 10} more")
                
                if len(actual_missing) < len(missing_keys):
                    ignored_count = len(missing_keys) - len(actual_missing)
                    logger.info(f"ℹ️  {ignored_count} keys ignored (in _keys_to_ignore_on_load_missing)")
            
            # Log unexpected keys
            if unexpected_keys:
                logger.warning(f"⚠️  {len(unexpected_keys)} unexpected keys in checkpoint (will be ignored):")
                for key in unexpected_keys[:10]:
                    logger.warning(f"      - {key}")
                if len(unexpected_keys) > 10:
                    logger.warning(f"      ... and {len(unexpected_keys) - 10} more")
            
            # Check if model has text_encoder to provide appropriate log message
            has_text_encoder = hasattr(model, "text_encoder")
            if has_text_encoder:
                logger.info("✓ Model state dict loaded (text_encoder uses pretrained weights if excluded)")
            else:
                logger.info("✓ Model state dict loaded (SeamlessM4Tv2ForSpeechToText)")
                
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                logger.error(f"❌ Failed to load model state dict: {e}")
                logger.error("This may happen if checkpoint was saved with different model architecture")
                if strict:
                    logger.info("Retrying with strict=False...")
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
                    logger.warning("✓ Loaded with strict=False (some keys may be missing or unexpected)")
                else:
                    raise
            else:
                raise
    else:
        # Checkpoint doesn't have "model" key - might be in different format
        logger.warning("⚠️  Checkpoint does not contain 'model' key")
        logger.warning(f"   Top-level keys in checkpoint: {list(checkpoint.keys())[:10]}")
        if len(checkpoint.keys()) > 10:
            logger.warning(f"   ... and {len(checkpoint.keys()) - 10} more keys")
        raise ValueError("Checkpoint format not recognized: missing 'model' key")
    
    # Load optimizer state dict
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("✓ Optimizer state dict loaded")
    
    # Load scheduler state dict
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info("✓ Scheduler state dict loaded")
    
    # Load scaler state dict
    if scaler is not None and "scaler" in checkpoint:
        try:
            scaler.load_state_dict(checkpoint["scaler"])
            logger.info("✓ GradScaler state dict loaded")
        except Exception as e:
            logger.warning(f"Failed to load GradScaler state dict: {e}")
    
    # Return metadata
    metadata = {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "stage": checkpoint.get("stage", None),
    }
    
    logger.info(f"Loaded checkpoint: epoch={metadata['epoch']}, step={metadata['step']}, stage={metadata['stage']}")
    
    return metadata