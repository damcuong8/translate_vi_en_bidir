from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import torch
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from functools import partial


def get_deepspeed_config():
    return {
        "train_batch_size": 16,
        "gradient_accumulation_steps": 2,
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "allgather_bucket_size": 5e7,
            "reduce_bucket_size": 5e7,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": False
            }
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-4,
                "betas": [0.9, 0.998],
                "eps": 1e-6,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": 1000,
                "cos_min_ratio": 0.05,
                "warmup_num_steps": 100
            }
        },
        "torch_autocast": {
            "enabled": True,
            "dtype": "float16",
        },
        "flops_profiler": {
            "enabled": True,
            "profile_step": 1,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
            "output_file": "flops_profiler.txt",
        },
        "wandb": {
            "enabled": True,
            "project": "Translate-Vi-En",
            "name": "deepspeed_run"
        }
    }

def get_kaggle_config():
    """Configuration optimized for Kaggle DataParallel training"""
    return { 
        # Training Parameters
        "num_epochs": 20,
        
        # Kaggle paths (update these for your dataset)
        "tokenizer_path": "/kaggle/input/tokenizer/Tokenizer_ENVI",
        "train_hf_dataset_path": "/kaggle/input/pho-mt-token-text/pho_mt_full/train",
        "val_hf_dataset_path": "/kaggle/input/pho-mt-token-text/pho_mt_full/dev",
        "test_hf_dataset_path": "/kaggle/input/pho-mt-token-text/pho_mt_full/test",
        "dev_flores_hf_dataset_path": "/kaggle/input/flores-tokenized-text/flores_tokenized/flores_dev",
        "test_flores_hf_dataset_path": "/kaggle/input/flores-tokenized-text/flores_tokenized/flores_devtest",
        "experiment_name": "runs/envi_model_kaggle",
        "checkpoint_path": "/kaggle/working/checkpoints",

        # Language tokens
        "lang_token_map": {
            "eng": "__eng__",
            "vie": "__vie__"
        },

        # Training Config
        "use_fsdp": True,
        "train_batch_size": 16,
        "val_batch_size": 16,
        "test_batch_size": 16,
        "dev_flores_batch_size": 16,
        "test_flores_batch_size": 16,
        "experiment_name": "runs/envi_model_kaggle",

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-4,
                "betas": [0.9, 0.998],
                "eps": 1e-6,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "cos_min_ratio": 0.05,
                "warmup_num_steps_ratio": 0.03
            }
        },

        # Gradient Accumulation
        "gradient_accumulation_steps": 1,
        
        # Gradient Clipping
        "max_grad_norm": 1.0,
        
        # AMP (Automatic Mixed Precision)
        "use_amp": True,
        "amp_dtype": "fp16",  # "fp16" or "bf16"
        
        # Checkpoint Config
        "save_steps": 500,  # Save checkpoint every N steps
        "save_total_limit": 10,  # Keep only last N checkpoints
        "save_optimizer_state": True,  # Include optimizer/scheduler in checkpoint
        
        # FSDP Config
        "sharding_strategy": "SHARD_GRAD_OP",
        "cpu_offload": True,
        "use_mixed_precision": False,
        "mixed_precision_dtype": "fp16",
        "use_torch_compile": True,
        
        # Wandb Config
        "wandb": {
            "enabled": True,
            "project": "Translate-Vi-En",
            "name": "fsdp_run"
        }
    }


@dataclass
class FSDPConfig:

    # Sharding strategies
    SHARDING_STRATEGIES = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    
    @staticmethod
    def get_mixed_precision_policy(dtype: str = "fp16"):
        """Get mixed precision policy for FSDP"""
        if dtype == "bf16":
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif dtype == "fp16":
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        else:
            return None

    @staticmethod
    def get_auto_wrap_policy():
        """Get auto wrap policy for transformer layers"""
        from model import (
            EncoderBlock,
            DecoderBlock,
            Transformer
        )
        import torch.nn as nn
        
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        
        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                EncoderBlock,
                DecoderBlock
            },
        )