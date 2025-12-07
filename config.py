from pathlib import Path


def get_deepspeed_config():
    return {
        "train_batch_size": 64,
        "gradient_accumulation_steps": 2,
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "allgather_bucket_size": 5e7,
            "reduce_bucket_size": 5e7,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
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
            "project": "Translate-Vi-En"
        }
    }

def get_kaggle_config():
    """Configuration optimized for Kaggle DataParallel training"""
    return { 
        # Training Parameters
        "num_epochs": 20,
        "early_stopping_patience": 5,
        "warmup_steps": 2000,
        
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
        }
    }
