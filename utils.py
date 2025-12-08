import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, BackwardPrefetch
from config import FSDPConfig

def wrap_model_with_fsdp(model: nn.Module, config: dict) -> nn.Module:
    """
    Wrap model with FSDP for distributed training.
    
    Args:
        model: The model to wrap
        config: Training configuration
        
    Returns:
        FSDP-wrapped model (or original model if FSDP not enabled)
    """
    if not config['use_fsdp'] or dist.get_world_size() == 1:
        print("FSDP not enabled or single GPU training, returning unwrapped model")
        return model
    
    # Get rank for logging
    rank = int(os.environ.get('LOCAL_RANK', 0))
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
    
    print("Wrapping model with FSDP")
    
    # Get sharding strategy
    sharding_strategy = FSDPConfig.SHARDING_STRATEGIES.get(
        config['sharding_strategy'],
        FSDPConfig.SHARDING_STRATEGIES["FULL_SHARD"]
    )
    print(f"Sharding strategy: {config['sharding_strategy']}")
    
    # Get mixed precision policy
    mixed_precision_policy = None
    if config['use_mixed_precision']:
        mixed_precision_policy = FSDPConfig.get_mixed_precision_policy(
            config['mixed_precision_dtype']
        )
        print(f"Using mixed precision: {config['mixed_precision_dtype']}")
    
    # Get auto wrap policy
    auto_wrap_policy = FSDPConfig.get_auto_wrap_policy()
    
    # CPU offload config
    cpu_offload_config = CPUOffload(offload_params=True) if config['cpu_offload'] else None
    if config['cpu_offload']:
        print("CPU offload enabled")
    

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_POST,
        cpu_offload=cpu_offload_config,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
    )
    
    print(f"✓ Model wrapped with FSDP")
    return model


def create_cosine_scheduler(
    optimizer: optim.Optimizer,
    config: dict,
    num_training_steps: int,
) -> lr_scheduler.LambdaLR:
    """
    Create LR scheduler: linear warmup -> cosine decay to min_lr_ratio.
    
    Args:
        optimizer: torch optimizer
        config: training config
        num_training_steps: total training steps (batches)
        warmup_steps: steps for linear warmup; if None uses config defaults
        min_lr_ratio: final LR ratio relative to base_lr; if None uses config default
        
    Returns:
        a lr_scheduler.LambdaLR scheduler
    """
    min_lr_ratio = config['scheduler']['params']['cos_min_ratio']
    warmup_num_steps_ratio = config['scheduler']['params']['warmup_num_steps_ratio']
    warmup_steps = int(num_training_steps * warmup_num_steps_ratio)
    
    def lr_lambda(step: int):
        # warmup phase: linear 0 -> 1
        if step < warmup_steps:
            multiplier = float(step) / float(max(1, warmup_steps))
            return multiplier
        # cosine decay phase: 1 -> min_lr_ratio
        progress = float(step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        multiplier = float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)
        return multiplier

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(
        f"Created cosine scheduler with {warmup_steps} warmup steps out of {num_training_steps} total steps, "
        f"min_lr_ratio={min_lr_ratio}"
    )

    return scheduler