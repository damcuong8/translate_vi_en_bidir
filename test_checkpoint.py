#!/usr/bin/env python3
"""
Test script for checkpoint saving and loading (Simple + FSDP/DCP).
Includes detailed state dict inspection and parameter comparison.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import logging
import argparse
import json
from collections import defaultdict
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

from checkpoint_utils import save_checkpoint, load_checkpoint
from model import build_transformer, ModelConfig
from utils import wrap_model_with_fsdp
from config import FSDPConfig

# Try to import safetensors, but don't fail if missing
try:
    from safetensors.torch import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions for Inspection ---

def print_state_dict_summary(checkpoint_state_dict, model_state_dict=None, model=None):
    """
    Print a comprehensive summary of checkpoint state dict.
    """
    print("\n" + "="*80)
    print("📊 CHECKPOINT STATE DICT SUMMARY")
    print("="*80)
    
    # Group keys by component
    def group_keys(keys):
        groups = defaultdict(list)
        for key in sorted(keys):
            if "." in key:
                component = key.split(".")[0]
            else:
                component = "root"
            groups[component].append(key)
        return groups
    
    checkpoint_keys = set(checkpoint_state_dict.keys())
    checkpoint_groups = group_keys(checkpoint_keys)
    
    # Calculate total parameters and size
    total_params = 0
    total_size_mb = 0
    component_stats = {}
    
    for component, keys in checkpoint_groups.items():
        component_params = 0
        component_size = 0
        for key in keys:
            tensor = checkpoint_state_dict[key]
            if isinstance(tensor, torch.Tensor):
                num_params = tensor.numel()
                size_bytes = tensor.numel() * tensor.element_size()
                component_params += num_params
                component_size += size_bytes
                total_params += num_params
                total_size_mb += size_bytes / (1024**2)
        component_stats[component] = {
            "keys": len(keys),
            "params": component_params,
            "size_mb": component_size / (1024**2)
        }
    
    # Print checkpoint info
    print(f"\n📦 Checkpoint Info:")
    print(f"   Total keys: {len(checkpoint_keys)}")
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   Total size: {total_size_mb:.2f} MB")
    
    # Print component breakdown
    print(f"\n🔧 Components:")
    for component in sorted(component_stats.keys()):
        stats = component_stats[component]
        print(f"   [{component}]")
        print(f"      Keys: {stats['keys']}")
        print(f"      Parameters: {stats['params']:,} ({stats['params']/1e6:.2f}M)")
        print(f"      Size: {stats['size_mb']:.2f} MB")
    
    # Compare with model state dict if provided
    if model_state_dict is not None:
        model_keys = set(model_state_dict.keys())
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        
        # Check for shape mismatches
        matched_keys = checkpoint_keys & model_keys
        shape_mismatches = []
        for key in matched_keys:
            ckpt_val = checkpoint_state_dict[key]
            model_val = model_state_dict[key]
            if isinstance(ckpt_val, torch.Tensor) and isinstance(model_val, torch.Tensor):
                if ckpt_val.shape != model_val.shape:
                    shape_mismatches.append((key, ckpt_val.shape, model_val.shape))
        
        print(f"\n🔍 Comparison with Model State Dict:")
        print(f"   Model keys: {len(model_keys)}")
        print(f"   Matched keys: {len(matched_keys)}")
        print(f"   Missing keys: {len(missing_keys)}")
        print(f"   Unexpected keys: {len(unexpected_keys)}")
        print(f"   Shape mismatches: {len(shape_mismatches)}")
        
        if missing_keys:
            ignored_keys = []
            if model is not None:
                ignored_keys = getattr(model, "_keys_to_ignore_on_load_missing", [])
            
            actual_missing = [k for k in missing_keys if not any(k.startswith(prefix) for prefix in ignored_keys)]
            if actual_missing:
                print(f"\n   ⚠️  Missing keys (not in ignore list): {len(actual_missing)}")
                missing_groups = group_keys(actual_missing)
                for component, keys in sorted(missing_groups.items()):
                    print(f"      [{component}]: {len(keys)} keys")
                    for key in keys[:3]:
                        print(f"         - {key}")
                    if len(keys) > 3:
                        print(f"         ... and {len(keys) - 3} more")
            
            if len(actual_missing) < len(missing_keys):
                ignored_count = len(missing_keys) - len(actual_missing)
                print(f"   ℹ️  {ignored_count} missing keys are in ignore list (expected)")
        
        if unexpected_keys:
            print(f"\n   ⚠️  Unexpected keys: {len(unexpected_keys)}")
            unexpected_groups = group_keys(unexpected_keys)
            for component, keys in sorted(unexpected_groups.items()):
                print(f"      [{component}]: {len(keys)} keys")
                for key in keys[:3]:
                    print(f"         - {key}")
                if len(keys) > 3:
                    print(f"         ... and {len(keys) - 3} more")
        
        if shape_mismatches:
            print(f"\n   ❌ Shape mismatches: {len(shape_mismatches)}")
            for key, ckpt_shape, model_shape in shape_mismatches[:5]:
                print(f"      - {key}: checkpoint {ckpt_shape} vs model {model_shape}")
            if len(shape_mismatches) > 5:
                print(f"      ... and {len(shape_mismatches) - 5} more")

    print("="*80 + "\n")

def compare_model_params_before_after(model_state_dict_before, model_state_dict_after, checkpoint_state_dict=None):
    """
    Compare model parameters before and after loading checkpoint.
    """
    print("\n" + "="*80)
    print("🔄 MODEL PARAMETERS COMPARISON (Before vs After Loading Checkpoint)")
    print("="*80)
    
    keys_before = set(model_state_dict_before.keys())
    keys_after = set(model_state_dict_after.keys())
    
    # Check if keys changed
    if keys_before != keys_after:
        print(f"\n⚠️  WARNING: Model keys changed after loading checkpoint!")
        print(f"   Keys before: {len(keys_before)}")
        print(f"   Keys after: {len(keys_after)}")
    else:
        print(f"\n✓ Model keys unchanged: {len(keys_before)} keys")
    
    # Compare parameter values
    common_keys = keys_before & keys_after
    unchanged_params = []
    changed_params = []
    shape_mismatches = []
    
    print(f"\n📊 Parameter Value Comparison:")
    print(f"   Common keys: {len(common_keys)}")
    
    for key in sorted(common_keys):
        tensor_before = model_state_dict_before[key]
        tensor_after = model_state_dict_after[key]
        
        # Check shape
        if tensor_before.shape != tensor_after.shape:
            shape_mismatches.append((key, tensor_before.shape, tensor_after.shape))
            continue
        
        # Check if values are identical
        if torch.equal(tensor_before, tensor_after):
            unchanged_params.append(key)
        else:
            # Calculate difference statistics
            diff = tensor_before - tensor_after
            max_diff = torch.abs(diff).max().item()
            mean_diff = torch.abs(diff).mean().item()
            changed_params.append((key, max_diff, mean_diff, tensor_before.shape))
    
    # Report unchanged parameters
    if unchanged_params:
        print(f"\n   ✓ Unchanged parameters: {len(unchanged_params)}")
        if len(unchanged_params) <= 5:
            for key in unchanged_params:
                 print(f"      - {key}")
    
    # Report changed parameters
    if changed_params:
        print(f"\n   ⚠️  Changed parameters: {len(changed_params)}")
        # Sort by max difference
        changed_params.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n   Top 10 parameters with largest changes:")
        for i, (key, max_diff, mean_diff, shape) in enumerate(changed_params[:10], 1):
            print(f"      {i}. {key} {shape}")
            print(f"         Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    else:
        print(f"\n   ⚠️  WARNING: No parameters changed! This might indicate:")
        print(f"      - Checkpoint was not loaded correctly")
        print(f"      - Model was already initialized with checkpoint weights")
    
    print("="*80 + "\n")

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint directory in a given directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    subdirs = []
    for entry in os.listdir(checkpoint_dir):
        full_entry = os.path.join(checkpoint_dir, entry)
        if os.path.isdir(full_entry) and entry.startswith("checkpoint-"):
            try:
                step_val = int(entry.split("-")[-1])
                subdirs.append((step_val, full_entry))
            except ValueError:
                pass
    
    if not subdirs:
        return None
    
    subdirs.sort(key=lambda x: x[0])
    return subdirs[-1][1]

# --- Simple Model for Testing ---
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.fc(x)

# --- Distributed Utils ---
def setup_dist():
    if dist.is_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("gloo", rank=0, world_size=1)

def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0

# --- Test Functions ---

def test_simple_save(output_dir="./test_output_simple"):
    rank = get_rank()
    if rank != 0:
        if dist.is_initialized(): dist.barrier()
        return

    print("\n=== Testing Simple Save (Non-FSDP) ===")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    config = {
        "output_dir": output_dir,
        "save_total_limit": 2,
        "use_fsdp": False,
        "checkpoint_dir": os.path.join(output_dir, "checkpoints")
    }
    
    # Save step 1
    print("Saving step 1...")
    save_checkpoint(model, optimizer, None, epoch=0, step=1, config=config, rank=0, global_step=1)
    
    # Save step 2
    print("Saving step 2...")
    save_checkpoint(model, optimizer, None, epoch=0, step=2, config=config, rank=0, global_step=2)
    
    # Save step 3 (cleanup step 1)
    print("Saving step 3...")
    save_checkpoint(model, optimizer, None, epoch=0, step=3, config=config, rank=0, global_step=3)
    
    ckpt_dir = config["checkpoint_dir"]
    latest = find_latest_checkpoint(ckpt_dir)
    print(f"Latest checkpoint found: {latest}")
    assert latest is not None
    assert "checkpoint-3" in latest
    
    # Verify cleanup
    all_ckpts = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
    print(f"Checkpoints in dir: {all_ckpts}")
    assert len(all_ckpts) == 2, f"Expected 2 checkpoints, found {len(all_ckpts)}"
    
    # Test Loading with verification
    print("\n--- Verifying Load ---")
    loaded_model = SimpleModel()
    # Initial state
    state_before = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    
    # Load
    metadata = load_checkpoint(latest, loaded_model, device='cpu')
    print(f"Loaded metadata: {metadata}")
    
    state_after = {k: v.clone() for k, v in loaded_model.state_dict().items()}
    
    # Compare
    compare_model_params_before_after(state_before, state_after)
    
    # Check strict equality with original model
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2)
    print("✓ Simple Save Test Passed!")
    
    if dist.is_initialized(): dist.barrier()

def test_fsdp_save(output_dir="./test_output_fsdp"):
    setup_dist()
    rank = get_rank()
    
    if rank == 0:
        print("\n=== Testing FSDP/DCP Save ===")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    # Config
    model_config = ModelConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=256,
        head_dim=32,
        num_attention_heads=4
    )
    
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    
    # Build Model
    model = build_transformer(config=model_config).to(device)
    
    # FSDP Config
    config = {
        "output_dir": output_dir,
        "save_total_limit": 2,
        "use_fsdp": True,
        "save_optimizer_state": True,
        "sharding_strategy": "FULL_SHARD",
        "use_mixed_precision": False,
        "cpu_offload": False,
        "checkpoint_dir": os.path.join(output_dir, "checkpoints")
    }
    
    # Wrap
    if dist.get_world_size() == 1:
        # Manual wrap for single GPU FSDP test
        auto_wrap_policy = FSDPConfig.get_auto_wrap_policy()
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=False),
            use_orig_params=True
        )
    else:
        model = wrap_model_with_fsdp(model, config)
        
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    
    if dist.is_initialized(): dist.barrier()
    
    # Save
    if rank == 0: print("Saving FSDP step 1...")
    save_checkpoint(model, optimizer, None, epoch=0, step=1, config=config, rank=rank, global_step=1)
    
    ckpt_dir = config["checkpoint_dir"]
    latest = find_latest_checkpoint(ckpt_dir)
    
    if rank == 0:
        print(f"Latest checkpoint found: {latest}")
        assert latest is not None
        assert os.path.isdir(latest)
    
    # Verification Loading
    if dist.is_initialized(): dist.barrier()
    
    loaded_model = build_transformer(config=model_config).to(device)
    # Must wrap to load FSDP
    if dist.get_world_size() == 1:
        loaded_model = FSDP(
            loaded_model,
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=False),
            use_orig_params=True
        )
    else:
        loaded_model = wrap_model_with_fsdp(loaded_model, config)
    
    loaded_optimizer = optim.AdamW(loaded_model.parameters(), lr=0.01)
    
    # Capture state before load (warning: FSDP state dict behavior depends on configuration)
    # For robust comparison, we might need full_state_dict if using FSDP, 
    # but here we just want to see if parameters change.
    
    # Load
    if rank == 0: print("Loading FSDP checkpoint...")
    
    # Note: load_checkpoint with use_dcp=True will load into the FSDP model
    meta = load_checkpoint(latest, loaded_model, loaded_optimizer, config=config)
    
    if rank == 0:
        print(f"Loaded Meta: {meta}")
        assert meta['global_step'] == 1
        
        print("Verifying forward pass match...")
        
    # Forward pass verification
    dummy_input = torch.randint(0, 1000, (2, 16)).to(device)
    dummy_mask = torch.ones((2, 16)).to(device)
    
    model.eval()
    loaded_model.eval()
    
    with torch.no_grad():
        out1, _, _, _ = model(dummy_input, dummy_mask, dummy_input, dummy_mask, dummy_input)
        out2, _, _, _ = loaded_model(dummy_input, dummy_mask, dummy_input, dummy_mask, dummy_input)
    
    if rank == 0:
        diff = (out1 - out2).abs().max()
        print(f"Max output difference: {diff}")
        assert torch.allclose(out1, out2, atol=1e-5)
        print("✓ FSDP Save Test Passed!")
    
    cleanup_dist()

def main():
    parser = argparse.ArgumentParser(description="Test Checkpoint Saving & Loading")
    parser.add_argument("--test", choices=["simple", "fsdp", "all"], default="all", help="Which test to run")
    parser.add_argument("--inspect", type=str, help="Path to a checkpoint to inspect (skips tests if provided)")
    args = parser.parse_args()
    
    if args.inspect:
        print(f"Inspecting checkpoint: {args.inspect}")
        # Inspection mode
        checkpoint_path = args.inspect
        # Use simple model for inspection just to load structure if generic
        # Or better, just load state dict directly if possible, but load_checkpoint expects a model
        # We'll use a dummy SimpleModel if it's a simple checkpoint, or try to load as dict
        
        try:
            # Try loading as raw dict first
            if os.path.isdir(checkpoint_path):
                 print("Directory checkpoint detected (DCP). Loading via load_checkpoint requires model structure.")
                 print("Skipping detailed inspection for DCP without model definition.")
            else:
                 state_dict = torch.load(checkpoint_path, map_location="cpu")
                 if "model" in state_dict:
                     print_state_dict_summary(state_dict["model"])
                 else:
                     print_state_dict_summary(state_dict)
        except Exception as e:
            print(f"Failed to inspect: {e}")
        return

    if args.test in ["simple", "all"]:
        test_simple_save()
    
    if args.test in ["fsdp", "all"]:
        test_fsdp_save()

if __name__ == "__main__":
    if "RANK" in os.environ:
        setup_dist()
    main()
