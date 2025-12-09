import os
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import logging
from checkpoint_utils import save_checkpoint, load_checkpoint, find_latest_checkpoint
import torch.distributed as dist

# Setup simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.fc(x)

def setup_dist():
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        # Use gloo for testing on CPU/Single GPU without heavy setup
        dist.init_process_group("gloo", rank=0, world_size=1)

def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()

import sys

# Setup logging to file
logging.basicConfig(filename='test_output.log', level=logging.INFO, force=True)

def log(msg):
    print(msg)
    logging.info(msg)

def test_simple_save():
    log("\n=== Testing Simple Save (Non-FSDP) ===")
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    config = {
        "output_dir": "./test_output_simple",
        "save_total_limit": 2,
        "use_fsdp": False
    }
    
    if os.path.exists(config["output_dir"]):
        shutil.rmtree(config["output_dir"])
        
    # Save step 1
    log("Saving step 1...")
    save_checkpoint(
        model=model, optimizer=optimizer, scheduler=None,
        epoch=0, step=1, config=config, rank=0, global_step=1
    )
    
    # Save step 2
    log("Saving step 2...")
    save_checkpoint(
        model=model, optimizer=optimizer, scheduler=None,
        epoch=0, step=2, config=config, rank=0, global_step=2
    )
    
    # Save step 3 (Should trigger cleanup of step 1)
    log("Saving step 3...")
    save_checkpoint(
        model=model, optimizer=optimizer, scheduler=None,
        epoch=0, step=3, config=config, rank=0, global_step=3
    )
    
    # Check if files exist
    ckpt_dir = os.path.join(config["output_dir"], "checkpoints")
    latest = find_latest_checkpoint(ckpt_dir)
    log(f"Latest checkpoint found: {latest}")
    assert latest is not None
    assert "checkpoint-3" in latest
    
    # Check cleanup (limit 2)
    all_ckpts = os.listdir(ckpt_dir)
    log(f"Checkpoints in dir: {all_ckpts}")
    # We expect checkpoint-2.pt and checkpoint-3.pt
    assert len(all_ckpts) == 2
    
    # Load
    loaded_model = SimpleModel()
    load_checkpoint(latest, loaded_model)
    
    # Compare weights
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2)
    
    log("✓ Simple Save Test Passed!")

def test_fsdp_save():
    log("\n=== Testing FSDP/DCP Save ===")
    try:
        setup_dist()
    except Exception as e:
        log(f"Skipping FSDP test due to dist init failure: {e}")
        return

    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    config = {
        "output_dir": "./test_output_fsdp",
        "save_total_limit": 2,
        "use_fsdp": True
    }
    
    if os.path.exists(config["output_dir"]):
        shutil.rmtree(config["output_dir"])
        
    # Save step 1
    log("Saving FSDP step 1...")
    save_checkpoint(
        model=model, optimizer=optimizer, scheduler=None,
        epoch=0, step=1, config=config, rank=0, global_step=1
    )
    
    # Check
    ckpt_dir = os.path.join(config["output_dir"], "checkpoints")
    latest = find_latest_checkpoint(ckpt_dir)
    log(f"Latest checkpoint found: {latest}")
    assert latest is not None
    assert os.path.isdir(latest)
    
    # Load
    loaded_model = SimpleModel()
    loaded_optimizer = optim.Adam(loaded_model.parameters(), lr=0.01)
    
    meta = load_checkpoint(latest, loaded_model, loaded_optimizer)
    log(f"Loaded Meta: {meta}")
    assert meta['global_step'] == 1
    
    # Compare weights
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2)
        
    cleanup_dist()
    log("✓ FSDP Save Test Passed!")

if __name__ == "__main__":
    test_simple_save()
    test_fsdp_save()

