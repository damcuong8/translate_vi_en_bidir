import os
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import logging
from checkpoint_utils import save_checkpoint, load_checkpoint, find_latest_checkpoint
import torch.distributed as dist
from model import build_transformer, ModelConfig
from utils import wrap_model_with_fsdp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.fc(x)

def setup_dist():
    if dist.is_initialized():
        return

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Running via torchrun
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
    else:
        # Standalone single-process run
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("gloo", rank=0, world_size=1)

def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    return 1

def test_simple_save():
    rank = get_rank()
    # Only rank 0 runs simple save test
    if rank != 0:
        if dist.is_initialized():
            dist.barrier()
        return

    print("\n=== Testing Simple Save (Non-FSDP) ===")
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
    print("Saving step 1...")
    save_checkpoint(
        model=model, optimizer=optimizer, scheduler=None,
        epoch=0, step=1, config=config, rank=0, global_step=1
    )
    
    # Save step 2
    print("Saving step 2...")
    save_checkpoint(
        model=model, optimizer=optimizer, scheduler=None,
        epoch=0, step=2, config=config, rank=0, global_step=2
    )
    
    # Save step 3 (Should trigger cleanup of step 1)
    print("Saving step 3...")
    save_checkpoint(
        model=model, optimizer=optimizer, scheduler=None,
        epoch=0, step=3, config=config, rank=0, global_step=3
    )
    
    # Check if files exist
    ckpt_dir = os.path.join(config["output_dir"], "checkpoints")
    latest = find_latest_checkpoint(ckpt_dir)
    print(f"Latest checkpoint found: {latest}")
    assert latest is not None
    assert "checkpoint-3" in latest
    
    # Check cleanup (limit 2)
    all_ckpts = os.listdir(ckpt_dir)
    print(f"Checkpoints in dir: {all_ckpts}")
    # We expect checkpoint-2.pt and checkpoint-3.pt
    assert len(all_ckpts) == 2
    
    # Load
    loaded_model = SimpleModel()
    # Ensure loaded model is on same device as simple model (CPU)
    load_checkpoint(latest, loaded_model, device='cpu')
    
    # Compare weights
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2)
    
    print("✓ Simple Save Test Passed!")
    
    if dist.is_initialized():
        dist.barrier()

def test_fsdp_save():
    # Setup dist if not already
    setup_dist()
    rank = get_rank()
    world_size = get_world_size()
    
    if rank == 0:
        print("\n=== Testing FSDP/DCP Save ===")
    
    # Use actual Transformer model config
    model_config = ModelConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=256,
        head_dim=32,
        num_attention_heads=4
    )
    
    device_id = rank % torch.cuda.device_count() if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")

    # Build model
    model = build_transformer(config=model_config).to(device)
    
    # FSDP Config
    config = {
        "output_dir": "./test_output_fsdp",
        "save_total_limit": 2,
        "use_fsdp": True,
        "save_optimizer_state": True,
        "sharding_strategy": "FULL_SHARD",
        "use_mixed_precision": False,
        "cpu_offload": True,
        "checkpoint_dir": "./test_output_fsdp/checkpoints" # Ensure this is set
    }
    
    # Wrap with FSDP
    # Note: wrap_model_with_fsdp checks config['use_fsdp'] and dist.get_world_size()
    # If world_size=1 (single GPU test), it might skip FSDP wrapping unless we force it or if dist is init.
    # Our setup_dist ensures dist is initialized even for world_size=1
    
    # However, wrap_model_with_fsdp returns early if dist.get_world_size() == 1.
    # We might need to bypass that check for testing FSDP on single GPU if we want to test FSDP saving logic.
    # OR we just rely on standard torch.save if not wrapped.
    # But the user specifically wants to test FSDP saving.
    # FSDP can work on single device if initialized.
    
    # Let's mock world size check if needed, but let's try standard path first.
    # If wrap_model_with_fsdp returns unwrapped model, save_checkpoint will use standard save.
    
    # Hack: Force wrap even if world_size=1 for testing purposes if needed.
    # But let's see if we can use existing utils.
    
    # The utils.py says: if not config['use_fsdp'] or dist.get_world_size() == 1: return model
    # So for single process test, it returns unwrapped model.
    # We need to manually wrap it for this test to verify FSDP saving logic specifically.
    
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
    from config import FSDPConfig
    
    if dist.get_world_size() == 1:
        # Manually wrap for single GPU FSDP test
        auto_wrap_policy = FSDPConfig.get_auto_wrap_policy()
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=True),
            use_orig_params=True
        )
    else:
        model = wrap_model_with_fsdp(model, config)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    
    if rank == 0:
        if os.path.exists(config["output_dir"]):
            shutil.rmtree(config["output_dir"])
    
    if dist.is_initialized():
        dist.barrier()
        
    # Save step 1
    if rank == 0:
        print("Saving FSDP step 1...")
    
    # Note: save_checkpoint handles barrier internally
    save_checkpoint(
        model=model, optimizer=optimizer, scheduler=None,
        epoch=0, step=1, config=config, rank=rank, global_step=1
    )
    
    # Check
    ckpt_dir = config["checkpoint_dir"]
    latest = find_latest_checkpoint(ckpt_dir)
    if rank == 0:
        print(f"Latest checkpoint found: {latest}")
        assert latest is not None
        assert os.path.isdir(latest)
    
    # Load
    # To load FSDP checkpoint, we need a model structure.
    loaded_model = build_transformer(config=model_config).to(device)
    
    # We must wrap the loaded model with FSDP to load FSDP checkpoint properly into it
    # (DCP loads into the state dict of the object, FSDP object handles sharding)
    if dist.get_world_size() == 1:
        loaded_model = FSDP(
            loaded_model,
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=True),
            use_orig_params=True
        )
    else:
        loaded_model = wrap_model_with_fsdp(loaded_model, config)
        
    loaded_optimizer = optim.AdamW(loaded_model.parameters(), lr=0.01)
    
    meta = load_checkpoint(latest, loaded_model, loaded_optimizer)
    
    if rank == 0:
        print(f"Loaded Meta: {meta}")
        assert meta['global_step'] == 1
    
    # Compare weights
    # Note: Accessing parameters directly on FSDP model gathers them if use_orig_params=True?
    # Or we verify by running a forward pass and checking output?
    # Comparing parameters directly might be tricky with FSDP sharding.
    
    # Let's check if we can run a forward pass
    if rank == 0:
        print("Verifying forward pass...")
    
    dummy_input = torch.randint(0, 1000, (2, 16)).to(device)
    dummy_mask = torch.ones((2, 16)).to(device)
    dummy_labels = torch.randint(0, 1000, (2, 16)).to(device)
    
    # Ensure both in eval mode
    model.eval()
    loaded_model.eval()
    
    with torch.no_grad():
        out1, _, _, _ = model(dummy_input, dummy_mask, dummy_input, dummy_mask, dummy_labels)
        out2, _, _, _ = loaded_model(dummy_input, dummy_mask, dummy_input, dummy_mask, dummy_labels)
    
    if rank == 0:
        print(f"Max difference: {(out1 - out2).abs().max()}")
        assert torch.allclose(out1, out2, atol=1e-5)
        print("✓ FSDP Save Test Passed!")

    cleanup_dist()

if __name__ == "__main__":
    # If running with torchrun, setup_dist must be called first to handle coordination
    if "RANK" in os.environ:
        setup_dist()
        
    test_simple_save()
    test_fsdp_save()
