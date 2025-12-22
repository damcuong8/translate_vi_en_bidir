import argparse
import json
import os
import sys
import wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to training config json")
    parser.add_argument("--ds_config", type=str, required=False, help="Path to deepspeed config json")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="local GPU rank supplied by DeepSpeed launcher")
    parser.add_argument("--single_gpu", action="store_true", help="Use single GPU training mode")
    parser.add_argument("--resume_from_checkpoint", type=str, default="./checkpoints/checkpoint-12144", help="Path to checkpoint to resume from")
    
    wandb.login(key="d5f9bf0b4e6741e7f8daf108d0e2b8efdcc23eb1")
    
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = json.load(f)

    # Override config with resume path if provided
    if args.resume_from_checkpoint:
        config['resume_from_checkpoint'] = args.resume_from_checkpoint

    # Load ds_config if provided (needed for DeepSpeed)
    ds_config = None
    if args.ds_config and os.path.exists(args.ds_config):
        with open(args.ds_config) as f:
            ds_config = json.load(f)

    # Check for single GPU training (flag or config)
    if args.single_gpu or config.get('use_single_gpu', False):
        print("Training mode: Single GPU")
        from train_single import train_single
        train_single(config=config)
    elif config.get('use_fsdp', False):
        print("Training mode: FSDP")
        from train_fsdp import train_fsdp
        train_fsdp(config=config)
    else:
        print("Training mode: DeepSpeed")
        from train_deepspeed import train_deepspeed
        train_deepspeed(config=config, ds_config=ds_config)

if __name__ == "__main__":
    main()
