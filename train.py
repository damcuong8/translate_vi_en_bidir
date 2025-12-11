import argparse
import json
import os
import sys
import wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to training config json")
    parser.add_argument("--ds_config", type=str, required=False, help="Path to deepspeed config json")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="local GPU rank supplied by DeepSpeed launcher")
    
    wandb.login(key="d5f9bf0b4e6741e7f8daf108d0e2b8efdcc23eb1")
    
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = json.load(f)

    # Load ds_config if provided
    ds_config = None
    if args.ds_config and os.path.exists(args.ds_config):
        with open(args.ds_config) as f:
            ds_config = json.load(f)

    if config.get('use_fsdp', False):
        print("Training mode: FSDP")
        from train_fsdp import train_fsdp
        train_fsdp(config=config)
    else:
        print("Training mode: DeepSpeed")
        from train_deepspeed import train_deepspeed
        train_deepspeed(config=config, ds_config=ds_config)

if __name__ == "__main__":
    main()
