import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="Path to training config json")
    parser.add_argument("--ds_config", type=str, required=False, help="Path to deepspeed config json")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="local GPU rank supplied by DeepSpeed launcher")
    
    # We parse known args to get config paths, but we also want to pass args to the called functions if needed.
    # However, the called functions in the other files accept config objects, not args.
    # But if we run them as scripts, they parse args.
    
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
