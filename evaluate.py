import argparse
import torch
from tqdm import tqdm
import time
import os
import json
import hashlib
from pathlib import Path
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from datasets import load_from_disk

from model import build_transformer, ModelConfig
from config import get_kaggle_config
from checkpoint_utils import load_checkpoint, _has_dcp_artifacts, _resolve_checkpoint_dir
from transformers import AutoTokenizer
from dataset import Collator

from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore

class EvaluationDataset(Dataset):
    """Dataset for evaluating a single translation direction."""
    def __init__(self, dataset_path, tokenizer, src_lang_token, tgt_lang_token, direction="en_to_vi", max_seq_len=152):
        """
        Args:
            dataset_path: Path to the dataset on disk
            tokenizer: Tokenizer instance
            src_lang_token: Source language token (e.g., "__eng__")
            tgt_lang_token: Target language token (e.g., "__vie__")
            direction: "en_to_vi" or "vi_to_en"
            max_seq_len: Maximum sequence length
        """
        self.ds = load_from_disk(dataset_path)
        original_count = len(self.ds)
        print(f"Original dataset count: {original_count}")
        
        # Setup writable cache directory for filtering
        cache_dir = "/tmp/translate_vi_en_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create a consistent cache filename based on dataset path and parameters
        safe_name = os.path.basename(dataset_path)
        ds_hash = hashlib.md5(dataset_path.encode()).hexdigest()[:8]
        cache_file_name = os.path.join(cache_dir, f"filtered_{safe_name}_{ds_hash}_{max_seq_len}.arrow")

        # Filter sequences longer than max_seq_len
        self.ds = self.ds.filter(
            lambda x: (len(x["input_ids_en"]) <= max_seq_len) and (len(x["input_ids_vi"]) <= max_seq_len),
            num_proc=min(os.cpu_count(), 4),
            cache_file_name=cache_file_name
        )
        filtered_count = len(self.ds)
        print(f"Filtered dataset count: {filtered_count}")
        print(f"Removed {original_count - filtered_count} samples due to length > {max_seq_len}")
        
        self.tokenizer = tokenizer
        self.direction = direction
        
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        
        self.src_token_id = tokenizer.convert_tokens_to_ids(src_lang_token)
        self.tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_token)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        item = self.ds[idx]
        
        # Determine source and target based on direction
        if self.direction == "en_to_vi":
            # English to Vietnamese
            src_ids = [self.src_token_id] + item["input_ids_en"] + [self.eos_id]
            tgt_ids = [self.bos_id, self.tgt_token_id] + item["input_ids_vi"] + [self.eos_id]
            src_text = item["en"]
            tgt_text = item["vi"]
        else:
            # Vietnamese to English
            src_ids = [self.src_token_id] + item["input_ids_vi"] + [self.eos_id]
            tgt_ids = [self.bos_id, self.tgt_token_id] + item["input_ids_en"] + [self.eos_id]
            src_text = item["vi"]
            tgt_text = item["en"]
        
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def _ensure_torch_checkpoint(path: str) -> str:
    """Convert DCP shards to pytorch_model.bin if needed."""
    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        candidate = os.path.join(path, "pytorch_model.bin")
        if os.path.isfile(candidate):
            return candidate

        metadata_file = os.path.join(path, ".metadata")
        has_dcp = os.path.isfile(metadata_file) or any(
            entry.startswith("__") and entry.endswith(".distcp")
            for entry in os.listdir(path)
            if os.path.isdir(os.path.join(path, entry))
        )

        if has_dcp:
            print("  Detected DCP checkpoint; converting to pytorch_model.bin ...")
            try:
                # Use torch's built-in dcp_to_torch_save function
                # Load state dict from DCP and save to torch format
                dcp_to_torch_save(
                    dcp_checkpoint_dir=path,
                    torch_save_path=candidate,
                )
                print(f"  ✓ Materialized Torch checkpoint at {candidate}")
                return candidate
            except Exception as exc:
                print(f"  ❌ Failed to convert DCP checkpoint: {exc}")
                import traceback
                traceback.print_exc()

    return path

def load_model_and_tokenizer(config_path=None, model_path=None, device='cuda', save_converted_path=None):
    """
    Load model and tokenizer.
    Handles DCP checkpoints by converting/loading them via checkpoint_utils.
    Optionally saves the converted checkpoint to a single file.
    """
    # Load config
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        print("Warning: No config file provided or found. Using default Kaggle config.")
        config_dict = get_kaggle_config()
    
    # Initialize tokenizer
    tokenizer_path = config_dict.get('tokenizer_path')
    if not tokenizer_path or not os.path.exists(tokenizer_path):
        # Fallback to local tokenizer.py if path invalid
        print(f"Tokenizer path {tokenizer_path} not found. Attempting to use local tokenizer...")
        # Note: This requires the tokenizer to be importable or available
        # For now, we assume the user provides a valid path or we fail gracefully
        # Try to find a tokenizer directory in current dir
        if os.path.exists("tokenizer"):
             tokenizer_path = "tokenizer"
        else:
             print("Warning: Could not find tokenizer. Please specify valid tokenizer_path in config.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer from {tokenizer_path}: {e}")
        raise

    # Build model config
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size)
    for key, value in config_dict.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
            
    # Build model
    print("Building model...")
    model = build_transformer(config=model_config)
    model = model.to(device)
    
    # Load weights
    if model_path:
        print(f"Loading checkpoint from: {model_path}")
        
        # Check if DCP
        is_dcp = False
        try:
            resolved_path = _resolve_checkpoint_dir(model_path)
            if _has_dcp_artifacts(resolved_path):
                is_dcp = True
                print("Detected Distributed Checkpoint (DCP).")
        except:
            pass
            
        # load_checkpoint handles both and loads into model
        load_checkpoint(
            checkpoint_path=model_path,
            model=model,
            optimizer=None,
            scheduler=None,
            config=config_dict,
            strict=False # Allow missing keys
        )
        
        # Save converted if requested
        if save_converted_path:
            print(f"Saving converted model to {save_converted_path}...")
            torch.save(model.state_dict(), save_converted_path)
            print("Saved.")
        elif is_dcp:
            print("Note: Model loaded from DCP. To save as standard torch checkpoint, provide --save_converted_path")
            
    else:
        print("Warning: No model path provided. Model initialized with random weights.")

    model.eval()
    return model, tokenizer, config_dict

def encode_input(tokenizer, text, src_lang_token, device):
    """
    Encode input text with source language token.
    Format: [src_lang_token] + text + [eos]
    """
    # Get token IDs
    src_lang_id = tokenizer.convert_tokens_to_ids(src_lang_token)
    
    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Construct sequence: [src_lang_token] + tokens + [eos]
    input_ids = [src_lang_id] + tokens + [tokenizer.eos_token_id]
    
    return torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0) # Batch size 1

def greedy_decode(model, encoder_input, encoder_mask, tokenizer, tgt_lang_token, max_len, device):
    """
    Greedy decoding for a single sequence.
    """
    # Encoder
    encoder_output, _ = model.encoder(encoder_input, mask=encoder_mask)
    
    # Decoder Input: [bos, tgt_lang_token]
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang_token)
    decoder_input = torch.tensor([[tokenizer.bos_token_id, tgt_lang_id]], dtype=torch.long, device=device)
    
    for _ in range(max_len):
        # Create decoder mask (padding mask - all ones since no padding in greedy single batch)
        # But we need to match shape expected by model: (B, 1, 1, S) or (B, S) to be converted
        # The model's forward converts (B, S) -> (B, 1, 1, S)
        # Here we have no padding, so mask is all ones
        tgt_mask = torch.ones((1, decoder_input.size(1)), dtype=torch.long, device=device)
        
        # Decoder forward
        # Note: model.decoder returns decoder_output. We project it using lm_head
        decoder_output, _ = model.decoder(decoder_input, encoder_output, tgt_mask=None, src_mask=encoder_mask) # tgt_mask handled?
        
        # Wait, model.decoder expects tgt_mask for attention.
        # If passed None, it might fail or assume full attention.
        # model.py: tgt_mask = (tgt_attention_mask == 0)...
        # In DecoderBlock: mask=tgt_mask.
        # If we pass tgt_mask=None to decoder, it goes to AttentionBlock.
        # AttentionBlock: mask=None.
        # If causal_mask=True (which it is for decoder), it adds causal mask.
        # So None is fine for greedy decode (no padding).
        
        prob = model.lm_head(model.norm(decoder_output[:, -1]))
        _, next_token = torch.max(prob, dim=1)
        next_token = next_token.item()
        
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]], device=device)], dim=1)
        
        if next_token == tokenizer.eos_token_id:
            break
            
    # Exclude BOS and Lang Token from output
    return decoder_input.squeeze().tolist()[2:] 

def batch_translate_with_dataloader(model, dataloader, tokenizer, tgt_lang_token, device, max_len=256):
    """
    Translate using a DataLoader.
    Returns predictions and references.
    """
    model.eval()
    predictions = []
    references = []
    
    tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_token)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Translating"):
            src_inputs = batch["src_input_ids"].to(device)
            src_masks = batch["src_attention_mask"].to(device)
            src_texts = batch["src_text"]
            ref_texts = batch["tgt_text"]
            
            # Save references
            references.extend(ref_texts)
            
            # Encoder
            enc_mask_expanded = (src_masks == 0).unsqueeze(1).unsqueeze(2)
            encoder_output, _ = model.encoder(src_inputs, mask=enc_mask_expanded)
            
            # Greedy Decode
            batch_size = src_inputs.size(0)
            decoder_input = torch.tensor(
                [[tokenizer.bos_token_id, tgt_token_id]] * batch_size, 
                dtype=torch.long, 
                device=device
            )
            
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            for _ in range(max_len):
                decoder_output, _ = model.decoder(
                    decoder_input, 
                    encoder_output, 
                    tgt_mask=None, 
                    src_mask=enc_mask_expanded
                )
                
                logits = model.lm_head(model.norm(decoder_output[:, -1]))
                next_tokens = torch.argmax(logits, dim=-1)
                
                finished |= (next_tokens == tokenizer.eos_token_id)
                decoder_input = torch.cat([decoder_input, next_tokens.unsqueeze(1)], dim=1)
                
                if finished.all():
                    break
            
            # Decode to text
            for seq in decoder_input:
                tokens = seq[2:].tolist()
                try:
                    eos_index = tokens.index(tokenizer.eos_token_id)
                    tokens = tokens[:eos_index]
                except ValueError:
                    pass
                
                text = tokenizer.decode(tokens)
                predictions.append(text)
    
    return predictions, references

def calculate_metrics(predictions, references):
    metrics = {}
    
    # BLEU
    try:
        bleu = BLEUScore()
        # BLEU expects tokenized
        metrics['bleu'] = bleu(predictions, [[r] for r in references]).item()
    except Exception as e:
        print(f"BLEU Error: {e}")
        metrics['bleu'] = 0.0
        
    # WER
    try:
        wer = WordErrorRate()
        metrics['wer'] = wer(predictions, references).item()
    except Exception as e:
        print(f"WER Error: {e}")
        metrics['wer'] = 1.0
        
    # CER
    try:
        cer = CharErrorRate()
        metrics['cer'] = cer(predictions, references).item()
    except Exception as e:
        print(f"CER Error: {e}")
        metrics['cer'] = 1.0
        
    return metrics

def evaluate_direction(model, tokenizer, config, dataset_path, src_lang, tgt_lang, device):
    print(f"\nEvaluating Direction: {src_lang} -> {tgt_lang}")
    print(f"Dataset: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print("Dataset path not found. Skipping.")
        return
    
    # Get language tokens
    lang_token_map = config.get('lang_token_map', {})
    if not lang_token_map:
        lang_token_map = {
            "eng": "__eng__",
            "vie": "__vie__",
        }
    
    src_token = lang_token_map.get(src_lang, lang_token_map.get(src_lang[:2], f"__{src_lang}__"))
    tgt_token = lang_token_map.get(tgt_lang, lang_token_map.get(tgt_lang[:2], f"__{tgt_lang}__"))
    
    print(f"Using language tokens: {src_token} -> {tgt_token}")
    
    # Determine direction
    if src_lang in ["eng", "en"] and tgt_lang in ["vie", "vi"]:
        direction = "en_to_vi"
    elif src_lang in ["vie", "vi"] and tgt_lang in ["eng", "en"]:
        direction = "vi_to_en"
    else:
        # Default to en_to_vi
        direction = "en_to_vi"
        print(f"Warning: Unknown language pair {src_lang}->{tgt_lang}, defaulting to en_to_vi")
    
    # Create dataset and dataloader
    eval_dataset = EvaluationDataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        src_lang_token=src_token,
        tgt_lang_token=tgt_token,
        direction=direction,
        max_seq_len=config.get('max_seq_len', 152)
    )
    
    collator = Collator(tokenizer)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.get('test_batch_size', 32),
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )
    
    print(f"Evaluating {len(eval_dataset)} samples...")
    
    # Translate using dataloader
    predictions, references = batch_translate_with_dataloader(
        model=model,
        dataloader=eval_dataloader,
        tokenizer=tokenizer,
        tgt_lang_token=tgt_token,
        device=device,
        max_len=config.get('max_seq_len', 256)
    )
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, references)
    
    print(f"Results for {src_lang}->{tgt_lang}:")
    print(f"BLEU: {metrics['bleu']:.4f}")
    print(f"WER: {metrics['wer']:.4f}")
    print(f"CER: {metrics['cer']:.4f}")
    
    # Save examples
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = output_dir / f"eval_{src_lang}_{tgt_lang}_{timestamp}.txt"
    
    # Get some examples for output
    example_sources = []
    for i in range(min(10, len(eval_dataset))):
        item = eval_dataset[i]
        example_sources.append(item["src_text"])
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Metrics:\nBLEU: {metrics['bleu']}\nWER: {metrics['wer']}\nCER: {metrics['cer']}\n\n")
        f.write("Examples:\n")
        for i in range(min(10, len(predictions))):
            f.write(f"Src: {example_sources[i]}\nRef: {references[i]}\nPred: {predictions[i]}\n\n")
    print(f"Saved details to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config json')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint (file or dir)')
    parser.add_argument('--tokenizer_path', type=str, help='Path to tokenizer (overrides config)')
    parser.add_argument('--save_converted_path', type=str, help='Path to save converted standard checkpoint')
    
    # Evaluation dataset
    parser.add_argument('--dataset_path', type=str, help='Path to test dataset (processed with load_from_disk)')
    parser.add_argument('--bidirectional', action='store_true', help='Evaluate both directions (en->vi and vi->en)')
    parser.add_argument('--src_lang', type=str, default='eng', help='Source language (eng or vie)')
    parser.add_argument('--tgt_lang', type=str, default='vie', help='Target language (vie or eng)')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        
        print(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Ensure it's a torch checkpoint (convert DCP if needed)
        checkpoint_path = _ensure_torch_checkpoint(checkpoint_path)
    
    model, tokenizer, config = load_model_and_tokenizer(args.config, checkpoint_path, device, args.save_converted_path)
    
    if args.tokenizer_path:
        # Re-load tokenizer if specified
        print(f"Reloading tokenizer from {args.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Evaluation
    if args.dataset_path:
        if args.bidirectional:
            # Evaluate both directions
            print("\n=== Evaluating En -> Vi ===")
            evaluate_direction(model, tokenizer, config, args.dataset_path, 'eng', 'vie', device)
            print("\n=== Evaluating Vi -> En ===")
            evaluate_direction(model, tokenizer, config, args.dataset_path, 'vie', 'eng', device)
        else:
            # Single direction
            evaluate_direction(model, tokenizer, config, args.dataset_path, args.src_lang, args.tgt_lang, device)
    else:
        print("Please provide --dataset_path for evaluation.")
        print("Example: python evaluate.py --checkpoint model.pt --dataset_path ./test_dataset --bidirectional")

if __name__ == "__main__":
    main()
