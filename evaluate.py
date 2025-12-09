import argparse
import torch
from tqdm import tqdm
import time
import os
import json
from pathlib import Path
from typing import List, Optional

# Imports from codebase
from model import build_transformer, ModelConfig
from config import get_kaggle_config
from checkpoint_utils import load_checkpoint, _has_dcp_artifacts, _resolve_checkpoint_dir
from transformers import AutoTokenizer

# Import metrics
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore

def dcp_to_torch_save(dcp_path, output_path):
    """
    Convert a Distributed Checkpoint (DCP) to a standard PyTorch binary file.
    """
    print(f"Converting DCP {dcp_path} to {output_path}...")
    device = 'cpu' # Use CPU for conversion
    
    # Load default config
    config_dict = get_kaggle_config()
    
    # Tokenizer for vocab size
    tokenizer_path = config_dict.get('tokenizer_path')
    if not tokenizer_path or not os.path.exists(tokenizer_path):
            if os.path.exists("tokenizer"):
                tokenizer_path = "tokenizer"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Warning: Could not load tokenizer for conversion: {e}")
        # Fallback to default vocab size if tokenizer fails? 
        # But ModelConfig default is 24000.
        # Let's hope it works or user provided valid path in config/default.
        raise
        
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size)
    
    # Build model
    model = build_transformer(config=model_config)
    model = model.to(device)
    
    # Load DCP
    load_checkpoint(
        checkpoint_path=dcp_path,
        model=model,
        optimizer=None,
        scheduler=None,
        config=config_dict,
        strict=False
    )
    
    # Save
    torch.save(model.state_dict(), output_path)
    print("Conversion complete.")

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
                dcp_to_torch_save(path, candidate)
                print(f"  ✓ Materialized Torch checkpoint at {candidate}")
                return candidate
            except Exception as exc:
                print(f"  ❌ Failed to convert DCP checkpoint: {exc}")

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

def batch_translate(model, sentences, tokenizer, src_lang, tgt_lang, device, batch_size=32, max_len=500, config=None):
    """
    Translate a list of sentences.
    """
    model.eval()
    translations = []
    
    # Get language tokens
    lang_token_map = config.get('lang_token_map', {}) if config else {}
    if not lang_token_map:
        lang_token_map = {
            "eng": "__eng__",
            "vie": "__vie__",
        }
    
    src_token = lang_token_map.get(src_lang, lang_token_map.get(src_lang[:2], f"__{src_lang}__"))
    tgt_token = lang_token_map.get(tgt_lang, lang_token_map.get(tgt_lang[:2], f"__{tgt_lang}__"))
    
    print(f"Translating {len(sentences)} sentences from {src_lang} ({src_token}) to {tgt_lang} ({tgt_token})...")

    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch_sentences = sentences[i:i+batch_size]
            
            # Prepare batch inputs
            # [src_token] + content + [eos]
            src_token_id = tokenizer.convert_tokens_to_ids(src_token)
            
            batch_input_ids = []
            for sent in batch_sentences:
                tokens = tokenizer.encode(sent, add_special_tokens=False)
                # Truncate if too long (optional)
                if len(tokens) > max_len - 2:
                    tokens = tokens[:max_len-2]
                batch_input_ids.append([src_token_id] + tokens + [tokenizer.eos_token_id])
                
            # Pad
            max_batch_len = max(len(ids) for ids in batch_input_ids)
            padded_input_ids = []
            attention_masks = []
            
            for ids in batch_input_ids:
                pad_len = max_batch_len - len(ids)
                padded_ids = ids + [tokenizer.pad_token_id] * pad_len
                mask = [1] * len(ids) + [0] * pad_len
                padded_input_ids.append(padded_ids)
                attention_masks.append(mask)
                
            src_inputs = torch.tensor(padded_input_ids, dtype=torch.long, device=device)
            src_masks = torch.tensor(attention_masks, dtype=torch.long, device=device)
            
            # Create proper mask for model: (B, 1, 1, S)
            # The model expects (B, S) where 0 is pad, and converts it internally.
            # But wait, model.py Transformer.forward does:
            # src_mask = (src_attention_mask == 0).unsqueeze(1).unsqueeze(2)
            # So we pass (B, S) src_masks (1 for valid, 0 for pad).
            
            # Encoder
            # Encoder.forward takes (x, mask)
            # mask should be (B, 1, 1, S) or compatible. 
            # Transformer.forward handles the conversion. Here we call encoder directly.
            # We must convert mask manually.
            enc_mask_expanded = (src_masks == 0).unsqueeze(1).unsqueeze(2)
            encoder_output, _ = model.encoder(src_inputs, mask=enc_mask_expanded)
            
            # Greedy Decode for batch
            # Initial decoder input: [bos, tgt_token]
            tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_token)
            decoder_input = torch.tensor([[tokenizer.bos_token_id, tgt_token_id]] * len(batch_sentences), dtype=torch.long, device=device)
            
            finished = torch.zeros(len(batch_sentences), dtype=torch.bool, device=device)
            
            for _ in range(max_len):
                # Decoder mask not needed for attention within decoder (causal handled), 
                # but we need to mask padding if we were padding decoder input (we aren't, all same length grow together)
                # However, for Cross-Attention, we need src_mask (enc_mask_expanded).
                
                # Decoder
                decoder_output, _ = model.decoder(decoder_input, encoder_output, tgt_mask=None, src_mask=enc_mask_expanded)
                
                # Project last token
                logits = model.lm_head(model.norm(decoder_output[:, -1]))
                next_tokens = torch.argmax(logits, dim=-1)
                
                # Update finished status
                finished |= (next_tokens == tokenizer.eos_token_id)
                
                # Append
                decoder_input = torch.cat([decoder_input, next_tokens.unsqueeze(1)], dim=1)
                
                if finished.all():
                    break
            
            # Decode to text
            for j, seq in enumerate(decoder_input):
                # Skip [bos, tgt_token] (first 2)
                # Stop at first eos
                tokens = seq[2:].tolist()
                try:
                    eos_index = tokens.index(tokenizer.eos_token_id)
                    tokens = tokens[:eos_index]
                except ValueError:
                    pass
                
                text = tokenizer.decode(tokens)
                translations.append(text)
                
    return translations

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

def evaluate_direction(model, tokenizer, config, src_file, ref_file, src_lang, tgt_lang, device):
    print(f"\nEvaluating Direction: {src_lang} -> {tgt_lang}")
    print(f"Source: {src_file}")
    print(f"Reference: {ref_file}")
    
    if not os.path.exists(src_file) or not os.path.exists(ref_file):
        print("Source or reference file not found. Skipping.")
        return
        
    with open(src_file, 'r', encoding='utf-8') as f:
        sources = f.read().splitlines()
    with open(ref_file, 'r', encoding='utf-8') as f:
        references = f.read().splitlines()
        
    if len(sources) != len(references):
        print("Warning: Source and reference length mismatch. Truncating to shorter.")
        min_len = min(len(sources), len(references))
        sources = sources[:min_len]
        references = references[:min_len]
        
    predictions = batch_translate(
        model, sources, tokenizer, 
        src_lang=src_lang, tgt_lang=tgt_lang, 
        device=device,
        batch_size=config.get('test_batch_size', 32),
        config=config
    )
    
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
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Metrics:\nBLEU: {metrics['bleu']}\nWER: {metrics['wer']}\nCER: {metrics['cer']}\n\n")
        f.write("Examples:\n")
        for i in range(min(10, len(predictions))):
            f.write(f"Src: {sources[i]}\nRef: {references[i]}\nPred: {predictions[i]}\n\n")
    print(f"Saved details to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config json')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint (file or dir)')
    parser.add_argument('--tokenizer_path', type=str, help='Path to tokenizer (overrides config)')
    parser.add_argument('--save_converted_path', type=str, help='Path to save converted standard checkpoint')
    
    # Evaluation files
    parser.add_argument('--test_en', type=str, help='Path to English/Source test file')
    parser.add_argument('--test_vi', type=str, help='Path to Vietnamese/Target test file')
    
    # Optional direct specification
    parser.add_argument('--src_file', type=str, help='Source file for single direction')
    parser.add_argument('--ref_file', type=str, help='Reference file for single direction')
    parser.add_argument('--src_lang', type=str, default='eng')
    parser.add_argument('--tgt_lang', type=str, default='vie')
    
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

    # Direction 1: En -> Vi (or Src -> Tgt)
    if args.test_en and args.test_vi:
        # Bidirectional evaluation
        evaluate_direction(model, tokenizer, config, args.test_en, args.test_vi, 'en', 'vi', device)
        evaluate_direction(model, tokenizer, config, args.test_vi, args.test_en, 'vi', 'en', device)
    elif args.src_file and args.ref_file:
        # Single direction
        evaluate_direction(model, tokenizer, config, args.src_file, args.ref_file, args.src_lang, args.tgt_lang, device)
    else:
        print("Please provide --test_en and --test_vi for bidirectional evaluation, or --src_file and --ref_file for single direction.")

if __name__ == "__main__":
    main()
