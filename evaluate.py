import argparse
import torch
from tqdm import tqdm
import time
import os
import json
import csv
import hashlib
import re
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

import sacrebleu
from comet import download_model, load_from_checkpoint

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
        # First try the original location
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
            
            # Save converted checkpoint in the same folder as the original checkpoint
            candidate_writable = os.path.join(path, "pytorch_model.bin")
            
            # Check if already converted
            if os.path.isfile(candidate_writable):
                print(f"  ✓ Using existing converted checkpoint at {candidate_writable}")
                return candidate_writable
            
            try:
                # Use torch's built-in dcp_to_torch_save function
                # Load state dict from DCP and save to torch format in the same folder
                print(f"  Converting to {candidate_writable}")
                dcp_to_torch_save(
                    dcp_checkpoint_dir=path,
                    torch_save_path=candidate_writable,
                )
                print(f"  ✓ Materialized Torch checkpoint at {candidate_writable}")
                return candidate_writable
            except PermissionError as exc:
                # If cannot write to original location, fallback to /tmp
                print(f"  ⚠ Cannot write to checkpoint folder (may be read-only), using /tmp instead...")
                writable_dir = "/tmp/converted_checkpoints"
                os.makedirs(writable_dir, exist_ok=True)
                checkpoint_name = os.path.basename(path.rstrip('/'))
                candidate_writable = os.path.join(writable_dir, f"{checkpoint_name}_pytorch_model.bin")
                
                if os.path.isfile(candidate_writable):
                    print(f"  ✓ Using cached converted checkpoint at {candidate_writable}")
                    return candidate_writable
                
                print(f"  Converting to {candidate_writable}")
                dcp_to_torch_save(
                    dcp_checkpoint_dir=path,
                    torch_save_path=candidate_writable,
                )
                print(f"  ✓ Materialized Torch checkpoint at {candidate_writable}")
                return candidate_writable
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

def batch_translate_with_dataloader(model, dataloader, tokenizer, tgt_lang_token, device, max_len=256, num_beams=1, length_penalty=1.0):
    """
    Translate using a DataLoader with model.generate().
    Returns predictions and references.
    """
    model.eval()
    predictions = []
    references = []
    
    tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_lang_token)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Translating")):
            src_inputs = batch["src_input_ids"].to(device)
            src_masks = batch["src_attention_mask"].to(device)
            ref_texts = batch["tgt_text"]
            
            # --- DEBUG: Inspect Evaluation Data ---
            if i == 0:
                print("\n--- DEBUG: First Evaluation Batch ---")
                print(f"Src IDs: {src_inputs[0].tolist()}")
                print(f"Src Text (Decoded): {tokenizer.decode(src_inputs[0], skip_special_tokens=False)}")
                print(f"Ref Text (Original): {ref_texts[0]}")
                print(f"Beam Size: {num_beams}, Length Penalty: {length_penalty}")
                print("-------------------------------------\n")
            # --------------------------------------

            # Save references
            references.extend(ref_texts)
            
            # Prepare start tokens: [BOS, tgt_lang_token]
            batch_size = src_inputs.size(0)
            tgt_start_ids = torch.tensor(
                [[tokenizer.bos_token_id, tgt_token_id]] * batch_size,
                dtype=torch.long,
                device=device
            )
            
            # Generate translations using model.generate()
            generated = model.generate(
                src_input_ids=src_inputs,
                src_attention_mask=src_masks,
                tgt_start_ids=tgt_start_ids,
                max_len=max_len,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=num_beams,
                length_penalty=length_penalty
            )
            
            # Decode to text (skip BOS and lang token, stop at EOS)
            for seq in generated:
                tokens = seq[2:].tolist()  # Skip [BOS, lang_token]
                try:
                    eos_index = tokens.index(tokenizer.eos_token_id)
                    tokens = tokens[:eos_index]
                except ValueError:
                    pass
                
                text = tokenizer.decode(tokens)
                predictions.append(text)
    
    return predictions, references

def calculate_metrics(predictions, references, sources=None, comet_model=None):
    """
    Calculate BLEU, chrF++, and COMET metrics.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations (single-reference corpus)
        sources: List of source texts (required for COMET)
        comet_model: Loaded COMET model (optional, will load if None)
    """
    metrics = {}
    
    # Prepare references for sacrebleu: list of reference corpora
    # Here we only have 1 reference, so wrap once: [references]
    list_of_references = [references]
    
    # BLEU using sacrebleu
    try:
        bleu = sacrebleu.corpus_bleu(predictions, list_of_references)
        # sacrebleu>=2 returns BLEUScore without .signature; try to extract if available
        bleu_sig = ""
        if hasattr(bleu, "signature"):
            bleu_sig = str(bleu.signature)
        elif hasattr(bleu, "format"):
            try:
                fmt = bleu.format()
                bleu_sig = str(fmt.get("signature", ""))
            except Exception:
                bleu_sig = ""
        metrics['bleu'] = {
            "score": float(bleu.score),
            "signature": bleu_sig,
        }
    except Exception as e:
        print(f"BLEU Error: {e}")
        metrics['bleu'] = {
            "score": 0.0,
            "signature": "",
        }

    # chrF++ using sacrebleu
    try:
        chrf_plus = sacrebleu.corpus_chrf(predictions, list_of_references, word_order=2)
        chrf_sig = ""
        if hasattr(chrf_plus, "signature"):
            chrf_sig = str(chrf_plus.signature)
        elif hasattr(chrf_plus, "format"):
            try:
                fmt = chrf_plus.format()
                chrf_sig = str(fmt.get("signature", ""))
            except Exception:
                chrf_sig = ""
        metrics['chrf_plus'] = {
            "score": float(chrf_plus.score),
            "signature": chrf_sig,
        }
    except Exception as e:
        print(f"chrF++ Error: {e}")
        metrics['chrf_plus'] = {
            "score": 0.0,
            "signature": "",
        }

    # COMET metric
    if sources is not None and comet_model is not None:
        try:
            # Prepare data for COMET (list of dicts)
            comet_data = [
                {"src": src, "mt": pred, "ref": ref}
                for src, pred, ref in zip(sources, predictions, references)
            ]
            
            # Calculate COMET scores
            comet_output = comet_model.predict(
                comet_data,
                batch_size=256,
                gpus=1 if torch.cuda.is_available() else 0,
            )
            # Use system_score as the main COMET metric
            metrics['comet'] = float(comet_output.system_score)
        except Exception as e:
            print(f"COMET Error: {e}")
            import traceback
            traceback.print_exc()
            metrics['comet'] = None
    else:
        if sources is None:
            print("Warning: Sources not provided, skipping COMET metric.")
        elif comet_model is None:
            print("Warning: COMET model not loaded, skipping COMET metric.")
        metrics['comet'] = None
        
    return metrics

def extract_step_from_checkpoint(checkpoint_path: str) -> str:
    """Extract step number from checkpoint path (e.g., 'checkpoints/checkpoint-84987' -> '84987')."""
    # Try to find pattern like checkpoint-12345 or checkpoint_12345
    match = re.search(r'checkpoint[-_](\d+)', checkpoint_path)
    if match:
        return match.group(1)
    # Fallback: try to extract any number at the end
    match = re.search(r'(\d+)$', os.path.basename(checkpoint_path.rstrip('/')))
    if match:
        return match.group(1)
    # If no step found, use timestamp as fallback
    return time.strftime("%Y%m%d-%H%M%S")

def extract_dataset_name(dataset_path: str) -> str:
    """Extract and clean dataset name from dataset path."""
    # Get the last component of the path
    dataset_name = os.path.basename(dataset_path.rstrip('/'))
    
    # If empty or just a dot, try parent directory
    if not dataset_name or dataset_name == '.':
        dataset_name = os.path.basename(os.path.dirname(dataset_path.rstrip('/')))
    
    # Clean the name: replace spaces and special chars with underscores
    # Keep only alphanumeric, underscore, and hyphen
    dataset_name = re.sub(r'[^\w\-]', '_', dataset_name)
    # Remove multiple consecutive underscores
    dataset_name = re.sub(r'_+', '_', dataset_name)
    # Remove leading/trailing underscores
    dataset_name = dataset_name.strip('_')
    
    # If still empty, use a default name
    if not dataset_name:
        dataset_name = "dataset"
    
    return dataset_name

def evaluate_direction(model, tokenizer, config, dataset_path, src_lang, tgt_lang, device, checkpoint_path=None, num_beams=1, length_penalty=1.0, comet_model=None):
    print(f"\nEvaluating Direction: {src_lang} -> {tgt_lang}")
    print(f"Dataset: {dataset_path}")
    print(f"Beam Search: beams={num_beams}, length_penalty={length_penalty}")
    
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
        batch_size=config.get('test_batch_size', 64),
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    
    print(f"Evaluating {len(eval_dataset)} samples...")
    
    # Translate using dataloader
    predictions, references = batch_translate_with_dataloader(
        model=model,
        dataloader=eval_dataloader,
        tokenizer=tokenizer,
        tgt_lang_token=tgt_token,
        device=device,
        max_len=config.get('max_seq_len', 256),
        num_beams=num_beams,
        length_penalty=length_penalty
    )
    
    # Get all sources for metrics calculation
    all_sources = []
    for i in range(len(eval_dataset)):
        item = eval_dataset[i]
        all_sources.append(item["src_text"])
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, references, sources=all_sources, comet_model=comet_model)
    
    print(f"Results for {src_lang}->{tgt_lang}:")
    print(f"BLEU: {metrics['bleu']['score']:.4f}")
    print(f"  signature: {metrics['bleu']['signature']}")
    print(f"chrF++: {metrics['chrf_plus']['score']:.4f}")
    print(f"  signature: {metrics['chrf_plus']['signature']}")
    if metrics['comet'] is not None:
        print(f"COMET: {metrics['comet']:.4f}")
    else:
        print(f"COMET: N/A")
    
    # Save results
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Extract step number from checkpoint path
    step_number = extract_step_from_checkpoint(checkpoint_path) if checkpoint_path else time.strftime("%Y%m%d-%H%M%S")
    
    # Extract dataset name from dataset path
    dataset_name = extract_dataset_name(dataset_path)
    
    # Save all predictions to CSV file
    pred_file = output_dir / f"predictions_{src_lang}_{tgt_lang}_{dataset_name}_{step_number}.csv"
    with open(pred_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prediction'])
        for pred in predictions:
            writer.writerow([pred])
    print(f"Saved predictions to {pred_file}")
    
    # Save all references to CSV file
    ref_file = output_dir / f"references_{src_lang}_{tgt_lang}_{dataset_name}_{step_number}.csv"
    with open(ref_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['reference'])
        for ref in references:
            writer.writerow([ref])
    print(f"Saved references to {ref_file}")
    
    # Save all sources to CSV file
    src_file = output_dir / f"sources_{src_lang}_{tgt_lang}_{dataset_name}_{step_number}.csv"
    with open(src_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source'])
        for src in all_sources:
            writer.writerow([src])
    print(f"Saved sources to {src_file}")
    
    # Save summary with metrics and examples
    summary_file = output_dir / f"summary_{src_lang}_{tgt_lang}_{dataset_name}_{step_number}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Summary: {src_lang} -> {tgt_lang}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Step: {step_number}\n")
        f.write(f"Total samples: {len(predictions)}\n")
        f.write(f"\nMetrics:\n")
        f.write(f"BLEU: {metrics['bleu']['score']:.4f}\n")
        f.write(f"BLEU signature: {metrics['bleu']['signature']}\n")
        f.write(f"chrF++: {metrics['chrf_plus']['score']:.4f}\n")
        f.write(f"chrF++ signature: {metrics['chrf_plus']['signature']}\n")
        if metrics['comet'] is not None:
            f.write(f"COMET: {metrics['comet']:.4f}\n")
        else:
            f.write(f"COMET: N/A\n")
        f.write(f"\n{'='*80}\n")
        f.write(f"Sample Examples (first 10):\n")
        f.write(f"{'='*80}\n\n")
        for i in range(min(10, len(predictions))):
            f.write(f"Example {i+1}:\n")
            f.write(f"Source:     {all_sources[i]}\n")
            f.write(f"Reference:  {references[i]}\n")
            f.write(f"Prediction: {predictions[i]}\n")
            f.write(f"{'-'*80}\n\n")
    print(f"Saved summary to {summary_file}")
    
    # Save as JSON for easy parsing
    results_json = {
        "direction": f"{src_lang}->{tgt_lang}",
        "step": step_number,
        "dataset": dataset_name,
        "total_samples": len(predictions),
        "metrics": {
            "bleu": {
                "score": float(metrics['bleu']['score']),
                "signature": metrics['bleu']['signature'],
            },
            "chrf_plus": {
                "score": float(metrics['chrf_plus']['score']),
                "signature": metrics['chrf_plus']['signature'],
            },
            "comet": float(metrics['comet']) if metrics['comet'] is not None else None,
        },
        "files": {
            "predictions": str(pred_file),
            "references": str(ref_file),
            "sources": str(src_file),
            "summary": str(summary_file)
        }
    }
    json_file = output_dir / f"results_{src_lang}_{tgt_lang}_{dataset_name}_{step_number}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON results to {json_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="train_config.json", help='Path to config json')
    parser.add_argument('--checkpoint', type=str, default="checkpoints/checkpoint-131410", help='Path to checkpoint (file or dir)')
    parser.add_argument('--tokenizer_path', type=str, default="/home/uet/cuongdam/ST/Tokenizer_ENVI/", help='Path to tokenizer (overrides config)')
    parser.add_argument('--save_converted_path', type=str, help='Path to save converted standard checkpoint')
    
    # Evaluation dataset
    parser.add_argument('--dataset_path', type=str, default="/home/uet/cuongdam/ST/flores_tokenized/flores_devtest", help='Path to test dataset (processed with load_from_disk)')
    parser.add_argument('--bidirectional', default=True, help='Evaluate both directions (en->vi and vi->en)')
    parser.add_argument('--src_lang', type=str, default='eng', help='Source language (eng or vie)')
    parser.add_argument('--tgt_lang', type=str, default='vie', help='Target language (vie or eng)')
    
    parser.add_argument('--beam_size', type=int, default=5, help='Number of beams for beam search')
    parser.add_argument('--length_penalty', type=float, default=1.0, help='Length penalty for beam search')
    
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

    # Load COMET model once (will be reused for all directions)
    comet_model = None
    try:
        print("Loading COMET model...")
        model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(model_path)
        print("COMET model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load COMET model: {e}")
        print("Continuing without COMET metric...")

    # Evaluation
    if args.dataset_path:
        # Get original checkpoint path before conversion
        original_checkpoint_path = args.checkpoint if args.checkpoint else None
        if args.bidirectional:
            # Evaluate both directions
            print("\n=== Evaluating En -> Vi ===")
            evaluate_direction(model, tokenizer, config, args.dataset_path, 'eng', 'vie', device, original_checkpoint_path, num_beams=args.beam_size, length_penalty=args.length_penalty, comet_model=comet_model)
            print("\n=== Evaluating Vi -> En ===")
            evaluate_direction(model, tokenizer, config, args.dataset_path, 'vie', 'eng', device, original_checkpoint_path, num_beams=args.beam_size, length_penalty=args.length_penalty, comet_model=comet_model)
        else:
            # Single direction
            evaluate_direction(model, tokenizer, config, args.dataset_path, args.src_lang, args.tgt_lang, device, original_checkpoint_path, num_beams=args.beam_size, length_penalty=args.length_penalty, comet_model=comet_model)
    else:
        print("Please provide --dataset_path for evaluation.")
        print("Example: python evaluate.py --checkpoint model.pt --dataset_path ./test_dataset --bidirectional")

if __name__ == "__main__":
    main()
