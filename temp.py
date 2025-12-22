#!/usr/bin/env python3
"""
Script để tokenize lại flores_dev và flores_devtest với tokenizer mới.
"""

import os
import json
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
from tqdm import tqdm


def tokenize_texts(tokenizer, texts, add_special_tokens=False):
    """Tokenize một batch các text."""
    # Tokenize without special tokens (we'll add them later in dataset)
    encoded = tokenizer(
        texts,
        add_special_tokens=add_special_tokens,
        return_attention_mask=False,
        return_token_type_ids=False,
        padding=False,
        truncation=False
    )
    return encoded["input_ids"]


def retokenize_dataset(dataset_path, tokenizer_path, output_path=None):
    """
    Load dataset và tokenize lại các text với tokenizer mới.
    
    Args:
        dataset_path: Đường dẫn đến dataset đã được tokenize trước đó
        tokenizer_path: Đường dẫn đến tokenizer mới
        output_path: Đường dẫn để lưu dataset đã tokenize lại (mặc định ghi đè dataset_path)
    """
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    print(f"\nLoading dataset from: {dataset_path}")
    try:
        dataset = load_from_disk(dataset_path)
    except Exception as e:
        print(f"Error loading dataset with load_from_disk: {e}")
        print("Trying alternative loading method...")
        # Thử load bằng cách khác nếu cần
        raise e
    
    print(f"Dataset loaded. Number of examples: {len(dataset)}")
    print(f"Dataset features: {dataset.features}")
    
    # Kiểm tra các field cần thiết
    required_fields = ["en", "vi"]
    for field in required_fields:
        if field not in dataset.column_names:
            raise ValueError(f"Dataset must contain '{field}' field. Found fields: {dataset.column_names}")
    
    # Lấy tất cả text
    print("\nExtracting texts...")
    en_texts = dataset["en"]
    vi_texts = dataset["vi"]
    
    print(f"Number of English texts: {len(en_texts)}")
    print(f"Number of Vietnamese texts: {len(vi_texts)}")
    
    # Tokenize lại
    print("\nTokenizing English texts...")
    input_ids_en = []
    for text in tqdm(en_texts, desc="Tokenizing EN"):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        input_ids_en.append(ids)
    
    print("\nTokenizing Vietnamese texts...")
    input_ids_vi = []
    for text in tqdm(vi_texts, desc="Tokenizing VI"):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        input_ids_vi.append(ids)
    
    # Tạo dataset mới với input_ids đã được tokenize lại
    print("\nCreating new dataset...")
    new_data = {
        "en": en_texts,
        "vi": vi_texts,
        "input_ids_en": input_ids_en,
        "input_ids_vi": input_ids_vi
    }
    
    new_dataset = Dataset.from_dict(new_data)
    print(f"New dataset created with {len(new_dataset)} examples")
    
    # Lưu dataset
    if output_path is None:
        output_path = dataset_path
    
    print(f"\nSaving dataset to: {output_path}")
    new_dataset.save_to_disk(output_path)
    print(f"Dataset saved successfully!")
    
    # In một số thống kê
    print("\n=== Tokenization Statistics ===")
    en_lengths = [len(ids) for ids in input_ids_en]
    vi_lengths = [len(ids) for ids in input_ids_vi]
    
    print(f"English token lengths:")
    print(f"  Mean: {sum(en_lengths) / len(en_lengths):.2f}")
    print(f"  Min: {min(en_lengths)}")
    print(f"  Max: {max(en_lengths)}")
    
    print(f"\nVietnamese token lengths:")
    print(f"  Mean: {sum(vi_lengths) / len(vi_lengths):.2f}")
    print(f"  Min: {min(vi_lengths)}")
    print(f"  Max: {max(vi_lengths)}")
    
    return new_dataset


def main():
    """Main function để tokenize lại cả flores_dev và flores_devtest."""
    
    # Đường dẫn tokenizer mới
    tokenizer_path = "./Tokenizer_ENVI"
    
    # Kiểm tra tokenizer path
    if not os.path.exists(tokenizer_path):
        # Thử đường dẫn tương đối
        tokenizer_path = "./Tokenizer_ENVI"
        if not os.path.exists(tokenizer_path):
            print(f"Error: Tokenizer not found at {tokenizer_path}")
            print("Please specify the correct tokenizer path.")
            return
    
    # Đường dẫn datasets
    base_dir = "./flores_tokenized"
    datasets_to_process = [
        ("flores_dev", os.path.join(base_dir, "flores_dev")),
        ("flores_devtest", os.path.join(base_dir, "flores_devtest"))
    ]
    
    # Process từng dataset
    for dataset_name, dataset_path in datasets_to_process:
        if not os.path.exists(dataset_path):
            print(f"\nWarning: Dataset {dataset_name} not found at {dataset_path}")
            print("Skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name}")
        print(f"{'='*60}")
        
        try:
            retokenize_dataset(
                dataset_path=dataset_path,
                tokenizer_path=tokenizer_path,
                output_path=dataset_path  # Ghi đè dataset cũ
            )
            print(f"\n✓ Successfully processed {dataset_name}")
        except Exception as e:
            print(f"\n✗ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("All datasets processed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

