import torch
import os
from torch.utils.data import Dataset
from datasets import load_from_disk


class BidirectionalDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, max_seq_len=512):
        self.ds = load_from_disk(dataset_path)
        
        # Filter sequences longer than max_seq_len
        self.ds = self.ds.filter(
            lambda x: (len(x["input_ids_en"]) <= max_seq_len) and (len(x["input_ids_vi"]) <= max_seq_len),
            num_proc=min(os.cpu_count(), 4)
        )
        
        self.tokenizer = tokenizer
        self.real_len = len(self.ds)
        
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        
        self.en_token_id = tokenizer.convert_tokens_to_ids("__eng__")
        self.vi_token_id = tokenizer.convert_tokens_to_ids("__vie__")

    def __len__(self):
        # for bidirectional dataset, we need to return the length of the dataset * 2
        return self.real_len * 2

    def __getitem__(self, idx):
        # if idx is in the first half -> En -> Vi
        if idx < self.real_len:
            item = self.ds[idx]
            # Source: <__vie__> + VI
            src_ids = [self.vi_token_id] + item["input_ids_en"]
            src_text = item["vi"]
            # Target: <__eng__> + EN
            tgt_ids = [self.en_token_id] + item["input_ids_vi"]
            tgt_text = item["en"]
        # if idx is in the second half -> Vi -> En
        else:
            # map index back to the original range
            real_idx = idx - self.real_len
            item = self.ds[real_idx]
            
            # Source: <__eng__> + EN
            src_ids = [self.en_token_id] + item["input_ids_vi"]
            src_text = item["en"]
            # Target: <__vie__> + VI
            tgt_ids = [self.vi_token_id] + item["input_ids_en"]
            tgt_text = item["vi"]

        # add EOS to the end of each sentence
        src_ids = src_ids + [self.eos_id]
        
        tgt_out = [self.bos_id] + tgt_ids + [self.eos_id]

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_out, dtype=torch.long),
            "src_text": src_text,
            "tgt_text": tgt_text
        }

class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        src_batch = [{"input_ids": x["src_ids"]} for x in batch]
        tgt_batch = [{"input_ids": x["tgt_ids"]} for x in batch]

        # Padding
        src_pad = self.tokenizer.pad(src_batch, pad_to_multiple_of=8, padding_side="right", return_attention_mask=True, return_tensors="pt")
        tgt_pad = self.tokenizer.pad(tgt_batch, pad_to_multiple_of=8, padding_side="right", return_attention_mask=True, return_tensors="pt")

        # Masks
        src_mask = src_pad["attention_mask"].long()
        tgt_mask = tgt_pad["attention_mask"].long()
        
        source_ids = src_pad["input_ids"]
        dec_input = tgt_pad["input_ids"][:, :-1]
        labels = tgt_pad["input_ids"][:, 1:]
        dec_mask = tgt_mask[:, :-1]

        return {
            "src_input_ids": source_ids,
            "src_attention_mask": src_mask,
            "tgt_input_ids": dec_input,
            "tgt_attention_mask": dec_mask,
            "labels": labels
        }
