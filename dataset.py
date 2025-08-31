import torch
import numpy as np
from torch.utils.data import Dataset
from tokenizers import Tokenizer
import random

class ViLoDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_text_file, tgt_text_file, tokenizer, max_len=None):
        super().__init__()
        self.src_data = np.memmap(src_file, dtype=np.uint16, mode='r')
        self.tgt_data = np.memmap(tgt_file, dtype=np.uint16, mode='r')
        with open(src_text_file, 'r', encoding='utf-8') as f:
            self.src_texts = f.readlines()
        with open(tgt_text_file, 'r', encoding='utf-8') as f:
            self.tgt_texts = f.readlines()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.unk_token = tokenizer.token_to_id("<unk>")
        assert len(self.src_data) == len(self.tgt_data) == len(self.src_texts) == len(self.tgt_texts)

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_tokens = self.src_data[idx]
        tgt_tokens = self.tgt_data[idx]
        return {
            "src_tokens": src_tokens.tolist(),
            "tgt_tokens": tgt_tokens.tolist(),
            "src_text": self.src_texts[idx].strip(),
            "tgt_text": self.tgt_texts[idx].strip()
        }

class CollateBatch:
    def __init__(
        self, eos_token_id,
        src_lang_token_id,
        tgt_lang_token_id,
    ):
        self.eos_token_id = eos_token_id
        self.src_lang_token_id = src_lang_token_id
        self.tgt_lang_token_id = tgt_lang_token_id

    def __call__(self, batch):
        src_seqlens = [len(item["src_tokens"]) + 2 for item in batch] # +2 for EOS and tag language
        tgt_seqlens = [len(item["tgt_tokens"]) + 2 for item in batch] # +2 for EOS and tag language
        # Get max lengths in this batch
        src_max_len = max(src_seqlens)
        tgt_max_len = max(tgt_seqlens)
        # cu_seqlen for flash attention
        src_cu_seqlen = torch.cumsum(torch.tensor([0] + src_seqlens), dim=0)
        tgt_cu_seqlen = torch.cumsum(torch.tensor([0] + tgt_seqlens), dim=0)
        # position for rope when pack qkv
        src_position_ids = torch.cat(
            [torch.arange(l, device=src_cu_seqlen.device) for l in src_seqlens]
        )
        tgt_position_ids = torch.cat(
            [torch.arange(l, device=tgt_cu_seqlen.device) for l in tgt_seqlens]
        )

        src_total_tokens = sum(src_seqlens)
        tgt_total_tokens = sum(tgt_seqlens)
        encoder_inputs = torch.zeros(src_total_tokens, dtype=torch.long, device=src_cu_seqlen.device)
        decoder_inputs = torch.zeros(tgt_total_tokens, dtype=torch.long, device=tgt_cu_seqlen.device)
        labels = torch.zeros(tgt_total_tokens, dtype=torch.long, device=tgt_cu_seqlen.device)
        src_texts = []
        tgt_texts = []
        for i, item in enumerate(batch):
            src_tokens = len(item["src_tokens"])
            tgt_tokens = len(item["tgt_tokens"])

            src = torch.tensor(
                [self.src_lang_token_id] + item["src_tokens"] + [self.eos_token_id],
                dtype=torch.long, device=src_cu_seqlen.device
            )
            tgt = torch.tensor(
                [self.tgt_lang_token_id] + item["tgt_tokens"] + [self.eos_token_id],
                dtype=torch.long, device=tgt_cu_seqlen.device
            )
            encoder_inputs[src_cu_seqlen[i]:src_cu_seqlen[i + 1]] = src
            decoder_inputs[tgt_cu_seqlen[i]:tgt_cu_seqlen[i + 1]] = tgt[:-1]
            labels[tgt_cu_seqlen[i]:tgt_cu_seqlen[i + 1]] = tgt[1:]
            src_texts.append(item.get("src_text", ""))
            tgt_texts.append(item.get("tgt_text", ""))
        return {
            "encoder_input": encoder_inputs,
            "decoder_input": decoder_inputs,
            "label": labels,
            "src_text": src_texts,
            "tgt_text": tgt_texts,
            "src_cu_seqlen": src_cu_seqlen,
            "tgt_cu_seqlen": tgt_cu_seqlen,
            "src_position_ids": src_position_ids,
            "tgt_position_ids": tgt_position_ids,
            "src_max_len": src_max_len,
            "tgt_max_len": tgt_max_len
        }
