import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import unpad_input, pad_input

class FlashAttentionBlock(nn.Module):
    def __init__(self, config, causal_mask=True):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.causal_mask = causal_mask
        self.softmax_scale = 1.0 / (config.head_dim ** 0.5)

    def forward(self, x, encoder_output=None, query_mask=None, encoder_mask=None):
        """
        x: [batch, seq_q, hidden]
        encoder_output: [batch, seq_k, hidden] (nếu là cross attention)
        query_mask: [batch, seq_q] (bool, True là token thật)
        encoder_mask: [batch, seq_k] (bool, True là token thật)
        """
        batch_size, seq_q, _ = x.shape
        is_cross = encoder_output is not None
        
        # 1. Projections
        q = self.w_q(x).view(batch_size, seq_q, self.num_heads, self.head_dim)
        if is_cross:
            k = self.w_k(encoder_output).view(batch_size, -1, self.num_heads, self.head_dim)
            v = self.w_v(encoder_output).view(batch_size, -1, self.num_heads, self.head_dim)
            # Cross attention thường KHÔNG dùng causal mask
            is_causal = False 
            kv_mask = encoder_mask
        else:
            k = self.w_k(x).view(batch_size, seq_q, self.num_heads, self.head_dim)
            v = self.w_v(x).view(batch_size, seq_q, self.num_heads, self.head_dim)
            is_causal = self.causal_mask
            kv_mask = query_mask

        # 2. Xử lý Unpadding cho Q và KV độc lập
        # Xử lý Q
        if query_mask is not None:
            q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_mask)
        else:
            # Tạo cu_seqlens giả lập nếu không có mask
            q_unpad = q.flatten(0, 1)
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seq_q, seq_q, device=q.device, dtype=torch.int32)
            max_seqlen_q = seq_q

        # Xử lý KV (Key/Value)
        if kv_mask is not None:
            k_unpad, _, cu_seqlens_k, max_seqlen_k = unpad_input(k, kv_mask)
            v_unpad, _, _, _ = unpad_input(v, kv_mask)
        else:
            k_unpad = k.flatten(0, 1)
            v_unpad = v.flatten(0, 1)
            seq_k = k.shape[1]
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seq_k, seq_k, device=k.device, dtype=torch.int32)
            max_seqlen_k = seq_k

        # 3. Flash Attention Varlen Core
        # Lưu ý: Truyền cu_seqlens_q và cu_seqlens_k riêng biệt
        output_unpad = flash_attn_varlen_func(
            q_unpad, k_unpad, v_unpad,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0 if not self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=is_causal
        )

        # 4. Đưa về lại shape ban đầu (Repadding)
        if query_mask is not None:
            output = pad_input(output_unpad, indices_q, batch_size, seq_q)
        else:
            output = output_unpad.view(batch_size, seq_q, self.num_heads, self.head_dim)

        return self.w_o(output.flatten(2))