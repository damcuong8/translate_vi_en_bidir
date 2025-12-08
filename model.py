# This version of transformer is inspired by deepseek

import torch
import torch.nn as nn
import torch.distributed as dist
import math
from torch.profiler import record_function
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from torch.nn.attention import sdpa_kernel, SDPBackend


@dataclass
class ModelConfig:
    num_hidden_layers: int = 9
    shared_experts: int = 1
    num_dense_layers: int = 1
    num_route_experts: int = 16
    num_activated_experts: int = 2
    vocab_size: int = 24000
    hidden_size: int = 512
    intermediate_size: int = 1360
    moe_intermediate_size: int = 384
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    initial_context_length: int = 512
    rope_theta: float = 10000.0
    rope_scaling_factor: float = 2.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 4.0
    initializer_range: float = 0.02
    gate_initializer_range: float = 0.01
    use_sdpa_kernel: bool = False
    w_load_loss: float = 0.01
    w_importance_loss: float = 0.01
    w_aux_loss: float = 0.01
    use_deepspeed_moe: bool = False
    use_fsdp_moe: bool = True


class RMSNorm(nn.Module):

    def __init__(
        self, dim: int, eps: float = 1e-6, device: torch.device | None = None
    ):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(
            torch.ones(dim, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.dim
        t, dtype = x.float(), x.dtype
        t = F.rms_norm(t, (self.dim,), self.weight, self.eps)
        return t.to(dtype)
    
def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    cos = cos[position_ids].to(x.dtype) # (batch, seq_len, head_dim // 2)
    sin = sin[position_ids].to(x.dtype)

    cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim // 2)
    sin = sin.unsqueeze(1)  # (batch, 1, seq_len, head_dim // 2)
    
    x1, x2 = torch.chunk(x, 2, dim=-1)

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    return torch.cat((o1, o2), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(
            self,
            head_dim: int,
            base: int,
            dtype: torch.dtype,
            initial_context_length: int,
            scaling_factor: float = 1.0,
            ntk_alpha: float = 1.0,
            ntk_beta: float = 32.0,
            device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = torch.exp(
            math.log(self.base) * torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device) 
        )

        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1
            )
        
            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * math.pi))
                / math.log(self.base)
            )

            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * math.pi))
                / math.log(self.base)
            )

            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolaton = 1.0 / freq

            ramp = (
                torch.arange(0, d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)

            mask = 1 - ramp.clamp(0, 1)

            inv_freq = (1 - mask) * interpolation + mask * extrapolaton
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq
    
        t = torch.arange(self.initial_context_length, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i, j -> ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        query = _apply_rotary_emb(query, self.cos_cached, self.sin_cached, position_ids)
        key = _apply_rotary_emb(key, self.cos_cached, self.sin_cached, position_ids)

        return query, key

# TODO code switch type of sdpa FA or torch sdpa
class AttentionBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        causal_mask: bool = True,
        device: torch.device | None = None
    ):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.causal_mask = causal_mask


        kv_dim = 2 * config.hidden_size
        self.q = (
            nn.Linear(config.hidden_size, config.hidden_size, device=device)
        )
        self.kv = (
            nn.Linear(config.hidden_size, kv_dim, device=device)
        )
        self.out = nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device
        )
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor | None = None,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        assert hidden_size == self.hidden_size, f"Expected hidden size {self.hidden_size}, got {hidden_size}"
        is_cross_attention = encoder_output is not None

        # Generate position_ids (0 to seq_len-1) for this batch
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        q = self.q(x)
        if is_cross_attention:
            kv = self.kv(encoder_output)
        else:
            kv = self.kv(x)
        k = kv[:, :, : self.hidden_size]
        v = kv[:, :, self.hidden_size :]

        # Reshape for Attention: [Batch, Seq, Heads, HeadDim] -> [Batch, Heads, Seq, HeadDim]
        q = q.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        if is_cross_attention:
            q, _ = self.rope(q, q, position_ids)
        else:
            q, k = self.rope(q, k, position_ids)
        
        # Ensure causal mask is only used for self-attention
        is_causal = self.causal_mask and not is_cross_attention
        
        if mask is not None:
            # SDPA does not support is_causal=True with a mask.
            # If we have a mask (e.g. for padding) and need causal masking,
            # we must merge the causal mask into the provided mask.
            if is_causal:
                seq_len = x.shape[1]
                causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool()
                mask = mask | causal_mask
            is_causal = False

        if self.config.use_sdpa_kernel:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                t = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal)
        else:
            t = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal)
        
        # Reshape back: [Batch, Heads, Seq, HeadDim] -> [Batch, Seq, Hidden]
        t = t.transpose(1, 2).contiguous().view(batch_size, seq_len, self.head_dim * self.num_attention_heads)
        t = self.out(t)
        return t


# TODO implement 5 trainable parameters
def swiglu(x, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(min=-limit, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(x_glu * alpha)
    return out_glu + (x_linear + 1)


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int
    ):
        super().__init__()
        self.hidden_size = dim
        self.intermediate_dim = intermediate_dim
        self.mlp1 = nn.Linear(dim, intermediate_dim * 2)
        self.mlp2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        assert x.shape[-1] == self.hidden_size, f"Expected hidden size {self.hidden_size}, got {x.shape[-1]}"
        t = swiglu(self.mlp1(x))
        t = self.mlp2(t)
        return t, 0.0

class MOEBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None
    ):
        super().__init__()

        assert config.num_route_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.num_route_experts = config.num_route_experts
        self.num_local_experts = config.num_route_experts // world_size
        self.num_activated_experts = config.num_activated_experts
        self.experts_start_idx = rank * self.num_local_experts
        self.experts_end_idx = self.experts_start_idx + self.num_local_experts
        self.w_load_loss = config.w_load_loss
        self.w_importance_loss = config.w_importance_loss
        self.w_aux_loss = config.w_aux_loss
        self.gate = nn.Linear(
            config.hidden_size, config.num_route_experts, device=device
        )

        self.shared_experts = MLP(
            config.hidden_size,
            config.moe_intermediate_size
        )
        
        self.experts = nn.ModuleList(
            [
                MLP(config.hidden_size, config.moe_intermediate_size) if self.experts_start_idx <= i < self.experts_end_idx else nn.Identity()
                for i in range(self.num_route_experts)
            ]
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        shape = x.size()
        x = x.view(-1, shape[-1])
        num_tokens, num_features = x.shape

        z, _ = self.shared_experts(x)

        gate_logits = self.gate(x)

        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_weights, experts_indices = torch.topk(gate_probs, self.num_activated_experts, dim=-1)
        # Normalize
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        y = torch.zeros_like(x)
        counts = torch.bincount(experts_indices.flatten(), minlength=self.num_route_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top_expert = torch.where(experts_indices == i)
            # expert returns tuple(tensor, float), take tensor
            exp_out, _ = expert(x[idx])
            y[idx] += exp_out * expert_weights[idx, top_expert]
        
        if world_size > 1:
            dist.all_reduce(y)
        output = (z + y).view(shape)
        
        # loss_load_balancing
        P = gate_probs.mean(dim=0)
        temp_counts = torch.bincount(experts_indices.flatten(), minlength=self.num_route_experts).float()
        D = temp_counts / (num_tokens * self.num_activated_experts)
        loss_load_balancing = self.w_load_loss * self.num_route_experts * (P * D).sum()
        
        # importance loss
        importance = gate_logits.sum(dim=0) 
        importance_mean = importance.mean()
        importance_std = torch.std(importance)
        cv = (importance_std / (importance_mean + 1e-6))
        importance_loss = (
            self.w_importance_loss * (cv ** 2)
        )

        aux_loss = self.w_aux_loss * (loss_load_balancing + importance_loss)

        return output, aux_loss


class FSDPMoEBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None
    ):
        super().__init__()
        self.num_route_experts = config.num_route_experts
        self.num_activated_experts = config.num_activated_experts
        self.w_load_loss = config.w_load_loss
        self.w_importance_loss = config.w_importance_loss
        self.w_aux_loss = config.w_aux_loss
        self.gate = nn.Linear(
            config.hidden_size, config.num_route_experts, device=device
        )

        self.shared_experts = MLP(
            config.hidden_size,
            config.moe_intermediate_size
        )
        
        self.experts = nn.ModuleList(
            [
                MLP(config.hidden_size, config.moe_intermediate_size)
                for i in range(self.num_route_experts)
            ]
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        shape = x.size()
        x = x.view(-1, shape[-1])
        num_tokens, num_features = x.shape

        z, _ = self.shared_experts(x)

        gate_logits = self.gate(x)

        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_weights, experts_indices = torch.topk(gate_probs, self.num_activated_experts, dim=-1)
        # Normalize
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        y = torch.zeros_like(x)
        counts = torch.bincount(experts_indices.flatten(), minlength=self.num_route_experts).tolist()
        
        for i in range(self.num_route_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top_expert = torch.where(experts_indices == i)
            # expert returns tuple(tensor, float), take tensor
            exp_out, _ = expert(x[idx])
            y[idx] += exp_out * expert_weights[idx, top_expert]
        
        output = (z + y).view(shape)
        
        # loss_load_balancing
        P = gate_probs.mean(dim=0)
        temp_counts = torch.bincount(experts_indices.flatten(), minlength=self.num_route_experts).float()
        D = temp_counts / (num_tokens * self.num_activated_experts)
        loss_load_balancing = self.w_load_loss * self.num_route_experts * (P * D).sum()
        
        # importance loss
        importance = gate_logits.sum(dim=0) 
        importance_mean = importance.mean()
        importance_std = torch.std(importance)
        cv = (importance_std / (importance_mean + 1e-6))
        importance_loss = (
            self.w_importance_loss * (cv ** 2)
        )

        aux_loss = self.w_aux_loss * (loss_load_balancing + importance_loss)

        return output, aux_loss


if ModelConfig.use_deepspeed_moe:
    from deepspeed.moe.layer import MoE

class DeepSpeedMoEBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None
    ):
        super().__init__()
        
        self.experts = nn.ModuleList([
            MLP(config.hidden_size, config.moe_intermediate_size) 
            for _ in range(config.num_route_experts)
        ])

        # DeepSpeed MoE Layer wrap
        # Nó sẽ tự động xử lý Gating (Top-K) và Routing (All-to-All communication)
        self.deepspeed_moe = MoE(
            hidden_size=config.hidden_size,
            expert=self.experts,
            num_experts=config.num_route_experts,
            ep_size=world_size, # ep_size = world_size (số GPU dùng cho Expert Parallel)
            k=config.num_activated_experts, # Top-k <= 2
            use_residual=False,
            min_capacity=0,
            noisy_gate_policy=None # Có thể chọn 'Jitter' hoặc 'RSample'
        )
        
        # Shared Expert (thường DeepSeek hay dùng 1 expert chạy trên tất cả token)
        # DeepSpeed MoE hiện tại tập trung vào routed experts. 
        # Shared expert nên để riêng bên ngoài lớp MoE này.
        self.shared_experts = MLP(
            config.hidden_size, 
            config.moe_intermediate_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, hidden]
        original_shape = x.shape
        x_flat = x.view(-1, original_shape[-1])
        
        # 1. Tính Shared Expert (luôn chạy trên local GPU cho mọi token)
        shared_output, _ = self.shared_experts(x_flat)
        
        # 2. Tính Routed Experts qua DeepSpeed
        # DeepSpeed MoE trả về: output, l_aux, exp_counts
        moe_output, aux_loss, _ = self.deepspeed_moe(x_flat)
        
        # 3. Cộng gộp
        final_output = shared_output + moe_output
        
        return final_output.view(original_shape), aux_loss

class EncoderBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, device=device)
        self.self_attention_block = AttentionBlock(config, causal_mask=False, device=device)
        self.ffn_norm = RMSNorm(config.hidden_size, device=device)
        self.feed_forward_block = MLP(config.hidden_size, config.intermediate_size) if layer_id < config.num_dense_layers else (
            DeepSpeedMoEBlock(config, device=device) if config.use_deepspeed_moe else (
                FSDPMoEBlock(config, device=device) if config.use_fsdp_moe else MOEBlock(config, device=device)
            )
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.self_attention_block(self.attn_norm(x), mask=mask)
        x_ffn, aux_loss = self.feed_forward_block(self.ffn_norm(x))
        x = x + x_ffn
        return x, aux_loss
    
class DecoderBlock(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, device=device)
        self.self_attention_block = AttentionBlock(config, causal_mask=True, device=device)
        self.cr_attn_norm = RMSNorm(config.hidden_size, device=device)
        self.cross_attention_block = AttentionBlock(config, causal_mask=False, device=device)
        self.ffn_norm = RMSNorm(config.hidden_size, device=device)
        self.feed_forward_block = MLP(config.hidden_size, config.intermediate_size) if layer_id < config.num_dense_layers else (
            DeepSpeedMoEBlock(config, device=device) if config.use_deepspeed_moe else (
                FSDPMoEBlock(config, device=device) if config.use_fsdp_moe else MOEBlock(config, device=device)
            )
        )

    def forward(self, x, encoder_output, tgt_mask: torch.Tensor | None = None, src_mask: torch.Tensor | None = None):
        x = x + self.self_attention_block(self.attn_norm(x), mask=tgt_mask)
        x = x + self.cross_attention_block(
            self.cr_attn_norm(x), encoder_output, mask=src_mask
        )
        x_ffn, aux_loss = self.feed_forward_block(self.ffn_norm(x))
        x = x + x_ffn
        return x, aux_loss

class Encoder(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        embedding: Optional[nn.Embedding] = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(
                config.vocab_size, config.hidden_size, device=device
            )
        if embedding is not None:
            self.embedding.weight = embedding.weight

        self.encoder_block = nn.ModuleList()
        for layer_id in range(config.num_hidden_layers):
            self.encoder_block.append(EncoderBlock(config, layer_id, device))

    def forward(self, x, mask: torch.Tensor | None = None):
        x = self.embedding(x)
        encoder_aux_loss = torch.tensor(0.0, device=x.device)
        for encoder in self.encoder_block:
            x, aux_loss = encoder(x, mask=mask)
            encoder_aux_loss += aux_loss
        return x, encoder_aux_loss

class Decoder(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        embedding: Optional[nn.Embedding] = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, device=device
        )
        if embedding is not None:
            self.embedding.weight = embedding.weight
            
        self.decoder_block = nn.ModuleList()
        for layer_id in range(config.num_hidden_layers):
            self.decoder_block.append(DecoderBlock(config, layer_id, device))

    def forward(self, x, encoder_output, tgt_mask: torch.Tensor | None = None, src_mask: torch.Tensor | None = None):
        x = self.embedding(x)
        decoder_aux_loss = torch.tensor(0.0, device=x.device)
        for decoder in self.decoder_block:
            x, aux_loss = decoder(x, encoder_output, tgt_mask=tgt_mask, src_mask=src_mask)
            decoder_aux_loss += aux_loss
        return x, decoder_aux_loss

class Transformer(nn.Module):

    _tied_weights_keys = [
        "lm_head.weight",
        "encoder.embedding.weight",
        "decoder.embedding.weight",
    ]

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()

        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        self.shared = self.embedding = nn.Embedding(
                config.vocab_size, config.hidden_size, device=device
            )
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device
        )
        self.encoder = Encoder(config, self.shared, device)
        self.decoder = Decoder(config, self.shared, device)

        self.tie_weights()

    def tie_weights(self):
        """
        Tie the weights of the head and the embeddings.
        """
        self.lm_head.weight = self.shared.weight
        if hasattr(self.encoder, "embedding"):
            self.encoder.embedding.weight = self.shared.weight
        if hasattr(self.decoder, "embedding"):
            self.decoder.embedding.weight = self.shared.weight
    
    def init_weights(self, config: ModelConfig):
        std_base = config.initializer_range
    
        num_layers = config.num_hidden_layers
        scaled_std = std_base / math.sqrt(2.0 * num_layers)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                use_std = std_base

                is_residual_output = False
                if "out" in name or "mlp2" in name:
                    is_residual_output = True
                
                if is_residual_output:
                    use_std = scaled_std
                
                if "gate" in name or "wg" in name:
                    use_std = config.gate_initializer_range

                nn.init.normal_(module.weight, mean=0.0, std=use_std)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std_base)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def forward(
        self,
        src_input_ids: torch.LongTensor,
        src_attention_mask: torch.LongTensor,
        tgt_input_ids: torch.LongTensor,
        tgt_attention_mask: torch.LongTensor,
        labels: torch.LongTensor,
    ) -> tuple[torch.Tensor, float, float]:
        
        # Prepare masks
        # src_attention_mask: (B, S). 1=valid, 0=pad
        # Encoder mask: (B, 1, 1, S) or broadcastable
        src_mask = (src_attention_mask == 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, S)
        
        # Decoder self-attention mask: padding only
        # AttentionBlock will handle adding causal mask if needed
        tgt_mask = (tgt_attention_mask == 0).unsqueeze(1).unsqueeze(2)

        encoder_output, encoder_aux_loss = self.encoder(src_input_ids, mask=src_mask)
        decoder_output, decoder_aux_loss = self.decoder(tgt_input_ids, encoder_output, tgt_mask=tgt_mask, src_mask=src_mask)
        logits = self.lm_head(self.norm(decoder_output))
        loss_lm = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return logits, loss_lm, encoder_aux_loss, decoder_aux_loss

def build_transformer(
    config: ModelConfig,
    device: torch.device | None = None,
) -> Transformer:

    transformer = Transformer(config, device)

    transformer.init_weights(config)
    
    return transformer
