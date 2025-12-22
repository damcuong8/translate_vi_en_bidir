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
import logging

logger = logging.getLogger(__name__)
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    num_encoder_layers: int = 12
    num_decoder_layers: int = 9
    shared_experts: int = 1
    num_dense_encoder_layers: int = 2
    num_dense_decoder_layers: int = 1
    num_route_experts: int = 16
    num_activated_experts: int = 2
    vocab_size: int = 24000
    hidden_size: int = 512
    intermediate_size: int = 1360
    moe_intermediate_size: int = 384
    head_dim: int = 64
    num_attention_heads: int = 8
    initial_context_length: int = 256
    rope_theta: float = 10000.0
    rope_scaling_factor: float = 1.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 1.0
    initializer_range: float = 0.02
    gate_initializer_range: float = 0.01
    w_load_loss: float = 0.01
    w_importance_loss: float = 0.01
    w_z_loss: float = 0.001
    w_aux_loss: float = 0.008
    use_deepspeed_moe: bool = False
    use_fsdp_moe: bool = True
    label_smoothing: float = 0.06
    attention_bias: bool = False
    use_flash_attn: bool = False
    use_sdpa_kernel: bool = False


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

    cos = cos.unsqueeze(2)  # (batch, seq_len, 1, head_dim // 2)
    sin = sin.unsqueeze(2)  # (batch, seq_len, 1, head_dim // 2)
    
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


try:
    from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    logger.warning("FlashAttention not found, using torch.nn.functional.scaled_dot_product_attention instead")

class AttentionBlock(nn.Module):
    def __init__(self, config, causal_mask: bool = True, device: torch.device | None = None):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.causal_mask = causal_mask

        # Linear layers
        self.w_q = nn.Linear(config.hidden_size, config.hidden_size, device=device, bias=config.attention_bias)
        self.w_k = nn.Linear(config.hidden_size, config.hidden_size, device=device, bias=config.attention_bias)
        self.w_v = nn.Linear(config.hidden_size, config.hidden_size, device=device, bias=config.attention_bias)
        self.w_o = nn.Linear(config.head_dim * config.num_attention_heads, config.hidden_size, device=device, bias=config.attention_bias)
        
        self.q_norm = RMSNorm(config.head_dim, device=device)
        self.k_norm = RMSNorm(config.head_dim, device=device)
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)
        
        self.rope = RotaryEmbedding(
            config.head_dim, config.rope_theta, torch.float32, config.initial_context_length,
            scaling_factor=config.rope_scaling_factor, ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta, device=device
        )

    def forward(self, x, encoder_output=None, query_mask=None, encoder_mask=None):
        batch_size, seq_len_q, _ = x.shape
        is_cross = encoder_output is not None
        device = x.device
        is_causal = self.causal_mask and not is_cross

        # Convert masks to boolean if provided (flash_attn expects bool, dataset provides long)
        # unpad_input requires boolean masks: True=valid token, False=padding
        if query_mask is not None:
            if query_mask.dtype != torch.bool:
                query_mask = query_mask.bool()
        if encoder_mask is not None:
            if encoder_mask.dtype != torch.bool:
                encoder_mask = encoder_mask.bool()

        # --- CASE 1: SELF-ATTENTION (QKVPACKED) ---
        if not is_cross:
            q = self.w_q(x).view(batch_size, seq_len_q, self.num_attention_heads, self.head_dim)
            k = self.w_k(x).view(batch_size, seq_len_q, self.num_attention_heads, self.head_dim)
            v = self.w_v(x).view(batch_size, seq_len_q, self.num_attention_heads, self.head_dim)

            # Apply RoPE & Norm
            position_ids = torch.arange(seq_len_q, device=device).unsqueeze(0).expand(batch_size, -1)
            q, k = self.rope(q, k, position_ids)
            q, k = self.q_norm(q), self.k_norm(k)

            # Stack into QKV packed: [B, S, 3, H, D]
            qkv = torch.stack([q, k, v], dim=2)

            if getattr(self.config, "use_flash_attn", True):
                # Unpad entire QKV block
                qkv_unpad, indices, cu_seqlens, max_s, _ = unpad_input(qkv, query_mask)
                
                output_unpad = flash_attn_varlen_qkvpacked_func(
                    qkv_unpad, cu_seqlens, max_s,
                    dropout_p=0.0, softmax_scale=self.softmax_scale, causal=is_causal
                )
                output = pad_input(output_unpad, indices, batch_size, seq_len_q)
            else:
                # Fallback to SDPA (cần tách Q, K, V)
                output = self._sdpa_fallback(q, k, v, query_mask, is_causal)

        # --- CASE 2: CROSS-ATTENTION (KVPACKED) ---
        else:
            q = self.w_q(x).view(batch_size, seq_len_q, self.num_attention_heads, self.head_dim)
            # K, V from encoder_output
            k = self.w_k(encoder_output).view(batch_size, -1, self.num_attention_heads, self.head_dim)
            v = self.w_v(encoder_output).view(batch_size, -1, self.num_attention_heads, self.head_dim)
            seq_len_k = k.shape[1]

            # Cross-attn: RoPE only for Q, Norm for both Q and K
            position_ids = torch.arange(seq_len_q, device=device).unsqueeze(0).expand(batch_size, -1)
            q, _ = self.rope(q, q, position_ids)
            q, k = self.q_norm(q), self.k_norm(k)

            # Stack into KV packed: [B, S_k, 2, H, D]
            kv = torch.stack([k, v], dim=2)

            if getattr(self.config, "use_flash_attn", True):
                # Unpad Q and KV separately because lengths are different
                q_unpad, indices_q, cu_seqlens_q, max_s_q, _ = unpad_input(q, query_mask)
                kv_unpad, _, cu_seqlens_k, max_s_k, _ = unpad_input(kv, encoder_mask)
                
                output_unpad = flash_attn_varlen_kvpacked_func(
                    q_unpad, kv_unpad, 
                    cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_s_q, max_seqlen_k=max_s_k,
                    dropout_p=0.0, softmax_scale=self.softmax_scale, causal=False
                )
                output = pad_input(output_unpad, indices_q, batch_size, seq_len_q)
            else:
                output = self._sdpa_fallback(q, k, v, encoder_mask, False)

        # 5. Output Projection
        output = output.reshape(batch_size, seq_len_q, -1)
        return self.w_o(output)

    def _sdpa_fallback(self, q, k, v, mask, is_causal):
        """
        Fallback to SDPA when flash_attn is not available.
        mask: boolean tensor of shape [B, S] or [B, 1, 1, S], True=valid token, False=padding
        SDPA expects: attn_mask with True=valid, False=padding (same convention)
        """
        # Convert layout to [B, H, S, D] for SDPA
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn_mask = mask.unsqueeze(1).unsqueeze(2) if mask is not None else None
        if self.config.use_sdpa_kernel:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=self.softmax_scale
                )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=self.softmax_scale
            )
        return out.transpose(1, 2)


def _inv_softplus(x: float) -> torch.Tensor:
    # Inverse of softplus for scalar initialization (numerically stable enough for our init values)
    return torch.log(torch.expm1(torch.tensor(x, dtype=torch.get_default_dtype())))


class SwiGLU5(nn.Module):
    """
    SwiGLU variant with 5 trainable scalars:
      - alpha: multiplier inside the sigmoid (init 1.702)
      - gate_scale: multiply the gate output (init 1.0)
      - up_shift: shift added to the "linear/up" path (init 1.0) -- replaces the constant +1
      - gate_clamp (positive via softplus): clamp limit for gate path (init 7.0)
      - up_clamp (positive via softplus): clamp limit for linear/up path (init 10.0)

    The clamps are parameterized via inverse-softplus to keep them strictly > 0.
    """

    def __init__(
        self,
        init_alpha: float = 1.702,
        init_gate_scale: float = 1.0,
        init_up_shift: float = 1.0,
        init_gate_clamp: float = 7.0,
        init_up_clamp: float = 7.0,
    ) -> None:
        super().__init__()
        # trainable scalars (must be 1D tensors for FSDP compatibility)
        self.alpha = nn.Parameter(torch.tensor([init_alpha]))
        self.gate_scale = nn.Parameter(torch.tensor([init_gate_scale]))
        self.up_shift = nn.Parameter(torch.tensor([init_up_shift]))
        # clamp parameters are stored as "raw" values and passed through softplus to ensure positivity
        # Convert to 1D tensor for FSDP compatibility
        gate_clamp_val = _inv_softplus(init_gate_clamp)
        up_clamp_val = _inv_softplus(init_up_clamp)
        self._gate_clamp_raw = nn.Parameter(gate_clamp_val.unsqueeze(0) if gate_clamp_val.dim() == 0 else gate_clamp_val)
        self._up_clamp_raw = nn.Parameter(up_clamp_val.unsqueeze(0) if up_clamp_val.dim() == 0 else up_clamp_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ensure parameters are same dtype/device as input
        dtype = x.dtype
        device = x.device
        alpha = self.alpha.to(dtype).to(device)
        gate_scale = self.gate_scale.to(dtype).to(device)
        up_shift = self.up_shift.to(dtype).to(device)
        gate_clamp = F.softplus(self._gate_clamp_raw).to(dtype).to(device)
        up_clamp = F.softplus(self._up_clamp_raw).to(dtype).to(device)

        # split on the last dim (expects even-sized final dim)
        x_glu, x_linear = x[..., ::2], x[..., 1::2]
        # clamp each path with their trainable clamp limits
        x_glu = torch.clamp(x_glu, min=-gate_clamp, max=gate_clamp)
        x_linear = torch.clamp(x_linear, min=-up_clamp, max=up_clamp)

        # gated nonlinearity
        out_glu = x_glu * torch.sigmoid(x_glu * alpha) * gate_scale
        # up_shift replaces the constant +1 and is trainable
        return out_glu * (x_linear + up_shift)


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
        # SwiGLU with 5 trainable scalars
        self.activation = SwiGLU5()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        assert x.shape[-1] == self.hidden_size, f"Expected hidden size {self.hidden_size}, got {x.shape[-1]}"
        t = self.activation(self.mlp1(x))
        t = self.mlp2(t)
        return t, 0.0

class MOEBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        is_encoder: bool,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.config = config
        self.layer_id = layer_id
        self.is_encoder = is_encoder

        assert config.num_route_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.num_route_experts = config.num_route_experts
        self.num_local_experts = config.num_route_experts // world_size
        self.num_activated_experts = config.num_activated_experts
        self.experts_start_idx = rank * self.num_local_experts
        self.experts_end_idx = self.experts_start_idx + self.num_local_experts
        self.w_load_loss = config.w_load_loss
        self.w_importance_loss = config.w_importance_loss
        self.w_z_loss = config.w_z_loss
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
        # Accumulate statistics across micro-batches for this block
        self._accumulated_stats = []
        # Reduce logging frequency for wandb (only log every N effective batches)
        self._log_step = 0
        self.log_interval = 50  # can be tuned if needed

    def get_statistics(self, gate_logits: torch.Tensor, gate_probs: torch.Tensor, experts_indices: torch.Tensor, num_tokens: int, valid_mask: torch.Tensor | None = None):
        """Extract statistics for auxiliary loss computation.
        
        Args:
            gate_logits: (num_tokens, num_experts)
            gate_probs: (num_tokens, num_experts)
            experts_indices: (num_tokens, k)
            num_tokens: total number of tokens (including pad)
            valid_mask: (num_tokens,) boolean mask, True for valid tokens, False for pad tokens
        """
        if valid_mask is not None:
            # Only keep valid tokens (exclude pad tokens)
            gate_logits = gate_logits[valid_mask]
            gate_probs = gate_probs[valid_mask]
            experts_indices = experts_indices[valid_mask]
            # Use shape[0] instead of sum().item() to avoid graph breaks with torch.compile
            num_tokens = gate_logits.shape[0]
        
        return {
            'gate_logits': gate_logits,  # (num_valid_tokens, num_experts)
            'gate_probs': gate_probs,  # (num_valid_tokens, num_experts)
            'experts_indices': experts_indices,  # (num_valid_tokens, k)
            'num_tokens': num_tokens
        }
    
    def compute_aux_loss_from_statistics(self, accumulated_stats: list) -> torch.Tensor:
        """Compute auxiliary loss from accumulated statistics across effective batch."""
        if not accumulated_stats:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Concatenate all statistics
        all_gate_probs = torch.cat([stats['gate_probs'] for stats in accumulated_stats], dim=0)
        all_experts_indices = torch.cat([stats['experts_indices'] for stats in accumulated_stats], dim=0)
        all_gate_logits = torch.cat([stats['gate_logits'] for stats in accumulated_stats], dim=0)
        # Use shape[0] instead of sum() to keep as tensor for torch.compile optimization
        # num_tokens is already an int from shape[0], so sum is fine, but we convert to tensor later if needed
        total_num_tokens = sum(stats['num_tokens'] for stats in accumulated_stats)

        # Token counts per expert across effective batch
        token_counts = torch.bincount(
            all_experts_indices.flatten(),
            minlength=self.num_route_experts,
        )
        
        # loss_load_balancing
        P = all_gate_probs.mean(dim=0)  # Mean over all tokens in effective batch
        temp_counts = torch.bincount(all_experts_indices.flatten(), minlength=self.num_route_experts).float()
        D = temp_counts / (total_num_tokens * self.num_activated_experts)
        loss_load_balancing = self.w_load_loss * self.num_route_experts * (P * D).sum()
        
        # importance loss
        importance = all_gate_logits.sum(dim=0)  # Sum over all tokens in effective batch
        importance_mean = importance.mean()
        importance_std = torch.std(importance)
        cv = (importance_std / (importance_mean + 1e-6))
        importance_loss = self.w_importance_loss * (cv ** 2)
        
        # z-loss
        z_loss = torch.logsumexp(all_gate_logits, dim=-1).pow(2).mean()
        weighted_z_loss = self.w_z_loss * z_loss
        
        aux_loss = self.w_aux_loss * (loss_load_balancing + importance_loss + weighted_z_loss)
        
        # Log token distribution for last encoder/decoder layer as 16 scalar series
        is_last_encoder = self.is_encoder and (self.layer_id == self.config.num_encoder_layers - 1)
        is_last_decoder = (not self.is_encoder) and (self.layer_id == self.config.num_decoder_layers - 1)
        if is_last_encoder or is_last_decoder:
            # Log only every `log_interval` effective batches to avoid spam
            self._log_step += 1
            if self._log_step % self.log_interval != 0:
                return aux_loss

            tag = "encoder_last" if self.is_encoder else "decoder_last"
            try:
                import wandb
                if wandb.run is not None:
                    metrics = {
                        f"moe/{tag}/num_tokens": int(total_num_tokens),
                    }
                    for i in range(self.num_route_experts):
                        metrics[f"moe/{tag}/expert_{i}_tokens"] = int(token_counts[i])
                    wandb.log(metrics, commit=False)
            except ImportError:
                pass

        return aux_loss

    def reset_accumulated_stats(self) -> None:
        """Reset accumulated statistics (called after using is_final=True)."""
        self._accumulated_stats = []

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        is_final: bool = False,
        return_stats: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | dict]:
        
        shape = x.size()
        x = x.view(-1, shape[-1])
        num_tokens, num_features = x.shape

        # Prepare valid mask for excluding pad tokens
        valid_mask = None
        if mask is not None:
            # mask shape: (B, 1, 1, S) or broadcastable, True=valid, False=pad
            # Flatten to (B*S,)
            valid_mask = (~mask).squeeze(1).squeeze(1).view(-1)  # ~mask because mask is True for valid

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
            y[idx] += exp_out * expert_weights[idx, top_expert].unsqueeze(-1)
        
        if world_size > 1:
            dist.all_reduce(y)
        output = (z + y).view(shape)
        
        # Always build statistics (excluding pad tokens via valid_mask if provided)
        stats = self.get_statistics(gate_logits, gate_probs, experts_indices, num_tokens, valid_mask)

        if return_stats:
            # Pure inspection/debug mode: just return stats for this batch
            return output, stats

        # If this is not the final micro-batch in effective batch, accumulate stats and return
        if not is_final:
            # Detach stats before accumulating to break gradient graph and prevent "backward twice" errors
            detached_stats = {
                'gate_logits': stats['gate_logits'].detach(),
                'gate_probs': stats['gate_probs'].detach(),
                'experts_indices': stats['experts_indices'].detach(),
                'num_tokens': stats['num_tokens']
            }
            self._accumulated_stats.append(detached_stats)
            aux_loss = torch.tensor(0.0, device=x.device)
            return output, aux_loss

        # Final micro-batch: compute aux loss over all accumulated stats + current stats
        # Use current stats (with gradient) + accumulated stats (detached) for accurate computation
        all_stats = self._accumulated_stats + [stats]
        aux_loss = self.compute_aux_loss_from_statistics(all_stats)
        self.reset_accumulated_stats()
        return output, aux_loss


class FSDPMoEBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        is_encoder: bool,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.is_encoder = is_encoder
        self.num_route_experts = config.num_route_experts
        self.num_activated_experts = config.num_activated_experts
        self.w_load_loss = config.w_load_loss
        self.w_importance_loss = config.w_importance_loss
        self.w_z_loss = config.w_z_loss
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
        # Reduce logging frequency for wandb (only log every N effective batches)
        self._log_step = 0
        self.log_interval = 50  # can be tuned if needed
        # Accumulate statistics across micro-batches for this block
        self._accumulated_stats = []

    def get_statistics(self, gate_logits: torch.Tensor, gate_probs: torch.Tensor, experts_indices: torch.Tensor, num_tokens: int, valid_mask: torch.Tensor | None = None):
        """Extract statistics for auxiliary loss computation.
        
        Args:
            gate_logits: (num_tokens, num_experts)
            gate_probs: (num_tokens, num_experts)
            experts_indices: (num_tokens, k)
            num_tokens: total number of tokens (including pad)
            valid_mask: (num_tokens,) boolean mask, True for valid tokens, False for pad tokens
        """
        if valid_mask is not None:
            # Only keep valid tokens (exclude pad tokens)
            gate_logits = gate_logits[valid_mask]
            gate_probs = gate_probs[valid_mask]
            experts_indices = experts_indices[valid_mask]
            # Use shape[0] instead of sum().item() to avoid graph breaks with torch.compile
            num_tokens = gate_logits.shape[0]
        
        return {
            'gate_logits': gate_logits,  # (num_valid_tokens, num_experts)
            'gate_probs': gate_probs,  # (num_valid_tokens, num_experts)
            'experts_indices': experts_indices,  # (num_valid_tokens, k)
            'num_tokens': num_tokens
        }
    
    def compute_aux_loss_from_statistics(self, accumulated_stats: list) -> torch.Tensor:
        """Compute auxiliary loss from accumulated statistics across effective batch."""
        if not accumulated_stats:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Concatenate all statistics
        all_gate_probs = torch.cat([stats['gate_probs'] for stats in accumulated_stats], dim=0)
        all_experts_indices = torch.cat([stats['experts_indices'] for stats in accumulated_stats], dim=0)
        all_gate_logits = torch.cat([stats['gate_logits'] for stats in accumulated_stats], dim=0)
        # Use shape[0] instead of sum() to keep as tensor for torch.compile optimization
        # num_tokens is already an int from shape[0], so sum is fine, but we convert to tensor later if needed
        total_num_tokens = sum(stats['num_tokens'] for stats in accumulated_stats)

        # Token counts per expert across effective batch
        token_counts = torch.bincount(
            all_experts_indices.flatten(),
            minlength=self.num_route_experts,
        )
        
        # loss_load_balancing
        P = all_gate_probs.mean(dim=0)  # Mean over all tokens in effective batch
        temp_counts = torch.bincount(all_experts_indices.flatten(), minlength=self.num_route_experts).float()
        D = temp_counts / (total_num_tokens * self.num_activated_experts)
        loss_load_balancing = self.w_load_loss * self.num_route_experts * (P * D).sum()
        
        # importance loss
        importance = all_gate_logits.sum(dim=0)  # Sum over all tokens in effective batch
        importance_mean = importance.mean()
        importance_std = torch.std(importance)
        cv = (importance_std / (importance_mean + 1e-6))
        importance_loss = self.w_importance_loss * (cv ** 2)
        
        # z-loss
        z_loss = torch.logsumexp(all_gate_logits, dim=-1).pow(2).mean()
        weighted_z_loss = self.w_z_loss * z_loss
        
        aux_loss = self.w_aux_loss * (loss_load_balancing + importance_loss + weighted_z_loss)
        
        # Log token distribution for last encoder/decoder layer as 16 scalar series
        is_last_encoder = self.is_encoder and (self.layer_id == self.config.num_encoder_layers - 1)
        is_last_decoder = (not self.is_encoder) and (self.layer_id == self.config.num_decoder_layers - 1)
        if is_last_encoder or is_last_decoder:
            # Log only every `log_interval` effective batches to avoid spam
            self._log_step += 1
            if self._log_step % self.log_interval != 0:
                return aux_loss

            tag = "encoder_last" if self.is_encoder else "decoder_last"
            try:
                import wandb
                if wandb.run is not None:
                    metrics = {
                        f"fsdp_moe/{tag}/num_tokens": int(total_num_tokens),
                    }
                    for i in range(self.num_route_experts):
                        metrics[f"fsdp_moe/{tag}/expert_{i}_tokens"] = int(token_counts[i])
                    wandb.log(metrics, commit=False)
            except ImportError:
                pass

        return aux_loss

    def reset_accumulated_stats(self) -> None:
        """Reset accumulated statistics (called after using is_final=True)."""
        self._accumulated_stats = []

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        is_final: bool = False,
        return_stats: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | dict]:
        
        shape = x.size()
        x = x.view(-1, shape[-1])
        num_tokens, num_features = x.shape

        # Prepare valid mask for excluding pad tokens
        valid_mask = None
        if mask is not None:
            # mask shape: (B, S) or broadcastable, True=valid, False=pad or 1=valid, 0=pad
            mask = mask.bool() if mask.dtype != torch.bool else mask
            # Flatten to (B*S,) and invert: True=valid, False=pad
            valid_mask = (mask).view(-1)

        z, _ = self.shared_experts(x)

        gate_logits = self.gate(x)

        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_weights, experts_indices = torch.topk(gate_probs, self.num_activated_experts, dim=-1)
        # Normalize
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        # Replace bincount with fixed-size scatter_add (avoid .tolist())
        flat_idx = experts_indices.flatten().long()
        device = flat_idx.device if flat_idx.numel() > 0 else x.device
        counts = torch.zeros(self.num_route_experts, device=device, dtype=torch.long)
        if flat_idx.numel() > 0:
            ones = torch.ones_like(flat_idx, dtype=torch.long)
            counts = counts.scatter_add(0, flat_idx, ones)

        y = torch.zeros_like(x)

        # Note: avoid branching that converts tensors to Python scalars inside graph.
        # Calling expert on empty index returns empty tensors — that's okay.
        for i in range(self.num_route_experts):
            # find which tokens select expert i among the k choices
            idx, top_expert = torch.where(experts_indices == i)
            if idx.numel() == 0:
                # skip cheap if no token selects this expert (still a tensor check)
                continue
            expert = self.experts[i]
            # expert returns tuple(tensor, float), take tensor
            exp_out, _ = expert(x[idx])
            y[idx] += exp_out * expert_weights[idx, top_expert].unsqueeze(-1)
        
        output = (z + y).view(shape)
        
        # Always build statistics (excluding pad tokens via valid_mask if provided)
        stats = self.get_statistics(gate_logits, gate_probs, experts_indices, num_tokens, valid_mask)

        if return_stats:
            # Pure inspection/debug mode: just return stats for this batch
            return output, stats

        # If this is not the final micro-batch in effective batch, accumulate stats and return
        if not is_final:
            # Detach stats before accumulating to break gradient graph and prevent "backward twice" errors
            detached_stats = {
                'gate_logits': stats['gate_logits'].detach(),
                'gate_probs': stats['gate_probs'].detach(),
                'experts_indices': stats['experts_indices'].detach(),
                'num_tokens': stats['num_tokens']
            }
            self._accumulated_stats.append(detached_stats)
            aux_loss = torch.tensor(0.0, device=x.device)
            return output, aux_loss

        # Final micro-batch: compute aux loss over all accumulated stats + current stats
        # Use current stats (with gradient) + accumulated stats (detached) for accurate computation
        all_stats = self._accumulated_stats + [stats]
        aux_loss = self.compute_aux_loss_from_statistics(all_stats)
        self.reset_accumulated_stats()
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
        self.feed_forward_block = MLP(config.hidden_size, config.intermediate_size) if layer_id < config.num_dense_encoder_layers else (
            DeepSpeedMoEBlock(config, device=device) if config.use_deepspeed_moe else (
                FSDPMoEBlock(config, layer_id, True, device=device) if config.use_fsdp_moe else MOEBlock(config, layer_id, True, device=device)
            )
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, is_final: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.self_attention_block(self.attn_norm(x), query_mask=mask)
        
        x_norm = self.ffn_norm(x)
        
        # Forward through feed-forward block.
        # For MoE blocks, accumulation and aux_loss are handled internally.
        # For plain MLP, aux_loss will be 0.0.
        if isinstance(self.feed_forward_block, (MOEBlock, FSDPMoEBlock)):
            x_ffn, aux_loss = self.feed_forward_block(x_norm, mask=mask, is_final=is_final)
        else:
            x_ffn, aux_loss = self.feed_forward_block(x_norm)
            if not isinstance(aux_loss, torch.Tensor):
                aux_loss = torch.tensor(0.0, device=x.device)
        
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
        self.feed_forward_block = MLP(config.hidden_size, config.intermediate_size) if layer_id < config.num_dense_decoder_layers else (
            DeepSpeedMoEBlock(config, device=device) if config.use_deepspeed_moe else (
                FSDPMoEBlock(config, layer_id, False, device=device) if config.use_fsdp_moe else MOEBlock(config, layer_id, False, device=device)
            )
        )

    def forward(self, x, encoder_output, tgt_mask: torch.Tensor | None = None, src_mask: torch.Tensor | None = None, is_final: bool = False):
        x = x + self.self_attention_block(self.attn_norm(x), query_mask=tgt_mask)
        x = x + self.cross_attention_block(
            self.cr_attn_norm(x), encoder_output, query_mask=tgt_mask, encoder_mask=src_mask
        )
        
        x_norm = self.ffn_norm(x)
        
        # Forward through feed-forward block.
        # For MoE blocks, accumulation and aux_loss are handled internally.
        # For plain MLP, aux_loss will be 0.0.
        if isinstance(self.feed_forward_block, (MOEBlock, FSDPMoEBlock)):
            x_ffn, aux_loss = self.feed_forward_block(x_norm, mask=tgt_mask, is_final=is_final)
        else:
            x_ffn, aux_loss = self.feed_forward_block(x_norm)
            if not isinstance(aux_loss, torch.Tensor):
                aux_loss = torch.tensor(0.0, device=x.device)
        
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
        
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(
                config.vocab_size, config.hidden_size, device=device
            )

        self.encoder_block = nn.ModuleList()
        for layer_id in range(config.num_encoder_layers):
            self.encoder_block.append(EncoderBlock(config, layer_id, device))

    def forward(self, x, mask: torch.Tensor | None = None, is_final: bool = False):
        x = self.embedding(x)
        encoder_aux_loss = torch.tensor(0.0, device=x.device)
        for encoder in self.encoder_block:
            x, aux_loss = encoder(x, mask=mask, is_final=is_final)
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
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(
                config.vocab_size, config.hidden_size, device=device
            )
            
        self.decoder_block = nn.ModuleList()
        for layer_id in range(config.num_decoder_layers):
            self.decoder_block.append(DecoderBlock(config, layer_id, device))

    def forward(self, x, encoder_output, tgt_mask: torch.Tensor | None = None, src_mask: torch.Tensor | None = None, is_final: bool = False):
        x = self.embedding(x)
        decoder_aux_loss = torch.tensor(0.0, device=x.device)
        for decoder in self.decoder_block:
            x, aux_loss = decoder(x, encoder_output, tgt_mask=tgt_mask, src_mask=src_mask, is_final=is_final)
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
        self.config = config

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
    
        # Tính scaled_std riêng cho encoder và decoder
        num_encoder_layers = config.num_encoder_layers
        num_decoder_layers = config.num_decoder_layers
        scaled_std_encoder = std_base / math.sqrt(2.0 * num_encoder_layers)
        scaled_std_decoder = std_base / math.sqrt(2.0 * num_decoder_layers)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                use_std = std_base

                is_residual_output = False
                if "out" in name or "mlp2" in name:
                    is_residual_output = True
                
                if is_residual_output:
                    # Xác định module thuộc encoder hay decoder
                    if "encoder" in name:
                        use_std = scaled_std_encoder
                    elif "decoder" in name:
                        use_std = scaled_std_decoder
                    else:
                        # Nếu không rõ, dùng giá trị trung bình hoặc encoder (conservative)
                        use_std = scaled_std_encoder
                
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
        is_final: bool = False,
    ) -> tuple[torch.Tensor, float, float]:
        
        encoder_output, encoder_aux_loss = self.encoder(src_input_ids, mask=src_attention_mask, is_final=is_final)
        decoder_output, decoder_aux_loss = self.decoder(tgt_input_ids, encoder_output, tgt_mask=tgt_attention_mask, src_mask=src_attention_mask, is_final=is_final)
        logits = self.lm_head(self.norm(decoder_output))

        loss_lm = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            labels.view(-1),
            label_smoothing=self.config.label_smoothing,
            ignore_index=-100
        )
        
        return logits, loss_lm, encoder_aux_loss, decoder_aux_loss
    
    @torch.no_grad()
    def generate(
        self,
        src_input_ids: torch.LongTensor,
        src_attention_mask: torch.LongTensor,
        tgt_start_ids: torch.LongTensor,
        max_len: int = 256,
        eos_token_id: int = 2,
        num_beams: int = 1,
        length_penalty: float = 1.0,
    ) -> torch.LongTensor:
        """
        Generation with Greedy or Beam Search support.
        
        Args:
            src_input_ids: Source input IDs (B, S)
            src_attention_mask: Source attention mask (B, S), 1=valid, 0=pad
            tgt_start_ids: Starting tokens for target (B, T), e.g., [BOS, lang_token]
            max_len: Maximum generation length
            eos_token_id: EOS token ID to stop generation
            num_beams: Number of beams for beam search. 1 means greedy decoding.
            length_penalty: Exponential penalty to the length. 1.0 means no penalty.
                            Values > 1.0 encourage longer sequences.
                            Values < 1.0 encourage shorter sequences.
            
        Returns:
            Generated sequences (B, T') where T' <= max_len
        """
        if num_beams == 1:
            return self._generate_greedy(
                src_input_ids, src_attention_mask, tgt_start_ids, max_len, eos_token_id
            )
        else:
            return self._generate_beam_search(
                src_input_ids, src_attention_mask, tgt_start_ids, max_len, eos_token_id, 
                num_beams, length_penalty
            )

    def _generate_greedy(
        self,
        src_input_ids: torch.LongTensor,
        src_attention_mask: torch.LongTensor,
        tgt_start_ids: torch.LongTensor,
        max_len: int = 256,
        eos_token_id: int = 2,
    ) -> torch.LongTensor:
        """Greedy generation implementation."""
        batch_size = src_input_ids.size(0)
        device = src_input_ids.device
        
        # Prepare source mask
        src_mask = (src_attention_mask == 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        
        # Encode source once
        encoder_output, _ = self.encoder(src_input_ids, mask=src_mask)
        
        # Initialize decoder input with start tokens
        decoder_input = tgt_start_ids  # (B, T_start)
        
        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Autoregressive generation
        for _ in range(max_len):
            # Decode current sequence
            decoder_output, _ = self.decoder(
                decoder_input, 
                encoder_output, 
                tgt_mask=None,  # Causal mask applied internally
                src_mask=src_mask
            )
            
            # Project last token to vocabulary
            logits = self.lm_head(self.norm(decoder_output[:, -1]))  # (B, V)
            next_tokens = torch.argmax(logits, dim=-1)  # (B,)
            
            # Mark finished sequences
            finished |= (next_tokens == eos_token_id)
            
            # Append next tokens
            decoder_input = torch.cat([decoder_input, next_tokens.unsqueeze(1)], dim=1)
            
            # Stop if all sequences finished
            if finished.all():
                break
        
        return decoder_input

    def _generate_beam_search(
        self,
        src_input_ids: torch.LongTensor,
        src_attention_mask: torch.LongTensor,
        tgt_start_ids: torch.LongTensor,
        max_len: int = 256,
        eos_token_id: int = 2,
        num_beams: int = 5,
        length_penalty: float = 1.0,
    ) -> torch.LongTensor:
        """Beam search generation implementation."""
        batch_size = src_input_ids.size(0)
        device = src_input_ids.device
        
        # 1. Expand inputs for beam search: (B, ...) -> (B * K, ...)
        src_input_ids = src_input_ids.repeat_interleave(num_beams, dim=0)
        src_mask = (src_attention_mask == 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        src_mask = src_mask.repeat_interleave(num_beams, dim=0)  # (B*K, 1, 1, S)
        
        # Encode source once (with expanded batch)
        encoder_output, _ = self.encoder(src_input_ids, mask=src_mask)
        
        # Initialize decoder input: (B, T) -> (B*K, T)
        decoder_input = tgt_start_ids.repeat_interleave(num_beams, dim=0)
        
        # 2. Initialize scores
        # beam_scores: (B, K). Beam 0 gets 0, others -inf to force picking beam 0 initially.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # (B * K)
        
        # Store finished sequences: list of lists of (score, tensor)
        finished_sequences = [[] for _ in range(batch_size)]

        # Track finished beams (shape: B*K). Finished beams should not be decoded further.
        beam_finished = torch.zeros((batch_size * num_beams,), dtype=torch.bool, device=device)

        # Track which batch elements are fully done (found enough finished beams)
        batch_done = torch.zeros((batch_size,), dtype=torch.bool, device=device)

        # Small helper for stable -inf
        neg_inf = torch.finfo(beam_scores.dtype).min

        # Determine effective max length based on model's context limit
        max_context_len = getattr(self.config, "initial_context_length", 256)
        current_len = decoder_input.size(1)
        # We can generate at most (max_context_len - current_len) tokens to fit in RoPE cache
        generation_steps = min(max_len, max_context_len - current_len)

        for step in range(generation_steps):
            # Early stop when every batch already has enough finished sequences.
            if batch_done.all():
                break

            vocab_size = getattr(self.lm_head, "out_features", None)
            if vocab_size is None:
                vocab_size = self.lm_head.weight.size(0)

            # Compute next-token logprobs only for active beams (not finished, not in done batches).
            # For finished beams we will later force EOS-only probability mass.
            active_mask = ~beam_finished
            if batch_done.any():
                # Freeze every beam belonging to a "done" batch so we avoid extra decoding for it.
                # (Done batches already have >= num_beams finished sequences stored.)
                done_beam_mask = batch_done.repeat_interleave(num_beams)
                active_mask = active_mask & ~done_beam_mask

            next_token_logprobs = torch.full(
                (batch_size * num_beams, vocab_size),
                fill_value=neg_inf,
                device=device,
                dtype=torch.float,
            )

            if active_mask.any():
                decoder_output_active, _ = self.decoder(
                    decoder_input[active_mask],
                    encoder_output[active_mask],
                    tgt_mask=None,
                    src_mask=src_mask[active_mask],
                )

                logits_active = self.lm_head(self.norm(decoder_output_active[:, -1]))  # (A, V)
                next_token_logprobs_active = F.log_softmax(logits_active, dim=-1)      # (A, V)
                next_token_logprobs[active_mask] = next_token_logprobs_active

            # Force finished beams (and beams in done batches) to only be able to emit EOS,
            # preventing them from being "expanded" into non-EOS tokens.
            freeze_mask = beam_finished
            if batch_done.any():
                freeze_mask = freeze_mask | batch_done.repeat_interleave(num_beams)
            if freeze_mask.any():
                next_token_logprobs[freeze_mask, :] = neg_inf
                next_token_logprobs[freeze_mask, eos_token_id] = 0.0
            
            # Calculate next scores
            # (B*K, V) = (B*K, 1) + (B*K, V)
            curr_scores = beam_scores.unsqueeze(1) + next_token_logprobs
            
            # Reshape to select top K per batch
            # (B, K * V)
            curr_scores = curr_scores.view(batch_size, num_beams * vocab_size)
            
            # Top K selection
            # topk_scores: (B, K)
            # topk_indices: (B, K) -> index in range [0, K*V - 1]
            topk_scores, topk_indices = torch.topk(curr_scores, num_beams, dim=1)
            
            # Convert linear indices back to (beam_idx, token_idx)
            # beam_indices: which beam in the previous step (0..K-1)
            beam_indices = topk_indices // vocab_size
            # token_indices: which token to add (0..V-1)
            token_indices = topk_indices % vocab_size
            
            # Calculate global beam indices for gathering
            # batch_base: [0, K, 2K, ...]
            batch_base = torch.arange(batch_size, device=device).unsqueeze(1) * num_beams
            global_beam_indices = batch_base + beam_indices # (B, K)
            
            # Flatten for next iteration
            global_beam_indices = global_beam_indices.view(-1)
            token_indices = token_indices.view(-1)
            beam_scores = topk_scores.view(-1)
            
            # 3. Update sequences
            # Gather previous sequences
            decoder_input = decoder_input[global_beam_indices]
            beam_finished = beam_finished[global_beam_indices]
            # Keep encoder-side tensors aligned with the current beam ordering.
            # This is important when encoder outputs are not identical across beams (e.g., dropout/train mode).
            encoder_output = encoder_output[global_beam_indices]
            src_mask = src_mask[global_beam_indices]
            # Append new tokens
            decoder_input = torch.cat([decoder_input, token_indices.unsqueeze(1)], dim=1)
            
            # 4. Check for EOS
            # Identify which beams just generated EOS
            is_eos = (token_indices == eos_token_id)

            # Only treat EOS as "newly finished" if this beam was not finished already.
            newly_finished = is_eos & (~beam_finished)
            if newly_finished.any():
                eos_indices = torch.nonzero(newly_finished).squeeze(1)
                for idx in eos_indices.tolist():
                    batch_idx = idx // num_beams
                    if len(finished_sequences[batch_idx]) >= num_beams:
                        beam_finished[idx] = True
                        continue

                    score = beam_scores[idx].item()
                    seq = decoder_input[idx]

                    # Apply length penalty: score = log_prob / (length ** alpha)
                    gen_len = seq.size(0) - tgt_start_ids.size(1)
                    penalty = (max(1, gen_len) ** length_penalty)
                    final_score = score / penalty

                    finished_sequences[batch_idx].append((final_score, seq))
                    beam_finished[idx] = True

            # Update batch_done after recording new finished beams
            for b in range(batch_size):
                if not batch_done[b] and (len(finished_sequences[b]) >= num_beams):
                    batch_done[b] = True
        
        # 5. Select best sequences
        final_outputs = []
        for i in range(batch_size):
            # If we have finished sequences, pick the best one
            if finished_sequences[i]:
                finished_sequences[i].sort(key=lambda x: x[0], reverse=True)
                best_seq = finished_sequences[i][0][1]
            else:
                # If no EOS found, pick the current best beam by score (not necessarily beam 0).
                scores_i = beam_scores.view(batch_size, num_beams)[i]  # (K,)
                best_beam = torch.argmax(scores_i).item()
                best_idx = i * num_beams + best_beam
                best_seq = decoder_input[best_idx]
            final_outputs.append(best_seq)
            
        # Pad sequences to form a batch tensor
        max_out_len = max(len(s) for s in final_outputs)
        padded_tensors = []
        for seq in final_outputs:
            if len(seq) < max_out_len:
                pad_size = max_out_len - len(seq)
                pad = torch.full((pad_size,), eos_token_id, device=device, dtype=torch.long)
                seq = torch.cat([seq, pad])
            padded_tensors.append(seq)
            
        return torch.stack(padded_tensors)

def build_transformer(
    config: ModelConfig,
    device: torch.device | None = None,
) -> Transformer:

    transformer = Transformer(config, device)

    transformer.init_weights(config)
    
    return transformer
