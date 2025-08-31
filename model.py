import torch
import torch.nn as nn
import math
from torch.profiler import record_function
from flash_attn.flash_attention_interface import flash_attn_unpadded_qkvpacked_func

@dataclass
class ModelConfig:
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    src_vocab_size: int = 201088
    tag_vocab_size: int = 30000
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 10000.0
    rope_scaling_factor: float = 2.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0

class RMSNorm(nn.Module):

    def __init__(
        self, num_features: int, eps: float = 1e-6, device: torch.device | None = None
    ):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.scale = nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)
    
def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    cos = cos[position_ids].to(x.dtype) # (total_tokens, head_dim // 2)
    sin = sin[position_ids].to(x.dtype)

    cos = cos.unsqueeze(1)  # (total_tokens, 1, head_dim // 2)
    sin = sin.unsqueeze(1)  # (total_tokens, 1, head_dim // 2)
    
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

class AttentionBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        causal_mask: bool = True,
        device: torch.device | None = None
    ):
        super().__init__()
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.causal_mask = causal_mask
        self.norm = RMSNorm(config.hidden_size, device=device)

        qkv_dim = 3 * config.hidden_size
        self.qkv = (
            nn.Linear(config.hidden_size, qkv_dim, device=device)
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

    def forward(self, x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_ids: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        total_tokens, hidden_size = x.shape
        assert hidden_size == self.hidden_size, f"Expected hidden size {self.hidden_size}, got {hidden_size}"
        t = self.norm(x)
        qkv = self.qkv(t)
        q = qkv[:, : self.hidden_size].contiguous()
        k = qkv[:, self.hidden_size : 2 * self.hidden_size].contiguous()
        v = qkv[:, 2 * self.hidden_size : 3 * self.hidden_size].contiguous()

        q = q.view(-1, self.num_attention_heads, self.head_dim)
        k = k.view(-1, self.num_attention_heads, self.head_dim)
        v = v.view(-1, self.num_attention_heads, self.head_dim)

        q, k = self.rope(q, k, position_ids) # Apply rotary embeddings

        qkv = torch.stack([q, k, v], dim=1)
        t,_ = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_seqlens = cu_seqlens,
            max_seqlen = max_seqlen, dropout_p = 0.0,
            causal = self.causal_mask
        )
        t = t.contiguous().view(-1, self.head_dim * self.num_attention_heads)
        t = self.out(t)
        t = t + x
        return t

# TODO implement 5 trainable parameters
def swiglu(x, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(min=-limit, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(x_glu * alpha)
    return out_glu + (x_linear + 1)

class MLPBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.w_load_loss = config.w_load_loss
        self.w_importance_loss = config.w_importance_loss
        self.w_aux_loss = config.w_aux_loss
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.gate = nn.Linear(
            config.hidden_size, config.num_experts, device=device
        )
        self.mlp1_weight = nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size * 2,
                    config.hidden_size,
                ),
                device=device,
            )
        )
        self.mlp1_bias = nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2),
                device=device,
            )
        )

        self.mlp2_weight = nn.Parameter(
            torch.empty(
                (
                    config.num_experts, 
                    config.hidden_size,
                    config.intermediate_size
                ),
                device=device,
            )
        )
        self.mlp2_bias = nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=device,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens, num_features = x.shape
        t = self.norm(x)
        gate_logits = self.gate(t)

        gate_probs = torch.nn.functional.softmax(gate_logits, dim=-1)
        expert_weights, experts_indices = torch.topk(gate_probs, self.experts_per_token, dim=-1)
        # Normalize
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        mlp1_weight = self.mlp1_weight[experts_indices, ...]
        mlp1_bias = self.mlp1_bias[experts_indices, ...]

        t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
        t = swiglu(t, limit=self.swiglu_limit)
        mlp2_weight = self.mlp2_weight[experts_indices, ...]
        mlp2_bias = self.mlp2_bias[experts_indices, ...]

        t = torch.einsum("beck,bek->bec", mlp2_weight, t) + mlp2_bias
        # Weighted sum of experts
        t = torch.einsum("bec,be->bc", t, expert_weights)
        t = t + x
        # loss_load_balancing
        P = gate_probs.mean(dim=0)
        D = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.experts_per_token):
            D += D.scatter_add_(
                dim=0, index=experts_indices[:, i], src=torch.ones(num_tokens, device=x.device),
            )
        D = D / (num_tokens * self.experts_per_token)
        loss_load_balancing = self.w_load_loss * self.num_experts * (P * D).sum()
        # importance loss
        importance = gate_logits.sum(dim=0)
        importance_mean = importance.mean()
        importance_std = torch.std(importance)
        cv = (importance_std / (importance_mean + 1e-6))
        importance_loss = (
            self.w_importance_loss * (cv ** 2)
        )

        aux_loss = self.w_aux_loss * (loss_load_balancing + importance_loss)

        return t, aux_loss

class EncoderBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.self_attention_block = AttentionBlock(config, device=device)
        self.feed_forward_block = MLPBlock(config, device=device)
    
    def forward(self, x, src_cu_seqlen, src_position_ids, src_max_len) -> torch.Tensor:
        x = self.self_attention_block(x, src_cu_seqlen, src_position_ids, src_max_len)
        x, aux_loss = self.feed_forward_block(x)
        return x, aux_loss
    
class DecoderBlock(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.self_attention_block = AttentionBlock(config, device=device)
        self.cross_attention_block = AttentionBlock(config, device=device)
        self.feed_forward_block = MLPBlock(config, device=device)

    def forward(self, x, encoder_output, tgt_cu_seqlen, tgt_position_ids, tgt_max_len):
        x = self.self_attention_block(x, tgt_cu_seqlen, tgt_position_ids, tgt_max_len)
        x = self.cross_attention_block(
            x, encoder_output, encoder_output,
            tgt_cu_seqlen, tgt_position_ids, tgt_max_len
        )
        x, aux_loss = self.feed_forward_block(x)
        return x, aux_loss

class Encoder(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            config.src_vocab_size, config.hidden_size, device=device
        )
        self.encoder_block = nn.ModuleList(
            [
                EncoderBlock(config, device)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(self, x, encoder_output, src_cu_seqlen, src_position_ids, src_max_len):
        x = self.embedding(x)
        encoder_aux_loss = 0.0
        for encoder in self.encoder_block:
            x, aux_loss = encoder(x, encoder_output, src_cu_seqlen, src_position_ids, src_max_len)
            encoder_aux_loss += aux_loss
        return x, encoder_aux_loss

class Decoder(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            config.tag_vocab_size, config.hidden_size, device=device
        )
        self.decoder_block = nn.ModuleList(
            [
                EncoderBlock(config, device)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(self, x, encoder_output, tgt_cu_seqlen, tgt_position_ids, tgt_max_len):
        x = self.embedding(x)
        decoder_aux_loss = 0.0
        for decoder in self.decoder_block:
            x, aux_loss = decoder(x, encoder_output, tgt_cu_seqlen, tgt_position_ids, tgt_max_len)
            decoder_aux_loss += aux_loss
        return x, decoder_aux_loss

class Transformer(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.unembedding = nn.Linear(
            config.hidden_size,
            config.tag_vocab_size,
            bias=False,
            device=device
        )
        self.encoder = Encoder(config, device)
        self.decoder = Decoder(config, device)

    def forward(
        self, src, tgt,
        src_cu_seqlen, tgt_cu_seqlen,
        src_position_ids, tgt_position_ids,
        src_max_len, tgt_max_len,
    ):
        encoder_output, encoder_aux_loss = self.encode(src, src_cu_seqlen, src_position_ids, src_max_len)
        decoder_output, decoder_aux_loss = self.decode(tgt, encoder_output, tgt_cu_seqlen, tgt_position_ids, tgt_max_len)
        logits = self.unembedding(decoder_output)
        return logits, encoder_aux_loss, decoder_aux_loss

def build_transformer(
    config: ModelConfig,
    device: torch.device | None = None,
) -> Transformer:

    # Create the transformer
    transformer = Transformer(config, device)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer