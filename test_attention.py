"""
Test script để kiểm tra AttentionBlock hoạt động đúng với SDPA và Flash Attention 2
So sánh outputs và tính toán thủ công một số bước để verify
"""

import torch
import torch.nn.functional as F
import math
from model import AttentionBlock, ModelConfig

def manual_attention(q, k, v, mask=None, causal=False, scale=None):
    """
    Tính toán attention thủ công để so sánh
    q, k, v: [B, H, S, D] hoặc [B, S, H, D]
    mask: [B, S] boolean, True=valid, False=padding
    """
    # Convert to [B, H, S, D] if needed
    if q.dim() == 4 and q.shape[1] != q.shape[-1]:  # [B, S, H, D]
        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    
    B, H, S_q, D = q.shape
    _, _, S_k, _ = k.shape
    
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    # Compute attention scores: [B, H, S_q, S_k]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # Apply mask if provided
    if mask is not None:
        # mask: [B, S_q] -> [B, 1, S_q, 1]
        # We need to broadcast to [B, 1, S_q, S_k]
        if mask.dim() == 2:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, S_q, 1]
            mask_expanded = mask_expanded.expand(-1, H, -1, S_k)  # [B, H, S_q, S_k]
        else:
            mask_expanded = mask  # Assume already correct shape
        
        # Mask out padding positions (mask=True means valid, so we want to keep those)
        # Set invalid positions to -inf
        scores = scores.masked_fill(~mask_expanded, float('-inf'))
    
    # Apply causal mask if needed
    if causal:
        causal_mask = torch.triu(torch.ones(S_q, S_k, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply to values
    output = torch.matmul(attn_weights, v)  # [B, H, S_q, D]
    
    return output, attn_weights


def test_attention_block():
    """Test AttentionBlock với các configurations khác nhau"""
    
    print("=" * 80)
    print("Testing AttentionBlock")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create config
    config = ModelConfig(
        vocab_size=1000,
        hidden_size=512,
        num_attention_heads=8,
        head_dim=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        intermediate_size=2048,
        initial_context_length=128,
    )
    # Add use_flash_attn attribute (not in ModelConfig but used by AttentionBlock)
    config.use_flash_attn = True
    
    batch_size = 2
    seq_len = 16
    hidden_size = config.hidden_size
    
    # Create dummy input với float32
    x_fp32 = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    
    # Create mask (some padding at the end)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    mask[0, 12:] = False  # First sequence: last 4 tokens are padding
    mask[1, 14:] = False  # Second sequence: last 2 tokens are padding
    
    print(f"\nInput shape: {x_fp32.shape}")
    print(f"Input dtype: {x_fp32.dtype}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask sum per seq: {mask.sum(dim=1)}")
    
    # Test 1: Self-attention với Flash Attention (Pure float32, no mixed precision)
    print("\n" + "=" * 80)
    print("TEST 1: Self-Attention với Flash Attention (Pure float32)")
    print("=" * 80)
    
    config.use_flash_attn = True
    attn_block_fa_fp32 = AttentionBlock(config, causal_mask=False, device=device)
    # Keep model in float32
    attn_block_fa_fp32 = attn_block_fa_fp32.to(device=device, dtype=torch.float32)
    attn_block_fa_fp32.eval()
    
    with torch.no_grad():
        try:
            # Note: Flash Attention typically requires bfloat16/fp16, so this might fail
            # But let's test to see what happens
            output_fa_fp32 = attn_block_fa_fp32(x_fp32, query_mask=mask)
            print(f"✓ Flash Attention (fp32) output shape: {output_fa_fp32.shape}")
            print(f"  Output dtype: {output_fa_fp32.dtype}")
            print(f"  Output stats: mean={output_fa_fp32.mean().item():.6f}, std={output_fa_fp32.std().item():.6f}")
            print(f"  Has NaN: {torch.isnan(output_fa_fp32).any().item()}")
            print(f"  Has Inf: {torch.isinf(output_fa_fp32).any().item()}")
            output_fa = output_fa_fp32
        except Exception as e:
            print(f"  ✗ Flash Attention with float32 failed (expected): {type(e).__name__}: {e}")
            print(f"  → Falling back to SDPA for comparison")
            output_fa = None
    
    # Test 1b: Self-attention với Flash Attention (Mixed Precision - bfloat16)
    print("\n" + "=" * 80)
    print("TEST 1b: Self-Attention với Flash Attention (Mixed Precision - bfloat16)")
    print("=" * 80)
    
    config.use_flash_attn = True
    attn_block_fa_amp = AttentionBlock(config, causal_mask=False, device=device)
    attn_block_fa_amp = attn_block_fa_amp.to(device=device, dtype=torch.float32)
    attn_block_fa_amp.eval()
    
    with torch.no_grad():
        try:
            from torch.amp import autocast
            # Use autocast for mixed precision
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                output_fa_amp = attn_block_fa_amp(x_fp32, query_mask=mask)
            print(f"✓ Flash Attention (mixed precision) output shape: {output_fa_amp.shape}")
            print(f"  Output dtype: {output_fa_amp.dtype}")
            print(f"  Output stats: mean={output_fa_amp.mean().item():.6f}, std={output_fa_amp.std().item():.6f}")
            print(f"  Has NaN: {torch.isnan(output_fa_amp).any().item()}")
            print(f"  Has Inf: {torch.isinf(output_fa_amp).any().item()}")
            if output_fa is not None:
                diff_amp = (output_fa_amp.float() - output_fa.float()).abs()
                mask_3d = mask.unsqueeze(-1).expand(-1, -1, hidden_size)
                valid_diff = diff_amp[mask_3d]
                print(f"  Max diff vs pure fp32: {valid_diff.max().item():.6f} (valid tokens)")
            output_fa = output_fa_amp if output_fa is None else output_fa_amp
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            if output_fa is None:
                output_fa = None
            print(f"✓ Flash Attention output shape: {output_fa.shape}")
            print(f"  Output stats: mean={output_fa.mean().item():.6f}, std={output_fa.std().item():.6f}")
            print(f"  Has NaN: {torch.isnan(output_fa).any().item()}")
            print(f"  Has Inf: {torch.isinf(output_fa).any().item()}")
            
            if torch.isnan(output_fa).any() or torch.isinf(output_fa).any():
                nan_count = torch.isnan(output_fa).sum().item()
                inf_count = torch.isinf(output_fa).sum().item()
                print(f"  ✗ ERROR: Output contains NaN/Inf! NaN: {nan_count}, Inf: {inf_count}")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            output_fa = None
    
    # Test 2: Self-attention với SDPA fallback (Pure float32)
    print("\n" + "=" * 80)
    print("TEST 2: Self-Attention với SDPA Fallback (Pure float32)")
    print("=" * 80)
    
    config.use_flash_attn = False
    attn_block_sdpa_fp32 = AttentionBlock(config, causal_mask=False, device=device)
    attn_block_sdpa_fp32 = attn_block_sdpa_fp32.to(device=device, dtype=torch.float32)
    # Copy weights from FA model for fair comparison (if available)
    if output_fa is not None:
        try:
            # Try to copy weights (they should be compatible if same dtype)
            attn_block_sdpa_fp32.load_state_dict(attn_block_fa_fp32.state_dict() if 'attn_block_fa_fp32' in locals() else attn_block_fa_amp.state_dict())
        except:
            pass
    attn_block_sdpa_fp32.eval()
    
    with torch.no_grad():
        try:
            output_sdpa_fp32 = attn_block_sdpa_fp32(x_fp32, query_mask=mask)
            print(f"✓ SDPA (fp32) output shape: {output_sdpa_fp32.shape}")
            print(f"  Output dtype: {output_sdpa_fp32.dtype}")
            print(f"  Output stats: mean={output_sdpa_fp32.mean().item():.6f}, std={output_sdpa_fp32.std().item():.6f}")
            print(f"  Has NaN: {torch.isnan(output_sdpa_fp32).any().item()}")
            print(f"  Has Inf: {torch.isinf(output_sdpa_fp32).any().item()}")
            output_sdpa = output_sdpa_fp32
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            output_sdpa = None
    
    # Test 2b: Self-attention với SDPA fallback (Mixed Precision)
    print("\n" + "=" * 80)
    print("TEST 2b: Self-Attention với SDPA Fallback (Mixed Precision - bfloat16)")
    print("=" * 80)
    
    config.use_flash_attn = False
    attn_block_sdpa_amp = AttentionBlock(config, causal_mask=False, device=device)
    attn_block_sdpa_amp = attn_block_sdpa_amp.to(device=device, dtype=torch.float32)
    # Copy weights
    if output_sdpa is not None:
        try:
            attn_block_sdpa_amp.load_state_dict(attn_block_sdpa_fp32.state_dict())
        except:
            pass
    attn_block_sdpa_amp.eval()
    
    with torch.no_grad():
        try:
            from torch.amp import autocast
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                output_sdpa_amp = attn_block_sdpa_amp(x_fp32, query_mask=mask)
            print(f"✓ SDPA (mixed precision) output shape: {output_sdpa_amp.shape}")
            print(f"  Output dtype: {output_sdpa_amp.dtype}")
            print(f"  Output stats: mean={output_sdpa_amp.mean().item():.6f}, std={output_sdpa_amp.std().item():.6f}")
            print(f"  Has NaN: {torch.isnan(output_sdpa_amp).any().item()}")
            print(f"  Has Inf: {torch.isinf(output_sdpa_amp).any().item()}")
            if output_sdpa is not None:
                diff_amp = (output_sdpa_amp.float() - output_sdpa.float()).abs()
                mask_3d = mask.unsqueeze(-1).expand(-1, -1, hidden_size)
                valid_diff = diff_amp[mask_3d]
                print(f"  Max diff vs pure fp32: {valid_diff.max().item():.6f} (valid tokens)")
            output_sdpa = output_sdpa_amp if output_sdpa is None else output_sdpa_amp
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            if output_sdpa is None:
                output_sdpa = None
            print(f"✓ SDPA output shape: {output_sdpa.shape}")
            print(f"  Output stats: mean={output_sdpa.mean().item():.6f}, std={output_sdpa.std().item():.6f}")
            print(f"  Has NaN: {torch.isnan(output_sdpa).any().item()}")
            print(f"  Has Inf: {torch.isinf(output_sdpa).any().item()}")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            output_sdpa = None
    
    # Compare outputs
    if output_fa is not None and output_sdpa is not None:
        print("\n" + "=" * 80)
        print("Comparing Flash Attention vs SDPA")
        print("=" * 80)
        
        # Convert to same dtype for comparison
        output_fa_comp = output_fa.float() if output_fa.dtype != torch.float32 else output_fa
        output_sdpa_comp = output_sdpa.float() if output_sdpa.dtype != torch.float32 else output_sdpa
        
        # Compute difference (only for valid tokens)
        diff = (output_fa_comp - output_sdpa_comp).abs()
        
        # Only compare valid tokens (use mask)
        mask_3d = mask.unsqueeze(-1).expand(-1, -1, hidden_size)  # [B, S, H]
        valid_diff = diff[mask_3d]
        
        print(f"Output dtypes - FA: {output_fa.dtype}, SDPA: {output_sdpa.dtype}")
        print(f"Max absolute difference: {diff.max().item():.6f}")
        print(f"Mean absolute difference (valid tokens): {valid_diff.mean().item():.6f}")
        print(f"Max absolute difference (valid tokens): {valid_diff.max().item():.6f}")
        
        # Check if they're close (allowing for numerical differences)
        close_threshold = 1e-1  # More lenient for mixed precision
        if valid_diff.max().item() < close_threshold:
            print(f"✓ Outputs are close (max diff < {close_threshold})")
        else:
            print(f"⚠ WARNING: Large differences between FA and SDPA (max diff = {valid_diff.max().item():.6f})")
            print(f"  This might be expected due to different implementations or mixed precision")
    
    # Test 3: Causal self-attention (Mixed Precision)
    print("\n" + "=" * 80)
    print("TEST 3: Causal Self-Attention với Flash Attention (Mixed Precision)")
    print("=" * 80)
    
    config.use_flash_attn = True
    attn_block_causal = AttentionBlock(config, causal_mask=True, device=device)
    attn_block_causal = attn_block_causal.to(device=device, dtype=torch.float32)
    attn_block_causal.eval()
    
    with torch.no_grad():
        try:
            from torch.amp import autocast
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                output_causal = attn_block_causal(x_fp32, query_mask=mask)
            print(f"✓ Causal Flash Attention output shape: {output_causal.shape}")
            print(f"  Output stats: mean={output_causal.mean().item():.6f}, std={output_causal.std().item():.6f}")
            print(f"  Has NaN: {torch.isnan(output_causal).any().item()}")
            print(f"  Has Inf: {torch.isinf(output_causal).any().item()}")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 4: Cross-attention (Mixed Precision)
    print("\n" + "=" * 80)
    print("TEST 4: Cross-Attention với Flash Attention (Mixed Precision)")
    print("=" * 80)
    
    config.use_flash_attn = True
    attn_block_cross = AttentionBlock(config, causal_mask=False, device=device)
    attn_block_cross = attn_block_cross.to(device=device, dtype=torch.float32)
    attn_block_cross.eval()
    
    # Encoder output (different sequence length) - float32
    encoder_output = torch.randn(batch_size, 20, hidden_size, device=device, dtype=torch.float32)
    encoder_mask = torch.ones(batch_size, 20, dtype=torch.bool, device=device)
    encoder_mask[0, 16:] = False  # First: last 4 tokens padding
    encoder_mask[1, 18:] = False  # Second: last 2 tokens padding
    
    with torch.no_grad():
        try:
            from torch.amp import autocast
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                output_cross = attn_block_cross(
                    x_fp32, 
                    encoder_output=encoder_output,
                    query_mask=mask,
                    encoder_mask=encoder_mask
                )
            print(f"✓ Cross Attention output shape: {output_cross.shape}")
            print(f"  Output stats: mean={output_cross.mean().item():.6f}, std={output_cross.std().item():.6f}")
            print(f"  Has NaN: {torch.isnan(output_cross).any().item()}")
            print(f"  Has Inf: {torch.isinf(output_cross).any().item()}")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 5: Manual computation comparison (small case)
    print("\n" + "=" * 80)
    print("TEST 5: Manual Computation Comparison (Small Case)")
    print("=" * 80)
    
    # Small case for manual computation
    B_small, H_small, S_small, D_small = 1, 2, 4, 8
    config_small = ModelConfig(
        vocab_size=100,
        hidden_size=H_small * D_small,
        num_attention_heads=H_small,
        head_dim=D_small,
        num_encoder_layers=1,
        num_decoder_layers=1,
        intermediate_size=32,
        initial_context_length=32,
    )
    config_small.use_flash_attn = False  # Use SDPA for easier comparison
    
    x_small = torch.randn(B_small, S_small, H_small * D_small, device=device, dtype=torch.float32)
    mask_small = torch.ones(B_small, S_small, dtype=torch.bool, device=device)
    
    attn_small = AttentionBlock(config_small, causal_mask=False, device=device)
    attn_small = attn_small.to(device=device, dtype=torch.float32)
    attn_small.eval()
    
    # Get Q, K, V manually
    with torch.no_grad():
        q = attn_small.w_q(x_small).view(B_small, S_small, H_small, D_small)
        k = attn_small.w_k(x_small).view(B_small, S_small, H_small, D_small)
        v = attn_small.w_v(x_small).view(B_small, S_small, H_small, D_small)
        
        # Apply RoPE and norm
        position_ids = torch.arange(S_small, device=device).unsqueeze(0).expand(B_small, -1)
        q, k = attn_small.rope(q, k, position_ids)
        q, k = attn_small.q_norm(q), attn_small.k_norm(k)
        
        # Manual attention
        output_manual, attn_weights = manual_attention(
            q, k, v, 
            mask=mask_small, 
            causal=False, 
            scale=attn_small.softmax_scale
        )
        
        # Apply output projection
        output_manual = output_manual.transpose(1, 2).contiguous().view(B_small, S_small, -1)
        output_manual = attn_small.w_o(output_manual)
        
        # SDPA output
        output_sdpa_small = attn_small(x_small, query_mask=mask_small)
        
        # Compare
        diff_small = (output_manual - output_sdpa_small).abs()
        print(f"Manual vs SDPA (small case):")
        print(f"  Max diff: {diff_small.max().item():.6f}")
        print(f"  Mean diff: {diff_small.mean().item():.6f}")
        
        if diff_small.max().item() < 1e-4:
            print(f"  ✓ Manual computation matches SDPA!")
        else:
            print(f"  ⚠ Differences detected (might be due to implementation details)")
            print(f"    Manual output stats: mean={output_manual.mean().item():.6f}, std={output_manual.std().item():.6f}")
            print(f"    SDPA output stats: mean={output_sdpa_small.mean().item():.6f}, std={output_sdpa_small.std().item():.6f}")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_attention_block()

