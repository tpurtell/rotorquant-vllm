"""
Triton kernels for IsoQuant decompress: nibble unpack -> gather centroids -> norm scale -> cast.

Decompress path (no inverse rotation): load nibble-packed bytes -> unpack indices
-> gather centroids -> scale by norm -> cast to target dtype.

Rotation is applied to Q once (O(1) per step), so K/V decompress in rotated space.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _iso_decompress_kernel(
    packed_ptr,
    norms_ptr,
    centroids_ptr,
    out_ptr,
    M,
    HALF_D,
    D,
    HALF_D_PAD: tl.constexpr,
):
    """Decompress kernel: one program per row.

    Steps per row:
      1. Load HALF_D nibble-packed bytes
      2. Unpack into two index streams (hi=even pos, lo=odd pos)
      3. Gather centroids, scale by norm
      4. Interleave and store: even=hi, odd=lo (first D elements)
    """
    pid = tl.program_id(0)

    p_offs = tl.arange(0, HALF_D_PAD)
    p_mask = p_offs < HALF_D

    # Load packed bytes
    packed = tl.load(packed_ptr + pid * HALF_D + p_offs, mask=p_mask, other=0).to(tl.int32)

    # Nibble unpack
    hi_idx = ((packed >> 4) & 0x0F).to(tl.int32)  # even positions
    lo_idx = (packed & 0x0F).to(tl.int32)          # odd positions

    # Centroid gather
    c_hi = tl.load(centroids_ptr + hi_idx).to(tl.float32)
    c_lo = tl.load(centroids_ptr + lo_idx).to(tl.float32)

    # Scale by norm
    norm = tl.load(norms_ptr + pid)
    c_hi = c_hi * norm
    c_lo = c_lo * norm

    # Interleave and store: even=hi, odd=lo
    out_base = pid * D
    tl.store(out_ptr + out_base + p_offs * 2, c_hi,
             mask=p_mask & (p_offs * 2 < D))
    tl.store(out_ptr + out_base + p_offs * 2 + 1, c_lo,
             mask=p_mask & (p_offs * 2 + 1 < D))


def iso_decompress(packed: torch.Tensor,
                   norms: torch.Tensor,
                   centroids: torch.Tensor,
                   dtype: torch.dtype) -> torch.Tensor:
    """Decompress nibble-packed uint8 + fp32 norms back to float tensor via IsoQuant.

    Args:
        packed:    nibble-packed uint8 indices of shape (N, H, HALF_D).
        norms:     fp32 norms of shape (N, H, 1).
        centroids: Lloyd-Max centroids of shape (n_levels,), dtype fp32.
        dtype:     Target output dtype (torch.float16 or torch.bfloat16).

    Returns:
        out: (N, H, D) tensor in target dtype, where D = HALF_D * 2.
    """
    N, H, HALF_D = packed.shape
    D = HALF_D * 2

    M = N * H
    HALF_D_PAD = triton.next_power_of_2(HALF_D)

    packed_cont = packed.reshape(M, HALF_D).contiguous()
    norms_cont = norms.reshape(M).contiguous().float()
    centroids_f32 = centroids.float().contiguous()

    out = torch.empty(M, D, device=packed.device, dtype=dtype)

    try:
        _iso_decompress_kernel[(M,)](
            packed_cont,
            norms_cont,
            centroids_f32,
            out,
            M,
            HALF_D,
            D,
            HALF_D_PAD=HALF_D_PAD,
        )
    except Exception:
        return _iso_decompress_cpu(packed, norms, centroids, dtype)

    return out.reshape(N, H, D).to(dtype)


def _iso_decompress_cpu(packed: torch.Tensor,
                        norms: torch.Tensor,
                        centroids: torch.Tensor,
                        dtype: torch.dtype) -> torch.Tensor:
    """CPU fallback for iso_decompress using PyTorch operations."""
    N, H, HALF_D = packed.shape
    D = HALF_D * 2

    M = N * H

    packed_flat = packed.reshape(M, HALF_D)
    norms_flat = norms.reshape(M).float()

    # Nibble unpack
    hi_idx = ((packed_flat >> 4) & 0x0F).long()  # even positions
    lo_idx = (packed_flat & 0x0F).long()          # odd positions

    # Gather centroids
    c_hi = centroids[hi_idx].float()  # (M, HALF_D)
    c_lo = centroids[lo_idx].float()  # (M, HALF_D)

    # Scale by norm
    c_hi = c_hi * norms_flat.unsqueeze(-1)
    c_lo = c_lo * norms_flat.unsqueeze(-1)

    # Interleave: even=hi, odd=lo
    out = torch.zeros(M, D, device=packed.device, dtype=torch.float32)
    out[:, 0::2] = c_hi
    out[:, 1::2] = c_lo

    return out.reshape(N, H, D).to(dtype)
