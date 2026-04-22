"""
Triton kernels for PlanarQuant decompress: nibble unpack → gather centroids → norm scale → cast.

Decompress path: nibble unpack packed bytes → gather centroid values → scale by norm → cast to dtype.
No inverse rotation needed (Q pre-rotation path, rotation applied to Q in attention impl).
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _planar_decompress_kernel(
    packed_ptr,
    norms_ptr,
    centroids_ptr,
    out_ptr,
    M,
    HALF_D,
    D,
    HALF_D_PAD: tl.constexpr,
    n_levels: tl.constexpr,
    BLOCK_HALF_D: tl.constexpr,
):
    """Decompress kernel: one program per row.

    Steps per row:
      1. Load HALF_D packed bytes, nibble unpack to get 2 index streams
      2. Gather centroid values for each index
      3. Scale by norm, interleave even/odd positions
      4. Store first D elements to output
    """
    pid = tl.program_id(0)

    # Load norm for this row
    norm = tl.load(norms_ptr + pid)

    # Load packed nibble bytes and unpack
    half_d_offs = tl.arange(0, BLOCK_HALF_D)
    packed_ptrs = packed_ptr + pid * HALF_D_PAD + half_d_offs
    packed_vals = tl.load(packed_ptrs, mask=half_d_offs < HALF_D, other=0).to(tl.int32)

    # Nibble unpack: high nibble = even index, low nibble = odd index
    hi_idx = (packed_vals >> 4) & 0x0F
    lo_idx = packed_vals & 0x0F

    # Gather centroid values
    q_hi = tl.load(centroids_ptr + hi_idx).to(tl.float32)
    q_lo = tl.load(centroids_ptr + lo_idx).to(tl.float32)

    # Scale by norm
    f_hi = q_hi * norm
    f_lo = q_lo * norm

    # Interleave: even positions = hi, odd positions = lo
    # Write to output at positions 2*g and 2*g+1
    for g in range(BLOCK_HALF_D):
        even_off = 2 * g
        odd_off = 2 * g + 1
        tl.store(out_ptr + pid * D + even_off, f_hi, mask=even_off < D)
        tl.store(out_ptr + pid * D + odd_off, f_lo, mask=odd_off < D)


def planar_decompress(packed: torch.Tensor,
                      norms: torch.Tensor,
                      centroids: torch.Tensor,
                      dtype: torch.dtype,
                      d_padded: int | None = None) -> torch.Tensor:
    """Decompress nibble-packed uint8 indices + fp32 norms back to float tensor via PlanarQuant.

    Args:
        packed: uint8 nibble-packed indices of shape (N, H, HALF_D).
        norms:  fp32 norms of shape (N, H, 1).
        centroids: Lloyd-Max centroids of shape (n_levels,), dtype fp32.
        dtype: Target output dtype (torch.float16 or torch.bfloat16).
        d_padded: Padded dimension (multiple of 2). Auto-computed if None.

    Returns:
        out: (N, H, D) tensor in target dtype.
    """
    N, H, HALF_D = packed.shape
    D = HALF_D * 2

    if d_padded is not None:
        D = d_padded

    M = N * H

    # Compute padded dimensions for Triton
    HALF_D_PAD = triton.next_power_of_2(HALF_D)

    packed_cont = packed.reshape(M, HALF_D).contiguous()
    norms_cont = norms.reshape(M).contiguous().float()
    centroids_f32 = centroids.float().contiguous()

    out = torch.empty(M, D, device=packed.device, dtype=torch.float32)

    BLOCK_HALF_D = min(HALF_D_PAD, 64)

    try:
        _planar_decompress_kernel[(M,)](
            packed_cont,
            norms_cont,
            centroids_f32,
            out,
            M,
            HALF_D,
            D,
            HALF_D_PAD=HALF_D_PAD,
            n_levels=centroids.shape[0],
            BLOCK_HALF_D=BLOCK_HALF_D,
        )
    except Exception:
        return _planar_decompress_cpu(packed, norms, centroids, dtype)

    return out.reshape(N, H, D).to(dtype)


def _planar_decompress_cpu(packed: torch.Tensor,
                           norms: torch.Tensor,
                           centroids: torch.Tensor,
                           dtype: torch.dtype) -> torch.Tensor:
    """CPU fallback for planar_decompress using PyTorch operations."""
    N, H, HALF_D = packed.shape
    D = HALF_D * 2
    M = N * H

    flat_packed = packed.reshape(M, HALF_D)
    norms_flat = norms.reshape(M).float()

    # Nibble unpack: high nibble = even index, low nibble = odd index
    hi_idx = ((flat_packed >> 4) & 0x0F).long()  # (M, HALF_D)
    lo_idx = (flat_packed & 0x0F).long()  # (M, HALF_D)

    # Gather centroid values
    q_hi = centroids[hi_idx]  # (M, HALF_D)
    q_lo = centroids[lo_idx]  # (M, HALF_D)

    # Interleave: even positions = hi, odd positions = lo
    result = torch.zeros(M, D, device=packed.device, dtype=torch.float32)
    result[:, 0::2] = q_hi
    result[:, 1::2] = q_lo

    # Scale by norms
    result = result * norms_flat.unsqueeze(-1)

    return result.reshape(N, H, D).to(dtype)
