"""
Triton kernel for PlanarQuant compress: tiled matmul with pre-split rotation columns.

Compress path: load X -> compute norm -> normalize -> tiled rotate via R^T
-> boundary bucketize -> nibble pack 2 indices into 1 byte.
Outputs: uint8 nibble-packed indices (M, HALF_D) and fp32 norms (M,).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _planar_compress_kernel(
    X,              # (M, D) fp16/bf16 input
    Rot_T_even,     # (D, HALF_D) fp32 even columns of R^T
    Rot_T_odd,      # (D, HALF_D) fp32 odd columns of R^T
    Boundaries,     # (N_BOUND,) fp32 boundaries
    Packed_out,     # (M, HALF_D) uint8 output
    Norms_out,      # (M,) fp32 output norms
    M,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
    N_BOUND: tl.constexpr,
    BLOCK_K: tl.constexpr,
    D_PAD: tl.constexpr,
    HALF_D_PAD: tl.constexpr,
):
    """Fused PlanarQuant compress: norm + normalize + tiled rotation + bucketize + pack.

    One program per row.  Computes L2 norm, normalizes, tiles the rotation
    matmul using pre-split even/odd column halves of R^T, performs linear-scan
    bucketize, and writes nibble-packed output plus one fp32 norm.
    """
    row = tl.program_id(0)
    if row >= M:
        return

    d_offs = tl.arange(0, D_PAD)
    d_mask = d_offs < D
    half_offs = tl.arange(0, HALF_D_PAD)
    half_mask = half_offs < HALF_D

    # Step 1: Load input and compute norm
    x = tl.load(X + row * D + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    norm = tl.sqrt(tl.sum(x * x))
    inv_norm = 1.0 / (norm + 1e-10)

    # Step 2: Tiled rotation with even/odd output split
    rotated_even = tl.zeros([HALF_D_PAD], dtype=tl.float32)
    rotated_odd  = tl.zeros([HALF_D_PAD], dtype=tl.float32)

    for k_start in range(0, D, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < D

        x_chunk = (
            tl.load(X + row * D + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            * inv_norm
        )

        re = tl.load(
            Rot_T_even + k_offs[:, None] * HALF_D + half_offs[None, :],
            mask=k_mask[:, None] & half_mask[None, :],
            other=0.0,
        )
        ro = tl.load(
            Rot_T_odd + k_offs[:, None] * HALF_D + half_offs[None, :],
            mask=k_mask[:, None] & half_mask[None, :],
            other=0.0,
        )

        rotated_even += tl.sum(x_chunk[:, None] * re, axis=0)
        rotated_odd  += tl.sum(x_chunk[:, None] * ro, axis=0)

    # Step 3: Bucketize (linear scan over sorted boundaries)
    idx_even = tl.zeros([HALF_D_PAD], dtype=tl.int32)
    idx_odd  = tl.zeros([HALF_D_PAD], dtype=tl.int32)
    for b in range(N_BOUND):
        boundary = tl.load(Boundaries + b)
        idx_even += (rotated_even >= boundary).to(tl.int32)
        idx_odd  += (rotated_odd  >= boundary).to(tl.int32)

    # Step 4: Pack nibbles -- high=even, low=odd
    packed = ((idx_even & 0xF) << 4) | (idx_odd & 0xF)

    # Step 5: Store packed indices and norm
    tl.store(Packed_out + row * HALF_D + half_offs, packed.to(tl.uint8), mask=half_mask)
    tl.store(Norms_out + row, norm)


def _next_pow2(n: int) -> int:
    """Round up to the nearest power of two."""
    return 1 << (n - 1).bit_length() if n > 0 else 1


def planar_compress(
    x: torch.Tensor,
    rot_T_even: torch.Tensor,
    rot_T_odd: torch.Tensor,
    boundaries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress (N, H, D) float tensor to nibble-packed uint8 + fp32 norms via PlanarQuant.

    Uses tiled matmul with pre-split even/odd rotation columns for contiguous loads.

    Args:
        x: Input tensor of shape (N, H, D), dtype fp16 or bf16.
        rot_T_even: (D, HALF_D) fp32 even columns of R^T.
        rot_T_odd: (D, HALF_D) fp32 odd columns of R^T.
        boundaries: Lloyd-Max boundaries of shape (n_bound,), dtype fp32.

    Returns:
        packed: (N, H, HALF_D) uint8 tensor of nibble-packed centroid indices.
        norms:  (N, H, 1) fp32 tensor of vector norms.
    """
    N, H, D = x.shape
    HALF_D = D // 2
    M = N * H

    if not x.is_cuda:
        return _planar_compress_cpu(x, rot_T_even, rot_T_odd, boundaries)

    x_flat = x.reshape(M, D).contiguous()

    packed = torch.empty(M, HALF_D, dtype=torch.uint8, device=x.device)
    norms = torch.empty(M, dtype=torch.float32, device=x.device)

    N_BOUND = boundaries.shape[0]
    BLOCK_K = min(32, D)
    D_PAD = _next_pow2(D)
    HALF_D_PAD = _next_pow2(HALF_D)

    grid = (M,)
    _planar_compress_kernel[grid](
        x_flat,
        rot_T_even,
        rot_T_odd,
        boundaries,
        packed,
        norms,
        M,
        D=D,
        HALF_D=HALF_D,
        N_BOUND=N_BOUND,
        BLOCK_K=BLOCK_K,
        D_PAD=D_PAD,
        HALF_D_PAD=HALF_D_PAD,
    )

    return packed.reshape(N, H, HALF_D), norms.reshape(N, H, 1)


def _planar_compress_cpu(
    x: torch.Tensor,
    rot_T_even: torch.Tensor,
    rot_T_odd: torch.Tensor,
    boundaries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU fallback for planar_compress using PyTorch operations."""
    N, H, D = x.shape
    HALF_D = D // 2
    M = N * H

    flat = x.reshape(M, D).float()

    raw_norms = torch.norm(flat, dim=-1, keepdim=True)
    normalized = flat / (raw_norms + 1e-10)

    # Reconstruct full rotation.T from even/odd halves
    rotation_T = torch.empty(D, D, dtype=torch.float32, device=x.device)
    rotation_T[:, 0::2] = rot_T_even
    rotation_T[:, 1::2] = rot_T_odd

    rotated = normalized @ rotation_T

    indices = torch.bucketize(rotated, boundaries)
    indices = indices.clamp(0, 2**4 - 1)

    idx_u8 = indices.to(torch.uint8)
    packed = (idx_u8[:, 0::2] << 4) | idx_u8[:, 1::2]

    return packed.reshape(N, H, HALF_D), raw_norms.reshape(N, H, 1)
