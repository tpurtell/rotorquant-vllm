"""
Triton kernels for PlanarQuant decompress: gather centroids → inverse Givens rotate → rescale.

Decompress path: load indices → gather centroids → inverse Givens: f0=c*q0+s*q1, f1=-s*q0+c*q1
→ rescale by norm → cast to target dtype.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _planar_decompress_kernel(
    indices_ptr,
    norms_ptr,
    rot2_ptr,
    centroids_ptr,
    out_ptr,
    M,
    D,
    D_padded,
    n_groups: tl.constexpr,
    n_levels: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Decompress kernel: one program per row, processes D_padded elements in pairs of 2.

    Steps per row:
      1. Load D indices, gather centroids
      2. For each 2D pair, apply inverse Givens: f0=c*q0+s*q1, f1=-s*q0+c*q1
      3. Rescale by norm, cast to dtype
      4. Store result (first D elements only)
    """
    pid = tl.program_id(0)

    # Load norm for this row
    norm = tl.load(norms_ptr + pid)

    # ── Process each 2D pair ─────────────────────────────────────────────
    for g in range(n_groups):
        g_base = g * 2

        # ── Gather centroids from indices ────────────────────────────────
        i0 = tl.load(indices_ptr + pid * D_padded + g_base + 0,
                     mask=g_base < D, other=0).to(tl.int32)
        i1 = tl.load(indices_ptr + pid * D_padded + g_base + 1,
                     mask=(g_base + 1) < D, other=0).to(tl.int32)

        q0 = tl.load(centroids_ptr + i0).to(tl.float32)
        q1 = tl.load(centroids_ptr + i1).to(tl.float32)

        # ── Inverse Givens rotation ──────────────────────────────────────
        cos_t = tl.load(rot2_ptr + g * 2 + 0)
        sin_t = tl.load(rot2_ptr + g * 2 + 1)

        f0 = cos_t * q0 + sin_t * q1
        f1 = -sin_t * q0 + cos_t * q1

        # ── Rescale by norm and store ────────────────────────────────────
        tl.store(out_ptr + pid * D_padded + g_base + 0, f0 * norm, mask=g_base < D)
        tl.store(out_ptr + pid * D_padded + g_base + 1, f1 * norm, mask=(g_base + 1) < D)


def planar_decompress(indices: torch.Tensor,
                      norms: torch.Tensor,
                      rot2: torch.Tensor,
                      centroids: torch.Tensor,
                      dtype: torch.dtype,
                      d_padded: int | None = None) -> torch.Tensor:
    """Decompress int8 indices + fp32 norms back to float tensor via PlanarQuant.

    Args:
        indices: int8 indices of shape (N, H, D_padded).
        norms:   fp32 norms of shape (N, H, 1).
        rot2: Rotation parameters of shape (n_groups, 2), dtype fp32. [cos, sin] per pair.
        centroids: Lloyd-Max centroids of shape (n_levels,), dtype fp32.
        dtype: Target output dtype (torch.float16 or torch.bfloat16).
        d_padded: Padded dimension (multiple of 2). Auto-computed if None.

    Returns:
        out: (N, H, D) tensor in target dtype.
    """
    N, H, D_padded = indices.shape
    D = D_padded
    n_groups = rot2.shape[0]
    if d_padded is not None:
        D_padded = d_padded

    M = N * H

    indices_cont = indices.reshape(M, D_padded).contiguous()
    norms_cont = norms.reshape(M).contiguous().float()
    rot2_f32 = rot2.float().contiguous()
    centroids_f32 = centroids.float().contiguous()

    out = torch.empty(M, D_padded, device=indices.device, dtype=torch.float32)

    BLOCK_D = min(triton.next_power_of_2(D_padded), 128)

    try:
        _planar_decompress_kernel[(M,)](
            indices_cont,
            norms_cont,
            rot2_f32,
            centroids_f32,
            out,
            M,
            D,
            D_padded,
            n_groups,
            centroids.shape[0],
            BLOCK_D=BLOCK_D,
        )
    except Exception:
        return _planar_decompress_cpu(indices, norms, rot2, centroids, dtype)

    return out.reshape(N, H, D_padded).to(dtype)


def _planar_decompress_cpu(indices: torch.Tensor,
                           norms: torch.Tensor,
                           rot2: torch.Tensor,
                           centroids: torch.Tensor,
                           dtype: torch.dtype) -> torch.Tensor:
    """CPU fallback for planar_decompress using PyTorch operations."""
    N, H, D_padded = indices.shape
    n_groups = rot2.shape[0]
    M = N * H

    flat_idx = indices.reshape(M, D_padded).long()
    norms_flat = norms.reshape(M).float()

    # Gather centroids
    values = centroids[flat_idx]  # (M, D_padded)

    # Reshape to 2D pairs
    pairs = values.reshape(M, n_groups, 2)  # (M, n_groups, 2)

    # Inverse Givens rotation
    cos_t = rot2[:, 0]  # (n_groups,)
    sin_t = rot2[:, 1]
    q0 = pairs[..., 0]  # (M, n_groups)
    q1 = pairs[..., 1]

    f0 = cos_t * q0 + sin_t * q1
    f1 = -sin_t * q0 + cos_t * q1
    result = torch.stack([f0, f1], dim=-1).reshape(M, D_padded)

    # Rescale by norms
    result = result * norms_flat.unsqueeze(-1)

    return result.reshape(N, H, D_padded).to(dtype)
