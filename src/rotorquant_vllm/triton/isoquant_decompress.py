"""
Triton kernels for IsoQuant decompress: gather centroids → inverse quaternion rotate → rescale.

Decompress path: load indices → gather centroids → conj(q_L) * v * q_R (full) or conj(q_L) * v (fast)
→ rescale by norm → cast to target dtype.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _iso_decompress_kernel(
    indices_ptr,
    norms_ptr,
    ql_ptr,
    qr_ptr,
    centroids_ptr,
    out_ptr,
    M,
    D,
    D_padded,
    n_groups: tl.constexpr,
    n_levels: tl.constexpr,
    has_qr: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Decompress kernel: one program per row, processes D_padded elements in groups of 4.

    Steps per row:
      1. Load D indices, gather centroids → quantized 4D values
      2. For each 4D group, apply inverse quaternion rotation
      3. Rescale by norm, cast to output dtype
      4. Store result (first D elements only)
    """
    pid = tl.program_id(0)

    # Load norm for this row
    norm = tl.load(norms_ptr + pid)

    # ── Process each 4D group ────────────────────────────────────────────
    for g in range(n_groups):
        g_base = g * 4

        # ── Gather centroids from indices ────────────────────────────────
        i0 = tl.load(indices_ptr + pid * D_padded + g_base + 0,
                     mask=g_base < D, other=0).to(tl.int32)
        i1 = tl.load(indices_ptr + pid * D_padded + g_base + 1,
                     mask=(g_base + 1) < D, other=0).to(tl.int32)
        i2 = tl.load(indices_ptr + pid * D_padded + g_base + 2,
                     mask=(g_base + 2) < D, other=0).to(tl.int32)
        i3 = tl.load(indices_ptr + pid * D_padded + g_base + 3,
                     mask=(g_base + 3) < D, other=0).to(tl.int32)

        v0 = tl.load(centroids_ptr + i0).to(tl.float32)
        v1 = tl.load(centroids_ptr + i1).to(tl.float32)
        v2 = tl.load(centroids_ptr + i2).to(tl.float32)
        v3 = tl.load(centroids_ptr + i3).to(tl.float32)

        # ── Inverse quaternion rotation ──────────────────────────────────
        # conj(q_L) * v
        qL_w = tl.load(ql_ptr + g * 4 + 0)
        qL_x = tl.load(ql_ptr + g * 4 + 1)
        qL_y = tl.load(ql_ptr + g * 4 + 2)
        qL_z = tl.load(ql_ptr + g * 4 + 3)

        conj_w = qL_w
        conj_x = -qL_x
        conj_y = -qL_y
        conj_z = -qL_z

        t_w = conj_w * v0 - conj_x * v1 - conj_y * v2 - conj_z * v3
        t_x = conj_w * v1 + conj_x * v0 + conj_y * v3 - conj_z * v2
        t_y = conj_w * v2 - conj_x * v3 + conj_y * v0 + conj_z * v1
        t_z = conj_w * v3 + conj_x * v2 - conj_y * v1 + conj_z * v0

        if has_qr:
            # Full mode: conj(q_L) * v * q_R
            qR_w = tl.load(qr_ptr + g * 4 + 0)
            qR_x = tl.load(qr_ptr + g * 4 + 1)
            qR_y = tl.load(qr_ptr + g * 4 + 2)
            qR_z = tl.load(qr_ptr + g * 4 + 3)

            # temp * q_R
            f0 = t_w * qR_w - t_x * qR_x - t_y * qR_y - t_z * qR_z
            f1 = t_w * qR_x + t_x * qR_w + t_y * qR_z - t_z * qR_y
            f2 = t_w * qR_y - t_x * qR_z + t_y * qR_w + t_z * qR_x
            f3 = t_w * qR_z + t_x * qR_y - t_y * qR_x + t_z * qR_w
        else:
            # Fast mode: conj(q_L) * v (single multiply)
            f0, f1, f2, f3 = t_w, t_x, t_y, t_z

        # ── Rescale by norm and store ────────────────────────────────────
        tl.store(out_ptr + pid * D_padded + g_base + 0, f0 * norm, mask=g_base < D)
        tl.store(out_ptr + pid * D_padded + g_base + 1, f1 * norm, mask=(g_base + 1) < D)
        tl.store(out_ptr + pid * D_padded + g_base + 2, f2 * norm, mask=(g_base + 2) < D)
        tl.store(out_ptr + pid * D_padded + g_base + 3, f3 * norm, mask=(g_base + 3) < D)


def iso_decompress(indices: torch.Tensor,
                   norms: torch.Tensor,
                   q_L: torch.Tensor,
                   q_R: torch.Tensor | None,
                   centroids: torch.Tensor,
                   dtype: torch.dtype,
                   mode: str = "fast",
                   d_padded: int | None = None) -> torch.Tensor:
    """Decompress int8 indices + fp32 norms back to float tensor via IsoQuant.

    Args:
        indices: int8 indices of shape (N, H, D_padded).
        norms:   fp32 norms of shape (N, H, 1).
        q_L: Left quaternions of shape (n_groups, 4), dtype fp32.
        q_R: Right quaternions of shape (n_groups, 4), dtype fp32. None for fast mode.
        centroids: Lloyd-Max centroids of shape (n_levels,), dtype fp32.
        dtype: Target output dtype (torch.float16 or torch.bfloat16).
        mode: 'full' (conj(q_L) * v * q_R) or 'fast' (conj(q_L) * v).
        d_padded: Padded dimension (multiple of 4). Auto-computed if None.

    Returns:
        out: (N, H, D) tensor in target dtype.
    """
    N, H, D_padded = indices.shape
    D = D_padded  # indices are already padded
    n_groups = q_L.shape[0]
    if d_padded is not None:
        D_padded = d_padded

    M = N * H
    has_qr = (q_R is not None) and (mode == "full")

    indices_cont = indices.reshape(M, D_padded).contiguous()
    norms_cont = norms.reshape(M).contiguous().float()
    ql_f32 = q_L.float().contiguous()

    if has_qr:
        qr_f32 = q_R.float().contiguous()
    else:
        qr_f32 = torch.ones(n_groups, 4, device=q_L.device, dtype=torch.float32)
    centroids_f32 = centroids.float().contiguous()

    out = torch.empty(M, D_padded, device=indices.device, dtype=torch.float32)

    BLOCK_D = min(triton.next_power_of_2(D_padded), 128)

    try:
        _iso_decompress_kernel[(M,)](
            indices_cont,
            norms_cont,
            ql_f32,
            qr_f32,
            centroids_f32,
            out,
            M,
            D,
            D_padded,
            n_groups,
            centroids.shape[0],
            has_qr,
            BLOCK_D=BLOCK_D,
        )
    except Exception:
        return _iso_decompress_cpu(indices, norms, q_L, q_R, centroids, dtype, mode)

    return out.reshape(N, H, D_padded).to(dtype)


def _iso_decompress_cpu(indices: torch.Tensor,
                        norms: torch.Tensor,
                        q_L: torch.Tensor,
                        q_R: torch.Tensor | None,
                        centroids: torch.Tensor,
                        dtype: torch.dtype,
                        mode: str = "fast") -> torch.Tensor:
    """CPU fallback for iso_decompress using PyTorch operations."""
    N, H, D_padded = indices.shape
    n_groups = q_L.shape[0]
    M = N * H
    has_qr = (q_R is not None) and (mode == "full")

    flat_idx = indices.reshape(M, D_padded).long()
    norms_flat = norms.reshape(M).float()

    # Gather centroids
    values = centroids[flat_idx]  # (M, D_padded)

    # Reshape to 4D groups
    groups = values.reshape(M, n_groups, 4)  # (M, n_groups, 4)

    # Inverse rotation: conj(q_L) * v
    qL = q_L.float()
    qL_conj = qL.clone()
    qL_conj[:, 1:] *= -1

    cw, cx, cy, cz = qL_conj.unbind(-1)
    v0, v1, v2, v3 = groups.unbind(-1)

    t_w = cw * v0 - cx * v1 - cy * v2 - cz * v3
    t_x = cw * v1 + cx * v0 + cy * v3 - cz * v2
    t_y = cw * v2 - cx * v3 + cy * v0 + cz * v1
    t_z = cw * v3 + cx * v2 - cy * v1 + cz * v0

    if has_qr:
        qR = q_R.float()
        bw, bx, by, bz = qR.unbind(-1)
        # temp * q_R
        f0 = t_w * bw - t_x * bx - t_y * by - t_z * bz
        f1 = t_w * bx + t_x * bw + t_y * bz - t_z * by
        f2 = t_w * by - t_x * bz + t_y * bw + t_z * bx
        f3 = t_w * bz + t_x * by - t_y * bx + t_z * bw
    else:
        f0, f1, f2, f3 = t_w, t_x, t_y, t_z

    result = torch.stack([f0, f1, f2, f3], dim=-1).reshape(M, D_padded)

    # Rescale by norms
    result = result * norms_flat.unsqueeze(-1)

    return result.reshape(N, H, D_padded).to(dtype)
