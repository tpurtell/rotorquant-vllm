"""
Triton kernels for IsoQuant compress: quaternion 4D rotation + Lloyd-Max quantization.

Compress path: load X -> compute norm -> normalize -> quaternion rotate -> find nearest centroids.
Outputs: nibble-packed uint8 indices (M, HALF_D) and fp32 norms (M,).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _iso_compress_kernel(
    x_ptr,
    ql_ptr,
    qr_ptr,
    centroids_ptr,
    packed_ptr,
    norms_ptr,
    M,
    D,
    D_padded,
    HALF_D,
    n_groups: tl.constexpr,
    n_levels: tl.constexpr,
    has_qr: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compress kernel: one program per row, processes D_padded elements in groups of 4.

    Steps per row:
      1. Load D elements (padded to D_padded), compute norm, normalize
      2. For each 4D group, apply quaternion rotation (full or fast mode)
      3. Find nearest centroid index for each rotated coordinate
      4. Pack 4 indices into 2 nibble-bytes, store as uint8
    """
    pid = tl.program_id(0)

    # ── Load input and compute norm ──────────────────────────────────────
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    x_ptrs = x_ptr + pid * D_padded + d_offs
    x = tl.load(x_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    # L2 norm via manual sum
    norm_sq = tl.sum(x * x)
    norm = tl.sqrt(norm_sq)
    norm = tl.maximum(norm, 1e-8)
    inv_norm = 1.0 / norm

    # Store norm for this row
    tl.store(norms_ptr + pid, norm)

    # ── Rotate and quantize each 4D group ────────────────────────────────
    for g in range(n_groups):
        g_base = g * 4

        # Load quaternion q_L (normalized unit quaternion, [w, x, y, z])
        qL_w = tl.load(ql_ptr + g * 4 + 0)
        qL_x = tl.load(ql_ptr + g * 4 + 1)
        qL_y = tl.load(ql_ptr + g * 4 + 2)
        qL_z = tl.load(ql_ptr + g * 4 + 3)

        # Load 4D vector block (normalize inline)
        v0 = tl.load(x_ptr + pid * D_padded + g_base + 0, mask=(g_base + 0) < D, other=0.0) * inv_norm
        v1 = tl.load(x_ptr + pid * D_padded + g_base + 1, mask=(g_base + 1) < D, other=0.0) * inv_norm
        v2 = tl.load(x_ptr + pid * D_padded + g_base + 2, mask=(g_base + 2) < D, other=0.0) * inv_norm
        v3 = tl.load(x_ptr + pid * D_padded + g_base + 3, mask=(g_base + 3) < D, other=0.0) * inv_norm

        # Forward rotation: temp = q_L * v
        t_w = qL_w * v0 - qL_x * v1 - qL_y * v2 - qL_z * v3
        t_x = qL_w * v1 + qL_x * v0 + qL_y * v3 - qL_z * v2
        t_y = qL_w * v2 - qL_x * v3 + qL_y * v0 + qL_z * v1
        t_z = qL_w * v3 + qL_x * v2 - qL_y * v1 + qL_z * v0

        if has_qr:
            # Full mode: rotated = temp * conj(q_R)
            qR_w = tl.load(qr_ptr + g * 4 + 0)
            qR_x = tl.load(qr_ptr + g * 4 + 1)
            qR_y = tl.load(qr_ptr + g * 4 + 2)
            qR_z = tl.load(qr_ptr + g * 4 + 3)

            r0 = t_w * qR_w + t_x * qR_x + t_y * qR_y + t_z * qR_z
            r1 = t_x * qR_w - t_y * qR_z + t_z * qR_y - t_w * qR_x
            r2 = t_y * qR_w + t_x * qR_z - t_z * qR_x - t_w * qR_y
            r3 = t_z * qR_w - t_x * qR_y + t_y * qR_x - t_w * qR_z
        else:
            # Fast mode: rotated = q_L * v (single multiply)
            r0, r1, r2, r3 = t_w, t_x, t_y, t_z

        # ── Quantize: find nearest centroid index for each component ─────
        # Scalar best indices (tl.where on scalar + block yields scalar when both branches scalar)
        best_i0 = tl.zeros((1,), dtype=tl.int32)
        best_i1 = tl.zeros((1,), dtype=tl.int32)
        best_i2 = tl.zeros((1,), dtype=tl.int32)
        best_i3 = tl.zeros((1,), dtype=tl.int32)

        c0 = tl.load(centroids_ptr).to(tl.float32)
        best_d0 = tl.abs(r0 - c0)
        best_d1 = tl.abs(r1 - c0)
        best_d2 = tl.abs(r2 - c0)
        best_d3 = tl.abs(r3 - c0)

        for i in tl.static_range(1, n_levels):
            ci = tl.load(centroids_ptr + i).to(tl.float32)
            d0 = tl.abs(r0 - ci)
            d1 = tl.abs(r1 - ci)
            d2 = tl.abs(r2 - ci)
            d3 = tl.abs(r3 - ci)

            m0 = d0 < best_d0
            m1 = d1 < best_d1
            m2 = d2 < best_d2
            m3 = d3 < best_d3

            best_d0 = tl.where(m0, d0, best_d0)
            best_d1 = tl.where(m1, d1, best_d1)
            best_d2 = tl.where(m2, d2, best_d2)
            best_d3 = tl.where(m3, d3, best_d3)
            best_i0 = tl.where(m0, i, best_i0)
            best_i1 = tl.where(m1, i, best_i1)
            best_i2 = tl.where(m2, i, best_i2)
            best_i3 = tl.where(m3, i, best_i3)

        # ── Nibble pack: pair (idx0,idx1) and (idx2,idx3) into 2 bytes ──
        # High nibble = even index, low nibble = odd index
        # best_i* are 1-element blocks; << and | produce 1-element blocks
        # We store them to 1-element block pointers
        packed_pair0 = (best_i0.to(tl.uint8) << 4) | best_i1.to(tl.uint8)
        packed_pair1 = (best_i2.to(tl.uint8) << 4) | best_i3.to(tl.uint8)

        packed_base = pid * HALF_D + g * 2
        # Store as 1-element blocks using tl.arange(0, 1) offsets
        off0 = tl.arange(0, 1)
        tl.store(packed_ptr + packed_base + off0,
                 packed_pair0, mask=(g * 2 + 0) < HALF_D)
        tl.store(packed_ptr + packed_base + 1 + off0,
                 packed_pair1, mask=(g * 2 + 1) < HALF_D)


def iso_compress(x: torch.Tensor,
                 q_L: torch.Tensor,
                 q_R: torch.Tensor | None,
                 centroids: torch.Tensor,
                 mode: str = "fast",
                 d_padded: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress (N, H, D) float tensor to nibble-packed uint8 + fp32 norms via IsoQuant.

    Args:
        x: Input tensor of shape (N, H, D), dtype fp16 or bf16.
        q_L: Left quaternions of shape (n_groups, 4), dtype fp32.
        q_R: Right quaternions of shape (n_groups, 4), dtype fp32. None for fast mode.
        centroids: Lloyd-Max centroids of shape (n_levels,), dtype fp32.
        mode: 'full' (q_L * v * conj(q_R)) or 'fast' (q_L * v).
        d_padded: Padded dimension (multiple of 4). Auto-computed from D if None.

    Returns:
        packed: (N, H, HALF_D) uint8 tensor of nibble-packed centroid indices.
        norms:  (N, H, 1) fp32 tensor of vector norms.
    """
    N, H, D = x.shape
    n_levels = centroids.shape[0]
    n_groups = q_L.shape[0]

    if d_padded is None:
        d_padded = ((D + 3) // 4) * 4
    HALF_D = d_padded // 2

    M = N * H
    has_qr = (q_R is not None) and (mode == "full")

    # Ensure contiguous layout and convert to fp32
    x_f32 = x.float().contiguous()

    # Pad to d_padded if needed
    if d_padded > D:
        x_f32 = torch.nn.functional.pad(x_f32, (0, d_padded - D))

    x_f32 = x_f32.reshape(M, d_padded)

    ql_f32 = q_L.float().contiguous()
    if has_qr:
        qr_f32 = q_R.float().contiguous()
    else:
        qr_f32 = torch.ones(n_groups, 4, device=q_L.device, dtype=torch.float32)
    centroids_f32 = centroids.float().contiguous()

    packed = torch.empty(M, HALF_D, device=x.device, dtype=torch.uint8)
    norms = torch.empty(M, device=x.device, dtype=torch.float32)

    BLOCK_D = min(triton.next_power_of_2(d_padded), 128)

    try:
        _iso_compress_kernel[(M,)](
            x_f32,
            ql_f32,
            qr_f32,
            centroids_f32,
            packed,
            norms,
            M,
            D,
            d_padded,
            HALF_D,
            n_groups,
            n_levels,
            has_qr,
            BLOCK_D=BLOCK_D,
        )
    except Exception:
        return _iso_compress_cpu(x, q_L, q_R, centroids, mode, d_padded)

    return packed.reshape(N, H, HALF_D), norms.reshape(N, H, 1)


def _iso_compress_cpu(x: torch.Tensor,
                      q_L: torch.Tensor,
                      q_R: torch.Tensor | None,
                      centroids: torch.Tensor,
                      mode: str = "fast",
                      d_padded: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU fallback for iso_compress using PyTorch operations."""
    N, H, D = x.shape
    n_groups = q_L.shape[0]

    if d_padded is None:
        d_padded = ((D + 3) // 4) * 4
    HALF_D = d_padded // 2

    M = N * H
    has_qr = (q_R is not None) and (mode == "full")

    x_f32 = x.float().contiguous()

    # Normalize on original D dimensions (before padding)
    flat = x_f32.reshape(M, D)
    norms = torch.norm(flat, dim=-1, keepdim=False).clamp(min=1e-8)  # (M,)
    flat_norm = flat / norms.unsqueeze(-1)

    # Pad normalized vectors to d_padded
    if d_padded > D:
        flat_norm = torch.nn.functional.pad(flat_norm, (0, d_padded - D))

    # Reshape into 4D groups
    groups = flat_norm.reshape(M, n_groups, 4)  # (M, n_groups, 4)

    # Forward rotation
    qL = q_L.float()  # (n_groups, 4)
    aw, ax, ay, az = qL.unbind(-1)
    v0, v1, v2, v3 = groups.unbind(-1)

    t_w = aw * v0 - ax * v1 - ay * v2 - az * v3
    t_x = aw * v1 + ax * v0 + ay * v3 - az * v2
    t_y = aw * v2 - ax * v3 + ay * v0 + az * v1
    t_z = aw * v3 + ax * v2 - ay * v1 + az * v0

    if has_qr:
        qR = q_R.float()
        bw, bx, by, bz = qR.unbind(-1)
        # temp * conj(q_R)
        r0 = t_w * bw + t_x * bx + t_y * by + t_z * bz
        r1 = t_x * bw - t_y * bz + t_z * by - t_w * bx
        r2 = t_y * bw + t_x * bz - t_z * bx - t_w * by
        r3 = t_z * bw - t_x * by + t_y * bx - t_w * bz
        rotated = torch.stack([r0, r1, r2, r3], dim=-1)
    else:
        rotated = torch.stack([t_w, t_x, t_y, t_z], dim=-1)

    # Find nearest centroid
    rotated_flat = rotated.reshape(M, d_padded)
    diffs = (rotated_flat.unsqueeze(-1) - centroids).abs()
    indices = diffs.argmin(dim=-1)  # (M, d_padded)

    # Nibble pack: (M, D_padded) -> (M, HALF_D)
    idx_u8 = indices.to(torch.uint8)
    packed = (idx_u8[:, 0::2] << 4) | idx_u8[:, 1::2]

    return packed.reshape(N, H, HALF_D), norms.reshape(N, H, 1)
