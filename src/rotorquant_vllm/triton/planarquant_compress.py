"""
Triton kernels for PlanarQuant compress: Givens 2D rotation + Lloyd-Max quantization + nibble packing.

Compress path: load X → compute norm → normalize → Givens rotate pairs → find nearest centroids
→ pack 2 indices into 1 byte (4-bit nibbles).
Outputs: uint8 nibble-packed indices (M, HALF_D) and fp32 norms (M,).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _planar_compress_kernel(
    x_ptr,
    rot2_ptr,
    centroids_ptr,
    packed_ptr,
    norms_ptr,
    M,
    D,
    D_padded,
    n_groups: tl.constexpr,
    n_levels: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compress kernel: one program per row, processes D_padded elements in pairs of 2.

    Steps per row:
      1. Load D elements (padded to D_padded), compute norm, normalize
      2. For each 2D pair, apply Givens rotation: r0=c*v0-s*v1, r1=s*v0+c*v1
      3. Find nearest centroid index for each rotated coordinate
      4. Pack 2 indices into 1 byte: packed[g] = (best_i0 << 4) | best_i1
      5. Store norm as fp32
    """
    pid = tl.program_id(0)

    # ── Load input and compute norm ──────────────────────────────────────
    d_offs = tl.arange(0, BLOCK_D)
    x_ptrs = x_ptr + pid * D_padded + d_offs
    x = tl.load(x_ptrs, mask=d_offs < D, other=0.0).to(tl.float32)

    norm_sq = tl.sum(x * x)
    norm = tl.sqrt(norm_sq)
    norm = tl.maximum(norm, 1e-8)
    inv_norm = 1.0 / norm

    # Store norm for this row
    tl.store(norms_ptr + pid, norm)

    # Load first centroid
    c0 = tl.load(centroids_ptr).to(tl.float32)

    # ── Rotate and quantize each 2D pair ─────────────────────────────────
    for g in range(n_groups):
        g_base = g * 2

        # Load rotation parameters [cos, sin]
        cos_t = tl.load(rot2_ptr + g * 2 + 0)
        sin_t = tl.load(rot2_ptr + g * 2 + 1)

        # Load vector pair, normalize
        v0 = tl.load(x_ptr + pid * D_padded + g_base + 0, mask=g_base < D, other=0.0) * inv_norm
        v1 = tl.load(x_ptr + pid * D_padded + g_base + 1, mask=(g_base + 1) < D, other=0.0) * inv_norm

        # Forward Givens rotation
        r0 = cos_t * v0 - sin_t * v1
        r1 = sin_t * v0 + cos_t * v1

        # ── Quantize: find nearest centroid index ────────────────────────
        best_i0 = tl.zeros((1,), dtype=tl.int32) + 0
        best_i1 = tl.zeros((1,), dtype=tl.int32) + 0

        best_d0 = tl.abs(r0 - c0)
        best_d1 = tl.abs(r1 - c0)

        for i in tl.static_range(1, n_levels):
            ci = tl.load(centroids_ptr + i).to(tl.float32)
            d0 = tl.abs(r0 - ci)
            d1 = tl.abs(r1 - ci)

            m0 = d0 < best_d0
            m1 = d1 < best_d1

            best_d0 = tl.where(m0, d0, best_d0)
            best_d1 = tl.where(m1, d1, best_d1)
            best_i0 = tl.where(m0, i, best_i0)
            best_i1 = tl.where(m1, i, best_i1)

        # ── Pack 2 indices into 1 byte (nibble packing) ──────────────────
        packed_val = (best_i0.to(tl.int32) << 4) | best_i1.to(tl.int32)
        tl.store(packed_ptr + pid * (D_padded // 2) + g,
                 packed_val.to(tl.uint8), mask=g_base < D)


def planar_compress(x: torch.Tensor,
                    rot2: torch.Tensor,
                    centroids: torch.Tensor,
                    d_padded: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress (N, H, D) float tensor to nibble-packed uint8 indices + fp32 norms via PlanarQuant.

    Args:
        x: Input tensor of shape (N, H, D), dtype fp16 or bf16.
        rot2: Rotation parameters of shape (n_groups, 2), dtype fp32. [cos, sin] per pair.
        centroids: Lloyd-Max centroids of shape (n_levels,), dtype fp32.
        d_padded: Padded dimension (multiple of 2). Auto-computed from D if None.

    Returns:
        packed: (N, H, HALF_D) uint8 tensor of nibble-packed centroid indices.
        norms:  (N, H, 1) fp32 tensor of vector norms.
    """
    N, H, D = x.shape
    n_levels = centroids.shape[0]
    n_groups = rot2.shape[0]

    if d_padded is None:
        d_padded = ((D + 1) // 2) * 2

    HALF_D = d_padded // 2
    M = N * H

    # Ensure contiguous layout and convert to fp32
    x_f32 = x.float().contiguous()

    # Pad to d_padded if needed
    if d_padded > D:
        x_f32 = torch.nn.functional.pad(x_f32, (0, d_padded - D))

    x_f32 = x_f32.reshape(M, d_padded)

    rot2_f32 = rot2.float().contiguous()
    centroids_f32 = centroids.float().contiguous()

    packed = torch.empty(M, HALF_D, device=x.device, dtype=torch.uint8)
    norms = torch.empty(M, device=x.device, dtype=torch.float32)

    BLOCK_D = min(triton.next_power_of_2(d_padded), 128)

    try:
        _planar_compress_kernel[(M,)](
            x_f32,
            rot2_f32,
            centroids_f32,
            packed,
            norms,
            M,
            D,
            d_padded,
            n_groups,
            n_levels,
            BLOCK_D=BLOCK_D,
        )
    except Exception:
        # CPU fallback
        return _planar_compress_cpu(x, rot2, centroids, d_padded)

    return packed.reshape(N, H, HALF_D), norms.reshape(N, H, 1)


def _planar_compress_cpu(x: torch.Tensor,
                         rot2: torch.Tensor,
                         centroids: torch.Tensor,
                         d_padded: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU fallback for planar_compress using PyTorch operations."""
    N, H, D = x.shape
    n_groups = rot2.shape[0]

    if d_padded is None:
        d_padded = ((D + 1) // 2) * 2

    HALF_D = d_padded // 2
    M = N * H

    x_f32 = x.float().contiguous()

    # Normalize on original D dimensions (before padding)
    flat = x_f32.reshape(M, D)
    norms = torch.norm(flat, dim=-1, keepdim=False).clamp(min=1e-8)  # (M,)
    flat_norm = flat / norms.unsqueeze(-1)

    # Pad normalized vectors to d_padded
    if d_padded > D:
        flat_norm = torch.nn.functional.pad(flat_norm, (0, d_padded - D))

    # Reshape into 2D pairs
    pairs = flat_norm.reshape(M, n_groups, 2)  # (M, n_groups, 2)

    # Forward Givens rotation
    cos_t = rot2[:, 0]  # (n_groups,)
    sin_t = rot2[:, 1]
    v0 = pairs[..., 0]  # (M, n_groups)
    v1 = pairs[..., 1]

    r0 = cos_t * v0 - sin_t * v1
    r1 = sin_t * v0 + cos_t * v1
    rotated = torch.stack([r0, r1], dim=-1)

    # Find nearest centroid
    rotated_flat = rotated.reshape(M, d_padded)
    diffs = (rotated_flat.unsqueeze(-1) - centroids).abs()
    indices = diffs.argmin(dim=-1)  # (M, d_padded)

    # Nibble pack: 2 indices per byte
    idx_u8 = indices.to(torch.uint8)
    packed = (idx_u8[:, 0::2] << 4) | idx_u8[:, 1::2]  # (M, HALF_D)

    return packed.reshape(N, H, HALF_D), norms.reshape(N, H, 1)
