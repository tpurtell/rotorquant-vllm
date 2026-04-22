"""Fused paged RotorQuant INT8 prefill attention -- IMMA tensor core path.

Reads RotorQuant-compressed blocks directly from vLLM's paged block table,
decompresses in SRAM, re-quantizes Q and K to INT8 per-tile, and computes
Q@K^T via IMMA tensor cores (``mma.sync.aligned.m16n8k32.s8.s8.s32``).
P@V accumulation stays in FP16/BF16 HMMA.

Designed for **prefill** (BLOCK_M=64, compute-bound) where INT8 tensor
cores provide 1.3-2x speedup over FP16.  Decode uses the separate
FP16 path (memory-bound where INT8 provides no benefit).

The kernel operates in **rotated space**: caller pre-rotates Q by
``R^T`` and post-rotates the output by ``R``.

Autotune: 4 configs (BLOCK_N in {16, 32} x stages=1 x warps {4, 8}).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune configs — BLOCK_M=64 (constexpr), BLOCK_N in {16, 32}
# ---------------------------------------------------------------------------

_FUSED_INT8_PREFILL_CONFIGS = [
    triton.Config({"BLOCK_N": BN}, num_stages=1, num_warps=w)
    for BN in [16, 32]
    for w in [4, 8]
]


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.autotune(configs=_FUSED_INT8_PREFILL_CONFIGS, key=["HEAD_DIM"])
@triton.jit
def _fused_paged_rq_int8_prefill_kernel(
    # ── Queries (pre-rotated by R^T) ──
    Q_rot,
    # ── Compressed KV cache (paged) ──
    KV_cache,
    # ── Page table and sequence metadata ──
    Block_table,
    Seq_lens,
    # ── RotorQuant codebooks (separate K and V centroids) ──
    K_Centroids,
    V_Centroids,
    # ── Output ──
    Out,
    # ── Stride parameters ──
    stride_qz,
    stride_qh,
    stride_qk,
    stride_cache_block,
    stride_cache_token,
    stride_bt_seq,
    stride_bt_block,
    stride_oz,
    stride_oh,
    stride_ok,
    # ── Compile-time constants ──
    sm_scale,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HALF_D: tl.constexpr,
    K_NORM_OFFSET: tl.constexpr,
    V_IDX_OFFSET: tl.constexpr,
    V_NORM_OFFSET: tl.constexpr,
    NUM_TOKENS,
    # ── Tiling (autotuned) ──
    BLOCK_N: tl.constexpr = 32,
    BLOCK_M: tl.constexpr = 64,
):
    """Fused paged RotorQuant INT8 prefill attention kernel.

    One program per (Q-tile, query head).  Loads BLOCK_M=64 queries,
    loops over all KV tiles with in-tile nibble decompression + INT8
    re-quantization, and computes attention with online softmax.

    ``NUM_TOKENS`` is a runtime parameter to avoid per-sequence-length
    recompilation.
    """
    # Grid: (cdiv(num_tokens, BLOCK_M), H_Q)
    off_tile = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_h_kv = off_h_q // (H_Q // H_KV)

    # Query positions for this tile
    offs_m = off_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_d_half = tl.arange(0, HALF_D)

    # Load Q tile [BLOCK_M, HEAD_DIM]
    q_base = Q_rot + off_h_q * stride_qh
    q_mask = offs_m < NUM_TOKENS
    q_tile = tl.load(
        q_base + offs_m[:, None] * stride_qz + offs_d[None, :] * stride_qk,
        mask=q_mask[:, None],
        other=0.0,
    )

    # INT8 Q quantization (per Q-tile, once in outer loop)
    q_abs_max = tl.max(tl.abs(q_tile)) + 1e-10
    q_scale_val = q_abs_max / 127.0
    q_int8 = (q_tile / q_scale_val + 0.5).to(tl.int8)

    # Sequence length
    seq_len = tl.load(Seq_lens)

    # fp32 online softmax state — per row (BLOCK_M rows)
    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], value=1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale: tl.constexpr = sm_scale * 1.44269504

    # === Main KV tile loop ===
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_valid = offs_n < seq_len

        # -- Block table lookup --
        logical_block = offs_n // BLOCK_SIZE
        within_block = offs_n % BLOCK_SIZE
        physical_block = tl.load(
            Block_table + logical_block * stride_bt_block,
            mask=kv_valid,
            other=0,
        )

        # -- Compute base byte address per token --
        token_base = (
            physical_block * stride_cache_block
            + within_block * stride_cache_token
        )

        # -- K decompression (nibble-packed → fp16/bf16) --
        k_idx_addr = (
            KV_cache
            + token_base[:, None]
            + off_h_kv * HALF_D
            + offs_d_half[None, :]
        )
        k_packed = tl.load(k_idx_addr, mask=kv_valid[:, None], other=0)

        k_hi = (k_packed >> 4).to(tl.int32)
        k_lo = (k_packed & 0x0F).to(tl.int32)

        k_fp16 = tl.join(
            tl.load(K_Centroids + k_hi), tl.load(K_Centroids + k_lo)
        ).reshape(BLOCK_N, HEAD_DIM)

        # K norms (fp32 bitcast from 4 uint8 bytes)
        k_norm_byte_addr = KV_cache + token_base + K_NORM_OFFSET + off_h_kv * 4
        b0 = tl.load(k_norm_byte_addr, mask=kv_valid, other=0).to(tl.int32)
        b1 = tl.load(k_norm_byte_addr + 1, mask=kv_valid, other=0).to(tl.int32)
        b2 = tl.load(k_norm_byte_addr + 2, mask=kv_valid, other=0).to(tl.int32)
        b3 = tl.load(k_norm_byte_addr + 3, mask=kv_valid, other=0).to(tl.int32)
        k_norm_bits = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        k_norms = k_norm_bits.to(tl.float32, bitcast=True)

        k_fp16 = (k_fp16 * k_norms[:, None]).to(Q_rot.dtype.element_ty)

        # -- Q @ K^T via IMMA tensor cores --
        # INT8 K re-quantization (per K-tile)
        k_abs_max = tl.max(tl.abs(k_fp16)) + 1e-10
        k_scale_val = k_abs_max / 127.0
        k_int8 = (k_fp16 / k_scale_val + 0.5).to(tl.int8)

        # INT8 matmul → IMMA tensor cores
        score_int32 = tl.dot(q_int8, tl.trans(k_int8))
        qk = score_int32.to(tl.float32) * (q_scale_val * k_scale_val)

        qk = qk * qk_scale

        # Causal masking: query at position m can only attend to keys at n <= m
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        combined_mask = causal_mask & kv_valid[None, :] & q_mask[:, None]
        qk = tl.where(combined_mask, qk, float("-inf"))

        # -- Online softmax update (per row) --
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        acc = acc * alpha[:, None]
        l_ij = tl.sum(p, 1)

        # -- V decompression (nibble-packed → fp16/bf16) --
        v_idx_addr = (
            KV_cache
            + token_base[:, None]
            + V_IDX_OFFSET
            + off_h_kv * HALF_D
            + offs_d_half[None, :]
        )
        v_packed = tl.load(v_idx_addr, mask=kv_valid[:, None], other=0)

        v_hi = (v_packed >> 4).to(tl.int32)
        v_lo = (v_packed & 0x0F).to(tl.int32)
        v = tl.join(
            tl.load(V_Centroids + v_hi), tl.load(V_Centroids + v_lo)
        ).reshape(BLOCK_N, HEAD_DIM)

        # V norms (fp32 bitcast from 4 uint8 bytes)
        v_norm_byte_addr = KV_cache + token_base + V_NORM_OFFSET + off_h_kv * 4
        vb0 = tl.load(v_norm_byte_addr, mask=kv_valid, other=0).to(tl.int32)
        vb1 = tl.load(v_norm_byte_addr + 1, mask=kv_valid, other=0).to(tl.int32)
        vb2 = tl.load(v_norm_byte_addr + 2, mask=kv_valid, other=0).to(tl.int32)
        vb3 = tl.load(v_norm_byte_addr + 3, mask=kv_valid, other=0).to(tl.int32)
        v_norm_bits = vb0 | (vb1 << 8) | (vb2 << 16) | (vb3 << 24)
        v_norms = v_norm_bits.to(tl.float32, bitcast=True)

        v = (v * v_norms[:, None]).to(Q_rot.dtype.element_ty)

        # -- P @ V accumulation (FP16/BF16 HMMA) --
        acc += tl.dot(p.to(v.dtype), v).to(tl.float32)

        l_i = l_i * alpha + l_ij
        m_i = m_new

    # Epilogue: normalize and store
    acc = acc / l_i[:, None]
    o_base = Out + off_h_q * stride_oh
    tl.store(
        o_base + offs_m[:, None] * stride_qz + offs_d[None, :] * stride_ok,
        acc.to(Q_rot.dtype.element_ty),
        mask=q_mask[:, None],
    )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def fused_paged_rq_int8_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    k_centroids: torch.Tensor,
    v_centroids: torch.Tensor,
    rotation: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    sm_scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused paged RotorQuant INT8 prefill attention.

    Pre-rotates Q by ``rotation^T``, launches the INT8 prefill kernel
    that decompresses RotorQuant blocks in-tile and uses IMMA tensor
    cores for Q@K^T, then post-rotates the output to return to
    original space.

    Designed for prefill (multiple queries per sequence, compute-bound).
    For decode (single query, memory-bound), use the decompress +
    Flash Attention path instead.

    Args:
        q: Query ``[num_tokens, H_Q, head_dim]`` fp16/bf16.
        kv_cache: Packed paged cache ``[num_blocks, block_size, total_bytes]``
            uint8.
        block_table: Page table ``[max_num_blocks_per_seq]`` int32.
            Must represent exactly one sequence (single-sequence kernel).
        seq_lens: Sequence lengths ``[1]`` int32.  Must have exactly one
            entry (single-sequence kernel).
        k_centroids: K codebook ``[16]`` fp32.
        v_centroids: V codebook ``[16]`` fp32.
        rotation: Orthogonal rotation ``[head_dim, head_dim]`` fp32.
        num_kv_heads: Number of KV heads.
        head_dim: Head dimension (e.g. 128).
        block_size: vLLM page size (tokens per block).
        sm_scale: Softmax scale.  Defaults to ``1 / sqrt(head_dim)``.
        out: Optional pre-allocated output ``[num_tokens, H_Q, head_dim]``.

    Returns:
        Attention output ``[num_tokens, H_Q, head_dim]`` in original space.

    Raises:
        ValueError: If ``seq_lens`` contains more than one sequence
            (kernel hardcodes single-sequence access).
    """
    num_tokens, H_Q, D = q.shape

    assert D == head_dim
    assert H_Q % num_kv_heads == 0
    assert kv_cache.dtype == torch.uint8
    assert block_table.dtype == torch.int32
    assert seq_lens.dtype == torch.int32

    # Kernel hardcodes single-sequence access; reject multi-sequence inputs.
    if seq_lens.numel() != 1:
        raise ValueError(
            f"fused_paged_rq_int8_prefill supports only a single sequence, "
            f"but got seq_lens.numel() == {seq_lens.numel()}."
        )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    half_D = head_dim // 2

    # Byte layout constexprs (same as decompress path)
    k_norm_offset = num_kv_heads * half_D
    v_idx_offset = k_norm_offset + num_kv_heads * 4
    v_norm_offset = v_idx_offset + num_kv_heads * half_D

    # Pre-rotate Q by R^T
    q_rot = torch.matmul(q.float(), rotation.T).to(q.dtype)

    out_rot = torch.empty_like(q) if out is None else out

    grid = (triton.cdiv(num_tokens, 64), H_Q)

    _fused_paged_rq_int8_prefill_kernel[grid](
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        k_centroids,
        v_centroids,
        out_rot,
        q_rot.stride(0),
        q_rot.stride(1),
        q_rot.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        out_rot.stride(0),
        out_rot.stride(1),
        out_rot.stride(2),
        sm_scale=sm_scale,
        H_Q=H_Q,
        H_KV=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        HALF_D=half_D,
        K_NORM_OFFSET=k_norm_offset,
        V_IDX_OFFSET=v_idx_offset,
        V_NORM_OFFSET=v_norm_offset,
        NUM_TOKENS=num_tokens,
    )

    # Post-rotate: convert from rotated space back to original space
    result = torch.matmul(out_rot.float(), rotation).to(q.dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result
