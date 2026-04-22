"""Fused paged RotorQuant decode attention kernel.

Combines nibble unpack, centroid gather, norm scale, and attention
in a single kernel pass.  Operates in **rotated space** — the caller
pre-rotates Q by R^T and post-rotates output by R.  No HBM writes of
decompressed K/V.

Key difference from turboquant: rotorquant nibble packing interleaves
(even=hi, odd=lo), so the kernel splits Q into even/odd halves and
accumulates separately rather than using tl.join.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune configs: BLOCK_N in {32, 64} x stages {2, 3} x warps {4, 8}
# ---------------------------------------------------------------------------
_FUSED_DECODE_CONFIGS = [
    triton.Config({"BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BN in [32, 64]
    for s in [2, 3]
    for w in [4, 8]
]


@triton.autotune(configs=_FUSED_DECODE_CONFIGS, key=["HEAD_DIM"])
@triton.jit
def _fused_paged_rq_decode_kernel(
    # ── Queries (pre-rotated by R^T) ──
    Q_rot,
    # ── Compressed KV cache (paged) ──
    KV_cache,
    # ── Page table and sequence metadata ──
    Block_table,
    Seq_lens,
    # ── Codebooks ──
    Centroids_K,
    Centroids_V,
    # ── Output (rotated space) ──
    Out,
    # ── Strides ──
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
    BLOCK_N: tl.constexpr = 32,
):
    """Fused paged RQ decode attention kernel.

    One program per (sequence, query head).  Tiles over sequence length,
    decompresses K/V from paged cache in-tile, computes Q@K^T with online
    softmax, accumulates P@V.  Output is in rotated space.
    """
    off_seq = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_h_kv = off_h_q // (H_Q // H_KV)

    seq_len = tl.load(Seq_lens + off_seq)

    # ── Even / odd head-dim offsets for nibble unpack ──────────────────
    offs_d_half = tl.arange(0, HALF_D)
    offs_d_even = offs_d_half * 2
    offs_d_odd = offs_d_half * 2 + 1

    # Load Q split into even/odd halves (HALF_D,) each
    q_base = Q_rot + off_seq * stride_qz + off_h_q * stride_qh
    q_even = tl.load(q_base + offs_d_even * stride_qk)
    q_odd = tl.load(q_base + offs_d_odd * stride_qk)

    # fp32 online softmax state
    m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([1], dtype=tl.float32) + 1.0
    acc_even = tl.zeros([HALF_D], dtype=tl.float32)
    acc_odd = tl.zeros([HALF_D], dtype=tl.float32)

    qk_scale: tl.constexpr = sm_scale * 1.44269504

    # ── Main KV tile loop ──────────────────────────────────────────────
    for start_n in range(0, seq_len, BLOCK_N):
        offs_t = start_n + tl.arange(0, BLOCK_N)
        kv_valid = offs_t < seq_len

        # -- Block table lookup --
        logical_block = offs_t // BLOCK_SIZE
        within_block = offs_t % BLOCK_SIZE
        physical_block = tl.load(
            Block_table
            + off_seq * stride_bt_seq
            + logical_block * stride_bt_block,
            mask=kv_valid,
            other=0,
        )

        # -- Token base byte address --
        token_base = (
            physical_block * stride_cache_block
            + within_block * stride_cache_token
        )

        # ── K decompression ────────────────────────────────────────────
        k_idx_addr = (
            KV_cache
            + token_base[:, None]
            + off_h_kv * HALF_D
            + offs_d_half[None, :]
        )
        k_packed = tl.load(k_idx_addr, mask=kv_valid[:, None], other=0)

        k_hi = (k_packed >> 4).to(tl.int32)
        k_lo = (k_packed & 0x0F).to(tl.int32)

        k_c_hi = tl.load(Centroids_K + k_hi).to(tl.float32)
        k_c_lo = tl.load(Centroids_K + k_lo).to(tl.float32)

        # K norms: 4 uint8 bytes → fp32 bitcast (little-endian)
        k_norm_addr = KV_cache + token_base + K_NORM_OFFSET + off_h_kv * 4
        kb0 = tl.load(k_norm_addr, mask=kv_valid, other=0).to(tl.int32)
        kb1 = tl.load(k_norm_addr + 1, mask=kv_valid, other=0).to(tl.int32)
        kb2 = tl.load(k_norm_addr + 2, mask=kv_valid, other=0).to(tl.int32)
        kb3 = tl.load(k_norm_addr + 3, mask=kv_valid, other=0).to(tl.int32)
        k_norm_bits = kb0 | (kb1 << 8) | (kb2 << 16) | (kb3 << 24)
        k_norms = k_norm_bits.to(tl.float32, bitcast=True)

        k_hi = (k_c_hi * k_norms[:, None]).to(Q_rot.dtype.element_ty)
        k_lo = (k_c_lo * k_norms[:, None]).to(Q_rot.dtype.element_ty)

        # Q @ K^T via even/odd split (avoids tl.join which is non-interleaved)
        qk = (
            tl.sum(q_even[None, :] * k_hi, axis=1)
            + tl.sum(q_odd[None, :] * k_lo, axis=1)
        ) * qk_scale
        qk = tl.where(kv_valid, qk, float("-inf"))

        # ── Online softmax update ──────────────────────────────────────
        m_ij = tl.max(qk, 0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new)
        acc_even = acc_even * alpha
        acc_odd = acc_odd * alpha
        l_ij = tl.sum(p, 0)

        # ── V decompression ────────────────────────────────────────────
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

        v_c_hi = tl.load(Centroids_V + v_hi).to(tl.float32)
        v_c_lo = tl.load(Centroids_V + v_lo).to(tl.float32)

        # V norms
        v_norm_addr = KV_cache + token_base + V_NORM_OFFSET + off_h_kv * 4
        vb0 = tl.load(v_norm_addr, mask=kv_valid, other=0).to(tl.int32)
        vb1 = tl.load(v_norm_addr + 1, mask=kv_valid, other=0).to(tl.int32)
        vb2 = tl.load(v_norm_addr + 2, mask=kv_valid, other=0).to(tl.int32)
        vb3 = tl.load(v_norm_addr + 3, mask=kv_valid, other=0).to(tl.int32)
        v_norm_bits = vb0 | (vb1 << 8) | (vb2 << 16) | (vb3 << 24)
        v_norms = v_norm_bits.to(tl.float32, bitcast=True)

        v_hi = (v_c_hi * v_norms[:, None]).to(Q_rot.dtype.element_ty)
        v_lo = (v_c_lo * v_norms[:, None]).to(Q_rot.dtype.element_ty)

        # P @ V accumulation via even/odd split
        acc_even += tl.sum(p[:, None] * v_hi, axis=0)
        acc_odd += tl.sum(p[:, None] * v_lo, axis=0)

        l_i = l_i * alpha + l_ij
        m_i = m_new

    # ── Epilogue: normalize and store (rotated space) ──────────────────
    acc_even = (acc_even / l_i).to(Q_rot.dtype.element_ty)
    acc_odd = (acc_odd / l_i).to(Q_rot.dtype.element_ty)

    o_base = Out + off_seq * stride_oz + off_h_q * stride_oh
    tl.store(o_base + offs_d_even * stride_ok, acc_even)
    tl.store(o_base + offs_d_odd * stride_ok, acc_odd)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def fused_paged_rq_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    centroids_k: torch.Tensor,
    centroids_v: torch.Tensor,
    rotation: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    sm_scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused paged RotorQuant decode attention.

    Pre-rotates Q by ``rotation^T``, launches the fused paged kernel that
    decompresses KV blocks in-tile from the page table, then post-rotates
    the output by ``rotation`` to return to original space.

    Args:
        q: Query ``(num_seqs, H_Q, head_dim)`` fp16/bf16.
        kv_cache: Packed paged cache ``(num_blocks, block_size, total_bytes)``
            uint8.
        block_table: Page table ``(num_seqs, max_blocks_per_seq)`` int32.
        seq_lens: Sequence lengths ``(num_seqs,)`` int32.
        centroids_k: K codebook ``(n_levels,)`` fp32.
        centroids_v: V codebook ``(n_levels,)`` fp32.
        rotation: Orthogonal rotation ``(head_dim, head_dim)`` fp32.
        num_kv_heads: Number of KV heads.
        head_dim: Head dimension (e.g. 128).
        block_size: vLLM page size (tokens per block).
        sm_scale: Softmax scale.  Defaults to ``1 / sqrt(head_dim)``.
        out: Optional pre-allocated output ``(num_seqs, H_Q, head_dim)``.

    Returns:
        Attention output ``(num_seqs, H_Q, head_dim)`` in original space.
    """
    num_seqs, H_Q, D = q.shape

    assert D == head_dim, f"D={D} != head_dim={head_dim}"
    assert H_Q % num_kv_heads == 0
    assert kv_cache.dtype == torch.uint8
    assert block_table.dtype == torch.int32
    assert seq_lens.dtype == torch.int32

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    half_D = head_dim // 2

    # Byte layout offsets (same as compressor)
    k_norm_offset = num_kv_heads * half_D
    v_idx_offset = k_norm_offset + num_kv_heads * 4
    v_norm_offset = v_idx_offset + num_kv_heads * half_D

    # Pre-rotate Q by R^T
    q_rot = torch.matmul(q.float(), rotation.T).to(q.dtype)

    # Scratch buffer for rotated-space output
    out_rot = torch.empty_like(q_rot)

    grid = (num_seqs, H_Q)

    _fused_paged_rq_decode_kernel[grid](
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids_k.float().contiguous(),
        centroids_v.float().contiguous(),
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
    )

    # Post-rotate: convert from rotated space back to original space
    result = torch.matmul(out_rot.float(), rotation).to(q.dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result
