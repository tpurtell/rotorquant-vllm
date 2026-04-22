"""RotorQuant compressed KV cache attention backend for vLLM.

Integrates RotorQuant's IsoQuant (4D quaternion) and PlanarQuant (2D Givens)
KV cache compression into vLLM as a drop-in plugin.  Stores compressed KV
cache as uint8 bytes with nibble-packed layout:

    [K_indices(all heads, nibble-packed) | K_norms(all heads) |
     V_indices(all heads, nibble-packed) | V_norms(all heads)]

Q pre-rotation + output post-rotation: rotation matrix applied once per step
to Q (forward) and attention output (inverse), avoiding O(cache_len) matmuls.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import TYPE_CHECKING

import torch
from vllm.v1.attention.backend import AttentionCGSupport, AttentionImplBase
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

from rotorquant_vllm.quantization.isoquant import IsoQuantMSE
from rotorquant_vllm.quantization.planarquant import PlanarQuantMSE
from rotorquant_vllm.triton.isoquant_compress import iso_compress
from rotorquant_vllm.triton.isoquant_decompress import iso_decompress
from rotorquant_vllm.triton.planarquant_compress import planar_compress
from rotorquant_vllm.triton.planarquant_decompress import planar_decompress

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadataBuilder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RQ_DEFAULT_BITS = 3
RQ_DEFAULT_MODE = "iso"  # 'iso' or 'planar'
RQ_SEED = 42
RQ_NORM_BYTES = 4  # fp32


# ---------------------------------------------------------------------------
# Env var parsing
# ---------------------------------------------------------------------------


def _parse_rq_mode() -> str:
    """Parse RQ_MODE env var: 'iso' or 'planar'."""
    mode = os.environ.get("RQ_MODE", RQ_DEFAULT_MODE).lower()
    if mode not in ("iso", "planar"):
        raise ValueError(
            f"RQ_MODE must be 'iso' or 'planar', got '{mode}'"
        )
    return mode


def _parse_rq_bits() -> int:
    """Parse RQ_BITS env var: 3 or 4."""
    raw = os.environ.get("RQ_BITS", str(RQ_DEFAULT_BITS))
    bits = int(raw)
    if bits not in (3, 4):
        raise ValueError(f"RQ_BITS must be 3 or 4, got '{bits}'")
    return bits


def _parse_kv_bits() -> tuple[int, int]:
    """Parse RQ_K_BITS and RQ_V_BITS env vars (asymmetric override).

    Returns:
        (k_bits, v_bits) — defaults to (RQ_BITS, RQ_BITS) when absent.
    """
    default_bits = _parse_rq_bits()
    k_raw = os.environ.get("RQ_K_BITS")
    v_raw = os.environ.get("RQ_V_BITS")

    k_bits = int(k_raw) if k_raw else default_bits
    v_bits = int(v_raw) if v_raw else default_bits

    if k_bits not in (3, 4):
        raise ValueError(f"RQ_K_BITS must be 3 or 4, got '{k_bits}'")
    if v_bits not in (3, 4):
        raise ValueError(f"RQ_V_BITS must be 3 or 4, got '{v_bits}'")

    return k_bits, v_bits


def _parse_iso_mode() -> str:
    """Parse RQ_ISO_MODE env var: 'fast' or 'full'."""
    mode = os.environ.get("RQ_ISO_MODE", "fast").lower()
    if mode not in ("fast", "full"):
        raise ValueError(
            f"RQ_ISO_MODE must be 'fast' or 'full', got '{mode}'"
        )
    return mode




# ---------------------------------------------------------------------------
# Fused paged decode feature gate
# ---------------------------------------------------------------------------

_fused_paged_kernel_available = False
_fused_paged_rq_decode_fn = None
try:
    from rotorquant_vllm.triton.fused_paged_rq_decode import (
        fused_paged_rq_decode as _fused_paged_rq_decode_fn,
    )

    _fused_paged_kernel_available = True
except (ImportError, RuntimeError) as exc:
    logger.info("Fused paged RQ decode kernel unavailable: %s", exc)


def _parse_rq_fused_paged_env() -> bool:
    """Parse ``RQ_USE_FUSED_PAGED`` environment variable.

    Returns:
        ``True`` when the env var is set to a truthy value
        (``"1"``, ``"true"``, ``"yes"``; case-insensitive).
        ``False`` for everything else including absent.
    """
    return os.environ.get("RQ_USE_FUSED_PAGED", "").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Cache size helpers — nibble-packed
# ---------------------------------------------------------------------------


def _rq_bytes_per_component(head_dim: int, num_kv_heads: int) -> int:
    """Bytes for one component (K or V): nibble-packed indices + norms.

    Nibble-packed: 2 indices per byte, so half_d = head_dim // 2 bytes per head.
    Norms: 4 bytes per head (fp32).
    """
    half_d = head_dim // 2
    return num_kv_heads * half_d + num_kv_heads * RQ_NORM_BYTES


def _rq_bytes_per_token_kv(head_dim: int, num_kv_heads: int) -> int:
    """Total bytes per token (K + V combined, all heads).

    For 8 heads, head_dim=128:
        8 * (64 + 4) * 2 = 1088 bytes vs 4096 FP16 K+V = ~3.76x compression
    """
    return _rq_bytes_per_component(head_dim, num_kv_heads) * 2


def _rq_padded_slot_bytes(head_dim: int) -> int:
    """Padded per-head slot (K+V combined) for hybrid model page alignment.

    Returns ``next_power_of_2(raw_slot)`` so RQ pages are divisible by
    Mamba layer pages in hybrid models (e.g. Qwen3.5).
    Padding bytes are unused by kernels.
    """
    from vllm.utils.math_utils import next_power_of_2

    return next_power_of_2(_rq_bytes_per_token_kv(head_dim, 1))


def _rq_total_bytes(head_dim: int, num_kv_heads: int) -> int:
    """Total bytes per token slot (padded per head * num heads)."""
    return num_kv_heads * _rq_padded_slot_bytes(head_dim)
# ---------------------------------------------------------------------------
# Rotation matrix builders
# ---------------------------------------------------------------------------


def _build_iso_R(quantizer, d: int, device, mode: str = 'fast') -> torch.Tensor:
    """Build rotation matrix from IsoQuant quaternion parameters.

    Returns D×D block-diagonal matrix R where R maps unit vectors to rotated space.
    """
    n_groups = quantizer.n_groups
    D_pad = quantizer.d_padded
    R = torch.eye(D_pad, dtype=torch.float32, device=device)
    for g in range(n_groups):
        base = g * 4
        qL = quantizer.q_L[g]  # (4,) [w,x,y,z]
        ql_w, ql_x, ql_y, ql_z = qL[0], qL[1], qL[2], qL[3]

        # Left multiply matrix (q_L * v):
        M_left = torch.tensor([
            [ql_w, -ql_x, -ql_y, -ql_z],
            [ql_x,  ql_w, -ql_z,  ql_y],
            [ql_y,  ql_z,  ql_w, -ql_x],
            [ql_z, -ql_y,  ql_x,  ql_w],
        ], dtype=torch.float32, device=device)

        if mode == 'full':
            qR = quantizer.q_R[g]
            qr_w, qr_x, qr_y, qr_z = qR[0], qR[1], qR[2], qR[3]
            # Right multiply by conj(q_R):
            M_right = torch.tensor([
                [qr_w,  qr_x,  qr_y,  qr_z],
                [-qr_x, qr_w, -qr_z,  qr_y],
                [-qr_y, qr_z,  qr_w, -qr_x],
                [-qr_z, -qr_y, qr_x,  qr_w],
            ], dtype=torch.float32, device=device)
            R[base:base + 4, base:base + 4] = M_left @ M_right
        else:
            # Fast mode: left mult by q_L, then right mult by conj(q_L)
            M_conj_L = torch.tensor([
                [ql_w,  ql_x,  ql_y,  ql_z],
                [-ql_x, ql_w, -ql_z,  ql_y],
                [-ql_y, ql_z,  ql_w, -ql_x],
                [-ql_z, -ql_y, ql_x,  ql_w],
            ], dtype=torch.float32, device=device)
            R[base:base + 4, base:base + 4] = M_left @ M_conj_L
    return R[:d, :d]


def _build_planar_R(quantizer, d: int, device) -> torch.Tensor:
    """Build rotation matrix from PlanarQuant Givens parameters.

    Returns D×D block-diagonal matrix R where R maps unit vectors to rotated space.
    """
    n_groups = quantizer.n_groups
    D_pad = quantizer.d_padded
    R = torch.eye(D_pad, dtype=torch.float32, device=device)
    for g in range(n_groups):
        base = g * 2
        c = quantizer.rot2[g, 0]
        s = quantizer.rot2[g, 1]
        R[base:base + 2, base:base + 2] = torch.tensor(
            [[c, -s], [s, c]], dtype=torch.float32, device=device
        )
    return R[:d, :d]


def _build_rotation_matrix(quantizer, d: int, device) -> torch.Tensor:
    """Build full D×D rotation matrix from quantizer block parameters."""
    if hasattr(quantizer, 'q_L'):
        # IsoQuantMSE
        mode = getattr(quantizer, 'mode', 'fast')
        return _build_iso_R(quantizer, d, device, mode)
    else:
        # PlanarQuantMSE
        return _build_planar_R(quantizer, d, device)


# ---------------------------------------------------------------------------
# KV cache spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class RQFullAttentionSpec(FullAttentionSpec):
    """KV cache spec with RotorQuant compressed page size.

    Overrides ``real_page_size_bytes`` so the block allocator provisions
    buffers sized for the packed RotorQuant format (nibble-packed indices + fp32 norms).
    Uses ``dtype=torch.uint8`` for the uint8 byte cache.
    """

    @property
    def real_page_size_bytes(self) -> int:  # noqa: D102
        return self.block_size * _rq_total_bytes(self.head_size, self.num_kv_heads)


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------


class RQMetadataBuilder(FlashAttentionMetadataBuilder):
    """Metadata builder for RotorQuant: no CUDA graph support (v0).

    Inherits all metadata-building logic from FlashAttentionMetadataBuilder.
    Only overrides CUDA graph support level.
    """

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: object,
        kv_cache_spec: object,
    ) -> AttentionCGSupport:
        """Report CUDA graph support: single-token decode when fused available.

        When fused paged decode is available and enabled, decode goes through
        the fused kernel path which is CG-safe.  Otherwise, decode uses
        dynamic decompress operations incompatible with CUDA graphs.
        """
        # Support CUDA graphs for single-token decode when fused paged kernel is available
        if _fused_paged_kernel_available and _parse_rq_fused_paged_env():
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        return AttentionCGSupport.NEVER



# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class RQAttentionBackend(FlashAttentionBackend):
    """RotorQuant compressed KV cache attention backend.

    Stores KV cache as packed uint8 bytes (nibble-packed indices + fp32 norms).
    Each forward() call decompresses cached K/V to FP16, runs Flash
    Attention, then returns the output with Q pre-rotation and output post-rotation.
    """

    forward_includes_kv_cache_update = True

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[AttentionImplBase]:
        return RQAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        return RQMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """Packed cache: ``(num_blocks, block_size, total_bytes)``.

        Last dimension packs K and V data for all heads as raw bytes::

            [K_indices(all heads, nibble-packed) | K_norms(all heads) |
             V_indices(all heads, nibble-packed) | V_norms(all heads)]
        """
        total_bytes = _rq_total_bytes(head_size, num_kv_heads)
        return (num_blocks, block_size, total_bytes)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """Raise to trigger identity fallback in reshape."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Attention implementation
# ---------------------------------------------------------------------------


class RQAttentionImpl(FlashAttentionImpl):
    """RotorQuant attention: compress -> store -> decompress -> Flash Attention.

    Stores nibble-packed bytes (4-bit indices + fp32 norms) in a uint8 cache.
    Q pre-rotation + output post-rotation avoids O(cache_len) inverse rotations.

    Each ``forward()`` call:

    1. Compresses incoming K/V tokens (nibble-packed, rotated space).
    2. Scatter-writes packed bytes to the uint8 cache.
    3. Pre-rotates Q by R^T (forward rotation into rotated space).
    4. Paged-decompresses cached K/V to compute dtype (no inverse rotation).
    5. Calls ``flash_attn_varlen_func`` with rotated Q and decompressed K/V.
    6. Post-rotates output by R (inverse rotation back to original space).
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize RotorQuant attention with compression primitives."""
        super().__init__(*args, **kwargs)

        head_size = self.head_size
        num_kv_heads = self.num_kv_heads

        # Parse env vars for mode and bits
        rq_mode = _parse_rq_mode()
        iso_mode = _parse_iso_mode()
        k_bits, v_bits = _parse_kv_bits()
        self._k_bits = k_bits
        self._v_bits = v_bits
        self._rq_mode = rq_mode

        # Determine target device
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        device = (
            vllm_config.device_config.device
            if vllm_config is not None
            else torch.device("cpu")
        )

        # Create quantizers
        if rq_mode == "iso":
            k_quantizer = IsoQuantMSE(
                head_size, k_bits, seed=RQ_SEED, mode=iso_mode, device=device
            )
            v_quantizer = (
                k_quantizer
                if v_bits == k_bits
                else IsoQuantMSE(
                    head_size, v_bits, seed=RQ_SEED, mode=iso_mode, device=device
                )
            )
        else:
            k_quantizer = PlanarQuantMSE(
                head_size, k_bits, seed=RQ_SEED, device=device
            )
            v_quantizer = (
                k_quantizer
                if v_bits == k_bits
                else PlanarQuantMSE(head_size, v_bits, seed=RQ_SEED, device=device)
            )

        # Ensure all quantizer buffers are on the target device
        self._k_quantizer = k_quantizer
        self._v_quantizer = v_quantizer

        for buf_name in ("centroids",):
            for quantizer in (k_quantizer, v_quantizer):
                buf = getattr(quantizer, buf_name, None)
                if buf is not None and buf.device != device:
                    buf.data = buf.to(device)

        # Mode-specific buffers
        if rq_mode == "iso":
            for buf_name in ("q_L",):
                for quantizer in (k_quantizer, v_quantizer):
                    buf = getattr(quantizer, buf_name, None)
                    if buf is not None and buf.device != device:
                        buf.data = buf.to(device)
            if iso_mode == "full":
                for buf_name in ("q_R",):
                    for quantizer in (k_quantizer, v_quantizer):
                        buf = getattr(quantizer, buf_name, None)
                        if buf is not None and buf.device != device:
                            buf.data = buf.to(device)
        else:
            for buf_name in ("rot2",):
                for quantizer in (k_quantizer, v_quantizer):
                    buf = getattr(quantizer, buf_name, None)
                    if buf is not None and buf.device != device:
                        buf.data = buf.to(device)

        # Build rotation matrix R (original -> rotated space)
        # Q pre-rotation: q_rot = q @ R^T
        # Output post-rotation: out_orig = out_rot @ R
        R = _build_rotation_matrix(k_quantizer, head_size, device)
        self._rq_rot = R
        self._rq_rot_T = R.T.contiguous()
        # Pre-split R^T into even/odd columns for tiled compress kernel
        self._rq_rot_T_even = self._rq_rot_T[:, 0::2].contiguous()  # (D, HALF_D) fp32
        self._rq_rot_T_odd = self._rq_rot_T[:, 1::2].contiguous()   # (D, HALF_D) fp32
        # Half dimension for nibble packing
        self._half_d = head_size // 2

        # Byte layout offsets (nibble-packed)
        k_idx_per_head = self._half_d
        self._k_idx_end = num_kv_heads * k_idx_per_head
        self._k_norm_end = self._k_idx_end + num_kv_heads * RQ_NORM_BYTES
        self._v_idx_end = self._k_norm_end + num_kv_heads * k_idx_per_head
        self._total_bytes = _rq_total_bytes(head_size, num_kv_heads)

        # Pre-allocated buffers flag and limits
        self._cg_buffers_ready = False
        self._max_model_len = (
            vllm_config.model_config.max_model_len if vllm_config is not None else 8192
        )
        self._max_prefill_len = (
            vllm_config.scheduler_config.max_num_batched_tokens
            if vllm_config is not None else 2048
        )


        # Fused paged decode feature gate.
        self._fused_paged_available = (
            _parse_rq_fused_paged_env() and _fused_paged_kernel_available
        )

        # Log compression ratio
        fp16_total = 2 * num_kv_heads * head_size * 2  # K+V in fp16
        compression = fp16_total / self._total_bytes
        logger.info(
            "RQAttentionImpl: %s mode, %d KV heads, head_size=%d, "
            "k_bits=%d, v_bits=%d, %d bytes/token (%.2fx compression vs FP16)",
            rq_mode,
            num_kv_heads,
            head_size,
            k_bits,
            v_bits,
            self._total_bytes,
            compression,
        )
        logger.info(
            "Fused paged RQ decode: %s",
            "enabled" if self._fused_paged_available else "disabled",
        )

    # ------------------------------------------------------------------
    # Pre-allocated scratch buffers
    # ------------------------------------------------------------------

    def _init_cg_buffers(self, kv_cache, compute_dtype):
        """Lazy-allocate scratch buffers bounded by max_model_len."""
        num_blocks, block_size, _ = kv_cache.shape
        max_tokens = min(self._max_model_len, num_blocks * block_size)
        device = kv_cache.device
        H = self.num_kv_heads
        D = self.head_size
        half_D = self._half_d

        # Decompress buffers (paged decompress output)
        self._cg_decompress_k = torch.empty(
            max_tokens, H, D, dtype=compute_dtype, device=device
        )
        self._cg_decompress_v = torch.empty_like(self._cg_decompress_k)

        # Prefill buffers
        prefill_tokens = min(self._max_prefill_len, max_tokens)
        self._cg_prefill_k = torch.empty(
            prefill_tokens, H, D, dtype=compute_dtype, device=device
        )
        self._cg_prefill_v = torch.empty_like(self._cg_prefill_k)
        self._max_prefill_blocks = prefill_tokens // block_size

        # Compress output buffers
        self._cg_compress_packed = torch.empty(
            1, H, half_D, dtype=torch.uint8, device=device
        )
        self._cg_compress_norms = torch.empty(
            1, H, 1, dtype=torch.float32, device=device
        )

        # Q rotation buffers
        self._cg_q_rot = torch.empty(
            1, self.num_heads, D, dtype=torch.float32, device=device
        )
        self._cg_q_rot_cast = torch.empty(
            1, self.num_heads, D, dtype=compute_dtype, device=device
        )

        # Compress row buffer
        self._cg_compress_row = torch.empty(
            1, self._total_bytes, dtype=torch.uint8, device=device
        )

        self._cg_buffers_ready = True

    # ------------------------------------------------------------------
    # Compress path — nibble-packed
    # ------------------------------------------------------------------

    def _compress_and_store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        compress_out=None,
        row_out=None,
    ) -> None:
        """Compress K/V to nibble-packed format and scatter-write to cache.

        Args:
            key: ``(N, H, D)`` new key tokens.
            value: ``(N, H, D)`` new value tokens.
            kv_cache: ``(NB, BS, total_bytes)`` uint8 packed cache.
            slot_mapping: ``(num_actual_tokens,)`` flat slot indices.
            compress_out: Optional (packed_buf, norms_buf) tuple for pre-allocated output.
            row_out: Optional pre-allocated row buffer ``(N, total_bytes)`` uint8.
        """
        N = key.shape[0]
        H = self.num_kv_heads
        num_actual = slot_mapping.shape[0]

        # Use pre-allocated buffer or create temporary
        row = row_out[:N] if row_out is not None else torch.empty(
            N, self._total_bytes, dtype=torch.uint8, device=key.device
        )

        # Compress K — tiled matmul with pre-split R^T columns
        if self._rq_mode == 'iso':
            k_packed, k_norms = iso_compress(
                key, self._rq_rot_T_even, self._rq_rot_T_odd,
                self._k_quantizer.boundaries,
            )
        else:
            k_packed, k_norms = planar_compress(
                key, self._rq_rot_T_even, self._rq_rot_T_odd,
                self._k_quantizer.boundaries,
            )

        # K indices: pack per-head nibble regions into contiguous bytes
        k_packed_flat = k_packed.reshape(N, H * self._half_d)
        row[:, :self._k_idx_end] = k_packed_flat
        # K norms: fp32 -> uint8 view
        row[:, self._k_idx_end:self._k_norm_end] = (
            k_norms.reshape(N, H).contiguous().view(torch.uint8)
        )

        # Compress V — tiled matmul with pre-split R^T columns
        if self._rq_mode == 'iso':
            v_packed, v_norms = iso_compress(
                value, self._rq_rot_T_even, self._rq_rot_T_odd,
                self._v_quantizer.boundaries,
            )
        else:
            v_packed, v_norms = planar_compress(
                value, self._rq_rot_T_even, self._rq_rot_T_odd,
                self._v_quantizer.boundaries,
            )

        # V indices: pack per-head nibble regions
        v_packed_flat = v_packed.reshape(N, H * self._half_d)
        row[:, self._k_norm_end:self._v_idx_end] = v_packed_flat
        # V norms: fp32 -> uint8 view
        row[:, self._v_idx_end:self._total_bytes] = (
            v_norms.reshape(N, H).contiguous().view(torch.uint8)
        )

        # Scatter-write to flat cache using slot_mapping
        flat_cache = kv_cache.view(-1, kv_cache.shape[-1])
        flat_cache[slot_mapping[:num_actual], :self._total_bytes] = row[:num_actual]

    # ------------------------------------------------------------------
    # Decompress path — nibble unpack, no rotation
    # ------------------------------------------------------------------

    def _decompress_cache(
        self,
        kv_cache: torch.Tensor,
        compute_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress the full packed cache to compute dtype (rotated space).

        No inverse rotation — K/V stored in rotated space.

        Args:
            kv_cache: ``(NB, BS, total_bytes)`` uint8 packed cache.
            compute_dtype: Output dtype.

        Returns:
            (key_cache, value_cache) each ``(NB, BS, H, D)``.
        """
        NB, BS, _ = kv_cache.shape
        H = self.num_kv_heads
        D = self.head_size
        half_D = self._half_d
        num_tokens = NB * BS

        flat = kv_cache.reshape(num_tokens, -1)

        # Extract K per head and decompress
        k_packed_list = []
        for h in range(H):
            start = h * half_D
            k_packed_list.append(flat[:, start:start + half_D])
        k_packed = torch.stack(k_packed_list, dim=1)  # (M, H, half_D)

        k_norms_raw = flat[:, self._k_idx_end:self._k_norm_end].contiguous().view(
            torch.float32
        ).reshape(-1, H, 1)

        # Decompress K (no rotation)
        if self._rq_mode == 'iso':
            k_out = iso_decompress(
                k_packed, k_norms_raw, self._k_quantizer.centroids, compute_dtype
            )
        else:
            k_out = planar_decompress(
                k_packed, k_norms_raw, self._k_quantizer.centroids, compute_dtype,
                d_padded=self._k_quantizer.d_padded,
            )

        # Extract V per head and decompress
        v_packed_list = []
        for h in range(H):
            start = self._k_norm_end + h * half_D
            end = start + half_D
            v_packed_list.append(flat[:, start:end])
        v_packed = torch.stack(v_packed_list, dim=1)  # (M, H, half_D)

        v_norms_raw = flat[:, self._v_idx_end:self._total_bytes].contiguous().view(
            torch.float32
        ).reshape(-1, H, 1)

        # Decompress V (no rotation)
        if self._rq_mode == 'iso':
            v_out = iso_decompress(
                v_packed, v_norms_raw, self._v_quantizer.centroids, compute_dtype
            )
        else:
            v_out = planar_decompress(
                v_packed, v_norms_raw, self._v_quantizer.centroids, compute_dtype,
                d_padded=self._v_quantizer.d_padded,
            )

        return (
            k_out.reshape(NB, BS, H, D),
            v_out.reshape(NB, BS, H, D),
        )

    def _decompress_cache_paged(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        compute_dtype: torch.dtype,
        out_k: torch.Tensor | None = None,
        out_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompress only the physical blocks referenced by block_table.

        Paged decompression: extract unique blocks, decompress them,
        build remapped block table for Flash Attention.

        Args:
            kv_cache: ``(NB, BS, total_bytes)`` uint8 packed cache.
            block_table: ``(batch, max_blocks_per_seq)`` int32 block table.
            seq_lens: ``(batch,)`` sequence lengths.
            compute_dtype: Output dtype.
            out_k: Pre-allocated buffer ``(max_tokens, H, D)`` or None.
            out_v: Pre-allocated buffer ``(max_tokens, H, D)`` or None.

        Returns:
            (key_cache, value_cache, remapped_block_table) where
            key/value are ``(num_unique_blocks, BS, H, D)``.
        """
        NB, BS, _ = kv_cache.shape
        H = self.num_kv_heads
        D = self.head_size
        half_D = self._half_d

        # Extract valid block indices from block_table using seq_lens
        max_blocks_per_seq = block_table.shape[1]
        blocks_needed = (seq_lens + BS - 1) // BS  # ceil division
        col_idx = torch.arange(max_blocks_per_seq, device=block_table.device).unsqueeze(0)
        valid_mask = col_idx < blocks_needed.unsqueeze(1)
        valid_block_indices = block_table[valid_mask]

        unique_blocks = torch.unique(valid_block_indices, sorted=True)
        num_unique = unique_blocks.numel()

        # Select referenced blocks
        max_cap = out_k.shape[0] // BS if out_k is not None else 0

        selected = kv_cache[unique_blocks]  # (num_unique, BS, total_bytes)
        flat = selected.reshape(num_unique * BS, -1)

        # Extract and decompress K
        k_packed_list = []
        for h in range(H):
            start = h * half_D
            k_packed_list.append(flat[:, start:start + half_D])
        k_packed = torch.stack(k_packed_list, dim=1)  # (M, H, half_D)

        k_norms_raw = flat[:, self._k_idx_end:self._k_norm_end].contiguous().view(
            torch.float32
        ).reshape(-1, H, 1)

        if self._rq_mode == 'iso':
            k_decomp = iso_decompress(
                k_packed, k_norms_raw, self._k_quantizer.centroids, compute_dtype
            )
        else:
            k_decomp = planar_decompress(
                k_packed, k_norms_raw, self._k_quantizer.centroids, compute_dtype,
                d_padded=self._k_quantizer.d_padded,
            )

        # Extract and decompress V
        v_packed_list = []
        for h in range(H):
            start = self._k_norm_end + h * half_D
            end = start + half_D
            v_packed_list.append(flat[:, start:end])
        v_packed = torch.stack(v_packed_list, dim=1)  # (M, H, half_D)

        v_norms_raw = flat[:, self._v_idx_end:self._total_bytes].contiguous().view(
            torch.float32
        ).reshape(-1, H, 1)

        if self._rq_mode == 'iso':
            v_decomp = iso_decompress(
                v_packed, v_norms_raw, self._v_quantizer.centroids, compute_dtype
            )
        else:
            v_decomp = planar_decompress(
                v_packed, v_norms_raw, self._v_quantizer.centroids, compute_dtype,
                d_padded=self._v_quantizer.d_padded,
            )

        # Store in pre-allocated buffers if available
        k_tokens = num_unique * BS
        if out_k is not None and num_unique <= max_cap:
            out_k[:k_tokens, :, :].copy_(k_decomp)
            k_out = out_k[:k_tokens, :, :]
        else:
            k_out = k_decomp

        if out_v is not None and num_unique <= max_cap:
            out_v[:k_tokens, :, :].copy_(v_decomp)
            v_out = out_v[:k_tokens, :, :]
        else:
            v_out = v_decomp

        # Build remapped block table: old physical -> compact 0..N-1
        remap = torch.zeros(NB, dtype=block_table.dtype, device=block_table.device)
        remap[unique_blocks] = torch.arange(
            num_unique, dtype=block_table.dtype, device=block_table.device
        )
        remapped_block_table = remap[block_table]

        return (
            k_out.reshape(num_unique, BS, H, D),
            v_out.reshape(num_unique, BS, H, D),
            remapped_block_table,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        """RotorQuant forward: compress -> store -> decompress -> Flash Attention.

        Q pre-rotation + output post-rotation path:
        1. Compress and store new K/V (nibble-packed, rotated space).
        2. Pre-rotate Q by R^T into rotated space.
        3. Paged decompress K/V (rotated space, no inverse rotation).
        4. Flash Attention with rotated Q and decompressed K/V.
        5. Post-rotate output by R back to original space.
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Fused output quantization is not supported with RotorQuant backend"
            )

        # Edge cases
        if attn_metadata is None or kv_cache is None:
            return output.zero_()

        # Encoder attention: delegate to parent
        from vllm.v1.attention.backend import AttentionType

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[: attn_metadata.num_actual_tokens],
                key[: attn_metadata.num_actual_tokens],
                value[: attn_metadata.num_actual_tokens],
                output[: attn_metadata.num_actual_tokens],
                attn_metadata,
                layer,
            )

        num_actual_tokens = attn_metadata.num_actual_tokens
        is_decode = (num_actual_tokens == 1)

        # Lazy-init buffers
        if not self._cg_buffers_ready:
            self._init_cg_buffers(kv_cache, query.dtype)

        if is_decode:
            # Decode path
            if key is not None and value is not None:
                self._compress_and_store(
                    key, value, kv_cache, attn_metadata.slot_mapping,
                    compress_out=(self._cg_compress_packed, self._cg_compress_norms),
                    row_out=self._cg_compress_row,
                )

            # Q pre-rotation: q_rot = q @ R^T
            q_slice = query[:1]
            torch.matmul(q_slice.float(), self._rq_rot_T, out=self._cg_q_rot[:1])
            self._cg_q_rot_cast[:1].copy_(self._cg_q_rot[:1])
            q_rot = self._cg_q_rot_cast[:1]

            # Paged decompress K/V
            key_cache, value_cache, fa_block_table = self._decompress_cache_paged(
                kv_cache, attn_metadata.block_table, attn_metadata.seq_lens,
                query.dtype,
                out_k=self._cg_decompress_k,
                out_v=self._cg_decompress_v,
            )
        else:
            # Prefill path
            if key is not None and value is not None:
                self._compress_and_store(
                    key, value, kv_cache, attn_metadata.slot_mapping
                )

            # Q pre-rotation
            q_slice = query[:num_actual_tokens]
            q_rot = (q_slice.float() @ self._rq_rot_T).to(q_slice.dtype)

            # Paged decompress K/V
            key_cache, value_cache, fa_block_table = self._decompress_cache_paged(
                kv_cache, attn_metadata.block_table, attn_metadata.seq_lens,
                query.dtype,
                out_k=self._cg_prefill_k,
                out_v=self._cg_prefill_v,
            )

        # Flash Attention
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        if attn_metadata.use_cascade:
            raise NotImplementedError("RQ does not yet support cascade attention")

        descale_shape = (attn_metadata.query_start_loc.shape[0] - 1, self.num_kv_heads)
        q_descale = layer._q_scale.expand(descale_shape)
        k_descale = layer._k_scale.expand(descale_shape)
        v_descale = layer._v_scale.expand(descale_shape)

        flash_attn_varlen_func(
            q=q_rot,
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=list(self.sliding_window)
            if self.sliding_window is not None
            else None,
            block_table=fa_block_table,
            softcap=self.logits_soft_cap,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
            s_aux=self.sinks,
        )

        # Output post-rotation: out_orig = out_rot @ R
        out_slice = output[:num_actual_tokens]
        output[:num_actual_tokens] = (out_slice.float() @ self._rq_rot).to(out_slice.dtype)

        return output


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_original_get_kv_cache_spec = None


def register_rq_backend() -> None:
    """Register RotorQuant as the CUSTOM attention backend.

    In addition to registering the backend class, this monkey-patches
    ``Attention.get_kv_cache_spec`` so that decoder attention layers
    return :class:`RQFullAttentionSpec` (with ``dtype=torch.uint8``
    and RotorQuant-sized pages) instead of the standard ``FullAttentionSpec``.

    Called automatically by the ``vllm.general_plugins`` entry point,
    or manually before starting vLLM::

        from rotorquant_vllm.vllm import register_rq_backend
        register_rq_backend()
        # then start vLLM with --attention-backend CUSTOM
    """
    global _original_get_kv_cache_spec  # noqa: PLW0603

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "rotorquant_vllm.vllm.rq_backend.RQAttentionBackend",
    )

    # Register RQFullAttentionSpec in the KV cache manager mapping.
    # vLLM uses exact type() match, not isinstance(), so subclasses
    # of FullAttentionSpec must be explicitly added.
    from vllm.v1.core.single_type_kv_cache_manager import spec_manager_map

    if RQFullAttentionSpec not in spec_manager_map:
        spec_manager_map[RQFullAttentionSpec] = spec_manager_map[FullAttentionSpec]

    # Monkey-patch Attention.get_kv_cache_spec to return RQ spec
    from vllm.model_executor.layers.attention.attention import Attention

    if _original_get_kv_cache_spec is None:
        _original_get_kv_cache_spec = Attention.get_kv_cache_spec

    def _rq_get_kv_cache_spec(self, vllm_config):
        spec = _original_get_kv_cache_spec(self, vllm_config)
        if isinstance(spec, FullAttentionSpec) and not isinstance(
            spec, RQFullAttentionSpec
        ):
            kwargs = {f.name: getattr(spec, f.name) for f in dc_fields(spec)}
            kwargs["dtype"] = torch.uint8
            return RQFullAttentionSpec(**kwargs)
        return spec

    Attention.get_kv_cache_spec = _rq_get_kv_cache_spec
    logger.info("RQ attention backend registered as CUSTOM (compressed cache)")
