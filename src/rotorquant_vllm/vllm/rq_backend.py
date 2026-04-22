"""RotorQuant compressed KV cache attention backend for vLLM.

Integrates RotorQuant's IsoQuant (4D quaternion) and PlanarQuant (2D Givens)
KV cache compression into vLLM as a drop-in plugin.  Stores compressed KV
cache as uint8 bytes with layout:

    [K_indices(all heads) | K_norms(all heads) | V_indices(all heads) | V_norms(all heads)]

Each component per head: ``head_dim`` int8 indices + 1 fp32 norm.
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
# Cache size helpers
# ---------------------------------------------------------------------------


def _rq_bytes_per_component(head_dim: int, num_kv_heads: int) -> int:
    """Bytes for one component (K or V): all heads' int8 indices + all heads' fp32 norms.

    Layout: [K_indices_0 | K_indices_1 | ... | K_norm_0 | K_norm_1 | ...]
    where indices are int8 (1 byte/coord) and norms are fp32 (4 bytes/head).
    """
    return num_kv_heads * head_dim + num_kv_heads * RQ_NORM_BYTES


def _rq_bytes_per_token_kv(head_dim: int, num_kv_heads: int) -> int:
    """Total bytes per token (K + V combined, all heads).

    For 8 heads, head_dim=128:
        8 * (128 + 4) * 2 = 2112 bytes vs 4096 FP16 K+V = ~1.94x compression
    """
    return _rq_bytes_per_component(head_dim, num_kv_heads) * 2


# ---------------------------------------------------------------------------
# KV cache spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class RQFullAttentionSpec(FullAttentionSpec):
    """KV cache spec with RotorQuant compressed page size.

    Overrides ``real_page_size_bytes`` so the block allocator provisions
    buffers sized for the packed RotorQuant format (int8 indices + fp32 norms).
    Uses ``dtype=torch.uint8`` for the uint8 byte cache.
    """

    @property
    def real_page_size_bytes(self) -> int:  # noqa: D102
        return self.block_size * _rq_bytes_per_token_kv(self.head_size, self.num_kv_heads)


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
        """No CUDA graph support for v0 (dynamic compress/decompress)."""
        return AttentionCGSupport.NEVER


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class RQAttentionBackend(FlashAttentionBackend):
    """RotorQuant compressed KV cache attention backend.

    Stores KV cache as packed uint8 bytes (int8 indices + fp32 norms).
    Each forward() call decompresses cached K/V to FP16, runs Flash
    Attention, then returns the output.
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

            [K_indices(all heads) | K_norms(all heads) |
             V_indices(all heads) | V_norms(all heads)]
        """
        total_bytes = _rq_bytes_per_token_kv(head_size, num_kv_heads)
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

    Stores packed bytes (int8 indices + fp32 norms) in a uint8 cache.
    Each ``forward()`` call:

    1. Compresses incoming K/V tokens via rotation + Lloyd-Max quantization.
    2. Scatter-writes packed bytes to the uint8 cache.
    3. Decompresses the cache to FP16 (gather centroids + inverse rotate + rescale).
    4. Calls ``flash_attn_varlen_func`` with the FP16 data.
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

        # Common buffers for both modes
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

        # Byte layout offsets within the last dimension of the packed cache.
        # Layout per-token: [K_indices | K_norms | V_indices | V_norms]
        #   K_indices: num_kv_heads * head_size bytes (int8)
        #   K_norms:   num_kv_heads * RQ_NORM_BYTES (fp32 viewed as uint8)
        #   V_indices: num_kv_heads * head_size bytes (int8)
        #   V_norms:   num_kv_heads * RQ_NORM_BYTES (fp32 viewed as uint8)
        k_idx_end = num_kv_heads * head_size
        k_norm_end = k_idx_end + num_kv_heads * RQ_NORM_BYTES
        v_idx_end = k_norm_end + num_kv_heads * head_size
        total_bytes = v_idx_end + num_kv_heads * RQ_NORM_BYTES

        self._k_idx_end = k_idx_end
        self._k_norm_end = k_norm_end
        self._v_idx_end = v_idx_end
        self._total_bytes = total_bytes

        # Log compression ratio
        fp16_total = 2 * num_kv_heads * head_size * 2  # K+V in fp16
        compression = fp16_total / total_bytes
        logger.info(
            "RQAttentionImpl: %s mode, %d KV heads, head_size=%d, "
            "k_bits=%d, v_bits=%d, %d bytes/token (%.2fx compression vs FP16)",
            rq_mode,
            num_kv_heads,
            head_size,
            k_bits,
            v_bits,
            total_bytes,
            compression,
        )

    # ------------------------------------------------------------------
    # Compress path
    # ------------------------------------------------------------------

    def _compress_tensor(
        self, x: torch.Tensor, quantizer
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress a tensor via rotation + Lloyd-Max quantization.

        Args:
            x: ``(N, H, D)`` input tensor (fp16/bf16).
            quantizer: IsoQuantMSE or PlanarQuantMSE instance.

        Returns:
            (indices, norms) where indices is ``(N, H, D)`` int8
            and norms is ``(N, H)`` fp32 (one norm per head).
        """
        v_q, indices_dict = quantizer.quantize(x)
        indices = indices_dict["indices"].to(torch.int8)  # (N, H, D)
        norms = indices_dict["_norms"].to(torch.float32)  # (N, H) — ensure fp32 for 4-byte storage
        return indices, norms

    def _compress_and_store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Compress K/V and scatter-write packed bytes to cache.

        Args:
            key: ``(N, H, D)`` new key tokens.
            value: ``(N, H, D)`` new value tokens.
            kv_cache: ``(NB, BS, total_bytes)`` uint8 packed cache.
            slot_mapping: ``(num_actual_tokens,)`` flat slot indices.
        """
        N = key.shape[0]
        num_actual = slot_mapping.shape[0]

        # Compress K and V
        k_indices, k_norms = self._compress_tensor(key, self._k_quantizer)
        v_indices, v_norms = self._compress_tensor(value, self._v_quantizer)

        # Build packed byte row per token:
        #   [K_indices(all heads) | K_norms(all heads) | V_indices(all heads) | V_norms(all heads)]
        row = torch.empty(N, self._total_bytes, dtype=torch.uint8, device=key.device)

        # Indices: (N, H, D) int8 → (N, H*D) uint8
        row[:, : self._k_idx_end] = k_indices.reshape(N, -1).to(torch.uint8)
        # Norms: (N, H) fp32 → view as uint8 → (N, H*4)
        row[:, self._k_idx_end : self._k_norm_end] = (
            k_norms.reshape(N, -1).contiguous().view(torch.uint8)
        )
        row[:, self._k_norm_end : self._v_idx_end] = v_indices.reshape(N, -1).to(torch.uint8)
        row[:, self._v_idx_end :] = (
            v_norms.reshape(N, -1).contiguous().view(torch.uint8)
        )

        # Scatter-write to flat cache using slot_mapping
        flat_cache = kv_cache.view(-1, kv_cache.shape[-1])
        flat_cache[slot_mapping[:num_actual], : self._total_bytes] = row[:num_actual]

    # ------------------------------------------------------------------
    # Decompress path
    # ------------------------------------------------------------------

    def _decompress_tensor(
        self, indices: torch.Tensor, norms: torch.Tensor, quantizer
    ) -> torch.Tensor:
        """Decompress quantized indices and norms back to fp16/bf16.

        Args:
            indices: ``(num_tokens, H, D)`` int8 indices.
            norms: ``(num_tokens, H)`` fp32 norms (one per head).
            quantizer: IsoQuantMSE or PlanarQuantMSE instance.

        Returns:
            ``(num_tokens, H, D)`` reconstructed tensor.
        """
        x_hat = quantizer.dequantize({"indices": indices.to(torch.int64), "_norms": norms})
        return x_hat

    def _decompress_cache(
        self,
        kv_cache: torch.Tensor,
        compute_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress the full packed cache to FP16.

        Args:
            kv_cache: ``(NB, BS, total_bytes)`` uint8 packed cache.
            compute_dtype: Output dtype (e.g., ``torch.bfloat16``).

        Returns:
            (key_cache, value_cache) each ``(NB, BS, H, D)``.
        """
        NB, BS, _ = kv_cache.shape
        H = self.num_kv_heads
        D = self.head_size
        num_tokens = NB * BS

        # Flatten to (num_tokens, total_bytes)
        flat = kv_cache.reshape(num_tokens, -1)

        # Extract K regions
        k_indices = flat[:, : self._k_idx_end].contiguous().to(torch.int8).reshape(-1, H, D)
        k_norms = (
            flat[:, self._k_idx_end : self._k_norm_end]
            .contiguous()
            .view(torch.float32)
            .reshape(-1, H)
        )

        # Extract V regions
        v_indices = (
            flat[:, self._k_norm_end : self._v_idx_end].contiguous().to(torch.int8).reshape(-1, H, D)
        )
        v_norms = (
            flat[:, self._v_idx_end : self._total_bytes]
            .contiguous()
            .view(torch.float32)
            .reshape(-1, H)
        )

        # Decompress
        key_out = self._decompress_tensor(k_indices, k_norms, self._k_quantizer).to(compute_dtype)
        value_out = self._decompress_tensor(v_indices, v_norms, self._v_quantizer).to(compute_dtype)

        return (
            key_out.reshape(NB, BS, H, D),
            value_out.reshape(NB, BS, H, D),
        )

    def _decompress_cache_paged(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        compute_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompress only the physical blocks referenced by block_table.

        Not CUDA-graph-safe: uses ``torch.unique`` for block selection.

        Args:
            kv_cache: ``(NB, BS, total_bytes)`` uint8 packed cache.
            block_table: ``(batch, max_blocks_per_seq)`` int32 block table.
            seq_lens: ``(batch,)`` sequence lengths.
            compute_dtype: Output dtype.

        Returns:
            (key_cache, value_cache, remapped_block_table) where
            key/value are ``(num_unique_blocks, BS, H, D)`` and
            remapped_block_table maps logical blocks to compact indices.
        """
        NB, BS, _ = kv_cache.shape
        H = self.num_kv_heads
        D = self.head_size

        # Extract valid block indices from block_table using seq_lens
        max_blocks_per_seq = block_table.shape[1]
        blocks_needed = (seq_lens + BS - 1) // BS  # ceil division
        col_idx = torch.arange(max_blocks_per_seq, device=block_table.device).unsqueeze(0)
        valid_mask = col_idx < blocks_needed.unsqueeze(1)
        valid_block_indices = block_table[valid_mask]

        unique_blocks = torch.unique(valid_block_indices, sorted=True)
        num_unique = unique_blocks.numel()

        # Gather referenced blocks and decompress
        selected = kv_cache[unique_blocks]  # (num_unique, BS, total_bytes)
        flat = selected.reshape(num_unique * BS, -1)

        k_indices = flat[:, : self._k_idx_end].contiguous().to(torch.int8).reshape(-1, H, D)
        k_norms = (
            flat[:, self._k_idx_end : self._k_norm_end]
            .contiguous()
            .view(torch.float32)
            .reshape(-1, H)
        )
        v_indices = (
            flat[:, self._k_norm_end : self._v_idx_end]
            .contiguous()
            .to(torch.int8)
            .reshape(-1, H, D)
        )
        v_norms = (
            flat[:, self._v_idx_end : self._total_bytes]
            .contiguous()
            .view(torch.float32)
            .reshape(-1, H)
        )

        key_out = self._decompress_tensor(k_indices, k_norms, self._k_quantizer).to(compute_dtype)
        value_out = self._decompress_tensor(v_indices, v_norms, self._v_quantizer).to(compute_dtype)

        key_cache = key_out.reshape(num_unique, BS, H, D)
        value_cache = value_out.reshape(num_unique, BS, H, D)

        # Build remapped block table: old physical → compact 0..N-1
        remap = torch.zeros(NB, dtype=block_table.dtype, device=block_table.device)
        remap[unique_blocks] = torch.arange(
            num_unique, dtype=block_table.dtype, device=block_table.device
        )
        remapped_block_table = remap[block_table]

        return key_cache, value_cache, remapped_block_table

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

        Handles edge cases (None kv_cache, encoder attention) by delegating
        to parent or returning zero output.
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Fused output quantization is not supported with RotorQuant backend"
            )

        # Profiling run — no metadata
        if attn_metadata is None:
            return output.zero_()

        # Warmup with no cache allocated yet
        if kv_cache is None:
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

        # Step 1: Compress and store new K/V
        if key is not None and value is not None:
            self._compress_and_store(key, value, kv_cache, attn_metadata.slot_mapping)

        # Step 2: Decompress cached K/V to FP16
        key_cache, value_cache = self._decompress_cache(kv_cache, query.dtype)

        # Step 3: Run Flash Attention
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        if attn_metadata.use_cascade:
            raise NotImplementedError("RotorQuant does not yet support cascade attention")

        flash_attn_varlen_func(
            q=query[:num_actual_tokens],
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
            block_table=attn_metadata.block_table,
            softcap=self.logits_soft_cap,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
            num_splits=attn_metadata.max_num_splits,
            s_aux=self.sinks,
        )

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
