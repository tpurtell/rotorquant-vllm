"""Integration tests for rotorquant-vllm."""
import torch
import pytest

from rotorquant_vllm.vllm import RQAttentionBackend, RQFullAttentionSpec, register_rq_backend
from rotorquant_vllm.vllm.rq_backend import (
    _build_iso_R,
    _build_planar_R,
    _rq_bytes_per_token_kv,
    _rq_padded_slot_bytes,
)
from rotorquant_vllm.quantization import IsoQuantMSE, PlanarQuantMSE
from rotorquant_vllm.triton import (
    iso_compress,
    iso_decompress,
    planar_compress,
    planar_decompress,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


class TestPluginRegistration:
    def test_register_backend(self):
        register_rq_backend()
        assert AttentionBackendEnum.CUSTOM.is_overridden()
        path = AttentionBackendEnum.CUSTOM.get_path()
        assert "rotorquant_vllm" in path

    def test_backend_name(self):
        assert RQAttentionBackend.get_name() == "CUSTOM"

    def test_backend_impl(self):
        from rotorquant_vllm.vllm.rq_backend import RQAttentionImpl

        assert RQAttentionBackend.get_impl_cls() is RQAttentionImpl

    def test_mm_prefix(self):
        assert RQAttentionBackend.supports_mm_prefix() is True


class TestCacheLayout:
    def test_nibble_packed_size(self):
        """D=128, H=8: 8*(64+4)*2 = 1088 raw, padded to 2048."""
        raw = _rq_bytes_per_token_kv(128, 8)
        assert raw == 1088
        padded = _rq_padded_slot_bytes(128)
        assert padded == 256  # next_power_of_2(136)
        total = 8 * padded
        assert total == 2048

    def test_cache_shape(self):
        shape = RQAttentionBackend.get_kv_cache_shape(100, 16, 8, 128)
        assert shape == (100, 16, 2048)

    def test_page_size(self):
        spec = RQFullAttentionSpec(
            block_size=16, num_kv_heads=8, head_size=128, dtype=torch.uint8
        )
        assert spec.real_page_size_bytes == 32768  # 16 * 2048


class TestCompressDecompress:
    def test_iso_roundtrip_cpu(self):
        iso = IsoQuantMSE(128, 3, seed=42, mode="fast")
        R = _build_iso_R(iso, 128, "cpu", "fast")
        R_T = R.T.contiguous()
        R_T_even = R_T[:, 0::2].contiguous()
        R_T_odd = R_T[:, 1::2].contiguous()

        x = torch.randn(2, 8, 128, dtype=torch.float16)
        packed, norms = iso_compress(x, R_T_even, R_T_odd, iso.boundaries)
        assert packed.shape == (2, 8, 64)
        assert packed.dtype == torch.uint8

        out = iso_decompress(packed, norms, iso.centroids, torch.float16)
        assert out.shape == (2, 8, 128)

    def test_planar_roundtrip_cpu(self):
        pq = PlanarQuantMSE(128, 3, seed=42)
        R = _build_planar_R(pq, 128, "cpu")
        R_T = R.T.contiguous()
        R_T_even = R_T[:, 0::2].contiguous()
        R_T_odd = R_T[:, 1::2].contiguous()

        x = torch.randn(2, 8, 128, dtype=torch.float16)
        packed, norms = planar_compress(x, R_T_even, R_T_odd, pq.boundaries)
        assert packed.shape == (2, 8, 64)
        out = planar_decompress(packed, norms, pq.centroids, torch.float16)
        assert out.shape == (2, 8, 128)

class TestRotationMatrices:
    def test_iso_orthogonality(self):
        iso = IsoQuantMSE(128, 3, seed=42, mode="fast")
        R = _build_iso_R(iso, 128, "cpu", "fast")
        err = (R @ R.T - torch.eye(128)).abs().max().item()
        assert err < 1e-5

    def test_planar_orthogonality(self):
        pq = PlanarQuantMSE(128, 3, seed=42)
        R = _build_planar_R(pq, 128, "cpu")
        err = (R @ R.T - torch.eye(128)).abs().max().item()
        assert err < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
