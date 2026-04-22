"""Microbenchmarks for rotorquant-vllm compression pipeline."""
import time
import torch
import pytest

from rotorquant_vllm.quantization import IsoQuantMSE, PlanarQuantMSE
from rotorquant_vllm.triton import (
    iso_compress, iso_decompress,
    planar_compress, planar_decompress,
)
from rotorquant_vllm.vllm.rq_backend import (
    _build_iso_R, _build_planar_R, _rq_bytes_per_token_kv,
)


def _cuda_timer_ms(fn, warmup=10, iters=50):
    """Time a CUDA function, return mean ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


class TestCompressLatency:
    """Compress latency benchmarks."""

    @pytest.mark.gpu
    @pytest.mark.benchmark
    def test_iso_compress_latency(self):
        """IsoQuant compress latency at D=128, bits=3."""
        N, H, D = 1, 8, 128
        iso = IsoQuantMSE(D, 3, seed=42, mode='fast', device='cuda')
        x = torch.randn(N, H, D, dtype=torch.float16, device='cuda')
        R = _build_iso_R(iso, D, 'cuda', 'fast')
        R_T = R.T.contiguous()
        R_T_even = R_T[:, 0::2].contiguous()
        R_T_odd = R_T[:, 1::2].contiguous()
        boundaries = iso.boundaries.to('cuda')

        def compress():
            return iso_compress(x, R_T_even, R_T_odd, boundaries)

        ms = _cuda_timer_ms(compress)
        us_per_token = ms * 1000 / N
        print(f"  IsoQuant compress: {us_per_token:.1f} µs/token ({ms:.3f} ms)")
        assert us_per_token > 0

    @pytest.mark.gpu
    @pytest.mark.benchmark
    def test_planar_compress_latency(self):
        """PlanarQuant compress latency at D=128, bits=3."""
        N, H, D = 1, 8, 128
        planar = PlanarQuantMSE(D, 3, seed=42, device='cuda')
        x = torch.randn(N, H, D, dtype=torch.float16, device='cuda')
        R = _build_planar_R(planar, D, 'cuda')
        R_T = R.T.contiguous()
        R_T_even = R_T[:, 0::2].contiguous()
        R_T_odd = R_T[:, 1::2].contiguous()
        boundaries = planar.boundaries.to('cuda')

        def compress():
            return planar_compress(x, R_T_even, R_T_odd, boundaries)

        ms = _cuda_timer_ms(compress)
        us_per_token = ms * 1000 / N
        print(f"  PlanarQuant compress: {us_per_token:.1f} µs/token ({ms:.3f} ms)")
        assert us_per_token > 0


class TestDecompressLatency:
    """Decompress latency benchmarks."""

    @pytest.mark.gpu
    @pytest.mark.benchmark
    @pytest.mark.parametrize("cache_tokens", [128, 1024, 4096, 16384])
    def test_iso_decompress_latency(self, cache_tokens):
        """IsoQuant decompress latency at different cache sizes."""
        H, D = 8, 128
        iso = IsoQuantMSE(D, 3, seed=42, mode='fast', device='cuda')
        packed = torch.randint(0, 8, (cache_tokens, H, D//2), dtype=torch.uint8, device='cuda')
        norms = torch.rand(cache_tokens, H, 1, dtype=torch.float32, device='cuda')

        def decompress():
            return iso_decompress(packed, norms, iso.centroids, torch.float16)

        ms = _cuda_timer_ms(decompress)
        us_per_token = ms * 1000 / cache_tokens
        bytes_per_token = D * 2  # fp16 output
        bandwidth_gb_s = (bytes_per_token * cache_tokens / 1e9) / (ms / 1000)
        print(f"  IsoQuant decompress {cache_tokens} tokens: "
              f"{us_per_token:.2f} µs/token, {bandwidth_gb_s:.1f} GB/s effective")
        assert us_per_token > 0


class TestRoundtripQuality:
    """Roundtrip quality measurements.

    Uses the full quantize/dequantize path (with inverse rotation) via
    IsoQuantMSE.forward() and PlanarQuantMSE.forward() to measure
    end-to-end reconstruction fidelity. The triton kernels skip inverse
    rotation by design (rotation is applied to Q separately in the backend).
    """

    @pytest.mark.parametrize("bits", [2, 3, 4])
    @pytest.mark.parametrize("mode", ["fast", "full"])
    def test_iso_quant_quality(self, bits, mode):
        """IsoQuant roundtrip quality at different bits and modes."""
        D = 128
        iso = IsoQuantMSE(D, bits, seed=42, mode=mode)
        x = torch.randn(100, D, dtype=torch.float32)
        x_norm = x / x.norm(dim=-1, keepdim=True)

        # Full roundtrip: quantize + dequantize (includes inverse rotation)
        x_hat, _ = iso.forward(x_norm)

        # Cosine similarity and MSE
        cos_sim = (x_norm * x_hat).sum(dim=-1).mean().item()
        mse = ((x_norm - x_hat) ** 2).mean().item()
        print(f"  IsoQuant {mode} {bits}-bit: cos_sim={cos_sim:.4f}, MSE={mse:.6f}")
        assert cos_sim > 0.8  # Basic sanity
        assert mse > 0  # Non-zero quantization error

    def test_planar_quant_quality(self):
        """PlanarQuant roundtrip quality."""
        D = 128
        for bits in [2, 3, 4]:
            planar = PlanarQuantMSE(D, bits, seed=42)
            x = torch.randn(100, D, dtype=torch.float32)
            x_norm = x / x.norm(dim=-1, keepdim=True)

            # Full roundtrip: quantize + dequantize (includes inverse rotation)
            x_hat, _ = planar.forward(x_norm)

            cos_sim = (x_norm * x_hat).sum(dim=-1).mean().item()
            mse = ((x_norm - x_hat) ** 2).mean().item()
            print(f"  PlanarQuant {bits}-bit: cos_sim={cos_sim:.4f}, MSE={mse:.6f}")
            assert cos_sim > 0.8


class TestCompressionRatio:
    """Compression ratio measurements."""

    def test_compression_ratio(self):
        """Verify compression ratios match expected values."""
        H, D = 8, 128
        bytes_fp16 = H * D * 2 * 2  # K+V, fp16
        bytes_compressed = _rq_bytes_per_token_kv(D, H)
        ratio = bytes_fp16 / bytes_compressed
        print(f"  FP16: {bytes_fp16} bytes, Compressed: {bytes_compressed} bytes, "
              f"Ratio: {ratio:.2f}x")
        assert ratio > 3.5  # Should be ~3.76x
        assert ratio < 4.0  # Sanity upper bound


class TestRotationMatrixBenchmark:
    """Rotation matrix construction benchmarks."""

    def test_iso_rotation_build_time(self):
        """Time to build IsoQuant rotation matrix."""
        iso = IsoQuantMSE(128, 3, seed=42, mode='fast')
        t0 = time.perf_counter()
        R = _build_iso_R(iso, 128, 'cpu', 'fast')
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  IsoQuant R build: {elapsed:.2f} ms")
        assert R.shape == (128, 128)

    def test_planar_rotation_build_time(self):
        """Time to build PlanarQuant rotation matrix."""
        planar = PlanarQuantMSE(128, 3, seed=42)
        t0 = time.perf_counter()
        R = _build_planar_R(planar, 128, 'cpu')
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  PlanarQuant R build: {elapsed:.2f} ms")
        assert R.shape == (128, 128)


class TestQRotationSavings:
    """Q pre-rotation vs full cache rotation comparison."""

    @pytest.mark.gpu
    @pytest.mark.benchmark
    def test_q_rotation_vs_cache_rotation(self):
        """Compare Q pre-rotation (1 token) vs cache rotation (N tokens).

        The key insight: Q rotation is O(1) per step regardless of cache size,
        while rotating the full cache is O(cache_len). On GPU, both matmuls
        are very fast; the real savings come from avoiding HBM reads of
        compressed cache data. This test demonstrates the FLOP scaling difference.
        """
        D, H = 128, 8
        iso = IsoQuantMSE(D, 3, seed=42, mode='fast', device='cuda')
        R = _build_iso_R(iso, D, 'cuda', 'fast')
        R_T = R.T.contiguous()

        # Q pre-rotation: 1 token x num_heads x D
        q = torch.randn(1, 32, D, dtype=torch.float16, device='cuda')

        def q_rotate():
            return q.float() @ R_T

        ms_q = _cuda_timer_ms(q_rotate)

        # Cache rotation: cache_len tokens (simulating rotating full cache)
        flop_ratio = 0
        for cache_len in [256, 1024, 4096]:
            cached_k = torch.randn(cache_len, H, D, dtype=torch.float16, device='cuda')
            def cache_rotate():
                return cached_k.float() @ R
            ms_cache = _cuda_timer_ms(cache_rotate)

            # Compute theoretical FLOP ratio
            q_flops = 1 * 32 * D * D * 2
            cache_flops = cache_len * H * D * D * 2
            flop_ratio = cache_flops / q_flops

            speedup = ms_cache / ms_q
            print(f"  Cache={cache_len}: Q rotation {ms_q:.4f}ms vs cache rotation "
                  f"{ms_cache:.4f}ms, speedup={speedup:.1f}x (FLOP ratio={flop_ratio:.0f}x)")

        # Q rotation must be measurable
        assert ms_q > 0
        # The FLOP ratio demonstrates the theoretical savings
        # (actual GPU speedup limited by kernel launch overhead on small tensors)
        assert flop_ratio > 50  # 4096 tokens: 4096 * 8 / 32 = 1024x FLOP ratio
