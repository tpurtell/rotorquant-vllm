"""Benchmark: compress kernel latency."""
import torch
import time
from rotorquant_vllm.quantization import IsoQuantMSE, PlanarQuantMSE
from rotorquant_vllm.vllm.rq_backend import _build_iso_R, _build_planar_R
from rotorquant_vllm.triton import iso_compress, planar_compress


def _split_rot_T(R):
    """Split R^T into even/odd column matrices for compress kernels."""
    R_T = R.T.contiguous()
    R_T_even = R_T[:, 0::2].contiguous()
    R_T_odd = R_T[:, 1::2].contiguous()
    return R_T_even, R_T_odd


def bench_iso_fast():
    print("=== IsoQuant compress (fast mode) ===")
    iso = IsoQuantMSE(128, 3, seed=42, mode='fast', device='cuda')
    R = _build_iso_R(iso, 128, 'cuda', 'fast')
    R_T_even, R_T_odd = _split_rot_T(R)

    for N in [1, 8, 64]:
        x = torch.randn(N, 8, 128, dtype=torch.float16, device='cuda')
        # Warmup
        for _ in range(10):
            iso_compress(x, R_T_even, R_T_odd, iso.boundaries)
        torch.cuda.synchronize()
        # Timed
        t0 = time.perf_counter()
        for _ in range(100):
            iso_compress(x, R_T_even, R_T_odd, iso.boundaries)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 100
        us_token = ms * 1000 / N
        print(f"  N={N:3d}: {ms:.3f}ms batch, {us_token:.1f} us/token")


def bench_iso_full():
    print("=== IsoQuant compress (full mode) ===")
    iso = IsoQuantMSE(128, 3, seed=42, mode='full', device='cuda')
    R = _build_iso_R(iso, 128, 'cuda', 'full')
    R_T_even, R_T_odd = _split_rot_T(R)

    for N in [1, 8, 64]:
        x = torch.randn(N, 8, 128, dtype=torch.float16, device='cuda')
        # Warmup
        for _ in range(10):
            iso_compress(x, R_T_even, R_T_odd, iso.boundaries)
        torch.cuda.synchronize()
        # Timed
        t0 = time.perf_counter()
        for _ in range(100):
            iso_compress(x, R_T_even, R_T_odd, iso.boundaries)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 100
        us_token = ms * 1000 / N
        print(f"  N={N:3d}: {ms:.3f}ms batch, {us_token:.1f} us/token")


def bench_planar():
    print("=== PlanarQuant compress ===")
    planar = PlanarQuantMSE(128, 3, seed=42, device='cuda')
    R = _build_planar_R(planar, 128, 'cuda')
    R_T_even, R_T_odd = _split_rot_T(R)

    for N in [1, 8, 64]:
        x = torch.randn(N, 8, 128, dtype=torch.float16, device='cuda')
        # Warmup
        for _ in range(10):
            planar_compress(x, R_T_even, R_T_odd, planar.boundaries)
        torch.cuda.synchronize()
        # Timed
        t0 = time.perf_counter()
        for _ in range(100):
            planar_compress(x, R_T_even, R_T_odd, planar.boundaries)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 100
        us_token = ms * 1000 / N
        print(f"  N={N:3d}: {ms:.3f}ms batch, {us_token:.1f} us/token")


def main():
    bench_iso_fast()
    print()
    bench_iso_full()
    print()
    bench_planar()


if __name__ == "__main__":
    main()
