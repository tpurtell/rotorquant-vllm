"""Benchmark: decompress kernel latency and bandwidth."""
import torch
import time
from rotorquant_vllm.quantization import IsoQuantMSE, PlanarQuantMSE
from rotorquant_vllm.triton import iso_decompress, planar_decompress


def bench_iso_decompress():
    print("=== IsoQuant decompress ===")
    iso = IsoQuantMSE(128, 3, seed=42, mode='fast', device='cuda')
    HALF_D = 64  # 128 / 2 for nibble packing

    for cache_tokens in [256, 1024, 4096, 16384]:
        packed = torch.randint(0, 8, (cache_tokens, 8, HALF_D), dtype=torch.uint8, device='cuda')
        norms = torch.rand(cache_tokens, 8, 1, dtype=torch.float32, device='cuda')
        # Warmup
        for _ in range(10):
            iso_decompress(packed, norms, iso.centroids, torch.float16)
        torch.cuda.synchronize()
        # Timed
        t0 = time.perf_counter()
        for _ in range(50):
            iso_decompress(packed, norms, iso.centroids, torch.float16)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 50
        us_token = ms * 1000 / cache_tokens
        bytes_token = 128 * 2  # fp16 output
        bw = (bytes_token * cache_tokens / 1e9) / (ms / 1000)
        print(f"  tokens={cache_tokens:5d}: {us_token:6.2f} us/token, {bw:5.1f} GB/s")


def bench_planar_decompress():
    print("=== PlanarQuant decompress ===")
    planar = PlanarQuantMSE(128, 3, seed=42, device='cuda')
    HALF_D = 64  # 128 / 2 for nibble packing

    for cache_tokens in [256, 1024, 4096, 16384]:
        packed = torch.randint(0, 8, (cache_tokens, 8, HALF_D), dtype=torch.uint8, device='cuda')
        norms = torch.rand(cache_tokens, 8, 1, dtype=torch.float32, device='cuda')
        # Warmup
        for _ in range(10):
            planar_decompress(packed, norms, planar.centroids, torch.float16)
        torch.cuda.synchronize()
        # Timed
        t0 = time.perf_counter()
        for _ in range(50):
            planar_decompress(packed, norms, planar.centroids, torch.float16)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 50
        us_token = ms * 1000 / cache_tokens
        bytes_token = 128 * 2  # fp16 output
        bw = (bytes_token * cache_tokens / 1e9) / (ms / 1000)
        print(f"  tokens={cache_tokens:5d}: {us_token:6.2f} us/token, {bw:5.1f} GB/s")


def main():
    bench_iso_decompress()
    print()
    bench_planar_decompress()


if __name__ == "__main__":
    main()
