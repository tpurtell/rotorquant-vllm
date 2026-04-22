"""Benchmark: roundtrip quality and compression ratio."""
import torch
from rotorquant_vllm.quantization import IsoQuantMSE, PlanarQuantMSE
from rotorquant_vllm.vllm.rq_backend import _build_iso_R, _build_planar_R, _rq_bytes_per_token_kv
from rotorquant_vllm.triton import iso_compress, iso_decompress, planar_compress, planar_decompress


def _split_rot_T(R):
    """Split R^T into even/odd column matrices for compress kernels."""
    R_T = R.T.contiguous()
    R_T_even = R_T[:, 0::2].contiguous()
    R_T_odd = R_T[:, 1::2].contiguous()
    return R_T_even, R_T_odd


def bench_compression_ratio():
    print("=== Compression ratio ===")
    for D in [64, 128, 256]:
        H = 8
        fp16_bytes = H * D * 2 * 2  # K+V
        compressed = _rq_bytes_per_token_kv(D, H)
        print(f"  D={D:3d}, H={H}: FP16={fp16_bytes:5d}B, compressed={compressed:5d}B, ratio={fp16_bytes / compressed:.2f}x")


def bench_iso_roundtrip():
    print("=== IsoQuant (fast) roundtrip quality ===")
    for bits in [2, 3, 4]:
        q = IsoQuantMSE(128, bits, seed=42, mode='fast')
        R = _build_iso_R(q, 128, 'cpu', 'fast')
        R_T_even, R_T_odd = _split_rot_T(R)

        x = torch.randn(100, 128, dtype=torch.float32)
        x_norm = x / x.norm(dim=-1, keepdim=True)

        packed, norms = iso_compress(x_norm.unsqueeze(0), R_T_even, R_T_odd, q.boundaries)
        decomp = iso_decompress(packed, norms, q.centroids, torch.float32).squeeze(0)

        # Decompress returns data in rotated space; rotate original for comparison
        x_rotated = x_norm @ R.T

        cos_sim = (x_rotated * decomp).sum(dim=-1).mean().item()
        mse = ((x_rotated - decomp) ** 2).mean().item()
        print(f"  iso(fast) {bits}-bit: cos_sim={cos_sim:.4f}, MSE={mse:.6f}")


def bench_iso_full_roundtrip():
    print("=== IsoQuant (full) roundtrip quality ===")
    for bits in [2, 3, 4]:
        q = IsoQuantMSE(128, bits, seed=42, mode='full')
        R = _build_iso_R(q, 128, 'cpu', 'full')
        R_T_even, R_T_odd = _split_rot_T(R)

        x = torch.randn(100, 128, dtype=torch.float32)
        x_norm = x / x.norm(dim=-1, keepdim=True)

        packed, norms = iso_compress(x_norm.unsqueeze(0), R_T_even, R_T_odd, q.boundaries)
        decomp = iso_decompress(packed, norms, q.centroids, torch.float32).squeeze(0)

        # Decompress returns data in rotated space; rotate original for comparison
        x_rotated = x_norm @ R.T

        cos_sim = (x_rotated * decomp).sum(dim=-1).mean().item()
        mse = ((x_rotated - decomp) ** 2).mean().item()
        print(f"  iso(full) {bits}-bit: cos_sim={cos_sim:.4f}, MSE={mse:.6f}")


def bench_planar_roundtrip():
    print("=== PlanarQuant roundtrip quality ===")
    for bits in [2, 3, 4]:
        q = PlanarQuantMSE(128, bits, seed=42)
        R = _build_planar_R(q, 128, 'cpu')
        R_T_even, R_T_odd = _split_rot_T(R)

        x = torch.randn(100, 128, dtype=torch.float32)
        x_norm = x / x.norm(dim=-1, keepdim=True)

        packed, norms = planar_compress(x_norm.unsqueeze(0), R_T_even, R_T_odd, q.boundaries)
        decomp = planar_decompress(packed, norms, q.centroids, torch.float32).squeeze(0)

        # Decompress returns data in rotated space; rotate original for comparison
        x_rotated = x_norm @ R.T

        cos_sim = (x_rotated * decomp).sum(dim=-1).mean().item()
        mse = ((x_rotated - decomp) ** 2).mean().item()
        print(f"  planar {bits}-bit: cos_sim={cos_sim:.4f}, MSE={mse:.6f}")


def main():
    bench_compression_ratio()
    print()
    bench_iso_roundtrip()
    print()
    bench_iso_full_roundtrip()
    print()
    bench_planar_roundtrip()


if __name__ == "__main__":
    main()
