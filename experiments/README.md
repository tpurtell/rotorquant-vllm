# RotorQuant Experiments

Benchmark scripts for the RotorQuant KV cache compression kernels.

## Running

All benchmarks require CUDA. Run from the project root:

```bash
python -m experiments.bench_compress
python -m experiments.bench_decompress
python -m experiments.bench_roundtrip
```

## Scripts

- **bench_compress.py** — Measures compress kernel latency across batch sizes (1, 8, 64) for IsoQuant (fast and full modes) and PlanarQuant.
- **bench_decompress.py** — Measures decompress kernel latency and effective memory bandwidth across cache sequence lengths (256 to 16384 tokens) for IsoQuant and PlanarQuant.
- **bench_roundtrip.py** — Reports compression ratios (FP16 vs nibble-packed) and roundtrip quality (cosine similarity and MSE) for 2-bit through 4-bit quantization across all modes.
