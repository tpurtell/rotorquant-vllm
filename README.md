# rotorquant-vllm

Drop-in vLLM plugin for RotorQuant KV cache compression — IsoQuant (quaternion 4D) and PlanarQuant (Givens 2D) block-diagonal rotations with Lloyd-Max scalar quantization.

## Install

```bash
pip install -e .
```

## Quick Start

```bash
# IsoQuant 3-bit (default)
vllm serve <model> --attention-backend CUSTOM

# PlanarQuant 4-bit
RQ_MODE=planar RQ_BITS=4 vllm serve <model> --attention-backend CUSTOM

# Asymmetric K/V
RQ_K_BITS=4 RQ_V_BITS=3 vllm serve <model> --attention-backend CUSTOM
```

No code changes required. The plugin registers automatically via vLLM's plugin system.

## Configuration

| Env Var | Values | Default | Description |
|---------|--------|---------|-------------|
| `RQ_MODE` | `iso`, `planar` | `iso` | Rotation method |
| `RQ_BITS` | `3`, `4` | `3` | Quantization bits |
| `RQ_K_BITS` | `3`, `4` | same as `RQ_BITS` | Key bits (asymmetric override) |
| `RQ_V_BITS` | `3`, `4` | same as `RQ_BITS` | Value bits (asymmetric override) |
| `RQ_ISO_MODE` | `fast`, `full` | `fast` | IsoQuant: single vs dual quaternion |

## How It Works

RotorQuant replaces TurboQuant's full d×d Walsh-Hadamard Transform with small block-diagonal rotations:

1. **Normalize** — extract and store vector norms separately
2. **Rotate** — via 4D quaternion (IsoQuant) or 2D Givens (PlanarQuant) blocks
3. **Quantize** — scalar Lloyd-Max on each rotated coordinate
4. **Inverse rotate** — reconstruct via inverse rotation on decompression

Block rotations are O(d) FMAs vs TurboQuant's O(d log d) butterfly network, with 44x fewer parameters (128 vs 16,384 per head at d=128).

### Cache Layout

Packed uint8 cache: `(num_blocks, block_size, total_bytes)` where each token slot stores:

```
[K_indices(128 bytes) | K_norms(4 bytes)] × num_heads
[V_indices(128 bytes) | V_norms(4 bytes)] × num_heads
```

For head_dim=128: 264 bytes vs 512 bytes FP16 = ~1.94x compression for K+V combined. K-only: ~1.94x.

## Architecture

```
src/rotorquant_vllm/
├── __init__.py                  # Package exports
├── quantization/
│   ├── lloyd_max.py             # Lloyd-Max optimal scalar quantizer
│   ├── isoquant.py              # IsoQuantMSE (quaternion 4D rotation)
│   └── planarquant.py           # PlanarQuantMSE (Givens 2D rotation)
├── triton/
│   ├── isoquant_compress.py     # IsoQuant compress Triton kernel
│   ├── isoquant_decompress.py   # IsoQuant decompress Triton kernel
│   ├── planarquant_compress.py  # PlanarQuant compress Triton kernel
│   └── planarquant_decompress.py # PlanarQuant decompress Triton kernel
└── vllm/
    └── rq_backend.py            # RQAttentionBackend, RQAttentionImpl, registration
```

## Status

Implemented:

- [x] Core quantization (IsoQuant, PlanarQuant)
- [x] Triton compress/decompress kernels with CPU fallbacks
- [x] Nibble-packed KV cache
- [x] Boundary-based bucketize in compress kernels
- [x] Pre-split rotation matrices for tiled compress
- [x] Tiled compress kernels
- [x] Paged decompression
- [x] Fused paged decode kernel
- [x] Fused INT8 prefill kernel
- [x] CUDA graph support for uniform single-token decode
- [x] Hybrid-model page alignment
- [x] VLM/mm_prefix support
- [x] Experiment scripts and integration tests

Remaining limitation:

- [ ] Cascade attention (matches turboquant-vllm: not implemented)
## Credits

This project was built from the RotorQuant and TurboQuant-vLLM work in the sibling repositories in this directory:

- [RotorQuant](https://github.com/scrya-com/rotorquant) — source quantization work and reference implementations for IsoQuant/PlanarQuant, Lloyd-Max codebooks, and the underlying RotorQuant design.
- [TurboQuantVLLM](https://github.com/Alberto-Codes/turboquant-vllm) — reference vLLM plugin architecture, Triton kernel structure, fused paged decode, INT8 prefill, and performance benchmarking patterns.

This repository is a RotorQuant-based vLLM backend built by adapting ideas and implementation patterns from both codebases to RotorQuant's block-diagonal rotation design.

## References

- [RotorQuant paper](https://www.scrya.com/rotorquant.pdf) — Clifford algebra vector quantization
- [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) — Google's KV cache compression
- [IsoQuant / PlanarQuant](https://github.com/ParaMind2025/isoquant) — Block-diagonal rotation quantizers

## License

MIT
