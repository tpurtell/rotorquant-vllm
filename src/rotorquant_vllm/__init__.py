"""RotorQuant KV cache compression for vLLM.

Drop-in vLLM plugin using IsoQuant (quaternion 4D rotation) and
PlanarQuant (Givens 2D rotation) for KV cache quantization.

Usage::

    pip install rotorquant-vllm

    # IsoQuant 3-bit (default)
    vllm serve <model> --attention-backend CUSTOM

    # PlanarQuant 4-bit
    RQ_MODE=planar RQ_BITS=4 vllm serve <model> --attention-backend CUSTOM

    # Asymmetric K/V
    RQ_K_BITS=4 RQ_V_BITS=3 vllm serve <model> --attention-backend CUSTOM
"""

from rotorquant_vllm.quantization import IsoQuantMSE, LloydMaxCodebook, PlanarQuantMSE
from rotorquant_vllm.vllm import RQAttentionBackend, RQFullAttentionSpec, register_rq_backend

__all__ = [
    "IsoQuantMSE",
    "LloydMaxCodebook",
    "PlanarQuantMSE",
    "RQAttentionBackend",
    "RQFullAttentionSpec",
    "register_rq_backend",
]
