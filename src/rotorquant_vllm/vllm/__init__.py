"""RotorQuant vLLM plugin integration.

Exports:
    RQAttentionBackend: Custom attention backend registered as CUSTOM.
    RQFullAttentionSpec: KV cache spec with RotorQuant page size.
    register_rq_backend: Callable to register the backend manually.

Usage:
    The backend registers automatically via the ``vllm.general_plugins``
    entry point when rotorquant-vllm is installed::

        pip install rotorquant-vllm
        vllm serve <model> --attention-backend CUSTOM

    Or register manually before starting vLLM::

        from rotorquant_vllm.vllm import register_rq_backend
        register_rq_backend()
"""

from rotorquant_vllm.vllm.rq_backend import (
    RQAttentionBackend,
    RQFullAttentionSpec,
    register_rq_backend,
)

__all__ = [
    "RQAttentionBackend",
    "RQFullAttentionSpec",
    "register_rq_backend",
]
