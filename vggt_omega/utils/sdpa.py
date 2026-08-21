"""Runtime controls for PyTorch scaled dot-product attention backends."""

from __future__ import annotations

from contextlib import nullcontext
import os
from typing import ContextManager

import torch


_BACKEND_ALIASES = {
    "flash": "FLASH_ATTENTION",
    "flash_attention": "FLASH_ATTENTION",
    "flash-attention": "FLASH_ATTENTION",
    "efficient": "EFFICIENT_ATTENTION",
    "mem_efficient": "EFFICIENT_ATTENTION",
    "memory_efficient": "EFFICIENT_ATTENTION",
    "math": "MATH",
    "cudnn": "CUDNN_ATTENTION",
}


def requested_sdpa_backend() -> str:
    """Return the requested SDPA backend policy.

    ``flash`` is the repository default because long-sequence inference should
    fail loudly if a path falls back to the quadratic math implementation.
    Set ``VGGT_SDPA_BACKEND=auto`` to restore PyTorch's normal dispatcher.
    """

    return os.environ.get("VGGT_SDPA_BACKEND", "flash").strip().lower()


def sdpa_kernel_from_env() -> ContextManager[None]:
    """Select the PyTorch SDPA backend requested by ``VGGT_SDPA_BACKEND``."""

    requested = requested_sdpa_backend()
    if requested in {"", "auto", "default", "pytorch"}:
        return nullcontext()

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except (ImportError, AttributeError) as exc:  # pragma: no cover - old PyTorch only.
        raise RuntimeError(
            "VGGT_SDPA_BACKEND requires torch.nn.attention.sdpa_kernel; "
            "set VGGT_SDPA_BACKEND=auto to use PyTorch's default dispatcher."
        ) from exc

    backends = []
    for item in requested.replace("+", ",").split(","):
        key = item.strip().lower()
        if not key:
            continue
        backend_name = _BACKEND_ALIASES.get(key)
        if backend_name is None:
            valid = ", ".join(sorted({"auto", *list(_BACKEND_ALIASES)}))
            raise ValueError(f"Unsupported VGGT_SDPA_BACKEND={requested!r}; valid values: {valid}")
        try:
            backends.append(getattr(SDPBackend, backend_name))
        except AttributeError as exc:
            raise RuntimeError(
                f"PyTorch {torch.__version__} does not expose SDPBackend.{backend_name}; "
                "set VGGT_SDPA_BACKEND=auto or choose another backend."
            ) from exc

    if not backends:
        return nullcontext()
    return sdpa_kernel(backends[0] if len(backends) == 1 else backends, set_priority=True)


def sdpa_runtime_status() -> dict[str, object]:
    """Small diagnostic payload for smoke tests and logs."""

    status: dict[str, object] = {"requested": requested_sdpa_backend()}
    if torch.cuda.is_available():
        try:
            status.update(
                flash_sdp_enabled=torch.backends.cuda.flash_sdp_enabled(),
                mem_efficient_sdp_enabled=torch.backends.cuda.mem_efficient_sdp_enabled(),
                math_sdp_enabled=torch.backends.cuda.math_sdp_enabled(),
            )
        except Exception as exc:  # pragma: no cover - backend API drift.
            status["backend_query_error"] = f"{type(exc).__name__}: {exc}"
    return status


__all__ = ["requested_sdpa_backend", "sdpa_kernel_from_env", "sdpa_runtime_status"]
