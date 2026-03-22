"""
dispatch.py — runtime backend selector for the Siddon ray tracer.

This module is the single import point for both ``siddon_ray_tracer`` and
``ray_tracing_fast``.  It reads the active backend from ``matRad.backend``
and monkey-patches the siddon module so that ``ray_tracing_fast`` (defined
in siddon.py) transparently uses the fast native implementation.

Usage in photon_svd_engine.py:
    from ...rayTracing.dispatch import siddon_ray_tracer
    ...
    from ...rayTracing.dispatch import ray_tracing_fast
"""

from __future__ import annotations

import importlib
import warnings

# -- Import the pure-Python module (baseline, always available) ---------------
from . import siddon as _siddon_mod
from .siddon import ray_tracing_fast, siddon_ray_tracer   # noqa: F401  (re-exported)

# Re-export so callers can do: from .dispatch import ray_tracing_fast
__all__ = ["siddon_ray_tracer", "ray_tracing_fast"]


def _patch(fn) -> None:
    """Replace siddon_ray_tracer in the siddon module (and in this namespace)."""
    global siddon_ray_tracer
    _siddon_mod.siddon_ray_tracer = fn
    siddon_ray_tracer = fn


def activate(backend: str | None = None) -> str:
    """
    Activate a backend.

    Parameters
    ----------
    backend : str or None
        One of 'python', 'cython', 'cpp', 'c'.
        If None, reads from matRad.backend.get_backend().

    Returns
    -------
    str
        The name of the backend that was actually activated (may differ from
        the requested one if the requested backend failed to load).
    """
    if backend is None:
        from .. import backend as _be_mod
        backend = _be_mod.get_backend()

    if backend == "python":
        # reset to original
        _patch(_siddon_mod.__dict__["_orig_siddon_ray_tracer"]
               if "_orig_siddon_ray_tracer" in _siddon_mod.__dict__
               else _siddon_py_original)
        return "python"

    try:
        if backend == "cython":
            from ._backends import siddon_cython as _be
            _patch(_be.siddon_ray_tracer)

        elif backend == "cpp":
            # compiled module lives inside _backends/siddon_cpp/
            import sys, os
            _cpp_dir = os.path.join(os.path.dirname(__file__), "_backends", "siddon_cpp")
            if _cpp_dir not in sys.path:
                sys.path.insert(0, _cpp_dir)
            import siddon_cpp as _be  # type: ignore
            _patch(_be.siddon_ray_tracer)

        elif backend == "c":
            from ._backends.siddon_ctypes import siddon_ray_tracer as _c_fn
            _patch(_c_fn)

        else:
            raise ValueError(f"Unknown backend: {backend!r}")

        return backend

    except (ImportError, OSError) as exc:
        warnings.warn(
            f"Backend '{backend}' could not be loaded ({exc}).  "
            "Falling back to pure-Python backend.",
            RuntimeWarning,
            stacklevel=2,
        )
        _patch(_siddon_py_original)
        return "python"


# Keep a reference to the original Python implementation
_siddon_py_original = _siddon_mod.siddon_ray_tracer

# Activate on import
_active_backend = activate()
