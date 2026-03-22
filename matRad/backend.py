"""
Backend selector for pyMatRad ray-tracing acceleration.

Priority:
  1. --backend <name>  CLI argument (parsed from sys.argv at import time)
  2. PYMATRAD_BACKEND  environment variable
  3. Default: 'python'

Valid names: 'python', 'cython', 'cpp', 'c'
"""

import os
import sys

_VALID_BACKENDS = ("python", "cython", "cpp", "c")
_backend_name: str = "python"


def _parse_cli() -> str | None:
    """Return backend name from --backend <name> in sys.argv, or None."""
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--backend" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--backend="):
            return arg.split("=", 1)[1]
    return None


def _init() -> str:
    cli = _parse_cli()
    if cli is not None:
        return cli
    return os.environ.get("PYMATRAD_BACKEND", "python")


_backend_name = _init()

if _backend_name not in _VALID_BACKENDS:
    import warnings
    warnings.warn(
        f"Unknown backend '{_backend_name}'. Valid choices: {_VALID_BACKENDS}. "
        "Falling back to 'python'.",
        RuntimeWarning,
        stacklevel=2,
    )
    _backend_name = "python"


def get_backend() -> str:
    """Return the active backend name."""
    return _backend_name


def set_backend(name: str) -> None:
    """Override the backend at runtime (e.g., for testing)."""
    global _backend_name
    if name not in _VALID_BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Valid: {_VALID_BACKENDS}")
    _backend_name = name
