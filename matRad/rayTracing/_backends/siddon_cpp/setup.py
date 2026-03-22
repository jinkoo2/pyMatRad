"""
Build script for the pybind11 C++ Siddon backend.

Usage:
    cd matRad/rayTracing/_backends/siddon_cpp
    python setup.py build_ext --inplace
    (or: python setup.py)       # direct MinGW build, bypasses setuptools

Produces siddon_cpp.<tag>.pyd (Windows) or siddon_cpp*.so (Linux/macOS).
"""

import os
import platform
import subprocess
import sys
import sysconfig

try:
    import pybind11
except ImportError:
    raise SystemExit("pybind11 not installed.  Run: pip install pybind11")

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_gxx():
    """Return path to g++ / x86_64-w64-mingw32-g++."""
    prefix = sysconfig.get_config_var("prefix") or sys.prefix
    candidates = [
        os.path.join(prefix, "Library", "bin", "x86_64-w64-mingw32-g++.exe"),
        os.path.join(prefix, "Library", "bin", "g++.exe"),
        "g++",
        "x86_64-w64-mingw32-g++",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
        # Try which / where
        which = subprocess.run(
            ["where", c] if platform.system() == "Windows" else ["which", c],
            capture_output=True, text=True,
        )
        if which.returncode == 0 and which.stdout.strip():
            return c
    return None


def build_direct():
    """Compile the pybind11 extension using the MinGW g++ directly."""
    gxx = _find_gxx()
    if not gxx:
        raise SystemExit("Could not find g++.  Install via conda: conda install gxx_win-64")

    inc_py  = sysconfig.get_path("include")
    inc_pb  = pybind11.get_include()
    lib_dir = sysconfig.get_config_var("prefix") or sys.prefix
    lib_dir = os.path.join(lib_dir, "libs")

    tag    = sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"
    out    = os.path.join(HERE, f"siddon_cpp{tag}")

    srcs = [
        os.path.join(HERE, "siddon.cpp"),
        os.path.join(HERE, "bindings.cpp"),
    ]

    cmd = [
        gxx,
        "-O3", "-shared", "-fPIC",
        "-std=c++17",
        f"-I{HERE}",
        f"-I{inc_py}",
        f"-I{inc_pb}",
        *srcs,
        f"-L{lib_dir}",
        f"-lpython{sysconfig.get_python_version().replace('.', '')}",
        "-o", out,
        "-lm",
    ]
    print("Building:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(result.returncode)
    print(f"Built: {out}")


if __name__ == "__main__":
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ("build_ext",)):
        build_direct()
    else:
        # Fallback: use setuptools with --inplace
        from setuptools import Extension, setup
        ext = Extension(
            name="siddon_cpp",
            sources=[
                os.path.join(HERE, "siddon.cpp"),
                os.path.join(HERE, "bindings.cpp"),
            ],
            include_dirs=[HERE, pybind11.get_include()],
            language="c++",
            extra_compile_args=["/O2", "/std:c++17"] if sys.platform == "win32"
                               else ["-O3", "-std=c++17"],
        )
        setup(name="siddon_cpp", ext_modules=[ext])
