"""
Build script for the Cython Siddon backend.

Usage:
    cd matRad/rayTracing/_backends
    python cython_setup.py build_ext --inplace
"""

from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize
except ImportError:
    raise SystemExit(
        "Cython is not installed.  Run: pip install cython"
    )

ext = Extension(
    name="siddon_cython",
    sources=["siddon_cython.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3"],
    language="c",
)

setup(
    name="siddon_cython",
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
        annotate=False,
    ),
)
