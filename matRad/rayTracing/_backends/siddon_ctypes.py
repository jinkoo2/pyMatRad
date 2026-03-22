"""
ctypes wrapper for the plain-C Siddon backend.

Loads siddon_c.so / siddon_c.dll from the siddon_c/ sub-directory.
Must be compiled first:
    cd matRad/rayTracing/_backends/siddon_c
    python build.py
"""

import ctypes
import os
import platform

import numpy as np

# ------------------------------------------------------------------ load lib
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR = os.path.join(_HERE, "siddon_c")

if platform.system() == "Windows":
    _LIB_FILE = os.path.join(_LIB_DIR, "siddon_c.dll")
else:
    _LIB_FILE = os.path.join(_LIB_DIR, "siddon_c.so")

if not os.path.isfile(_LIB_FILE):
    raise ImportError(
        f"C backend not built.  Run:\n"
        f"  cd {_LIB_DIR}\n"
        f"  python build.py"
    )

_lib = ctypes.CDLL(_LIB_FILE)

# void signature
_lib.siddon_ray_tracer_c.restype  = ctypes.c_int
_lib.siddon_ray_tracer_c.argtypes = [
    # iso
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    # resolution
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    # source
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    # target
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    # cube + shape
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    # outputs
    ctypes.POINTER(ctypes.c_double),   # alphas  (n_seg+1)
    ctypes.POINTER(ctypes.c_double),   # l       (n_seg)
    ctypes.POINTER(ctypes.c_double),   # rho     (n_seg)
    ctypes.POINTER(ctypes.c_longlong), # ix      (n_seg)
    ctypes.POINTER(ctypes.c_double),   # d12
    ctypes.c_int,                      # max_buf
]


def siddon_ray_tracer(isocenter_cube, resolution, source_point, target_point, cubes):
    """
    Drop-in replacement for siddon.siddon_ray_tracer using the C backend.
    """
    iso = np.asarray(isocenter_cube, dtype=np.float64)
    sp  = np.asarray(source_point,   dtype=np.float64)
    tp  = np.asarray(target_point,   dtype=np.float64)

    rx, ry, rz = resolution["x"], resolution["y"], resolution["z"]

    cube0 = np.asfortranarray(cubes[0], dtype=np.float64)
    Ny, Nx, Nz = int(cube0.shape[0]), int(cube0.shape[1]), int(cube0.shape[2])

    # Upper bound on segments: 3 * max(Ny, Nx, Nz) + 10
    max_buf = 3 * max(Ny, Nx, Nz) + 10

    alphas_buf = np.empty(max_buf + 1, dtype=np.float64)
    l_buf      = np.empty(max_buf,     dtype=np.float64)
    rho_buf    = np.empty(max_buf,     dtype=np.float64)
    ix_buf     = np.empty(max_buf,     dtype=np.int64)
    d12_val    = ctypes.c_double(0.0)

    flat_cube  = cube0.ravel(order="F")

    n_seg = _lib.siddon_ray_tracer_c(
        iso[0], iso[1], iso[2],
        rx, ry, rz,
        sp[0], sp[1], sp[2],
        tp[0], tp[1], tp[2],
        flat_cube.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Ny, Nx, Nz,
        alphas_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        l_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rho_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ix_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        ctypes.byref(d12_val),
        max_buf,
    )

    d12 = float(d12_val.value)
    empty = np.array([], dtype=np.float64)

    if n_seg <= 0:
        return empty, empty, [empty for _ in cubes], d12, np.array([], dtype=np.int64)

    alphas = alphas_buf[: n_seg + 1].copy()
    l      = l_buf[:n_seg].copy()
    ix     = ix_buf[:n_seg].copy()

    # Build rho for each cube (first cube already computed; others need lookups)
    rho_list = []
    for cube in cubes:
        flat = np.asfortranarray(cube, dtype=np.float64).ravel(order="F")
        rho_list.append(flat[ix - 1])

    return alphas, l, rho_list, d12, ix
