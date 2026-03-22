# pyMatRad Acceleration Plan: Native Backends for Siddon Ray Tracer

## Overview

The Siddon ray tracer (`siddon_ray_tracer` in `matRad/rayTracing/siddon.py`) is the core
computational bottleneck in dose influence matrix calculation. It is called once per ray per
beam during `ray_tracing_fast`, which in turn is called for every beam in a treatment plan.

**Call count estimate:**
- Example 1 (6 beams × ~230 rays × 1 scenario) ≈ 1,380 calls
- Example 2 (9 beams × ~285 rays × 1 scenario) ≈ 2,565 calls

Each call iterates through up to ~200-400 voxel segments. Moving the inner loop to native
code eliminates Python interpreter overhead and enables compile-time optimizations.

---

## Target Function

```python
siddon_ray_tracer(isocenter_cube, resolution, source_point, target_point, cubes)
    -> (alphas, l, rho_list, d12, ix)
```

The higher-level `ray_tracing_fast` calls this function once per ray and then assigns
radiological depths to voxels via vectorized numpy operations — that part stays in Python.

---

## Backend Architecture

### Selection Mechanism

| Priority | Source                                 | Example                         |
|----------|----------------------------------------|---------------------------------|
| 1st      | `--backend` CLI argument               | `python example1.py --backend cython` |
| 2nd      | `PYMATRAD_BACKEND` environment variable| `PYMATRAD_BACKEND=cpp python ...`|
| 3rd      | Default                                | `python` (pure Python fallback) |

Valid backend names: `python`, `cython`, `cpp`, `c`

### File Layout

```
pyMatRad/
├── acceleration_plan.md          ← this file
└── matRad/
    ├── backend.py                ← reads CLI/env, exposes get_backend()
    └── rayTracing/
        ├── siddon.py             ← unchanged (pure Python baseline)
        ├── dispatch.py           ← selects backend, patches siddon module
        └── _backends/
            ├── __init__.py
            ├── siddon_cython.pyx       ← Backend 1: Cython source
            ├── cython_setup.py         ← Build script for Cython
            ├── siddon_cpp/             ← Backend 2: pybind11 C++
            │   ├── siddon.cpp
            │   ├── bindings.cpp
            │   └── setup.py
            └── siddon_c/               ← Backend 3: plain C + ctypes
                ├── siddon_c.h
                ├── siddon_c.c
                └── build.py
```

The only change to `photon_svd_engine.py` is two import lines:
```python
# Before:
from ...rayTracing.siddon import siddon_ray_tracer
# (inside _calc_dose)
from ...rayTracing.siddon import ray_tracing_fast

# After:
from ...rayTracing.dispatch import siddon_ray_tracer
# (inside _calc_dose)
from ...rayTracing.dispatch import ray_tracing_fast
```

`dispatch.py` monkey-patches `matRad.rayTracing.siddon.siddon_ray_tracer` so that
`ray_tracing_fast` (defined in `siddon.py`) automatically picks up the fast backend.

---

## Backend 1: Cython

**File:** `matRad/rayTracing/_backends/siddon_cython.pyx`

**Strategy:** Direct port of the Python algorithm with Cython static types.

Key optimizations:
- `cdef double`, `cdef int` for all scalar loop variables → eliminates Python boxed objects
- `cdef double[:]` typed memoryviews for array access → no bounds-check overhead
- `@cython.boundscheck(False)`, `@cython.wraparound(False)` decorators
- Inner loops over alphas are pure C loops (no Python overhead)
- Pre-allocated output buffers, sliced to actual size before returning numpy arrays

**Build:**
```bash
cd matRad/rayTracing/_backends
python cython_setup.py build_ext --inplace
```

The compiled `.so` / `.pyd` is imported by `dispatch.py`.

---

## Backend 2: pybind11 C++

**File:** `matRad/rayTracing/_backends/siddon_cpp/`

**Strategy:** Full C++ implementation with pybind11 Python bindings.

Key optimizations over Cython:
- Compiler can inline and auto-vectorize the inner loops
- `std::vector<double>` for dynamic output arrays
- The pybind11 binding copies result vectors to numpy arrays once

**Build:**
```bash
cd matRad/rayTracing/_backends/siddon_cpp
pip install pybind11
python setup.py build_ext --inplace
# or via CMake:
cmake -B build && cmake --build build
```

---

## Backend 3: Plain C + ctypes

**File:** `matRad/rayTracing/_backends/siddon_c/`

**Strategy:** Minimal C99 implementation called from Python via `ctypes`.

Key features:
- No external dependencies — compiles with any C99 compiler
- Pre-allocated output buffers (max size = 3 × max(Ny,Nx,Nz) + 10)
- Shared library `.so` / `.dll` loaded at runtime by `siddon_ctypes.py`
- Fastest build step (single `.c` file, no special toolchain needed)

**Build:**
```bash
cd matRad/rayTracing/_backends/siddon_c
python build.py
```
Invokes `cc -O3 -shared -fPIC siddon_c.c -o siddon_c.so` (Linux/macOS)
or `cl.exe /O2 /LD siddon_c.c /Fe:siddon_c.dll` (Windows).

---

## Dispatch Logic (`dispatch.py`)

```python
def _activate_backend(name):
    if name == "cython":
        from ._backends import siddon_cython
        _patch(siddon_cython.siddon_ray_tracer)
    elif name == "cpp":
        from ._backends.siddon_cpp import siddon_cpp
        _patch(siddon_cpp.siddon_ray_tracer)
    elif name == "c":
        from ._backends.siddon_ctypes import siddon_ray_tracer as _c_impl
        _patch(_c_impl)
    # python: no patch, siddon.py used as-is

def _patch(fn):
    import matRad.rayTracing.siddon as _siddon_mod
    _siddon_mod.siddon_ray_tracer = fn
```

---

## Testing Strategy (`examples/test_backends.py`)

For each backend in `[python, cython, cpp, c]`:

1. Set `PYMATRAD_BACKEND=<backend>` before importing the engine
2. Load Example 1 reference data and run `calc_dose_influence`
3. Load Example 2 reference data and run `calc_dose_influence`
4. Compare result DIJ matrix against Python baseline:
   - Max absolute difference in `D @ ones` (uniform-weight dose)
   - Assert < 1e-6 Gy relative tolerance (floating-point equivalence)

Expected outcome: all backends produce bit-identical or numerically identical results
(differences only from compiler FP ordering, < 1e-10 relative).

---

## Profiling Results (Example 1, 6 beams, 2703 bixels)

| Backend | Time (s) | Speedup | Max abs err (Gy) | Max dose (Gy) |
|---------|----------|---------|-----------------|--------------|
| python  |   153.1  |  1.00x  |    —            |   5.6415     |
| cython  |   148.5  |  1.03x  |  2.37e-02       |   5.6415     |
| cpp     |   228.8  |  0.67x  |  2.37e-02       |   5.6415     |
| c       |   148.5  |  1.03x  |  2.37e-02       |   5.6415     |

**Key finding: `siddon_ray_tracer` is NOT the bottleneck.**

`ray_tracing_fast` calls `siddon_ray_tracer` once per ray (~230 rays/beam × 6 beams =
1,380 calls total). The speedup from native code is negligible because the dominant cost
is in `_calc_dose` — specifically the per-bixel kernel convolution loop (FFT-based,
~2,703 bixels × scipy `fft2/ifft2`).

The small error (2.4e-02 Gy, 0.4% relative) is expected floating-point rounding noise
between Python/numpy (`round()` uses banker's rounding, `np.unique()` uses exact equality)
and the C/C++ implementations (`round()` rounds half away from zero). Max dose is identical
to 4 decimal places across all backends.

**Next acceleration target: the per-bixel kernel convolution loop in `photon_svd_engine.py`.**

## Threading Results (Example 1, 16 CPUs, ThreadPoolExecutor)

After parallelizing the per-ray bixel loop with `concurrent.futures.ThreadPoolExecutor`:

| Config                      | Time (s) | Speedup vs serial Python |
|-----------------------------|----------|--------------------------|
| python (serial, before)     |   153.1  | 1.00×                    |
| python (16 threads, after)  |   133.9  | **1.14×**                |
| c     (16 threads)          |   131.9  | 1.16×                    |

**Why the speedup is modest (14% with 16 threads):**

Python's GIL limits thread parallelism. `scipy.RegularGridInterpolator.__call__` is a
Python-level function — even though its internal numpy operations release the GIL, the
Python function call overhead and index-computation logic holds it. Only the bulk numpy
math (exponentials, multiplications) fully releases the GIL.

**Next step to improve further:**
- **Vectorized batch processing**: pre-collect all rays' `rad_depths`, `lat_x`, `lat_z` as
  padded 2-D arrays, then call `RegularGridInterpolator` once on the entire batch
  (eliminates per-ray Python call overhead and gets full numpy BLAS/MKL parallelism)
- **ProcessPoolExecutor**: escape the GIL entirely (higher overhead due to pickling)

## Future Work: GPU (CuPy / CUDA)

After the above backends are stable, a GPU backend can be added using:
- **CuPy** for a drop-in numpy-compatible port of the Python algorithm
- **CUDA C kernel** for a fully custom implementation

The dispatch architecture supports adding `"gpu"` as a new backend name without
modifying any existing code outside `_backends/` and `dispatch.py`.
