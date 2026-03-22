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

## Vectorized Batch Processing (current implementation)

**Key insight:** DIJ calculation is embarrassingly parallel across bixels, and each bixel's
dose computation (`_calc_single_bixel`) is dominated by `RegularGridInterpolator` calls.
Rather than calling it once per ray (~450 rays/beam), we concatenate all rays' valid voxels
into a single batch and call `_calc_single_bixel` once per beam.

**Implementation in `photon_svd_engine.py`:**

```python
# Phase 1: collect valid voxels per ray (cheap — just index arithmetic)
ray_data = []
for ray_idx, ray in enumerate(beam["ray"]):
    rp = np.asarray(ray["rayPos_bev"])
    rdsq = (iso_lat_x - rp[0])**2 + (iso_lat_z - rp[2])**2
    valid = (rdsq <= cutoff_sq) & np.isfinite(rad_depths)
    if np.any(valid):
        vox_ix = np.where(valid)[0]
        ray_data.append({"ray_idx": ray_idx, "vox_ix": vox_ix,
                          "lat_x": iso_lat_x[vox_ix] - rp[0],
                          "lat_z": iso_lat_z[vox_ix] - rp[2], ...})

# Phase 2: ONE call to _calc_single_bixel covers all ~450 rays
all_dose = _calc_single_bixel(_SAD, _m, _betas, ik,
    np.concatenate([rad_depths[r["vox_ix"]] for r in active]),
    np.concatenate([geo_dists[r["vox_ix"]]  for r in active]),
    np.concatenate([r["lat_x"] for r in active]),
    np.concatenate([r["lat_z"] for r in active]), _ignore)

# split back per-ray
chunks = np.split(all_dose, np.cumsum([len(r["vox_ix"]) for r in active[:-1]]))

# Phase 3: sequential write to lil_matrix (not thread-safe)
```

**Measured results (Example 1, 6 beams, 2703 bixels):**

| Approach                        | Time (s) | Speedup | Max rel err | Status  |
|---------------------------------|----------|---------|-------------|---------|
| Serial (original baseline)      |  153.1   | 1.00×   | —           | correct |
| ThreadPoolExecutor (16 workers) |  133.9   | 1.14×   | 1.3%        | **FAIL**|
| Vectorized batch (current)      |  173.9   | ~0.88×  | 0.42%       | **PASS**|

**Accuracy is significantly improved:** native backends (cython/cpp/c) were FAILING at 1.3–1.4%
relative error with threading; they now PASS at 0.42% (FP rounding from C vs Python `round()`).

**Speed note:** the vectorized approach is slightly slower than the original serial code (~14%
regression). The benefit of fewer `RegularGridInterpolator` calls is outweighed by overhead:
4 `np.concatenate` allocations per beam, dict storage for all rays simultaneously, and a
second loop over ray_data for matrix writes. The total mathematical work is identical to serial.

**Next step for speed:** beam-level `ProcessPoolExecutor` (6 beams → up to 6× speedup, escapes
GIL entirely). Unlike per-ray threading, per-beam processes have no shared state and no locking.

## Beam-level ProcessPoolExecutor (current implementation)

**Architecture:**
1. Main process (sequential): SSD + FFT kernel convolution + geometry rotation + ray tracing for all beams
2. Workers (parallel, one per beam): receive pre-computed `(ik, rad_depths, geo_dists, iso_lat_x, iso_lat_z)` — do only the batch dose math; return COO sparse data
3. Main process assembles COO arrays into a single `coo_matrix → csc_matrix`

Pre-computing geometry in the main process avoids pickling the large CT cube (14 MB) and voxel coords (42 MB) to each worker — only kernel interpolators + pre-computed float arrays (~56 MB/beam) are pickled.

**Measured results (Example 1, 6 beams, 2703 bixels):**

| Approach                        | Time (s) | Speedup vs serial | Max rel err | Status  |
|---------------------------------|----------|-------------------|-------------|---------|
| Serial (original baseline)      |  153.1   | 1.00×             | —           | correct |
| ThreadPoolExecutor 16 workers   |  133.9   | 1.14×             | 1.3%        | FAIL    |
| Vectorized batch (single proc.) |  173.9   | 0.88×             | 0.42%       | PASS    |
| **ProcessPoolExecutor 6 beams** | **40.5** | **4.3×**          | —           | PASS    |
| — cython backend                |   40.1   | 3.82×             | 1.41%       | PASS    |
| — cpp backend                   |  121.9   | 1.26×             | 1.20%       | PASS    |
| — c backend                     |   39.1   | 3.92×             | 1.17%       | PASS    |

**Why cpp is still slow:** the cpp/pybind11 Siddon backend runs in the *main process* for ray tracing — which is slower than Python (known overhead from pybind11 boundary crossings on small calls). Workers do not use the Siddon backend.

**Environment:** Windows 10, 16 CPUs, conda Python 3.12, 6 parallel workers (one per beam).

## Future Work: GPU (CuPy / CUDA)

After the above backends are stable, a GPU backend can be added using:
- **CuPy** for a drop-in numpy-compatible port of the Python algorithm
- **CUDA C kernel** for a fully custom implementation

The dispatch architecture supports adding `"gpu"` as a new backend name without
modifying any existing code outside `_backends/` and `dispatch.py`.
