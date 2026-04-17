# GPU Acceleration in pyMatRad

## Hardware: NVIDIA GeForce RTX 5060 (GB206, Blackwell)

The RTX 5060 can accelerate several parts of the photon dose calculation pipeline.
This document describes where GPU would help, what the blockers are, and how to
get started.

---

## Where GPU Would Help

### 1. Per-ray dose math (`_calc_beam_worker`) — high value

The inner loop per beam is:
- `np.exp(-m * rd)`, `np.exp(-beta * rd)` — element-wise exponentials over
  ~2000-element arrays, repeated 1000+ times per beam
- Matrix multiply + sum over 3 scatter kernels
- `RegularGridInterpolator` calls for lateral kernel values

This is embarrassingly parallel and maps directly to GPU. **CuPy** (NVIDIA's
NumPy drop-in) would replace the numpy calls and run them on-device. The `exp`
operations are especially fast on GPU (dedicated transcendental function units).

### 2. Ray tracing (`ray_tracing_fast`) — high value

The KDTree query (`scipy.spatial.cKDTree`) and the vectorized Siddon depth
computation loop over rays are both parallelizable. GPU alternatives:

- **cuSpatial** (RAPIDS) has a GPU KD-tree
- A custom CUDA Siddon tracer (open-source implementations exist in C++/CUDA)
  would run all rays simultaneously

Ray tracing is typically the dominant cost in pencil beam engines.

### 3. Sparse matrix assembly — moderate value

**cuSPARSE** (via CuPy) can build CSC/CSR matrices on GPU and transfer to host
at the end. For large DIJ matrices this reduces assembly time and avoids the
CPU-side `np.concatenate` memory peak.

---

## Blockers for the RTX 5060 Specifically

### Driver / CUDA compatibility

The GB206 is Blackwell architecture. CuPy pre-built wheels for Blackwell were
added in **CuPy 13.x** (CUDA 12.8+). Check:

```bash
nvidia-smi           # CUDA version should be >= 12.8
python -c "import cupy; print(cupy.__version__)"
python -c "import cupy; print(cupy.cuda.Device(0).compute_capability)"
# Blackwell should report 120 (SM 12.0)
```

If CuPy is not available for your CUDA version, build from source against the
CUDA toolkit.

### VRAM budget

The RTX 5060 has 8–12 GB GDDR7. For this workload:

| Data | Size |
|------|------|
| WED CT cube at 5 mm grid | ~50–100 MB |
| Per-beam rad_depths / iso_lat_x/z / geo_dists | 4 × n_voxels × 8 bytes |
| Kernel convolution matrices | < 1 MB |

Everything fits comfortably. The WED CT and dose grid coordinates can be pinned
to device once and reused across all beams — no per-beam transfer overhead.

### `RegularGridInterpolator` — no direct CuPy equivalent

The kernel interpolators (`ik[ki](pts)`) use `scipy.interpolate.RegularGridInterpolator`.
CuPy does not have this. Options:

- Re-implement bilinear interpolation manually in CuPy (~10 lines)
- Use `cupyx.scipy.ndimage.map_coordinates`

This is the main missing piece before CuPy can handle the full worker loop.

---

## Expected Speedup

| Component | CPU baseline | GPU estimate | Notes |
|-----------|-------------|--------------|-------|
| Ray tracing (Siddon) | dominant | ~10× | Custom CUDA or cuSpatial |
| Per-beam exp + multiply | moderate | ~5–20× | CuPy trivial to apply |
| Kernel interpolation | moderate | ~5× | needs manual CuPy port |
| Sparse matrix assembly | small | ~2–3× | cuSPARSE |

Overall wall-clock speedup depends on Python/I/O overhead, but **5–15× on the
dose math** is realistic once the CUDA stack is working.

---

## Relevant Source Files

| File | GPU-relevant section |
|------|----------------------|
| `matRad/doseCalc/DoseEngines/photon_svd_engine.py` | `_calc_beam_worker` (Phase 2 dose math) |
| `matRad/rayTracing/siddon.py` | `ray_tracing_fast` (KDTree + Siddon loop) |
| `matRad/rayTracing/_backends/` | Drop-in backend for a CUDA Siddon tracer |

---

## Practical Starting Point

1. Verify the CUDA stack is working:
   ```bash
   nvcc --version
   nvidia-smi
   conda install -c conda-forge cupy cuda-version=12.8
   python -c "import cupy; cupy.zeros(1)"
   ```

2. Check compute capability:
   ```bash
   python -c "import cupy; print(cupy.cuda.Device(0).compute_capability)"
   ```

3. The fastest first win is replacing numpy in `_calc_beam_worker` with CuPy,
   since that is the per-beam computation path (and the OOM-sensitive section
   addressed in the streaming mode). It is pure array math with no data
   structure changes required.
