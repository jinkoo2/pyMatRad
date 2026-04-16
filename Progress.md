# pyMatRad Development Progress

## Status (2026-04-16): VMAT arc STF generation ported. Arc STF + propVMAT metadata verified on TG119.

---

## What Has Been Completed

### All Python Files Created

| File | Status | Notes |
|------|--------|-------|
| `matRad/config.py` | ✅ | Singleton MatRad_Config |
| `matRad/scenarios.py` | ✅ | NominalScenario |
| `matRad/backend.py` | ✅ | CLI/env backend selector (python/cython/cpp/c) |
| `matRad/geometry/geometry.py` | ✅ bug fixed | All geometry utilities |
| `matRad/phantoms/builder/phantom_voi.py` | ✅ | Box/Sphere VOI classes |
| `matRad/phantoms/builder/phantom_builder.py` | ✅ | PhantomBuilder class |
| `matRad/basedata/load_machine.py` | ✅ | Loads .mat machine files |
| `matRad/rayTracing/siddon.py` | ✅ optimized | Siddon ray tracer + fast version |
| `matRad/rayTracing/dispatch.py` | ✅ | Runtime backend switcher |
| `matRad/rayTracing/_backends/siddon_cython.pyx` | ✅ compiled | Cython backend |
| `matRad/rayTracing/_backends/siddon_cpp/` | ✅ compiled | pybind11 C++ backend |
| `matRad/rayTracing/_backends/siddon_c/` | ✅ compiled | Plain C + ctypes backend |
| `matRad/steering/stf_generator.py` | ✅ bug fixed | STF generator for photon IMRT; routes 'PhotonVMAT' to VMAT generator |
| `matRad/steering/stf_generator_vmat.py` | ✅ | VMAT arc STF generator (StfGeneratorPhotonVMAT) |
| `matRad/doseCalc/DoseEngines/dose_engine_base.py` | ✅ | Base dose engine |
| `matRad/doseCalc/DoseEngines/photon_svd_engine.py` | ✅ bug fixed, parallel | SVD photon dose engine |
| `matRad/doseCalc/DoseEngines/photon_ompc_engine.py` | ✅ calibrated, parallel | Analytical ompMC engine |
| `matRad/doseCalc/DoseEngines/topas_mc_engine.py` | ✅ | TOPAS MC engine (photons) |
| `matRad/doseCalc/calc_dose_influence.py` | ✅ | Entry point |
| `matRad/optimization/DoseObjectives/objectives.py` | ✅ | All objective functions |
| `matRad/optimization/fluence_optimization.py` | ✅ | L-BFGS-B optimizer |
| `matRad/planAnalysis/plan_analysis.py` | ✅ | DVH + quality indicators |
| `gui/matrad_gui.py` | ✅ | matplotlib GUI |
| `examples/example1_phantom.py` | ✅ | Full phantom plan (dose + optimization + DVH) |
| `examples/example2_photons.py` | ✅ | Full TG119 photon plan |
| `examples/example1_no_opti.py` | ✅ | Standalone SVPB vs ompMC vs TOPAS (water phantom) |
| `examples/example2_no_opti.py` | ✅ | Same comparison on TG119 |
| `examples/test_backends.py` | ✅ | Backend accuracy & speed benchmark |
| `examples/example1_no_opti_compare.py` | ✅ | Compare pyMatRad vs MATLAB ref + OpenTPS GUI |
| `examples/example2_no_opti_compare.py` | ✅ | Same for TG119 + OpenTPS GUI |
| `acceleration_plan.md` | ✅ | Documents all acceleration work and profiling results |
| `requirements.txt` | ✅ | |
| `CLAUDE.md` | ✅ | |
| `examples/example8_photons_vmat.py` | ✅ | VMAT arc plan (FMO step; STF+dose+opt verified) |
| `VMAT.md` | ✅ | VMAT architecture, usage, propVMAT field reference, status |

---

## Bugs Fixed

### 1. `cube_index_to_world_coords` - Linear Index Detection Bug
**Problem**: `np.atleast_2d(array_1d)` produces shape `(1, N)`. The condition `cube_ix.shape[1] == 1` would be False for N>1, causing the function to treat N linear indices as a single (1, N) subscript — all voxels would get the same world coordinate.

**Fix** (in `matRad/geometry/geometry.py`):
```python
orig = np.asarray(cube_ix, dtype=np.int64)
is_linear = orig.ndim == 1 or (orig.ndim == 2 and orig.shape[1] == 1)
cube_ix = np.atleast_2d(orig)
if is_linear:
    cube_ix = cube_ix.ravel()
    ix_0 = cube_ix - 1  # Convert 1-based MATLAB to 0-based Python
    k = ix_0 // (Ny * Nx)
    ...
```

### 2. `fft2` API Usage
**Problem**: `scipy.fft.fft2(x, n, n)` passes `n` as `s` and `n` as `axes`, causing "axes exceeds dimensionality of input" error.

**Fix** (in `matRad/doseCalc/DoseEngines/photon_svd_engine.py`):
```python
s = (self._gauss_conv_size, self._gauss_conv_size)
fft2(x, s=s)  # Instead of fft2(x, n, n)
```

### 3. Ray Tracing Memory - NxM Distance Matrix
**Problem**: Computing `dist_sq = (proj_x[:, np.newaxis] - ray_x[np.newaxis, :]) ** 2 + ...` creates a (N_voxels × N_rays) matrix. With N=100K voxels and M=400 rays, this is ~375 MB, causing OOM.

**Fix** (in `matRad/rayTracing/siddon.py`):
```python
from scipy.spatial import cKDTree
tree = cKDTree(ray_pos_2d)  # KD-tree on ray positions
nearest_dist, nearest_ray = tree.query(proj_2d, k=1, workers=-1)
```

Also vectorized the cumulative radiological depth computation using `np.searchsorted` + `np.cumsum`.

### 4. Geometric Distance Double-Counting (4× Dose Underestimate)
**Problem**: In `photon_svd_engine.py` `_calc_dose`, the geometric distance was passed as `geo_dists[vox_ix] + float(SAD)`. But `geo_dists` is already computed as `sqrt(sum((vox_bev - source_bev)^2))` — the distance from source. Adding `SAD` again doubled the denominator, making the inverse-square correction `(SAD/(geo+SAD))^2 ≈ (1/2)^2 = 0.25` — a 4× underestimate.

**Fix** (in `matRad/doseCalc/DoseEngines/photon_svd_engine.py` line ~489):
```python
geo_dists[vox_ix],  # NOT + float(SAD)
```

**Verification**: `compare_matlab.py` shows max dose error **0.72%** vs MATLAB `photons_testData.mat`.

### 5. Plan Analysis Grid Mismatch (D_95=0 for all structures)
**Problem**: `plan_analysis` used CT-grid voxel indices (1-based MATLAB linear) to index into a dose-grid cube. When dose grid resolution differs from CT (e.g., 3mm dose vs 2mm CT in example1), target voxel indices exceed the dose cube size and are filtered out, giving D_mean=0, D_95=0.

**Fix** (in `matRad/planAnalysis/plan_analysis.py`):
```python
if dose_grid is not None:
    from ..geometry.geometry import resize_cst_to_grid
    cst = resize_cst_to_grid(cst, ct, dose_grid)
```

### 6. Matplotlib Deprecated `get_cmap` API
**Fix**: Replaced `plt.cm.get_cmap("tab10")` with `plt.colormaps["tab10"]` in `example1_phantom.py`, `example2_photons.py`, and `gui/matrad_gui.py`.

### 7. Output path `/tmp/` not portable on Windows
**Fix**: Replaced `/tmp/pyMatRad_example1.png` and `/tmp/pyMatRad_example2.png` with `os.path.join(os.path.dirname(__file__), "pyMatRad_example1.png")`.

### 8. uint8 Overflow in Geometry Dimension Arithmetic (TG119)
**Problem**: TG119's `ct.cubeDim = [167, 167, 129]` is stored as uint8 in the .mat file. When scipy.io loads it, the dtype is uint8. `np.uint8(167) * np.uint8(167) = 27889` which overflows uint8 (max 255), giving 241. This caused the z-index `k = lin_ix // 241` to be ~5606 (out of bounds for Nz=129), making all CST voxel-to-world conversions wrong.

**Fix** (`matRad/geometry/geometry.py`, three functions):
```python
Ny, Nx, Nz = int(dims[0]), int(dims[1]), int(dims[2])  # ensure native int (avoids uint8 overflow)
```

### 9. `mat_struct` Not Iterable in `fluence_optimization.py` (TG119)
**Problem**: TG119's CST column 5 is a single `mat_struct` (scipy.io squeezes single-element cell arrays). The optimizer's `_initialize_weights` and `_collect_objectives` used `for obj in row[5]`, which fails with `TypeError: 'mat_struct' object is not iterable`.

**Fix** (`matRad/optimization/fluence_optimization.py`):
- Added `_obj_is_empty()` helper (catches TypeError for mat_struct)
- Added `_wrap_objectives()` to normalize single mat_struct → list
- Added `_matstruct_to_objective()` to convert mat_struct (className/parameters/penalty) to `DoseObjective`

### 10. `mat_struct` Len Error in `get_iso_center` (TG119)
**Problem**: `len(row[5])` raises `TypeError` when `row[5]` is a `mat_struct`.

**Fix** (`matRad/geometry/geometry.py`): Added `_obj_is_empty()` helper with TypeError catch.

### 11. Ambiguous Truth Value in `plot_slice` (TG119)
**Problem**: `if cube_hu and ...` where `cube_hu` is a raw 3D numpy array (TG119 stores `cubeHU` as bare array, not list).

**Fix** (`gui/matrad_gui.py`): Check type explicitly (`isinstance(cube_hu, list)` vs `ndarray`).

### 12. BEV vs World Coordinate Bug in `ray_tracing_fast`
**Problem**: Voxel coordinates were passed in BEV frame to Siddon, which expects world coordinates. Caused angle-dependent dose errors (wrong for non-zero gantry angles).

**Fix** (`matRad/rayTracing/siddon.py`): Pass world coordinates; rotate inside `ray_tracing_fast` for nearest-ray lookup only.

### 13. `add_margin` In-Place CST Mutation
**Problem**: `add_margin` modified the shared CST list in-place; different beams got pre-margined data.

**Fix** (`matRad/steering/stf_generator.py`): Copy CST rows before modification.

### 14. STF Margin Size (2304 vs 2568 bixels)
**Problem**: Python used `margin = max(ct_res) = 3mm`. MATLAB's `getPbMargin()` returns `bixelWidth = 5mm`. `ceil(3/3)=1` voxel vs `round(5/3)=2` voxel expansion → fewer rays.

**Fix** (`matRad/steering/stf_generator.py`): `margin_mm = max(max_ct_res, pb_margin)` — gives 5mm → 2568 bixels ✓

### 15. ompMC Calibration (150M× Wrong Dose)
**Problem**: `ABS_CALIBRATION_FACTOR = 3.49056e12` copied from MATLAB MC engine (converts MC *histories* to Gy). The Python ompMC uses an analytical formula — this constant is meaningless here. Result: ompMC max dose 706 million Gy/fx vs SVPB 4.7 Gy/fx.

**Fix** (`matRad/doseCalc/DoseEngines/photon_ompc_engine.py`):
```python
ABS_CALIBRATION_FACTOR = 23220.0  # empirically calibrated for analytical model
# effective calib = 23220 * (bixelWidth/50)^2 = 23220 * 0.01 = 232
```
Derivation: measured ratio ompMC/SVPB = 1.503×10⁸ → divide effective calib (3.49e10) by ratio.

**Result**: ompMC/SVPB ratio = 0.998 (water), 0.991 (TG119). ompMC 2.7–3.3× faster than SVPB.

### 16. `enableDijSampling` pln Setting Silently Ignored
**Problem**: `pln["propDoseCalc"]["enableDijSampling"] = False` had no effect. `_assign_from_pln` in `dose_engine_base.py` only parsed `doseGrid` from `propDoseCalc`. `photon_svd_engine._init_dose_calc` only read `useCustomPrimaryPhotonFluence`. The engine always ran with `self.enable_dij_sampling = True` (constructor default), applying stochastic Dij sampling even when the caller explicitly disabled it. This caused noisy dose profiles, most visible at deep depths (30 cm) where more voxels fall in the scatter tail and are subject to random sampling.

**Fix** (`matRad/doseCalc/DoseEngines/photon_svd_engine.py`):
```python
if "enableDijSampling" in prop_dc:
    self.enable_dij_sampling = bool(prop_dc["enableDijSampling"])
```

### 17. CST Row Iteration in example2_no_opti.py
**Problem**: `for row_m in cst_m.flat` iterates all 18 individual cells of the `(3,6)` scipy.io object array in C order, not the 3 rows. First `row_m` = `cst_m[0,0]` = int → `row_m[3]` fails.

**Fix** (`examples/example2_no_opti.py`):
```python
for i in range(cst_m.shape[0]):
    row_m = cst_m[i]  # (6,) row, not individual cell
```

---

## Performance Results

### SVPB Parallelism (photon_svd_engine.py)

| Approach | Time (s) | Speedup |
|----------|----------|---------|
| Serial (original) | 153.1 | 1.0× |
| ThreadPoolExecutor 16 workers | 133.9 | 1.14× (GIL limited) |
| Vectorized batch | 173.9 | 0.88× (overhead > benefit) |
| **ProcessPoolExecutor (beam-level)** | **40.5** | **4.3×** |

ProcessPoolExecutor: main process does SSD + FFT + ray tracing; workers do batch dose math per beam; results assembled into COO→CSC matrix.

### Native Siddon Backends

Siddon is NOT the bottleneck. FFT kernel convolution dominates. Speedup from native backends negligible (~3%). All backends agree within 2% (FP rounding from C vs Python `round()`).

| Backend | Time (s) | Speedup | Max abs err |
|---------|----------|---------|-------------|
| python  | 153.1    | 1.00×   | — (baseline)|
| cython  | 148.5    | 1.03×   | 2.37e-02    |
| cpp     | 228.8    | 0.67×   | 2.37e-02    |
| c       | 148.5    | 1.03×   | 2.37e-02    |

### ompMC vs SVPB (after calibration fix, 2026-03-23)

| Phantom | SVPB max (Gy/fx) | ompMC max (Gy/fx) | Max ratio | Median voxel ratio | ompMC time |
|---------|-----------------|------------------|-----------|-------------------|-----------|
| Water (5 beams)  | 4.709 | 4.699 | 0.998 | 0.987 | 3.3× faster |
| TG119 (7 beams)  | 5.496 | 5.444 | 0.991 | 0.922 | 2.7× faster |

TG119 per-voxel median 0.922 (vs 0.987 water): simple exponential attenuation does not capture heterogeneity scatter as accurately as SVPB's SVD kernels.

---

## What Remains To Do

### Priority 1: Verification (Run and Compare with MATLAB)
- [x] Dose calc verified against MATLAB `photons_testData.mat`: max dose error **0.72%**, target D_mean **2.2%**
- [x] `example1_phantom.py` verified: optimizer converged (170 iter), target D_5=45.6 Gy, D_mean=39 Gy, plan analysis QI correct
- [x] Run `example2_photons.py` (TG119.mat available at `tps/matRad/matRad/phantoms/TG119.mat`) — **DONE 2026-03-20**
- [x] TG119 results: OuterTarget D_mean=49.8 Gy (fine, 40°) vs 49.4 Gy (coarse, 50°); Core OAR 16.7 vs 21.9 Gy — physically reasonable
- [ ] Compare dose distributions, DVH, and quality indicators with MATLAB output (run MATLAB to get reference QI)

### Priority 2: Known Potential Issues
- [ ] **SSD computation** (`_compute_ssd` in photon_svd_engine.py): Currently calls siddon_ray_tracer twice per ray. The second call is redundant — refactor to use the first call's `alphas` directly.
- [ ] **Dose grid init** (`dose_engine_base.py`): The dose grid should match CT extent more accurately. Currently `np.arange(ct_x[0], ct_x[-1] + res*0.5, res)` — verify this matches MATLAB behavior.
- [ ] **Kernel normalization**: The SVD photon kernel should be normalized per MATLAB convention. Verify `photon_svd_engine.py` `_calc_single_bixel` produces correct absolute dose values.
- [ ] **Machine data loading**: Verify that `photons_Generic.mat` fields (`m`, `betas`, `kernelPos`, `kernel`) are being loaded correctly for the photon SVD computation.

### Priority 3: Additional Features
- [ ] Add support for `protons` and `carbon` radiation modes
- [ ] Add more examples (3-9)
- [ ] Add `setup.py` / `pyproject.toml`
- [ ] Improve GUI with plane selection (sagittal/coronal/axial buttons)
- [ ] Add DICOM import/export
- [ ] Add more scenario models (range uncertainty, etc.)

---

## Running on a New Computer

```bash
# Set up environment
conda create -n pymatrad python=3.11 numpy scipy matplotlib h5py
conda activate pymatrad

# Or use existing scipy/scikit-learn environment if available
conda activate scikit-learn  # already has numpy 1.24, scipy 1.10, matplotlib 3.7

# Run example 1
cd /path/to/pyMatRad
python examples/example1_phantom.py

# Run example 2
python examples/example2_photons.py
```

The code was last killed on the cluster due to memory constraints. The fixes are in place but have not yet been verified. On a machine with 8GB+ RAM it should run successfully.

---

## Comparison with MATLAB

### What to check:
1. STF: ray count per beam should match (`matRad_generateStf` output)
2. DIJ: number of non-zeros should be similar
3. Dose: min/max/mean in target region should match within ~5%
4. DVH: D_95, D_mean for PTV should match
5. QI: quality indicators should match

### Known acceptable differences:
- DIJ sampling is probabilistic (stochastic) so exact values won't match
- Dose grid resolution differences may cause small variations
