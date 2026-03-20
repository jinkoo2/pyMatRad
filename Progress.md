# pyMatRad Development Progress

## Status: In Progress (partially working, needs testing on better hardware)

---

## What Has Been Completed

### All Core Python Files Created

| File | Status | Notes |
|------|--------|-------|
| `matRad/config.py` | ✅ Done | Singleton MatRad_Config |
| `matRad/scenarios.py` | ✅ Done | NominalScenario |
| `matRad/geometry/geometry.py` | ✅ Done (bug fixed) | All geometry utilities |
| `matRad/phantoms/builder/phantom_voi.py` | ✅ Done | Box/Sphere VOI classes |
| `matRad/phantoms/builder/phantom_builder.py` | ✅ Done | PhantomBuilder class |
| `matRad/basedata/load_machine.py` | ✅ Done | Loads .mat machine files |
| `matRad/rayTracing/siddon.py` | ✅ Done (optimized) | Siddon ray tracer + fast version |
| `matRad/steering/stf_generator.py` | ✅ Done | STF generator for photon IMRT |
| `matRad/doseCalc/DoseEngines/dose_engine_base.py` | ✅ Done | Base dose engine |
| `matRad/doseCalc/DoseEngines/photon_svd_engine.py` | ✅ Done (bug fixed) | SVD photon dose engine |
| `matRad/doseCalc/calc_dose_influence.py` | ✅ Done | Entry point |
| `matRad/optimization/DoseObjectives/objectives.py` | ✅ Done | All objective functions |
| `matRad/optimization/fluence_optimization.py` | ✅ Done | L-BFGS-B optimizer |
| `matRad/planAnalysis/plan_analysis.py` | ✅ Done | DVH + quality indicators |
| `gui/matrad_gui.py` | ✅ Done | matplotlib GUI |
| `examples/example1_phantom.py` | ✅ Done | Phantom treatment plan |
| `examples/example2_photons.py` | ✅ Done | TG119 photon plan |
| `requirements.txt` | ✅ Done | |
| `CLAUDE.md` | ✅ Done | |

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

---

## What Remains To Do

### Priority 1: Verification (Run and Compare with MATLAB)
- [ ] Run `example1_phantom.py` on a system with enough memory (>4GB RAM recommended)
- [ ] Run `example2_photons.py` (requires TG119.mat available or falls back to synthetic phantom)
- [ ] Compare dose distributions, DVH, and quality indicators with MATLAB output

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
