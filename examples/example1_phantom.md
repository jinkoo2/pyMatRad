# Example 1: Synthetic Phantom Treatment Plan

## Overview

`example1_phantom.py` is a complete end-to-end radiation treatment planning example using a **synthetically generated phantom** (no real patient data). It ports `matRad_example1_phantom.m` to Python.

The script exercises every major module in pyMatRad in sequence:

```
PhantomBuilder → generate_stf → calc_dose_influence → fluence_optimization → plan_analysis → plot
```

---

## What the Code Does

### 1. Phantom Construction

A 200×200×100 voxel CT phantom (2 mm × 2 mm × 3 mm resolution) is created using `PhantomBuilder`. Three volumes of interest (VOIs) are added:

| VOI | Type | Shape | Dose Objective |
|-----|------|-------|----------------|
| Volume1 | TARGET | Sphere, r=20 mm | SquaredDeviation: hit 45 Gy |
| Volume2 | OAR | 60×30×60 mm box, offset –15 mm | SquaredOverdosing: minimize dose >0 Gy (penalty 400) |
| Volume3 | OAR | 60×30×60 mm box, offset +15 mm | SquaredOverdosing: minimize dose >0 Gy (penalty 10) |

Volume2 and Volume3 are placed symmetrically on either side of the target.

### 2. Plan Parameters

```python
gantryAngles = [0, 70, 140, 210, 280, 350]   # 6 beams, 70° spacing
bixelWidth   = 5 mm
numOfFractions = 30
doseGrid resolution = 3×3×3 mm
machine = "Generic"  # 6 MV photon
```

### 3. Beam Geometry (STF)

`generate_stf` computes for each gantry angle:
- Source position (at SAD = 1000 mm)
- Ray positions in BEV that cover the target with margin
- Bixel-to-voxel mapping

Typical output: ~700–900 total bixels across 6 beams.

### 4. Dose Influence Matrix (DIJ)

`calc_dose_influence` runs the **SVD photon pencil-beam engine**:
- Siddon ray tracing for radiological path length
- Kernel convolution (scatter + primary)
- Inverse-square law correction

Result is a sparse matrix `D ∈ ℝ^{N_voxels × N_bixels}` where `D[i,j]` = dose at voxel `i` from bixel `j` with unit fluence.

### 5. Fluence Optimization

`fluence_optimization` runs **L-BFGS-B** (bounded, no negative fluences) minimizing:

$$\min_{w \geq 0} \sum_k \lambda_k \cdot f_k(D w)$$

where:
- $w$ = bixel weights (fluence)
- $\lambda_k$ = penalty for structure $k$
- $f_k$ = squared deviation (target) or squared overdosing (OARs)

Typical convergence: ~200–400 iterations.

### 6. Plan Analysis

`plan_analysis` computes DVH and quality indicators (D_mean, D_95, D_5, V_dose) for each structure, remapping CST voxel indices from CT grid to dose grid resolution.

### 7. Visualization

Saves a PNG with:
- Axial dose slice at the isocenter
- DVH for all structures

Output: `examples/pyMatRad_example1.png`

---

## Bugs Fixed During Development

| Bug | Symptom | Fix |
|-----|---------|-----|
| `geo_dists + SAD` double-counting | 4× dose underestimate (max error 76%) | Removed `+ float(SAD)` in `photon_svd_engine.py` |
| CT-grid CST indices used in dose-grid cube | D_95=0 for all structures in plan_analysis | Added `resize_cst_to_grid(cst, ct, dose_grid)` in `plan_analysis.py` |
| `plt.cm.get_cmap()` deprecated | `matplotlib.MatplotlibDeprecationWarning` → crash | Changed to `plt.colormaps["tab10"]` |
| `/tmp/` output path on Windows | `FileNotFoundError` on Windows | Changed to `os.path.dirname(__file__)` |

---

## Results

After fixes, example1 runs end-to-end successfully:

- **DIJ**: sparse matrix with correct non-zero structure
- **Max dose error vs MATLAB reference**: **0.72%** (verified by `compare_matlab.py`)
- **Plan analysis**: D_95, D_mean, D_5 all return non-zero values for all structures
- **DVH**: target receives ~45 Gy (prescription), OARs receive significantly less

---

## How to Run

```bash
conda activate pyMatRad
cd C:\Users\jkim20\Desktop\projects\tps\pyMatRad
python examples/example1_phantom.py
```

Expected runtime: ~30–90 seconds depending on hardware.
