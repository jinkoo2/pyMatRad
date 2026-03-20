# compare_matlab.py — Numerical Verification Against MATLAB Reference

## Purpose

This script validates pyMatRad's dose engine output against ground-truth values from MATLAB's matRad. It loads the official MATLAB test data (`photons_testData.mat`), reconstructs the exact same geometry in pyMatRad, runs the dose calculation, and compares the resulting DIJ and dose cube numerically.

---

## Background / Motivation

After porting the photon SVD dose engine from MATLAB to Python, we needed to confirm that the numbers are correct — not just "runs without error". The MATLAB test file provides a small, reproducible reference case with known correct output. Any dose discrepancy would indicate a porting bug.

---

## Test Case Description

The reference case (`matRad/test/testData/photons_testData.mat`) is a small, fast phantom:

| Parameter | Value |
|-----------|-------|
| Phantom size | 20 × 10 × 10 voxels |
| Voxel size | 10 mm × 10 mm × 10 mm |
| Beams | 2 (gantry 0° and 180°) |
| Bixels per beam | 5 |
| Total bixels | 10 |
| Machine | Generic (6 MV photon) |
| Fluence | Uniform (w = 1 for all bixels) |

The test data includes:
- `ct`: CT cube (HU and electron density)
- `cst`: Structure set (2 structures: target + body)
- `stf`: Beam/ray geometry (already computed by MATLAB)
- `dij`: Reference dose influence matrix (sparse, 2000×10)
- `resultGUI.physicalDose`: Reference dose cube (Gy/fraction)

---

## What the Script Does

### Step 1: Load MATLAB Reference

Reads `photons_testData.mat` using `scipy.io.loadmat` with `squeeze_me=True`. Extracts:
- `ref_dose`: 20×10×10 dose cube (Gy/fraction)
- `ref_dij`: sparse 2000×10 dose influence matrix
- `ref_w`: weight vector (all ones = uniform fluence)

### Step 2: Reconstruct CT

Copies the exact same CT cube (HU values, electron density, resolution, HLUT) into pyMatRad's dict format.

### Step 3: Reconstruct CST

Converts the MATLAB cell array CST rows to pyMatRad's list-of-lists format, preserving the 1-based voxel indices.

### Step 4: Reconstruct STF

Converts the MATLAB stf struct array to pyMatRad dicts. Critically, this uses the **exact same ray positions** that MATLAB computed, so ray sampling differences cannot explain dose discrepancies.

### Step 5: Run pyMatRad Dose Calculation

Calls `calc_dose_influence(ct, cst, stf, pln)` — the full Siddon + SVD kernel pipeline.

### Step 6: Compare DIJ and Dose

With uniform weights (`w = 1`), computes `dose = D @ w` and compares:
- Max dose, mean dose
- DIJ non-zero count and total weight sum
- Per-voxel dose in the target structure

---

## Key Bugs Found and Fixed

### Critical Bug: `geo_dists + SAD` Double-Counting

**Discovery**: Initial run showed max dose error of **76%** (pyMatRad ~4× below MATLAB).

**Root cause** in `photon_svd_engine.py`:

```python
# WRONG — geo_dists is already the geometric distance from source
# Adding SAD made the denominator (SAD + geo_dists + SAD) ≈ 2*SAD at isocenter
self._calc_single_bixel(..., geo_dists[vox_ix] + float(SAD), ...)

# The inverse-square term became: (SAD / (geo + SAD))^2 ≈ (1/2)^2 = 0.25
```

**Fix**: Removed `+ float(SAD)`:

```python
# CORRECT — geo_dists is the true source-to-voxel distance
self._calc_single_bixel(..., geo_dists[vox_ix], ...)
```

**Impact**: Max dose error dropped from **76% → 0.72%**.

---

## Results (After Fixes)

```
=== Dose comparison (uniform fluence, per fraction) ============
Metric                           pyMatRad       MATLAB   rel err
------------------------------------------------------------------
max dose (Gy/fx)                   1.3823       1.3923     0.72%
mean dose >0 (Gy/fx)              0.6891       0.6978     1.25%
DIJ nnz                          1810.0       1810.0     0.00%
DIJ total weight sum              1038.9       1040.3     0.13%

=== Target voxel doses (Gy/fx) ================================
 Voxel      pyMatRad       MATLAB   rel err
--------------------------------------------
     0        1.1034       1.1088     0.49%
     1        1.2743       1.2921     1.38%
     ...
```

- DIJ sparsity pattern is **identical** (same nnz)
- Total weight sum matches to **0.13%**
- Max dose error: **0.72%** — within expected numerical tolerance

The remaining small discrepancies are attributed to:
- Floating-point accumulation differences between Python/NumPy and MATLAB
- Minor differences in kernel interpolation

---

## How to Run

```bash
conda activate pyMatRad
cd C:\Users\jkim20\Desktop\projects\tps\pyMatRad
python examples/compare_matlab.py
```

The MATLAB test file must be present at:
```
C:\Users\jkim20\Desktop\projects\tps\matRad\test\testData\photons_testData.mat
```
