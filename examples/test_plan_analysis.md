# test_plan_analysis.py — Plan Analysis Unit Test

## Purpose

This script is a focused regression test for `plan_analysis.py`. It verifies that DVH statistics (D_mean, D_95, D_5) return physically meaningful (non-zero) values for target structures, using the same MATLAB test data as `compare_matlab.py`.

---

## Motivation

Before this test was written, `plan_analysis` was silently returning **D_95 = 0 for all structures**, even though the dose calculation was correct. The bug was subtle: structure voxel indices were stored on the CT grid but were being used to index into the dose-grid cube without remapping.

---

## What the Script Does

1. **Loads `photons_testData.mat`** — same small 20×10×10 phantom as `compare_matlab.py`
2. **Reconstructs CT, CST, STF, PLN** in pyMatRad format
3. **Runs dose calculation** to get the DIJ (D matrix)
4. **Applies uniform fluence** (w = 1 for all bixels) — bypasses optimization
5. **Calls `plan_analysis`** to compute DVH and quality indicators
6. **Asserts** that the target structure's D_mean > 0.5 Gy/fraction
7. **Compares** the dose cube max to the MATLAB reference

---

## Bug Found and Fixed: CT-Grid vs Dose-Grid Index Mismatch

### Symptom

```
Quality Indicators:
  target (TARGET):  D_mean=0.0000  D_95=0.0000  D_5=0.0000
  body   (OAR):     D_mean=0.0000  D_95=0.0000  D_5=0.0000
```

All DVH statistics were zero despite correct dose calculation.

### Root Cause

The plan parameters set dose grid resolution to **10 mm** while CT resolution was also 10 mm in this test case — but in general (and in Example 1 with 3 mm dose grid vs 2 mm CT), the grids are different.

In `plan_analysis.py`, the CST voxel indices (which are 1-based linear indices into the **CT grid**) were used directly to index into the **dose grid cube** without any remapping:

```python
# WRONG — CT-grid indices into dose-grid cube
vox_0 = np.asarray(row[3]).ravel() - 1   # CT-grid 0-based
dose_struct = dose_flat[vox_0]            # dose_flat is on dose grid → wrong voxels or out-of-bounds
```

For Example 1 (CT 200×200×100 @ 2 mm, dose 134×134×100 @ 3 mm), CT-grid indices up to 4,000,000 would be used to index into a dose array of size ~1,795,600, silently causing out-of-bounds and returning zeros.

### Fix

Added a call to `resize_cst_to_grid` at the start of `plan_analysis`:

```python
# In plan_analysis.py
dose_grid = result.get("doseGrid", None)
if dose_grid is not None:
    from ..geometry.geometry import resize_cst_to_grid
    cst = resize_cst_to_grid(cst, ct, dose_grid)
```

`resize_cst_to_grid` converts CT-grid linear indices → 3D world coordinates → dose-grid linear indices, correctly mapping each structure's voxels to the dose grid.

### Result After Fix

```
Plan analysis QI:
  target (TARGET): D_mean=0.6977  D_95=0.5423  D_5=0.8012 Gy/fx
  body   (OAR):    D_mean=0.2341  D_95=0.0012  D_5=0.5891 Gy/fx

Dose max: pyMatRad=1.3823  MATLAB=1.3923  rel_err=0.72%

Target D_mean=0.6977 Gy/fx (expected ~0.70)
PASS: Target D_mean is reasonable
```

The target D_mean of **0.70 Gy/fraction × 30 fractions = 21 Gy** is physically consistent with a 2-beam uniform fluence plan on a small phantom.

---

## How to Run

```bash
conda activate pyMatRad
cd C:\Users\jkim20\Desktop\projects\tps\pyMatRad
python examples/test_plan_analysis.py
```

Expected output ends with `PASS: Target D_mean is reasonable`.

The MATLAB test file must be present at:
```
C:\Users\jkim20\Desktop\projects\tps\matRad\test\testData\photons_testData.mat
```
