# TrueBeam Machine Tuning Log

Record of pyMatRad TrueBeam photon machine build, parameter tuning, and
TG-51 absolute calibration.  Machine files live in `userdata/machines/`.

---

## Summary of Steps

| Step | Script | Date | Time |
|------|--------|------|------|
| 1. Build machines from GBD | `machineBuilder/build_truebeam.py` | — | — |
| 2. Tune run-time parameters (m, betas, fwhm) | `examples/tune_machine.py` | 2026-04-19 | ~4 h |
| 3. Rebuild kernels with tuned basis + TG-51 calibration | `examples/tune_machine.py --rebuild-kernels` | 2026-04-19 | ~1–2 h |

---

## Machine Parameters

### Before Tuning (original GBD build)

`fwhm_gauss = 6.0 mm` for all energies (hard-coded in `machineBuilder/build_truebeam.py`).
`m` and `betas` were auto-derived by log-linear regression on the zero-field TPR.
Original values below are from the MATLAB `photons_TrueBeam_*.mat` reference files.

| Energy | fwhm [mm] | m [mm⁻¹] | β₁ [mm⁻¹] | β₂ [mm⁻¹] | β₃ [mm⁻¹] |
|--------|-----------|----------|----------|----------|----------|
| TrueBeam_6X     | 6.000 | 0.004693 | 0.25420 | 0.01460 | 0.00469 |
| TrueBeam_6XFFF  | 6.000 | 0.005211 | 0.26786 | 0.01614 | 0.00521 |
| TrueBeam_10XFFF | 6.000 | 0.003865 | 0.15029 | 0.01163 | 0.00386 |
| TrueBeam_15X    | 6.000 | 0.003016 | 0.12251 | 0.00912 | 0.00302 |

Baseline GBD error (weighted RMSE of PDD at 5/10/20/30 cm + in-field profiles,
averaged over 3×3, 10×10, 20×20 fields):

| Energy | Baseline error |
|--------|---------------|
| TrueBeam_6X     | 2.628 |
| TrueBeam_6XFFF  | 2.330 |
| TrueBeam_10XFFF | 1.889 |
| TrueBeam_15X    | 1.894 |

---

### After Step 1 — Nelder-Mead Tuning of Run-Time Scalars

Script: `python examples/tune_machine.py` (all 3 fields, 60 iterations, ~4 hours)

Only the scalar parameters stored in `machine["data"]` were changed.
The lateral kernel weights W_ri (which determine field-size-dependent scatter)
were **not** rebuilt in this step.

| Energy | fwhm [mm] | m [mm⁻¹] | β₁ [mm⁻¹] | β₂ [mm⁻¹] | β₃ [mm⁻¹] | err before | err after | Δerr |
|--------|-----------|----------|----------|----------|----------|-----------|-----------|------|
| TrueBeam_6X     | 6.881 | 0.004817 | 0.21457 | 0.01689 | 0.00542 | 2.628 | 2.249 | −14% |
| TrueBeam_6XFFF  | 6.628 | 0.005322 | 0.22292 | 0.01526 | 0.00594 | 2.330 | 1.824 | −22% |
| TrueBeam_10XFFF | 8.015 | 0.003956 | 0.13986 | 0.01267 | 0.00462 | 1.889 | 1.390 | −26% |
| TrueBeam_15X    | 7.961 | 0.003019 | 0.11293 | 0.01164 | 0.00248 | 1.894 | 1.327 | −30% |

**Parameter changes relative to original:**

| Energy | Δfwhm | Δm | Δβ₁ | Δβ₂ | Δβ₃ |
|--------|-------|----|----|----|----|
| TrueBeam_6X     | +0.88 mm (+15%) | +0.000124 (+3%) | −0.040 (−16%) | +0.002 (+16%) | +0.001 (+15%) |
| TrueBeam_6XFFF  | +0.63 mm (+10%) | +0.000111 (+2%) | −0.045 (−17%) | −0.001 (−5%)  | +0.001 (+14%) |
| TrueBeam_10XFFF | +2.02 mm (+34%) | +0.000091 (+2%) | −0.010 (−7%)  | +0.001 (+9%)  | +0.001 (+20%) |
| TrueBeam_15X    | +1.96 mm (+33%) | +0.000003 (+0%) | −0.010 (−8%)  | +0.003 (+28%) | −0.001 (−18%) |

Key observations:
- **fwhm** increased for all energies, especially 10XFFF and 15X (+2 mm), indicating
  the original 6 mm penumbra was too narrow.
- **m** changed by < 3% for all energies — attenuation was already well-estimated.
- **β₁ decreased** for all energies — builds up slightly slower, improving d_max shape.
- **β₂, β₃** changes were moderate, affecting mid-depth and deep scatter.

---

### After Step 2 — Kernel Rebuild + TG-51 Calibration

Script: `python examples/tune_machine.py --rebuild-kernels`

The 501 radial kernel weight tables (W_ri for kernel1/kernel2/kernel3) were
re-fitted to the original GBD TPR data using the **tuned m/betas as fixed SVD
basis functions**.  This is NOT circular: the GBD TPR CSV files are unchanged;
only the decomposition basis is improved.  A consistent basis → better
least-squares fit of W_ri across all field sizes → reduced 3×3 vs 20×20 PDD
discrepancy.

Scalar parameters (fwhm, m, betas) are identical to Step 1.

**TG-51 calibration** (1 cGy/MU at d_max, 10×10 field, SSD = 100 cm):

| Energy | abs_calib [cGy/MU] | d_max [mm] |
|--------|-------------------|-----------|
| TrueBeam_6X     | 1.0351 | 16.0 |
| TrueBeam_6XFFF  | 1.0276 | 14.0 |
| TrueBeam_10XFFF | 1.0480 | 22.0 |
| TrueBeam_15X    | 1.0589 | 26.0 |

All `abs_calib` values are within the expected 1.00–1.06 cGy/MU range.
These are stored in `machine["meta"]["tg51"]` in each `.npy` file.

---

## Current Machine Files

All four files in `userdata/machines/` reflect the Step 2 state
(tuned scalars + rebuilt kernels + TG-51 calibration):

| File | Energy | SAD | abs_calib |
|------|--------|-----|-----------|
| `photons_TrueBeam_6X.npy`     | 6 MV        | 1000 mm | 1.0351 cGy/MU |
| `photons_TrueBeam_6XFFF.npy`  | 6 MV FFF    | 1000 mm | 1.0276 cGy/MU |
| `photons_TrueBeam_10XFFF.npy` | 10 MV FFF   | 1000 mm | 1.0480 cGy/MU |
| `photons_TrueBeam_15X.npy`    | 15 MV       | 1000 mm | 1.0589 cGy/MU |

---

## Reproduction Commands

```bash
# Step 1: Tune run-time parameters (hours; skips if checkpoints exist)
python examples/tune_machine.py

# Step 2: Rebuild kernels with tuned basis + calibrate (uses checkpoints)
python examples/tune_machine.py --rebuild-kernels

# Calibrate only (if step 2 calibration failed)
python examples/calibrate_machine.py --machine TrueBeam_6X --force
python examples/calibrate_machine.py --machine TrueBeam_6XFFF --force
python examples/calibrate_machine.py --machine TrueBeam_10XFFF --force
python examples/calibrate_machine.py --machine TrueBeam_15X --force

# Validate against GBD
python examples/validate_truebeam.py
```

Checkpoints (can be resumed after interruption):
`examples/cache/tune_cache/{energy}_3x3_10x10_20x20_ckpt.npy`

---

## Known Remaining Limitations

- **20×20 PDD undershoot** may persist at 3–5% at depths > 20 cm.
  The SVD 3-component model has limited degrees of freedom to simultaneously
  fit all field sizes.  Validation plots in `examples/validation_plots/`
  after running `validate_truebeam.py` will show the residual.
- **FFF primary fluence** (6XFFF, 10XFFF): off-axis profile accuracy at
  intermediate depths depends on the `primaryFluence` OAR read from the
  shallow GBD profile.  Manual adjustment of `machine["data"]["primaryFluence"]`
  may further improve FFF profile agreement.
