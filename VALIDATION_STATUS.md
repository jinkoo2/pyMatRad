# Validation Status — pyMatRad TrueBeam Photon Machine

**Last updated:** 2026-04-15

---

## Completed Work

### 1. Machine builder port (`matRad/machineBuilder/`)
- All 4 machines built and saved to `userdata/machines/`:
  - `photons_TrueBeam_6X.npy`
  - `photons_TrueBeam_6XFFF.npy`
  - `photons_TrueBeam_10XFFF.npy`
  - `photons_TrueBeam_15X.npy`
- Committed and pushed (commit `1b6c00e`)

### 2. Engine fixes
- **`matRad/doseCalc/DoseEngines/photon_svd_engine.py`** — three changes made:
  1. `dij["physicalDose"] = [None]` instead of `sp.lil_matrix(...)` (removes wasteful preallocation)
  2. `if n_workers <= 1: beam_results = [_calc_beam_worker(b) for b in bundles]` — runs in-process to avoid subprocess OOM
  3. `enableDijSampling` from `pln["propDoseCalc"]` is now correctly read and applied (was silently ignored before, causing stochastic noise in profiles even when set to `False`)

### 3. Validation script (`examples/validate_truebeam.py`)
- All 12 cases (4 energies × 3 field sizes) run successfully
- 24 PNG plots saved to `examples/validation_plots/`
- Key fix: **sparse VOI** — `build_water_phantom` now creates:
  - OAR = central axis (all depths) + profile-depth x-slices (not full 3D volume)
  - PTV = single-depth slab at y=5mm (for correct bixel placement, no dose grid bloat)
  - This reduces dose grid from millions of voxels to ~1000–3700 voxels → eliminates OOM

---

## Validation Results Summary

### PDD accuracy (pyMatRad vs GBD)
| Case | 5cm | 10cm | 20cm | vs MATLAB |
|------|-----|------|------|-----------|
| 6X 3×3 | −1.2% | −0.7% | +0.2% | within 0.5% |
| 6X 10×10 | −0.8% | −0.5% | +0.6% | within 0.6% |
| 6X 20×20 | −2.2% | −3.8% | −4.0% | within 0.8% |
| 6XFFF 3×3 | −1.5% | −0.6% | +0.2% | within 0.6% |
| 6XFFF 10×10 | −0.9% | −0.3% | +0.7% | within 0.2% |
| 6XFFF 20×20 | −2.4% | −3.4% | −3.2% | within 0.4% |
| 10XFFF 3×3 | −0.8% | −0.4% | +0.2% | within 0.5% |
| 10XFFF 10×10 | −0.3% | −0.1% | +0.8% | within 0.3% |
| 10XFFF 20×20 | −0.6% | −1.7% | −1.6% | within 0.3% |
| 15X 3×3 | −1.4% | −0.8% | −0.4% | within 0.4% |
| 15X 10×10 | −0.7% | −1.1% | −0.1% | within 1.0% |
| 15X 20×20 | +0.5% | −1.0% | −1.9% | within 0.8% |

**PDD conclusion:** Small/medium fields within 1–2% of GBD. Large 20×20 fields up to 4% error (expected limitation of pencil-beam model). pyMatRad matches MATLAB within ~1%.

### Profile FWHM — regular depths (5–30 cm)
- pyMatRad vs GBD: typically +0.5 to +1.5 cm wider (pencil-beam known limitation at depth)
- pyMatRad vs MATLAB: within 0.3–1.0 cm — **good agreement**

### ⚠️ Known Issue: Shallow-depth profiles
**10XFFF** (dmax=2.4cm) and **15X** (dmax=3.0cm) show near-zero FWHM at their shallow reference depth:
- 10XFFF at 2.4cm: py=0.39–0.79cm vs GBD=3–20cm ← **wrong**
- 15X at 3.0cm: py=0.35–0.74cm vs GBD=3–21cm ← **wrong**
- 6X at 1.5cm: py=2.69cm vs GBD=3.00cm ← slight under-penumbra, acceptable
- 6XFFF at 1.5cm: py=2.74cm vs GBD=2.99cm ← slight under-penumbra, acceptable

**Root cause investigation (incomplete when session ended):**

Depth-dose components look fine:
```
10XFFF: m=0.003865, betas=[0.1503, 0.01163, 0.003866]
  at rd=24mm: components=[0.908, 0.232, 0.085], sum=1.224  ← reasonable
  at rd=50mm: components=[0.845, 0.397, 0.159], sum=1.402
6X: m=0.004693, betas=[0.2542, 0.01460, 0.004692]
  at rd=15mm: components=[0.927, 0.190, 0.066], sum=1.182  ← reasonable
```

So the depth-dose kernel looks fine. The issue is likely one of:

**Hypothesis A — Ray tracing assigns NaN rad_depths to off-axis voxels at shallow depth:**
In `ray_tracing_fast` (siddon.py:378), the `valid_mask = nearest_dist <= effective_lateral_cutoff`.
At y=24mm (proj_x scaling = 1000/1024=0.977):
- A voxel at x=10mm has proj_x=9.77mm
- The nearest ray (bixel) at x=10mm has distance ~0.23mm → valid ✓
- BUT: what is `effective_lateral_cutoff` at this point? It's `kernel_cutoff + bixelWidth/sqrt(2) = 20 + 2/sqrt(2) = 21.4mm`.
- Voxels within 21.4mm of any ray should have valid rad_depth. For 3×3 field with bixels at ±14mm, voxels out to x=35mm should be fine.

This hypothesis doesn't explain a FWHM of 0.39cm (4mm).

**Hypothesis B — build_water_phantom PTV slab at iy=2 causes STF to not generate sufficient bixels:**
The PTV slab is at iy_ptv=2 (y=4mm). If the STF generator sees only a thin slab at y=4mm and generates bixels only for the projection at y=4mm, the beam coverage might be wrong. Check if STF generates correct number of bixels (should be 225 for 3×3, 441 for 10×10, 1681 for 20×20).

**Hypothesis C — The dose at y=24mm is actually correct, but the profile normalization makes it look narrow:**
At y=24mm (dmax for 10XFFF), the central-axis dose might be artificially high because the betas[2]=0.00386557 ≈ m=0.003865. This means betas[2]-m ≈ 0.0 → division near zero → large component for kernel 3. This spike in the depth-dose for the "electron contamination" component could make the central axis unnaturally high, while off-axis voxels (where the kernel3 contribution is filtered by the lateral kernel) appear lower. The profile normalized to central axis would then show a spike.

Check: `betas[2] = 0.00386557` and `m = 0.0038647`. These are essentially equal! `betas[2] - m = 0.0000`. This is the SPECIAL CASE in the engine:
```python
if abs(beta - m) < 1e-10:
    dose_component[:, ki] = m * all_rd * np.exp(-m * all_rd)
else:
    dose_component[:, ki] = (beta / (beta - m) * (exp(-m*rd) - exp(-beta*rd)))
```
The threshold is `1e-10`. With `betas[2]-m = 0.00386557 - 0.00386469 = 8.8e-7`, the absolute difference is 8.8e-7 > 1e-10, so the else branch runs. The division `beta/(beta-m) = 0.00386557/8.8e-7 = 4393` → **HUGE AMPLIFICATION!** This creates a pathologically large third component.

At rd=24mm:
`4393 * (exp(-0.003865*24) - exp(-0.003866*24))`
= `4393 * (0.9098 - 0.9097)` ≈ `4393 * 8.8e-7 * 24` ≈ `0.0928`

Actually that comes out OK. Let me recalculate more carefully...

Actually the manual calculation above showed sum=1.224 at rd=24mm which included betas[2]. So the sum is fine. The hypothesis C about near-equal betas might not be the issue.

**Most likely root cause: To be determined.** The session ended before identifying the exact cause.

---

## Files Modified in This Session

| File | Change |
|------|--------|
| `matRad/machineBuilder/__init__.py` | Created — module API |
| `matRad/machineBuilder/read_gbd_data.py` | Created — GBD CSV reader |
| `matRad/machineBuilder/kernel_calc.py` | Created — kernel calculation |
| `matRad/machineBuilder/build_truebeam.py` | Created — TrueBeam builder |
| `matRad/basedata/load_machine.py` | Modified — added .npy support |
| `matRad/__init__.py` | Modified — exposed machineBuilder |
| `matRad/doseCalc/DoseEngines/photon_svd_engine.py` | Modified — removed lil_matrix prealloc, in-process execution |
| `examples/validate_truebeam.py` | Created — 12-case validation script |

---

## Next Steps

1. **Investigate shallow-depth profile issue** (10XFFF at 2.4cm, 15X at 3.0cm):
   - Check if `betas[2] ≈ m` causes numerical issues in the kernel calculation
   - Compare with MATLAB intermediate results from s7/s8/s9 scripts

2. **Re-run validation** with `enableDijSampling=False` now correctly honoured — profiles at deep depths (30 cm) should be noticeably smoother

---

## How to Run Validation

```bash
source /gpfs/software/Anaconda/envs/scikit-learn/bin/activate
# OR:
/gpfs/software/Anaconda/envs/scikit-learn/bin/python3 examples/validate_truebeam.py
```

Plots saved to: `examples/validation_plots/`  
GBD data at: `/gpfs/projects/KimGroup/projects/tps/matRad/my_scripts/TrueBeamGBD`  
MATLAB results at: `/gpfs/projects/KimGroup/projects/tps/matRad/`
