# Machine Parameter Tuning

How to iteratively tune pyMatRad photon machine parameters to minimise
PDD and lateral-profile error vs Varian Golden Beam Data (GBD), and then
perform TG-51 absolute calibration.

---

## Overview

The SVD photon dose engine uses five scalar parameters that can be adjusted
after the machine file is built from commissioning data:

| Parameter | Machine key | Units | Effect |
|-----------|-------------|-------|--------|
| `penumbraFWHMatIso` | `data["penumbraFWHMatIso"]` | mm | Gaussian source FWHM at isocenter (lateral penumbra) |
| `m` | `data["m"]` | mm⁻¹ | Primary photon attenuation — depth-dose tail slope |
| `beta1` | `data["betas"][0]` | mm⁻¹ | Fast build-up decay (primary kernel component) |
| `beta2` | `data["betas"][1]` | mm⁻¹ | Mid-range scatter decay (secondary component) |
| `beta3` | `data["betas"][2]` | mm⁻¹ | Tail scatter decay (tertiary component) |

These five parameters appear explicitly in the SVD dose calculation at run-time,
so they can be changed **without** regenerating the 501 radial kernel tables.

---

## Physics Background

### SVD depth-dose model (Scholz 1994)

The dose at depth *d* for scatter component *k* is:

```
phi_k(d) = β_k / (β_k − m) · [exp(−m·d) − exp(−β_k·d)]
```

The total dose is:

```
dose(x, y, z)  =  Σ_k  W_ri(field_size, k)  ×  phi_k(d)  ×  kernel_k(r)  ×  (SAD/gd)²
```

where:
- `W_ri(field_size, k)` — **lateral kernel weights**: how much scatter component *k*
  contributes at a given field size (pre-computed from the GBD TPR, stored in the
  machine file as kernel1/kernel2/kernel3)
- `kernel_k(r)` — radial lateral kernel (also stored in the machine file)
- `phi_k(d)` — depth-dose envelope, computed at run-time from `m` and `betas`

### Two kinds of parameters

| Type | Parameters | Stored where | Requires kernel rebuild? |
|------|-----------|--------------|------------------------|
| **Run-time** | `m`, `betas`, `penumbraFWHMatIso` | `machine["data"]` scalars | No — used directly in dose calc |
| **Pre-computed** | W_ri (kernel weights) | `machine["data"]["kernel"][ssd]["kernel1..3"]` | Yes — fixed at build time |

---

## Why There Is a 3×3 vs 20×20 PDD Error

The kernel weights W_ri are fitted by solving a least-squares problem for each SSD:

```
GBD_TPR(depth, field_size)  ≈  Σ_k  W_ri[field_size, k]  ×  phi_k(depth, m, beta_k)
```

`generate_machine()` **auto-derives** m and betas from the GBD TPR via log-linear
regression.  If the auto-derived basis functions `phi_k` are a poor fit to the
actual physics across all depths, the least-squares solution for W_ri will be a
compromise that under-represents large-field scatter — producing the observed
3–4% PDD undershoot at 10–30 cm for 20×20 fields.

---

## Two-Step Correction Strategy

### Step 1 — Tune run-time parameters (fast, no rebuild)

`tune_machine.py` runs a Nelder-Mead optimiser that patches `m`, `betas`, and
`penumbraFWHMatIso` in memory, runs a full pyMatRad dose calculation for each
trial, and minimises the RMSE vs GBD.  Each evaluation takes minutes; the whole
optimisation runs in a few hours per energy.

This step improves the depth-dose envelope and lateral penumbra.  It does **not**
fix the relative 3×3 vs 20×20 difference, because W_ri was fit with the original
basis and remains unchanged.

### Step 2 — Rebuild kernels with the tuned basis (optional, ~5–15 min/energy)

`tune_machine.py --rebuild-kernels` takes the tuned m/betas and passes them to
`generate_machine()` as **fixed** basis functions.  `generate_machine()` then
re-solves the W_ri least-squares fit against the **same original GBD TPR data**
(not from pyMatRad output — that would be circular).

```
Before rebuild:  W_ri fit with auto-derived m/β  →  basis ≠ run-time m/β  →  mismatch
After rebuild:   W_ri fit with tuned m/β          →  basis = run-time m/β  →  consistent
```

A consistent basis gives the best possible least-squares fit to the GBD TPR across
all field sizes simultaneously, which is what reduces the 3×3 vs 20×20 discrepancy.

> **Note**: The input data for the rebuild is always the **original GBD TPR CSV** —
> not pyMatRad-calculated PDDs.  Using pyMatRad output as input would be circular
> and would not improve anything.

---

## Running the Tuning Script

```bash
conda activate scikit-learn
cd /path/to/pyMatRad

# Step 1 only: tune run-time parameters for all four TrueBeam energies
python examples/tune_machine.py

# Step 1 only: tune a single energy, 10×10 field only (fastest first pass)
python examples/tune_machine.py --machine TrueBeam_6X --field 10x10

# Steps 1 + 2: tune then rebuild kernels (recommended for production)
python examples/tune_machine.py --machine TrueBeam_6X --rebuild-kernels

# Dry run: compute baseline error only, no optimisation
python examples/tune_machine.py --machine TrueBeam_6X --dry-run

# Resume an interrupted run (checkpoints in examples/cache/tune_cache/)
python examples/tune_machine.py --machine TrueBeam_6X

# Force re-run ignoring checkpoints
python examples/tune_machine.py --machine TrueBeam_6X --force

# Save before/after comparison plots
python examples/tune_machine.py --plot-dir examples/cache/tune_plots
```

### Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--machine NAME` | all 4 | Machine name, e.g. `TrueBeam_6X` |
| `--field SIZE` | all 3 | Restrict optimizer to one field: `3x3`, `10x10`, or `20x20` |
| `--max-iter N` | 60 | Maximum Nelder-Mead iterations per energy |
| `--dry-run` | off | Compute baseline error only; do not save anything |
| `--force` | off | Ignore checkpoints; re-run from scratch |
| `--rebuild-kernels` | off | Re-fit W_ri to GBD TPR using tuned m/betas as fixed basis |
| `--no-calibrate` | off | Save tuned machine but skip TG-51 calibration |
| `--plot-dir DIR` | — | Directory for before/after comparison PNGs |
| `--output-md FILE` | `examples/tune_machine.md` | Path for the markdown result table |

---

## Optimisation Details

### Algorithm

Scipy `minimize` with `method="Nelder-Mead"` (derivative-free simplex).

- Parameters are **log-transformed** internally to enforce positivity: `x = log(param)`
- Tolerances: `xatol=0.005`, `fatol=0.01`, `adaptive=True`
- Checkpoints are saved after each energy to `examples/cache/tune_cache/`

### Objective function

```
error = pdd_weight × PDD_RMSE  +  profile_weight × Profile_RMSE
```

**PDD RMSE** — evaluated at 5, 10, 20, 30 cm vs GBD.
Depths ≥ 20 cm are weighted ×2 to prioritise correcting deep-dose errors.

**Profile RMSE** — within-field dose points vs GBD at each profile depth.

By default all three fields (3×3, 10×10, 20×20) contribute equally.
Use `--field 10x10` for a 3× faster first pass.

---

## Output Files

| File | Description |
|------|-------------|
| `userdata/machines/photons_{energy}.npy` | Tuned machine (patched or rebuilt) |
| `examples/tune_machine.md` | This file; results table appended after run |
| `examples/cache/tune_cache/{energy}_*.npy` | Optimizer checkpoint files |
| `--plot-dir/*.png` | Before/after PDD and profile comparison plots |

---

## After Tuning: TG-51 Calibration

`tune_machine.py` automatically runs TG-51 calibration after saving the tuned
machine (unless `--no-calibrate` is given).  Calibration computes:

```
abs_calib = 0.01 Gy/MU / dose(d_max, 10×10 field, SSD=100 cm, w=1 MU)
```

and stores it in `machine["meta"]["tg51"]`.  To re-run independently:

```bash
python examples/calibrate_machine.py --machine TrueBeam_6X --force
```

---

## Known Limitations

### Residual large-field PDD error after Step 1 only

After step 1 (patching only), typical residuals for 20×20 vs GBD:

| Energy | Error at 10 cm | Error at 30 cm |
|--------|----------------|----------------|
| 6X     | −3 to −4% | −4 to −5% |
| 6XFFF  | −2 to −3% | −3 to −4% |
| 10XFFF | −3 to −4% | −4 to −5% |
| 15X    | −2 to −3% | −3 to −4% |

These are driven by the mismatch between the auto-derived and tuned bases.
Running `--rebuild-kernels` (step 2) addresses this.

### FFF beam primary fluence

For 6XFFF and 10XFFF, profile accuracy at intermediate depths is limited by the
`primaryFluence` off-axis ratio derived from the GBD shallow profile.  If the
horn/softening shape is inaccurate, residual profile errors remain after penumbra
tuning.  The `primaryFluence` array can be manually edited in the `.npy` file.

---

## Tuned Parameters (auto-generated by `tune_machine.py`)

| Energy | fwhm [mm] | m [mm⁻¹] | β₁ [mm⁻¹] | β₂ [mm⁻¹] | β₃ [mm⁻¹] | err before | err after | abs_calib [cGy/MU] | d_max [mm] |
|--------|-----------|----------|----------|----------|----------|-----------|-----------|-------------------|-----------|
| TrueBeam_6X     | (run script) | — | — | — | — | — | — | — | — |
| TrueBeam_6XFFF  | (run script) | — | — | — | — | — | — | — | — |
| TrueBeam_10XFFF | (run script) | — | — | — | — | — | — | — | — |
| TrueBeam_15X    | (run script) | — | — | — | — | — | — | — | — |
