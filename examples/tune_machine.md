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
| `penumbraFWHMatIso` | `data["penumbraFWHMatIso"]` | mm | Gaussian source FWHM → lateral penumbra width |
| `m` | `data["m"]` | mm⁻¹ | Primary photon attenuation → depth-dose tail slope |
| `beta1` | `data["betas"][0]` | mm⁻¹ | Fast build-up decay (primary kernel) |
| `beta2` | `data["betas"][1]` | mm⁻¹ | Mid-range scatter decay (secondary kernel) |
| `beta3` | `data["betas"][2]` | mm⁻¹ | Tail scatter decay (tertiary kernel) |

These five parameters appear explicitly in the SVD dose calculation formula
(Scholz 1994) at run-time, so they can be changed **without** regenerating
the 501 radial kernel tables.

The lateral kernel weights (kernel1–3) that encode field-size-dependent scatter
are **not** changed by this script.  To improve 3×3 vs 20×20 PDD differences
the kernels must be rebuilt via `machineBuilder/build_truebeam.py` with updated
TPR data.

---

## Physics Background

### Depth-dose model (Scholz 1994)

For each scatter component *k*:

```
D_k(d) = β_k / (β_k − m) × [exp(−m·d) − exp(−β_k·d)]
```

The total dose at depth *d* is:

```
dose(d) = Σ_k [ D_k(d) × kernel_k(r) ] × (SAD / geo_dist)²
```

where `kernel_k(r)` is the lateral kernel at radial distance *r*
(convolved with the bixel fluence and Gaussian penumbra).

### Effect of each parameter

| Parameter | Too small | Too large |
|-----------|-----------|-----------|
| `m` | Tail dose too high | Tail dose too low |
| `beta1` | Build-up region too wide | Build-up too narrow (sharp peak) |
| `beta2` / `beta3` | More mid/deep scatter → higher dose at depth | Less scatter → lower dose at depth |
| `penumbraFWHMatIso` | Sharp penumbra, geometric field edge | Blurry penumbra, rounded field edge |

### Typical ranges (6 MV photons)

| Parameter | Typical range | Default TrueBeam_6X |
|-----------|---------------|---------------------|
| `penumbraFWHMatIso` | 4 – 10 mm | 6.0 mm |
| `m` | 0.002 – 0.007 mm⁻¹ | 0.003 – 0.004 mm⁻¹ |
| `beta1` | 0.1 – 2.0 mm⁻¹ | ~0.2 – 0.6 mm⁻¹ |
| `beta2` | 0.01 – 0.2 mm⁻¹ | ~0.02 – 0.06 mm⁻¹ |
| `beta3` | 0.003 – 0.02 mm⁻¹ | ~0.005 – 0.015 mm⁻¹ |

---

## Running the Tuning Script

```bash
conda activate scikit-learn
cd /path/to/pyMatRad

# Tune all four TrueBeam energies (multi-hour batch run)
python examples/tune_machine.py

# Tune a single energy
python examples/tune_machine.py --machine TrueBeam_6X

# Use only the 10×10 field (≈3× faster; good for initial pass)
python examples/tune_machine.py --machine TrueBeam_6X --field 10x10

# Dry run: compute baseline error only, no optimisation
python examples/tune_machine.py --machine TrueBeam_6X --dry-run

# Resume an interrupted run (checkpoint files are saved automatically)
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
| `--force` | off | Ignore checkpoints, re-run from scratch |
| `--no-calibrate` | off | Save tuned machine but skip TG-51 calibration |
| `--plot-dir DIR` | — | Directory for before/after comparison PNGs |
| `--output-md FILE` | `examples/machine_tuning.md` | Path for markdown result table |

---

## Optimisation Details

### Algorithm

Scipy `minimize` with `method="Nelder-Mead"` (derivative-free simplex method).

- Tolerances: `xatol=0.005`, `fatol=0.01`
- Adaptive simplex scaling enabled (`adaptive=True`)
- Parameters are log-transformed internally to ensure positivity

### Objective function

Weighted RMSE combining PDD and lateral-profile errors:

```
error = pdd_weight × PDD_RMSE + profile_weight × Profile_RMSE
```

**PDD RMSE** — evaluated at 5, 10, 20, 30 cm depth vs GBD:

```
PDD_RMSE = sqrt( mean_d [ w(d) × (pyMatRad(d) − GBD(d))² ] )
```

Depths ≥ 20 cm are weighted ×2 to preferentially correct deep-dose errors.

**Profile RMSE** — within-field dose points vs GBD at each profile depth.

### Fields used

By default all three fields are included:

| Field | Weight | Notes |
|-------|--------|-------|
| 3×3 cm² | 1/3 | Small-field penumbra |
| 10×10 cm² | 1/3 | TG-51 reference field |
| 20×20 cm² | 1/3 | Large-field scatter |

Use `--field 10x10` for a faster first pass (≈3× fewer dose calcs per iteration).

### Checkpointing

Each energy writes a checkpoint file to `examples/cache/tune_cache/`
when the optimiser terminates.  Subsequent runs load this checkpoint
automatically, so you can stop and resume without repeating calculations.

---

## Output Files

| File | Description |
|------|-------------|
| `userdata/machines/photons_{energy}.npy` | Tuned machine (overwrites or creates) |
| `examples/machine_tuning.md` | Parameter table + error summary |
| `examples/cache/tune_cache/{energy}_*.npy` | Optimiser checkpoint files |
| `--plot-dir/*.png` | Before/after PDD and profile plots |

---

## After Tuning: TG-51 Calibration

`tune_machine.py` automatically runs TG-51 calibration after tuning
(unless `--no-calibrate` is given).  The calibration computes
`abs_calib = 0.01 Gy/MU / dose(d_max)` at SSD=100 cm, 10×10 field,
and stores it in `machine["meta"]["tg51"]`.

To re-run calibration independently:

```bash
python examples/calibrate_machine.py --machine TrueBeam_6X --force
```

---

## Known Limitations

### Large-field PDD undershoot (20×20 vs GBD)

Typical values after default build:

| Energy | Error at 10 cm | Error at 30 cm |
|--------|----------------|----------------|
| 6X     | −3 to −4% | −4 to −5% |
| 6XFFF  | −2 to −3% | −3 to −4% |
| 10XFFF | −3 to −4% | −4 to −5% |
| 15X    | −2 to −3% | −3 to −4% |

This undershoot is driven by the lateral kernel weights (kernel1–3) which
encode field-size-dependent scatter from the GBD TPR data.  Tuning `m` and
`betas` adjusts the global depth-dose envelope but **cannot** selectively
correct the 20×20 response relative to 10×10.

**Resolution**: Rebuild the machine from TPR data using `build_truebeam.py`
with improved TPR measurement or post-processing, then re-tune.

### FFF beam primary fluence

For 6XFFF and 10XFFF beams, off-axis dose profile accuracy is limited by
the `primaryFluence` off-axis ratio (OAR) extracted from the GBD shallow
profile.  If the horn/softening shape is inaccurate, profiles at intermediate
depths will show residual errors even after penumbra tuning.  The
`primaryFluence` array can be manually adjusted in the `.npy` file.

---

## Tuned Parameters (run results)

*This table is auto-generated by `tune_machine.py --output-md`.*

| Energy | fwhm [mm] | m [mm⁻¹] | β₁ [mm⁻¹] | β₂ [mm⁻¹] | β₃ [mm⁻¹] | err before | err after | abs_calib [cGy/MU] | d_max [mm] |
|--------|-----------|----------|----------|----------|----------|-----------|-----------|-------------------|-----------|
| TrueBeam_6X     | (run script) | — | — | — | — | — | — | — | — |
| TrueBeam_6XFFF  | (run script) | — | — | — | — | — | — | — | — |
| TrueBeam_10XFFF | (run script) | — | — | — | — | — | — | — | — |
| TrueBeam_15X    | (run script) | — | — | — | — | — | — | — | — |
