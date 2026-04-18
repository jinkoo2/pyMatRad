# pyMatRad Machine Workflow

How to build a new machine data file from commissioning measurements and
then calibrate it for TG-51 absolute dose.

---

## Overview

A pyMatRad machine file is a Python dict saved as `.npy` (numpy pickle).
It contains the photon pencil-beam SVD kernel data that the dose engine needs:

```
machine/
├── meta/
│   ├── radiationMode       "photons"
│   ├── machine             "TrueBeam_6X"
│   ├── SAD                 1000.0  [mm]
│   ├── SCD                 345.0   [mm]   source-collimator distance
│   └── tg51/              ← added by calibrate_machine.py
│       ├── abs_calib       0.010351  [Gy/MU]
│       ├── d_max_mm        14.0
│       ├── ssd_ref_mm      1000.0
│       ├── field_mm        100.0
│       └── mu_ref          1.0
└── data/
    ├── energy              6.0   [MV]
    ├── m                   float  attenuation coefficient [mm⁻¹]
    ├── betas               (3,)   build-up exponential coefficients
    ├── kernelPos           (360,) radial kernel positions [mm]
    ├── kernel              list of 501 SSD-indexed dicts (kernel1..4)
    ├── penumbraFWHMatIso   float  [mm]
    ├── primaryFluence      (P, 2) radial off-axis fluence profile
    ├── surfaceDose         float
    └── electronRangeIntensity  float
```

Machine files live in two places (search order):

1. `pyMatRad/userdata/machines/`  — custom / site-specific (`.npy` preferred)
2. `matRad/matRad/basedata/`      — original MATLAB data (`.mat`)

Within each folder `.npy` takes priority over `.mat`.

---

## Step 1 — Build a Machine from Commissioning Data

Use `matRad.machineBuilder.generate_machine()` when you have raw beam
commissioning measurements (TPR tables, output factors, off-axis profile).

### What you need

| Input | Symbol | Typical values |
|-------|--------|----------------|
| Tissue Phantom Ratio table | `tpr[depth, field_size]` | depths 0–300 mm, fields 0–400 mm |
| Output factors | `of[field_size]` | fields 10–400 mm |
| Primary fluence off-axis profile | `pf[r]` | r = 0–200 mm |
| SAD | `params["SAD"]` | 1000 mm |
| Beam energy | `params["photon_energy"]` | 6 MV |
| Source FWHM (penumbra) | `params["fwhm_gauss"]` | 5–8 mm |
| Source-collimator distance | `params["source_collimator_distance"]` | 345 mm |

### Script

```python
import numpy as np
from matRad.machineBuilder import generate_machine, save_machine

# ── Commissioning data (replace with your measurements) ──────────────────

params = {
    "SAD":                        1000.0,   # mm
    "photon_energy":              6.0,      # MV
    "fwhm_gauss":                 6.0,      # mm
    "electron_range_intensity":   0.001,
    "source_collimator_distance": 345.0,    # mm
}

# TPR: rows = depths [mm], cols = field sizes [mm]
tpr_depths_mm      = np.array([0, 10, 20, 30, 50, 100, 150, 200, 250, 300], dtype=float)
tpr_field_sizes_mm = np.array([10, 30, 50, 100, 200, 300, 400], dtype=float)
tpr = np.array([...])   # shape (10, 7)  — measured TPR values

# Output factors  (normalised to 1.0 at 100×100 mm)
of_mm   = np.array([10, 20, 30, 50, 100, 200, 300, 400], dtype=float)
of_vals = np.array([0.89, 0.94, 0.96, 0.98, 1.00, 1.04, 1.07, 1.08])

# Primary fluence (flat beam → all ones; use measured profile for FFF beams)
pf_r    = np.arange(0, 201, dtype=float)   # 0–200 mm
pf_vals = np.ones_like(pf_r)               # flat for flattened beams

# ── Generate machine ──────────────────────────────────────────────────────

machine = generate_machine(
    name               = "MyLinac_6X",
    params             = params,
    tpr_field_sizes_mm = tpr_field_sizes_mm,
    tpr_depths_mm      = tpr_depths_mm,
    tpr                = tpr,
    of_mm              = of_mm,
    of_vals            = of_vals,
    pf_r               = pf_r,
    pf_vals            = pf_vals,
)

save_machine(machine, "userdata/machines/photons_MyLinac_6X.npy")
print("Machine saved.")
```

### What `generate_machine` does internally

1. **Inserts a zero-field-size TPR column** by linear extrapolation.
2. **Fits the attenuation coefficient µ** from a log-linear regression on
   the post-build-up tail of the TPR for the zero-field case.
3. **Fits three build-up betas** (`β₁, β₂, β₃`) by placing the depth-dose
   peak at three reference positions (Scholz 1994 SVD decomposition).
4. **Applies small-field output-factor correction** via convolution of the
   primary fluence profile with a Gaussian penumbra model.
5. **Generates 501 radial kernels** (one per SSD from 500 to 1000 mm) using
   the output-factor-corrected TPR.

---

## Step 2 — TG-51 Absolute Calibration

After building a machine, run `calibrate_machine.py` to compute and store
the TG-51 absolute calibration factor `abs_calib` [Gy/MU].

### What TG-51 defines

> At the reference conditions (SSD = 100 cm, 10×10 cm field, depth = d_max),
> **1 MU delivers 1 cGy**.

Because the SVD engine normalises dose internally, the engine-computed dose at
these reference conditions is generally not exactly 1 cGy/MU.  The
calibration factor corrects for this discrepancy:

```
dose_engine(d_max, 10×10, SSD=100 cm, w=1)  =  D_ref   (engine units)
abs_calib  =  0.01 Gy/MU  /  D_ref
```

When `import_rtplan_fluence` later computes bixel weights for an Eclipse plan:

```
w_i  =  open_frac_i × beam_MU × abs_calib
```

so that `dose = dij @ w` returns dose in physical Gy.

### Running the calibration

```bash
# Calibrate and update the .npy file
python examples/calibrate_machine.py --machine TrueBeam_6X

# Dry run (compute only, no file write)
python examples/calibrate_machine.py --machine TrueBeam_6X --dry-run

# Save depth-dose + lateral profile plot
python examples/calibrate_machine.py --machine TrueBeam_6X \
    --plot-dir examples/cache/calib_plots

# Force overwrite of existing calibration entry
python examples/calibrate_machine.py --machine TrueBeam_6X --force
```

### What the script does

1. **Loads the machine** with `load_machine()` to get SAD and kernel data.
2. **Builds a 200×300×200 mm water phantom** at 2 mm isotropic resolution.
   The isocenter is placed on the entrance surface so that SSD = 1000 mm:

   ```
   SSD  = y_surface − source_y  =  y_surface − (iso_y − SAD)  =  1000 mm
   →  iso_y  =  y_surface + SAD − 1000
   ```

   For standard SAD = 1000 mm: iso is exactly at the phantom surface.
   For other SADs the formula still ensures SSD = 1000 mm.

3. **Generates STF** — single AP beam (gantry 0°, couch 0°), 5 mm bixels,
   isocenter at phantom surface.
4. **Computes the full dij** dose-influence matrix.
5. **Selects 10×10 cm bixels** — sets `w = 1` for all bixels whose BEV
   position satisfies `|x_bev| ≤ 50 mm` and `|z_bev| ≤ 50 mm`, `w = 0`
   for all others.  This models a perfect open 10×10 cm field with 1 MU.
6. **Computes the forward dose** `dose = dij @ w`.
7. **Extracts the central-axis depth profile** at `x = iso_x, z = iso_z`.
8. **Finds d_max** (depth of dose maximum) and the dose there.
9. **Computes** `abs_calib = 0.01 / dose(d_max)` [Gy/MU].
10. **Writes** `machine["meta"]["tg51"]` to the `.npy` file.

### Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--machine NAME` | required | Machine name, e.g. `TrueBeam_6X` |
| `--radiation-mode MODE` | `photons` | Radiation modality |
| `--dry-run` | off | Compute but do not write to file |
| `--force` | off | Overwrite existing tg51 entry without prompting |
| `--plot-dir DIR` | — | Save PDD + lateral profile PNG here |

### Stored tg51 dict

```python
machine["meta"]["tg51"] = {
    "abs_calib":  0.010351,   # Gy/MU
    "d_max_mm":   14.0,       # mm
    "ssd_ref_mm": 1000.0,     # mm  (100 cm)
    "field_mm":   100.0,      # mm  (10 cm square)
    "mu_ref":     1.0,        # MU
}
```

`_tg51_abs_calib()` in `importer.py` reads this entry first; it falls back
to an inverse-square analytical estimate (≈4–6% less accurate) if the entry
is absent.

---

## Step 3 — Verify

Confirm the calibration is picked up by the importer:

```python
from matRad.basedata import load_machine
from matRad.dicom.importer import _tg51_abs_calib

machine   = load_machine({"radiationMode": "photons", "machine": "TrueBeam_6X"})
abs_calib = _tg51_abs_calib(machine)
print(f"abs_calib = {abs_calib*100:.4f} cGy/MU")   # should be ≈ 1.035
```

Then run a full Eclipse plan to check dose agreement:

```bash
python examples/import_eclipse_dicom.py --plan 7beam_IMRT \
    --eclipse-fluence --machine TrueBeam_6X
```

---

## Existing Machine Files

All machines in `userdata/machines/` already have a `tg51` entry:

| File | Energy | SAD | d_max | abs_calib |
|------|--------|-----|-------|-----------|
| `photons_TrueBeam_6X.npy` | 6 MV | 1000 mm | 14.0 mm | 1.0351 cGy/MU |
| `photons_TrueBeam_6XFFF.npy` | 6 MV FFF | 1000 mm | 12.0 mm | 1.0310 cGy/MU |
| `photons_TrueBeam_10XFFF.npy` | 10 MV FFF | 1000 mm | 22.0 mm | 1.0480 cGy/MU |
| `photons_TrueBeam_15X.npy` | 15 MV | 1000 mm | 26.0 mm | 1.0589 cGy/MU |

---

## Troubleshooting

### `abs_calib` is far from 1.0 cGy/MU

Typical values are 1.00–1.06 cGy/MU.  Large deviations (> 10%) suggest:
- Wrong `SAD` in the machine meta (check `machine["meta"]["SAD"]`).
- TPR data not normalised correctly (TPR should be 1.0 at d_max for 10×10 cm).
- Off-axis fluence profile not flat for a flattened beam.

### `No bixels found inside the 10×10 cm field`

generate_stf placed bixels outside the ±50 mm range.  This can happen if
the target VOI is too narrow.  Increase `_TARGET_WIDTH_N` in
`calibrate_machine.py` or check that the phantom dimensions match the SAD.

### Machine file is `.mat` only (no `.npy` in userdata)

`calibrate_machine.py` automatically creates a new `.npy` in
`userdata/machines/` when no `.npy` exists, copying the machine dict loaded
from the `.mat` source file.  Subsequent `load_machine()` calls will prefer
the `.npy` version.
