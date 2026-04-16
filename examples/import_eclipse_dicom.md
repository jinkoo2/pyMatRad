# import_eclipse_dicom.py — User Guide

Script to import Eclipse DICOM plans into pyMatRad, run dose calculations,
and compare against Eclipse RTDose.

---

## Location

```
pyMatRad/
├── examples/
│   └── import_eclipse_dicom.py    ← this script
└── matRad/
    └── dicom/
        └── importer.py            ← DICOM import library
```

Sample plans: `../_sample_plans/eclipse_tps/` (one directory up from pyMatRad root)

---

## Quick Start

```bash
cd /path/to/pyMatRad
conda activate pyMatRad

# Import only (no dose calc) — fast, good for inspecting structures/beams
python examples/import_eclipse_dicom.py --plan 7beam_IMRT --no-dose-calc

# Import + re-optimise fluence with matRad objectives
python examples/import_eclipse_dicom.py --plan 7beam_IMRT

# Import + reproduce Eclipse dose from MLC leaf sequences
python examples/import_eclipse_dicom.py --plan 7beam_IMRT --eclipse-fluence

# Plan with no CT — supply CT from a sibling directory
python examples/import_eclipse_dicom.py --plan ap_IMRT --ct-dir ap_sMLC --no-dose-calc

# Finer bixel grid and dose grid
python examples/import_eclipse_dicom.py --plan 7beam_IMRT --eclipse-fluence \
    --bixel-width 2.5 --dose-grid 3.0
```

---

## Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--plan NAME` | `7beam_IMRT` | Subdirectory name inside `_sample_plans/eclipse_tps/` |
| `--ct-dir DIR` | — | Separate CT directory (relative to `eclipse_tps/`). Needed for `ap_IMRT` and `ap_VMAT` which share the `ap_sMLC` CT. |
| `--no-dose-calc` | off | Skip STF generation, dij computation, and optimisation. Only imports and prints structure/beam summary. |
| `--eclipse-fluence` | off | Use Eclipse MLC leaf sequences to reproduce Eclipse dose rather than re-optimising fluence from scratch. |
| `--bixel-width MM` | `5.0` | Bixel side length in mm (square bixels). Independent of dose grid resolution. |
| `--dose-grid MM` | `5.0` | Isotropic dose calculation grid resolution in mm. |
| `--force` | off | Ignore all cached files and recompute from scratch. |
| `--cache-root DIR` | `examples/cache/` | Root directory for cache files. |

`--bixel-width` and `--dose-grid` are independent parameters:

- `--bixel-width` controls how finely the MLC aperture is sampled in BEV space.
  Smaller values capture more detail in the fluence map but increase the number
  of columns in the dij matrix and slow dose calculation.
- `--dose-grid` controls the spatial resolution of the dose cube output.

Cache files include both values in the tag (`dg5.0mm_bw5.0mm`) so changing
either parameter automatically triggers a recompute.

---

## Available Plans

| Plan | CT | Struct | Dose | Delivery | Notes |
|------|----|--------|------|----------|-------|
| `7beam_IMRT` | yes | yes | yes | Sliding-window DMLC, 7 beams 6 MV | Prostate, 232 slices |
| `ap_sMLC` | yes | yes | yes | Static MLC, 1 beam 6 MV | AP open field |
| `6x_10x10` | yes | yes | yes | Open field, 1 beam 6 MV | Reference field |
| `ap_IMRT` | no* | — | yes | IMRT, 1 beam | Needs `--ct-dir ap_sMLC` |
| `ap_VMAT` | no* | — | yes | VMAT arc, 1 beam | Needs `--ct-dir ap_sMLC` |
| `stereophan_ap_IMRT` | yes | — | — | — | CT only |
| `stereophan_ap_VMAT` | yes | — | — | — | CT only |
| `stereophan_IMRT_7beams` | yes | — | — | — | CT only |

\* `ap_IMRT` and `ap_VMAT` share the CT from `ap_sMLC`.

---

## What Each Mode Does

### `--no-dose-calc`

1. Scan folder for CT / RTStruct / RTPlan / RTDose files
2. Import CT → HU → relative electron density (RED)
3. Set RED = 0 for all voxels outside the BODY/External contour
4. Import RTStruct → rasterise contours → voxel indices → `cst`
5. Import RTPlan → beam angles, isocenter, SAD, MU per beam → `pln`
6. Import RTDose → interpolate onto CT grid → dose ndarray
7. Print structure list and beam summary

### Default (re-optimise)

Steps 1–7 above, then:

8. Set dose grid to `--dose-grid` mm, bixel width to `--bixel-width` mm, `numWorkers=1`
9. `generate_stf` — project TARGET voxels through each beam angle, snap to
   bixel grid → list of ray/bixel positions
10. `calc_dose_influence` — ray-trace + SVD kernel convolution → sparse dij matrix
11. `fluence_optimization` — L-BFGS-B minimises dose objectives → weight vector `w`
12. `dose = dij @ w` → dose cube
13. Compare vs Eclipse RTDose; save PNG comparison plots

### `--eclipse-fluence`

Steps 1–7 as above, then:

8. Set dose grid resolution and bixel width.
9. `stf_from_rtplan_aperture` — parse MLC jaw extents from `plan.dcm` and
   create a **field-covering** bixel grid (uniform, full jaw aperture) instead
   of the PTV-projected grid used by `generate_stf`.  This is critical: the
   Eclipse field aperture is typically much larger than the PTV footprint.
10. `calc_dose_influence` — ray-trace + kernel convolution → sparse dij matrix
11. `import_rtplan_fluence` — parse MLC `ControlPointSequence` from `plan.dcm`,
    compute per-bixel fluence weights accounting for partial bixel overlap,
    multiply by `beam_MU × abs_calib × numOfFractions` → weight vector `w`
12. `calc_dose_direct` — `dose = dij @ w` → dose cube (no optimisation)
13. Compare vs Eclipse RTDose; save PNG comparison plots

---

## Output

Comparison PNG plots saved to `examples/dicom_comparison_plots/`:

```
{plan}_{mode}_axial.png
{plan}_{mode}_coronal.png
{plan}_{mode}_sagittal.png
```

Each figure has three panels: **Eclipse RTDose | pyMatRad dose | Difference (matRad − Eclipse)**.

Slices pass through the **isocenter** (read from `pln["propStf"]["isoCenter"]`),
so you always see the high-dose region at the target.  The slice position is
printed to the console and shown in each figure's title, e.g.:

```
coronal  y = -210 mm
```

If the isocenter is unavailable, the script falls back to the dose-weighted
centroid of the Eclipse RTDose.

---

## Code Architecture

### `matRad/dicom/importer.py`

| Function | Purpose |
|----------|---------|
| `import_dicom(dir, ct_dir=None)` | One-shot: scans folder, calls all four importers; zeros RED outside BODY |
| `import_ct(files, hlut)` | Reads CT slices, sorts by z, converts HU→RED |
| `import_rtstruct(file, ct)` | Rasterises polygon contours into voxel indices |
| `import_rtplan(file, ct)` | Extracts beam geometry into `pln` dict |
| `import_rtdose(file, ct)` | Reads dose cube, applies scaling, interpolates to CT grid |
| `import_rtplan_fluence(file, stf, machine, num_fractions)` | Parses MLC leaf sequences → per-bixel weights (total plan dose) |
| `stf_from_rtplan_aperture(file, pln, bixel_width, machine)` | Field-covering bixel grid from DICOM jaw extents |
| `_parse_beam_mlc(beam_ds)` | Extracts jaw, leaf bounds, A/B arrays for one DICOM beam |
| `_fluence_at_bixels(mlc, x, z, MU, bixel_width)` | Integrates partial bixel overlap over all CP transitions |

### `matRad/doseCalc/calc_dose_direct.py`

```python
def calc_dose_direct(dij, w) -> dict:
    """dose = dij @ w  (sparse matrix-vector product)"""
```

Returns `{"physicalDose": ndarray(Ny,Nx,Nz), "w": w, "doseGrid": ...}`.

---

## How `import_rtplan_fluence` Works

For each beam, for each of the N−1 control-point transitions:

```
delta_w = (cum_w[i+1] - cum_w[i]) / cum_w[-1]   # normalized MU fraction
For each bixel at BEV (x, z):
  Sample t = [0, 1/(n_t-1), ..., 1]              # n_t=30 points per transition

  x-direction partial overlap (leaf opening):
    A(t) = A_i + t·(A_{i+1} − A_i)              # bank-A position at time t
    B(t) = B_i + t·(B_{i+1} − B_i)              # bank-B position at time t
    x_open(t) = clip(min(x+bw/2, B(t)) − max(x−bw/2, A(t)), 0, bw) / bw

  z-direction partial overlap (leaf selection):
    z_weight[lp] = clip(min(z+bw/2, bounds[lp+1]) − max(z−bw/2, bounds[lp]),
                        0, bw) / bw              # fraction of bixel in leaf row lp

  open_frac = mean_t( sum_lp( z_weight[lp] · x_open(t)[lp] ) )

  fluence[bixel] += delta_w × open_frac

fluence[bixel] ×= beam_MU × abs_calib × numOfFractions
```

### Partial bixel overlap

The fluence calculation accounts for partial overlap in both BEV dimensions:

- **x-direction (leaf opening)**: rather than a binary open/closed test at the
  bixel centre, the fraction of the bixel width `[x−bw/2, x+bw/2]` that lies
  inside the MLC opening `[A(t), B(t)]` is computed at each time sample.  A
  bixel whose centre is exactly on an MLC leaf edge receives 50% weight.

- **z-direction (leaf selection)**: a bixel straddling two leaf rows is weighted
  proportionally.  `z_weight[lp]` is the fraction of the bixel height covered
  by leaf pair `lp`; the final open fraction is the dot product of `z_weight`
  and the per-leaf x open fractions.

### Number of fractions

DICOM `BeamMeterset` stores per-fraction MU.  Eclipse RTDose stores the **total
plan dose** (sum over all fractions).  `import_rtplan_fluence` accepts a
`num_fractions` parameter (read from `pln["numOfFractions"]`) and multiplies the
weight vector by this factor so that `calc_dose_direct` returns total plan dose
directly comparable to Eclipse RTDose:

```python
w *= num_fractions   # single-fraction MU → total-plan equivalent
```

For a 20-fraction plan delivering 3 Gy/fraction, `num_fractions=20` and the
resulting dose cube represents the full 60 Gy plan dose.

### CumulativeMetersetWeight normalisation

DICOM allows `CumulativeMetersetWeight` to be stored either normalised to
`[0, 1]` or as absolute MU values.  `import_rtplan_fluence` normalises
robustly with:

```python
delta_w = np.diff(cum_w) / cum_w[-1]   # always sums to 1.0
```

**Works for all IMRT delivery types without special-casing:**

| Type | CPs | Leaf motion per CP gap | Result |
|------|-----|------------------------|--------|
| Sliding-window DMLC | many (e.g. 166) | continuous, small steps | fractional x_open from partial overlap |
| Static MLC | 2 | none (A(t) = const) | clean binary aperture |
| Step-and-shoot | 2 per segment | jump between segments | each segment's aperture, weighted by delta_w |

**Coordinate mapping** — both defined at isocenter, no magnification needed:

```
BEV rayPos_bev[0]  →  MLC x  (leaf-opening, A/B banks)
BEV rayPos_bev[2]  →  MLC y  (leaf selection, along leaf width)
```

**Jaw clipping**: bixels whose extent does not overlap `ASYMX`/`ASYMY` → fluence = 0.

---

## CT Preprocessing

### HU → Relative Electron Density

CT Hounsfield Units are converted to relative electron density (RED) via a
piecewise-linear lookup table (`_DEFAULT_HLUT`):

| HU | RED |
|----|-----|
| −1024 | 0.00 |
| −950 | 0.04 |
| −700 | 0.33 |
| −200 | 0.82 |
| 0 | 1.00 |
| 300 | 1.10 |
| 1500 | 1.47 |
| 3000 | 2.50 |

### BODY masking

After RTStruct import, `import_dicom` searches `cst` for a structure whose name
contains `"body"`, `"external"`, or `"outer contour"` (case-insensitive) and
sets `ct["cube"][0] = 0.0` for all voxels **outside** that contour.  This
prevents unphysical electron density values in air outside the patient from
contributing to ray-tracing.  A summary line is printed:

```
  RED zeroed outside BODY (1 230 456 voxels, 61.2%)
```

If no BODY structure is found the CT is used unmodified and a warning is printed.

---

## Memory Notes

A full clinical CT (512×512×232, 3 mm dose grid) creates ~4 M voxel dose grid.
The script always sets:

```python
pln["propDoseCalc"] = {
    "doseGrid":              {"resolution": {"x": 5.0, "y": 5.0, "z": 5.0}},
    "ignoreOutsideDensities": False,
    "numWorkers":            1,
}
```

- **5 mm grid**: reduces dose voxels from ~4 M to ~600 K
- **`numWorkers=1`**: runs dose workers in-process; subprocess OOM kills are
  invisible to Python (exit code −9) and caused the `BrokenProcessPool` error
  seen with the default multi-worker mode on clinical CT

To use a finer grid or more workers, override on the command line:

```bash
python examples/import_eclipse_dicom.py --plan 7beam_IMRT \
    --eclipse-fluence --dose-grid 3.0 --bixel-width 2.5
```

or in code after import:

```python
pln["propDoseCalc"]["doseGrid"] = {"resolution": {"x": 3.0, "y": 3.0, "z": 3.0}}
pln["propDoseCalc"]["numWorkers"] = 4
```
