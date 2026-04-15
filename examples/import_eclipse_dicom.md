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
```

---

## Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--plan NAME` | `7beam_IMRT` | Subdirectory name inside `_sample_plans/eclipse_tps/` |
| `--ct-dir DIR` | — | Separate CT directory (relative to `eclipse_tps/`). Needed for `ap_IMRT` and `ap_VMAT` which share the `ap_sMLC` CT. |
| `--no-dose-calc` | off | Skip STF generation, dij computation, and optimisation. Only imports and prints structure/beam summary. |
| `--eclipse-fluence` | off | Use Eclipse MLC leaf sequences to reproduce Eclipse dose rather than re-optimising fluence from scratch. |

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
2. Import CT → HU → relative electron density
3. Import RTStruct → rasterise contours → voxel indices → `cst`
4. Import RTPlan → beam angles, isocenter, SAD, MU per beam → `pln`
5. Import RTDose → interpolate onto CT grid → dose ndarray
6. Print structure list and beam summary

### Default (re-optimise)

Steps 1–6 above, then:

7. Set dose grid to 5 mm, `numWorkers=1` (avoids OOM on clinical CT)
8. `generate_stf` — project TARGET voxels through each beam angle, snap to
   5 mm bixel grid → list of ray/bixel positions
9. `calc_dose_influence` — ray-trace + SVD kernel convolution → sparse dij matrix
10. `fluence_optimization` — L-BFGS-B minimises dose objectives → weight vector `w`
11. `dose = dij @ w` → dose cube
12. Compare vs Eclipse RTDose; save PNG comparison plots

### `--eclipse-fluence`

Steps 1–9 as above, then:

10. `import_rtplan_fluence` — parse MLC `ControlPointSequence` from `plan.dcm`,
    integrate open fraction over all CP transitions at each bixel position,
    multiply by beam MU → weight vector `w` in MU
11. `calc_dose_direct` — `dose = dij @ w` → dose cube (no optimisation)
12. Compare vs Eclipse RTDose; save PNG comparison plots

---

## Output

Comparison PNG plots saved to `examples/dicom_comparison_plots/`:

```
{plan}_{mode}_axial.png
{plan}_{mode}_coronal.png
{plan}_{mode}_sagittal.png
```

Each figure has three panels: Eclipse RTDose | pyMatRad dose | Difference (matRad − Eclipse).

---

## Code Architecture

### `matRad/dicom/importer.py`

| Function | Purpose |
|----------|---------|
| `import_dicom(dir, ct_dir=None)` | One-shot: scans folder, calls all four importers |
| `import_ct(files, hlut)` | Reads CT slices, sorts by z, converts HU→RED |
| `import_rtstruct(file, ct)` | Rasterises polygon contours into voxel indices |
| `import_rtplan(file, ct)` | Extracts beam geometry into `pln` dict |
| `import_rtdose(file, ct)` | Reads dose cube, applies scaling, interpolates to CT grid |
| `import_rtplan_fluence(file, stf)` | Parses MLC leaf sequences → per-bixel weights |
| `_parse_beam_mlc(beam_ds)` | Extracts jaw, leaf bounds, A/B arrays for one DICOM beam |
| `_fluence_at_bixels(mlc, x, z, MU)` | Integrates open fraction over CP transitions |

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
delta_w = cum_w[i+1] - cum_w[i]          # fraction of total MU
For each bixel at BEV (x, z):
  lp = leaf pair covering z               # from LeafPositionBoundaries
  Sample t = [0, 1/(n_t-1), ..., 1]      # n_t=30 points
  A(t) = A_i + t·(A_{i+1} − A_i)
  B(t) = B_i + t·(B_{i+1} − B_i)
  open_frac = mean(A(t) < x < B(t))
  fluence[bixel] += delta_w × open_frac
fluence[bixel] ×= beam_MU                # units: MU per bixel
```

**Works for all IMRT delivery types without special-casing:**

| Type | CPs | Leaf motion per CP gap | Result |
|------|-----|------------------------|--------|
| Sliding-window DMLC | many (166) | continuous, small steps | fractional open_frac |
| Static MLC | 2 | none (A(t) = const) | binary {0, 1} aperture |
| Step-and-shoot | 2 per segment | jump between segments | each segment's binary aperture, weighted |

**Coordinate mapping** — both defined at isocenter, no magnification needed:

```
BEV rayPos_bev[0]  →  MLC x  (leaf-opening, A/B banks)
BEV rayPos_bev[2]  →  MLC y  (leaf selection, along leaf width)
```

**Jaw clipping**: bixels outside `ASYMX`/`ASYMY` → fluence = 0.

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

To use a finer grid or more workers, override these after import:

```python
pln["propDoseCalc"]["doseGrid"] = {"resolution": {"x": 3.0, "y": 3.0, "z": 3.0}}
pln["propDoseCalc"]["numWorkers"] = 4
```
