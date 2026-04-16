# VMAT Support in pyMatRad

## Overview

Volumetric Modulated Arc Therapy (VMAT) delivers dose as the gantry rotates continuously
around the patient, modulating aperture shape, aperture weight (dose rate), and gantry
rotation speed simultaneously.  Unlike fixed-gantry IMRT, VMAT optimises over a
continuous arc, so the treatment planning system must:

1. Sample the arc at multiple angles to build the dose influence matrix.
2. Optimise fluence maps at a coarse set of **FMO** (Fluence Map Optimisation) angles.
3. Sequence the fluence maps into MLC apertures at **DAO** (Direct Aperture Optimisation) angles.
4. Refine apertures and timing at the DAO angles with arc delivery constraints (leaf speed,
   gantry rotation speed, MU rate).

---

## Three-Level Angle Hierarchy

```
Arc anchors  [-180°, 180°]
    │
    │  _setup_arc_angles()
    ▼
Fine angles  ──────────────────────────────────────────────────────── (dose calculation)
  -180° -165° -150° -135° ... every maxGantryAngleSpacing deg

DAO angles   ────────────────────────────────────── (aperture optimisation control points)
  -180°       -150°       -120° ... every maxDAOGantryAngleSpacing deg

FMO angles   ──────────────────── (fluence map optimisation)
  -180°                   -120° ... every maxFMOGantryAngleSpacing deg
             (odd multiple of DAO spacing)
```

Every FMO beam is also a DAO beam. Every DAO beam is also a fine beam.
Fine beams that are neither FMO nor DAO are **interpolated** between their bracketing DAO beams.

---

## Quick Start

```python
pln = {
    "radiationMode": "photons",
    "machine": "Generic",
    "numOfFractions": 30,
    "propStf": {
        "gantryAngles": [-180, 180],       # arc anchor points [deg]
        "couchAngles":  [0, 0],
        "bixelWidth": 5,                    # [mm]
        "generator": "PhotonVMAT",          # selects arc STF generator
        "maxGantryAngleSpacing":    15,     # fine beam spacing [deg]
        "maxDAOGantryAngleSpacing": 30,     # DAO control-point spacing [deg]
        "maxFMOGantryAngleSpacing": 45,     # FMO control-point spacing [deg]
        "continuousAperture": False,        # False = step-and-shoot
    },
    "propOpt": {"runVMAT": True},
    "propDoseCalc": {"doseGrid": {"resolution": {"x": 3, "y": 3, "z": 3}}},
}

from matRad.steering.stf_generator import generate_stf
from matRad.doseCalc.calc_dose_influence import calc_dose_influence
from matRad.optimization.fluence_optimization import fluence_optimization

stf    = generate_stf(ct, cst, pln)        # arc STF with propVMAT metadata
dij    = calc_dose_influence(ct, cst, stf, pln)
result = fluence_optimization(dij, cst, pln)
```

See `examples/example8_photons_vmat.py` for a complete runnable script.

---

## Key Configuration Parameters

| Parameter | `propStf` key | Default | Description |
|-----------|--------------|---------|-------------|
| Arc anchors | `gantryAngles` | `[-180, 180]` | Start/finish angles [deg] |
| Couch angles | `couchAngles` | `[0, 0]` | One per anchor |
| Fine spacing | `maxGantryAngleSpacing` | `4` | Dose-calc beam spacing [deg] |
| DAO spacing | `maxDAOGantryAngleSpacing` | `8` | Aperture control-point spacing [deg] |
| FMO spacing | `maxFMOGantryAngleSpacing` | `32` | Fluence optimisation spacing [deg] |
| Delivery mode | `continuousAperture` | `False` | `True` = continuous, `False` = step-and-shoot |
| Generator | `generator` | `PhotonIMRT` | Must be `"PhotonVMAT"` to use arc generator |

---

## Full VMAT Pipeline

```
generate_stf()          → arc STF (fine + DAO + FMO angles, propVMAT metadata)
calc_dose_influence()   → sparse DIJ matrix (all fine-angle beams)
fluence_optimization()  → fluence maps at FMO beams           [FMO step]
matRad_arcSequencing    → apertures distributed to DAO beams  [NOT YET PORTED]
directApertureOptimization → refine apertures + timing        [NOT YET PORTED]
calcDeliveryMetrics     → leaf speed, gantry speed, MU rate   [NOT YET PORTED]
plan_analysis()         → DVH, quality indicators
```

---

## `propVMAT` Fields

Each beam in `stf[i]` has a `propVMAT` dict populated by `StfGeneratorPhotonVMAT`.

### All beams

| Field | Type | Description |
|-------|------|-------------|
| `FMOBeam` | bool | True if this beam is an FMO control point |
| `DAOBeam` | bool | True if this beam is a DAO control point |
| `beamParentFMOIndex` | int | Index of nearest FMO beam in `arcFMOGantryAngles` array |
| `beamParentGantryAngle` | float | Gantry angle of FMO parent |
| `beamParentIndex` | int | STF list index of FMO parent beam |
| `doseAngleBorders` | [float, float] | Angular sector attributed to this beam [deg] |
| `doseAngleBorderCentreDiff` | [float, float] | Distance from beam centre to each border |
| `doseAngleBordersDiff` | float | Total angular width of dose sector [deg] |

### DAO beams (additionally)

| Field | Type | Description |
|-------|------|-------------|
| `DAOIndex` | int | Ordinal position in DAO beam sequence (1-based) |
| `DAOAngleBorders` | [float, float] | DAO influence sector [deg] |
| `DAOAngleBorderCentreDiff` | [float, float] | Distance from beam centre to each DAO border |
| `DAOAngleBordersDiff` | float | Total DAO sector width [deg] |
| `timeFacCurr` | float | Fraction of DAO sector time covered by dose sector |
| `timeFac` | list | Time blending factors (2 elements for step-and-shoot; 3 for continuous) |
| `lastDAOIndex` | int | STF index of previous DAO beam |
| `nextDAOIndex` | int | STF index of next DAO beam |
| `beamChildrenIndex` | list[int] | STF indices of child (fine) beams |
| `beamChildrenGantryAngles` | list[float] | Gantry angles of child beams |
| `numOfBeamChildren` | int | Number of child beams |

### FMO beams (additionally)

| Field | Type | Description |
|-------|------|-------------|
| `FMOAngleBorders` | [float, float] | FMO influence sector [deg] |
| `FMOAngleBorderCentreDiff` | [float, float] | Distance from beam centre to each FMO border |
| `FMOAngleBordersDiff` | float | Total FMO sector width [deg] |
| `beamSubChildrenIndex` | list[int] | STF indices of non-DAO fine beams under this FMO |
| `beamSubChildrenGantryAngles` | list[float] | Gantry angles of non-DAO fine beams |
| `numOfBeamSubChildren` | int | Number of non-DAO sub-children |

### Non-DAO fine beams (additionally)

| Field | Type | Description |
|-------|------|-------------|
| `fracFromLastDAO` | float | Interpolation weight from previous DAO beam |
| `lastDAOIndex` | int | STF index of previous DAO beam |
| `nextDAOIndex` | int | STF index of next DAO beam |
| `fracFromLastDAO_I` | float | Leaf position interpolation fraction (dose sector start) |
| `fracFromLastDAO_F` | float | Leaf position interpolation fraction (dose sector end) |
| `fracFromNextDAO_I` | float | Complementary leaf position fraction (dose sector start) |
| `fracFromNextDAO_F` | float | Complementary leaf position fraction (dose sector end) |
| `timeFracFromLastDAO` | float | Time fraction attributed to previous DAO aperture |
| `timeFracFromNextDAO` | float | Time fraction attributed to next DAO aperture |

---

## Master Ray Set

All beams in a VMAT arc share the same set of ray positions (the **master ray set**).
This is the union of per-beam ray positions from the initial geometry pass, gap-filled
to ensure every leaf row has a contiguous set of bixel columns.  Using a common ray set
is required so that leaf positions can be smoothly interpolated across control points.

---

## Implementation Status

| Component | Status | File |
|-----------|--------|------|
| Arc STF generation (geometry + propVMAT) | ✅ ported | `matRad/steering/stf_generator_vmat.py` |
| Arc dose influence calculation | ✅ works (angle-agnostic) | `matRad/doseCalc/DoseEngines/photon_svd_engine.py` |
| Fluence map optimisation (FMO step) | ✅ works (dense-arc mode) | `matRad/optimization/fluence_optimization.py` |
| Arc leaf sequencing | ❌ not yet ported | MATLAB: `matRad/sequencing/matRad_arcSequencing.m` |
| Direct aperture optimisation (DAO) | ❌ not yet ported | MATLAB: `matRad/optimization/@matRad_OptimizationProblemVMAT/` |
| Delivery metrics (leaf/gantry speed) | ❌ not yet ported | MATLAB: `matRad/matRad_calcDeliveryMetrics.m` |

---

## MATLAB Reference

The Python implementation is a direct port of:

- `matRad/matRad/steering/matRad_StfGeneratorPhotonVMAT.m`
- `matRad/examples/matRad_example8_photonsVMAT.m`

The angle-hierarchy and `propVMAT` metadata structures match the MATLAB implementation
exactly so that future ports of sequencing and DAO can reuse the same data.
