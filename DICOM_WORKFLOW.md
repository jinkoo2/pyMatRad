# Eclipse DICOM Import & Dose Comparison Workflow

How to load Eclipse DICOM data (CT, RTStruct, RTPlan, RTDose) into matRad or
pyMatRad, run a dose calculation, and compare against the Eclipse reference dose.

---

## Supported Tools

| Task | matRad (MATLAB) | pyMatRad (Python) |
|------|----------------|-------------------|
| Load CT | `matRad_importDicomCt` | `matRad.dicom.import_ct` |
| Load RTStruct | `matRad_importDicomRtss` | `matRad.dicom.import_rtstruct` |
| Load RTPlan | `matRad_importDicomRTPlan` | `matRad.dicom.import_rtplan` |
| Load RTDose | `matRad_importDicomRTDose` | `matRad.dicom.import_rtdose` |
| One-shot import | `matRad_importDicom` | `matRad.dicom.import_dicom` |
| Import MLC fluence | — | `matRad.dicom.import_rtplan_fluence` |
| Run dose calc | `matRad_calcDoseInfluence` | `matRad.calc_dose_influence` |
| Forward-project weights | `matRad_calcDoseDirect` | `matRad.calc_dose_direct` |
| Re-optimise fluence | `matRad_fluenceOptimization` | `matRad.fluence_optimization` |
| Gamma analysis | `matRad_gammaIndex` | (not yet ported) |

---

## Sample Plans

All plans are at `../_sample_plans/eclipse_tps/` (relative to pyMatRad root).

| Directory | CT | Struct | Dose | Beams | Notes |
|-----------|----|--------|------|-------|-------|
| `7beam_IMRT/` | yes | yes | yes | 7 × 6 MV IMRT, sliding-window DMLC | Prostate, 232 CT slices |
| `ap_sMLC/` | yes | yes | yes | 1 × 6 MV, static MLC | AP beam |
| `6x_10x10/` | yes | yes | yes | 1 × 6 MV, open field | Reference 10×10 |
| `ap_IMRT/` | no* | — | yes | 1 × IMRT | Shares CT with `ap_sMLC` |
| `ap_VMAT/` | no* | — | yes | 1 × VMAT arc | Shares CT with `ap_sMLC` |
| `stereophan_ap_IMRT/` | yes | — | — | — | Stereo phantom CT only |
| `stereophan_ap_VMAT/` | yes | — | — | — | Stereo phantom CT only |
| `stereophan_IMRT_7beams/` | yes | — | — | — | Stereo phantom CT only |

\* `ap_IMRT` and `ap_VMAT` have no CT files; supply `ct_dir="ap_sMLC"` to `import_dicom`.

---

## MATLAB Workflow

```matlab
% ── 1. Import Eclipse DICOM ──────────────────────────────────────────────
dicomDir = '../_sample_plans/eclipse_tps/7beam_IMRT';
imp = matRad_DicomImporter(dicomDir);
imp = imp.matRad_importDicom();

ct          = imp.ct;
cst         = imp.cst;
pln         = imp.pln;
doseEclipse = imp.resultGUI.physicalDose;   % (Ny,Nx,Nz) [Gy], plan dose

% ── 2. Set machine ───────────────────────────────────────────────────────
% Eclipse plan stores only energy (6 MV) — map to the TrueBeam machine file
pln.machine       = 'TrueBeam_6X';
pln.radiationMode = 'photons';

% ── 3. Generate beam geometry (STF) from imported plan ───────────────────
stf = matRad_generateStf(ct, cst, pln);

% ── 4. Run matRad dose calculation ───────────────────────────────────────
dij = matRad_calcDoseInfluence(ct, cst, stf, pln);

% ── 5a. Re-optimize fluence (independent matRad plan) ────────────────────
resultMatRad = matRad_fluenceOptimization(dij, cst, pln);

% ── 5b. OR: apply Eclipse MU weights directly ────────────────────────────
%  (pyMatRad has import_rtplan_fluence; no direct MATLAB equivalent yet)

% ── 6. Compare dose cubes ────────────────────────────────────────────────
doseEclipse_interp = matRad_interpDicomDoseCube(ct, imp.ctGrid, ...
    doseEclipse, dij.doseGrid);

diff = resultMatRad.physicalDose - doseEclipse_interp;
disp(['Max abs diff: ', num2str(max(abs(diff(:)))), ' Gy']);
disp(['Mean abs diff: ', num2str(mean(abs(diff(:)))), ' Gy']);

% ── 7. Gamma analysis (3%/3mm) ───────────────────────────────────────────
res = [ct.resolution.x, ct.resolution.y, ct.resolution.z];
gamma = matRad_gammaIndex(resultMatRad.physicalDose, doseEclipse_interp, res, [3, 3]);
passRate = sum(gamma(:) <= 1) / sum(~isnan(gamma(:)));
fprintf('Gamma pass rate (3%%/3mm): %.1f%%\n', passRate*100);

% ── 8. DVH comparison ────────────────────────────────────────────────────
dvhMatRad  = matRad_calcDVH(cst, resultMatRad.physicalDose,  dij.doseGrid);
dvhEclipse = matRad_calcDVH(cst, doseEclipse_interp,         dij.doseGrid);
matRad_showDVH(dvhMatRad, cst, pln);
hold on;
matRad_showDVH(dvhEclipse, cst, pln, '--');
legend('matRad', 'Eclipse');
```

---

## Python (pyMatRad) Workflow — Re-optimise

Beam geometry taken from Eclipse; fluence re-optimised by matRad from scratch.

```python
import matRad
from matRad.dicom import import_dicom
import numpy as np

# ── 1. Import Eclipse DICOM ──────────────────────────────────────────────
result = import_dicom("../_sample_plans/eclipse_tps/7beam_IMRT")
ct           = result["ct"]
cst          = result["cst"]
pln          = result["pln"]
dose_eclipse = result["dose"]   # ndarray (Ny, Nx, Nz) [Gy]

# ── 2. Override machine & dose-calc settings ─────────────────────────────
pln["machine"] = "TrueBeam_6X"
pln["propDoseCalc"].update({
    "doseGrid":              {"resolution": {"x": 5.0, "y": 5.0, "z": 5.0}},
    "ignoreOutsideDensities": False,
    "numWorkers":            1,   # avoid OOM on full clinical CT
})

# ── 3. Generate STF ───────────────────────────────────────────────────────
stf = matRad.generate_stf(ct, cst, pln)

# ── 4. Dose influence matrix ─────────────────────────────────────────────
dij = matRad.calc_dose_influence(ct, cst, stf, pln)

# ── 5. Re-optimise fluence ────────────────────────────────────────────────
result_mr = matRad.fluence_optimization(dij, cst, pln)
dose_mr   = result_mr["physicalDose"]
```

---

## Python (pyMatRad) Workflow — Reproduce Eclipse Dose

Uses Eclipse MLC leaf sequences to reconstruct the delivered fluence map and
forward-project it through the matRad dose engine.  Works for:
- **Sliding-window DMLC** (e.g. `7beam_IMRT`) — continuous leaf motion, many CPs
- **Static MLC** (e.g. `ap_sMLC`) — 2 CPs, leaves fixed, full MU through open aperture

```python
import matRad
from matRad.dicom import import_dicom, import_rtplan_fluence
import numpy as np

# ── 1–4. Same import + dij steps as above ────────────────────────────────
result = import_dicom("../_sample_plans/eclipse_tps/7beam_IMRT")
ct, cst, pln = result["ct"], result["cst"], result["pln"]
pln["machine"] = "TrueBeam_6X"
pln["propDoseCalc"].update({
    "doseGrid": {"resolution": {"x": 5.0, "y": 5.0, "z": 5.0}},
    "ignoreOutsideDensities": False, "numWorkers": 1,
})
stf = matRad.generate_stf(ct, cst, pln)
dij = matRad.calc_dose_influence(ct, cst, stf, pln)

# ── 5. Import MLC fluence and forward-project ─────────────────────────────
w = import_rtplan_fluence("../_sample_plans/eclipse_tps/7beam_IMRT/plan.dcm", stf)
result_mr = matRad.calc_dose_direct(dij, w)
dose_mr   = result_mr["physicalDose"]

# ── 6. Compare ────────────────────────────────────────────────────────────
dose_eclipse = result["dose"]
mask = dose_eclipse > 0.05 * dose_eclipse.max()
diff = dose_mr - dose_eclipse
print(f"Mean |diff| (>5% max): {np.abs(diff[mask]).mean():.3f} Gy  "
      f"({np.abs(diff[mask]).mean() / dose_eclipse[mask].mean() * 100:.1f}%)")
print(f"Max  |diff|: {np.abs(diff).max():.3f} Gy")
```

---

## How `import_rtplan_fluence` Works

Converts DICOM MLC control-point sequences into per-bixel MU weights for
matRad's bixel grid.

```
DICOM control point i → i+1 (delta_w fraction of total MU):
  For each bixel at BEV position (x, z):
    lp   = leaf pair covering z  (from LeafPositionBoundaries)
    A(t) = A_i + t·(A_{i+1}−A_i)    t ∈ [0,1], sampled at 30 points
    B(t) = B_i + t·(B_{i+1}−B_i)
    open_frac = mean( A(t) < x < B(t) )
    fluence[bixel] += delta_w × open_frac

fluence[bixel] ×= beam_MU            → weight in MU per bixel
```

**Jaw clipping**: bixels outside `ASYMX`/`ASYMY` boundaries → fluence = 0.

**Delivery-type equivalence**:

| Delivery | CPs | Leaf motion | Behaviour |
|----------|-----|-------------|-----------|
| Sliding-window DMLC | many (e.g. 166) | continuous | open_frac ∈ (0,1) per bixel |
| Static MLC | 2 | none (A(t)=const) | open_frac ∈ {0, 1} — binary aperture |
| Step-and-shoot | 2×N segments | jumps between segments | handled identically |

**Coordinate mapping** (no SAD magnification needed — both defined at isocenter):

```
BEV rayPos_bev[0]  →  MLC x  (leaf-opening direction, A/B banks)
BEV rayPos_bev[2]  →  MLC y  (leaf-selection direction, along leaf width)
```

**Limitations**: leaf transmission and tongue-and-groove effect are not modelled
(typical error < 1% for IMRT fields).

---

## Data Structures

### `ct` dict

```python
ct = {
    "cube":       [np.ndarray],    # list[(Ny,Nx,Nz)] relative electron density
    "cubeHU":     [np.ndarray],    # list[(Ny,Nx,Nz)] Hounsfield units
    "cubeDim":    [Ny, Nx, Nz],
    "resolution": {"x": dx, "y": dy, "z": dz},  # mm
    "x":          np.ndarray,      # voxel centres (Nx,) [mm]
    "y":          np.ndarray,      # voxel centres (Ny,) [mm]
    "z":          np.ndarray,      # voxel centres (Nz,) [mm]
    "numOfCtScen": 1,
}
```

### `cst` list

```python
cst = [
    [idx, "PTV",    "TARGET", [voxel_indices], {"Priority":1, ...}, [objectives]],
    [idx, "Rectum", "OAR",    [voxel_indices], {"Priority":2, ...}, []],
    ...
]
```

### `pln` dict

```python
pln = {
    "radiationMode":  "photons",
    "machine":        "TrueBeam_6X",   # must match a file in userdata/machines/
    "numOfFractions": 20,
    "propStf": {
        "gantryAngles": np.array([205, 255, 305, 355, 45, 95, 145]),  # degrees
        "couchAngles":  np.array([0, 0, 0, 0, 0, 0, 0]),
        "isoCenter":    np.array([-2.02, -209.52, 36.17]),  # mm LPS
        "bixelWidth":   5.0,    # mm
        "SAD":          1000,   # mm
        "beamMU":       np.array([107.9, 106.6, ...]),  # MU per beam from DICOM
        "energies_MV":  np.array([6.0, 6.0, ...]),
    },
    "propDoseCalc": {
        "doseGrid": {"resolution": {"x": 5.0, "y": 5.0, "z": 5.0}},
        "ignoreOutsideDensities": False,
        "numWorkers": 1,        # 1 = in-process; avoids OOM on clinical CT
    },
}
```

### `dose` (Eclipse RTDose)

```python
dose_eclipse   # np.ndarray (Ny, Nx, Nz), interpolated onto CT grid [Gy]
               # DoseGridScaling applied automatically; plan-total dose
```

---

## Notes

- **Coordinate system**: DICOM LPS (Left-Posterior-Superior). pyMatRad uses the
  same convention — `ct.x` increases left, `ct.y` posteriorly, `ct.z` superiorly.
- **HU → density conversion**: uses the matRad default HLUT (8-point piecewise
  linear, air→water→bone).  Override by passing a custom `hlut` array to
  `import_dicom`.
- **Machine mapping**: the RTPlan stores only nominal energy (e.g. 6 MV).
  `import_rtplan` maps this to `TrueBeam_6X` automatically; override
  `pln["machine"]` if a different machine file is needed.
- **OOM on clinical CT**: a full 512×512×232 CT at 3 mm dose grid is ~4 M
  voxels.  Use `doseGrid resolution = 5 mm` and `numWorkers = 1` to avoid
  subprocess OOM kills.
- **Dose scaling**: Eclipse RTDose is plan-total.  Divide by
  `pln["numOfFractions"]` to get per-fraction dose.
- **Plans without CT** (`ap_IMRT`, `ap_VMAT`): pass
  `ct_dir="ap_sMLC"` to `import_dicom` — those plans share the `ap_sMLC` CT.
