# matRad Data Objects Reference

Core data structures used throughout matRad (MATLAB) and pyMatRad (Python).

---

## CT — Computed Tomography Image

Stores the patient CT image and geometry.

| Field | Shape / Type | Description |
|---|---|---|
| `cubeDim` | `[Ny, Nx, Nz]` | Voxel grid dimensions (y=rows, x=cols, z=slices) |
| `cubeHU` | cell of `(Ny,Nx,Nz)` float | Raw Hounsfield unit volumes, one per CT scenario |
| `cube` | cell of `(Ny,Nx,Nz)` float | Water-equivalent relative electron density (RED), derived from cubeHU via HLUT |
| `resolution` | struct `{x, y, z}` | Voxel spacing [mm] |
| `x`, `y`, `z` | 1D arrays | World coordinate of each voxel center along each axis [mm] |
| `numOfCtScen` | int | Number of CT scenarios (1 = nominal) |
| `hlut` | `(N,2)` array | Hounsfield lookup table: HU → RED pairs |

**Coordinate convention:**
- `cubeDim = [Ny, Nx, Nz]` — rows are y, columns are x, slices are z
- Linear index (MATLAB, 1-based, Fortran/column-major):
  `idx = i + (j-1)*Ny + (k-1)*Ny*Nx`  where i∈[1..Ny], j∈[1..Nx], k∈[1..Nz]
- Python: `idx = i + j*Ny + k*Ny*Nx`  (0-based)
- Origin defaults to geometric center of the volume

**HU → RED conversion:**
- Simple linear (used when no HLUT): `RED = max(0, 1 + HU/1000)`
  - Air (HU=−1000) → RED=0, Water (HU=0) → RED=1
- HLUT: piecewise-linear lookup from `ct.hlut` (preferred, used by MATLAB)

---

## CST — Clinical Structure Table

Defines anatomical regions of interest (targets, OARs) and their dose objectives.

**Format:** cell array, one row per structure.

| Column | Field | Type | Description |
|---|---|---|---|
| 1 | Index | int | Structure index |
| 2 | Name | string | Structure name, e.g. `"PTV"`, `"Lung_L"` |
| 3 | Type | string | `"TARGET"` or `"OAR"` |
| 4 | Voxels | `{V1, V2, ...}` | Cell of 1-based linear CT voxel indices, one per scenario |
| 5 | Properties | struct | Display color, visibility |
| 6 | Objectives | struct/cell | Dose objectives/constraints for the optimizer |

**Objectives (column 6) — common types:**

| Type | Parameters | Cost function |
|---|---|---|
| `SquaredDeviation` | penalty, d_ref | `penalty × (d − d_ref)²` — push dose to target |
| `SquaredOverdosing` | penalty, d_ref | `penalty × max(0, d − d_ref)²` — penalize overdose |
| `SquaredUnderdosing` | penalty, d_ref | `penalty × max(0, d_ref − d)²` — penalize underdose |
| `MeanDose` | penalty, d_ref | `penalty × (mean(d) − d_ref)²` |
| `MaxDose` | penalty, d_max | `penalty × max(0, d − d_max)²` |
| `MinDose` | penalty, d_min | `penalty × max(0, d_min − d)²` |
| `EUD` | penalty, d_ref, n | Equivalent uniform dose |

---

## PLN — Treatment Plan

High-level plan specification — inputs to STF generation, dose calc, and optimization.

| Field | Type | Description |
|---|---|---|
| `radiationMode` | string | `"photons"`, `"protons"`, `"carbon"` |
| `machine` | string | Machine name (e.g. `"Generic"`) |
| `numOfFractions` | int | Number of treatment fractions |
| `bioModel` | string | `"none"` (physical dose), `"LEM"`, `"MCN"`, etc. |
| `multScen` | string | `"nomScen"` (nominal), `"wcScen"`, `"rndScen"` |
| **`propStf`** | struct | Beam geometry inputs: |
| `propStf.gantryAngles` | `[1 × nBeams]` | Gantry angles [°] |
| `propStf.couchAngles` | `[1 × nBeams]` | Couch rotation angles [°] |
| `propStf.bixelWidth` | float | Pencil beam grid spacing at isocenter [mm] (typically 5) |
| `propStf.isoCenter` | `[nBeams × 3]` | Isocenter [x,y,z] per beam [mm] |
| **`propDoseCalc`** | struct | Dose engine inputs: |
| `propDoseCalc.doseGrid.resolution` | struct `{x,y,z}` | Dose grid voxel size [mm] |
| `propDoseCalc.kernelCutOff` | float | Max lateral kernel radius [mm] (default 200) |
| `propDoseCalc.geometricLateralCutOff` | float | Ray-tracing lateral cutoff [mm] (default 50) |
| **`propOpt`** | struct | Optimizer inputs: |
| `propOpt.runDAO` | bool | Run direct aperture optimization |
| `propOpt.optimizer` | string | `"IPOPT"`, `"fmincon"` |

---

## STF — Steering Information

Describes the beam geometry: source position, ray layout, and bixel grid for each beam.
Output of `matRad_generateStf` / `generate_stf`.

**Format:** array/list of structs, one per beam.

| Field | Type | Description |
|---|---|---|
| `gantryAngle` | float | Gantry rotation angle [°] |
| `couchAngle` | float | Couch rotation angle [°] |
| `SAD` | float | Source-to-axis distance [mm] (typically 1000) |
| `bixelWidth` | float | Pencil beam spacing at isocenter [mm] |
| `isoCenter` | `[x, y, z]` | Beam isocenter in world coordinates [mm] |
| `sourcePoint` | `[x, y, z]` | X-ray source position in world coordinates [mm] |
| `sourcePoint_bev` | `[x, y, z]` | Source position in beam's-eye-view coordinates |
| `numOfRays` | int | Number of rays (pencil beams) in this beam |
| `totalNumOfBixels` | int | Total bixels (= numOfRays for photons with 1 energy) |
| `ray` | array of structs | One entry per ray (see below) |

**Per-ray fields:**

| Field | Type | Description |
|---|---|---|
| `rayPos_bev` | `[x, z]` | Ray position at isocenter plane in BEV [mm] |
| `rayPos` | `[x, y, z]` | Ray position in world coordinates [mm] |
| `targetPoint` | `[x, y, z]` | Distal end of ray in world coordinates [mm] |
| `targetPoint_bev` | `[x, y, z]` | Distal end of ray in BEV [mm] |
| `SSD` | float | Source-to-surface distance for this ray [mm] |
| `energy` | array | Beam energy [MeV] (photons: typically `[6.0]`) |

**How rays are selected:**
1. Project all target voxels onto the beam's isocenter plane (BEV)
2. Quantize to bixel grid (spacing = `bixelWidth`)
3. Add margin around the projection (= `bixelWidth` for photons)
4. Only rays that hit target voxels are kept → different beams have different ray counts

---

## DIJ — Dose Influence Matrix

Stores the dose deposited per unit bixel weight at every dose-grid voxel.
This is the central object connecting beam weights to 3D dose.

| Field | Type | Description |
|---|---|---|
| `physicalDose` | cell of sparse matrices | `{D_scenario1, ...}` each `(nVoxels × nBixels)`, sparse float |
| `ctGrid` | struct | CT grid geometry (`x`, `y`, `z`, `resolution`, `dimensions`, `numOfVoxels`) |
| `doseGrid` | struct | Dose grid geometry (same fields; can differ from CT grid) |
| `numOfBeams` | int | Number of beams |
| `numOfRaysPerBeam` | `[1 × nBeams]` | Rays per beam |
| `totalNumOfRays` | int | Total ray count |
| `totalNumOfBixels` | int | Total bixel count (= number of columns in `physicalDose`) |
| `numOfScenarios` | int | Number of scenarios |
| `beamNum` | `[1 × nBixels]` | Beam index for each bixel column |
| `rayNum` | `[1 × nBixels]` | Ray index within beam for each bixel column |
| `bixelNum` | `[1 × nBixels]` | Bixel index within ray for each bixel column |

**The key relationship:**

```
dose_3d = reshape(D @ w, doseGrid.dimensions)

where:
  D   = dij.physicalDose{1}    shape: (nVoxels, nBixels), sparse
  w   = bixel weight vector     shape: (nBixels,)
  D@w = flat dose vector        shape: (nVoxels,)
```

**Bixel ordering in `w` and `D`:**
Bixels are packed sequentially: all bixels of beam 0, then beam 1, etc.

```
w = [w_beam0_bixel0, ..., w_beam0_bixelN | w_beam1_bixel0, ... | ...]
     ↑ offset = 0                          ↑ offset = nBixels_beam0
```

To extract weights for beam `k`:
```python
offset = sum(stf[i]['totalNumOfBixels'] for i in range(k))
count  = stf[k]['totalNumOfBixels']
w_beam_k = w[offset : offset + count]
```

---

## Bixel (Beam Element)

A bixel is the smallest independently-weighted unit of a treatment beam — one "pixel" of the beam field.

- **Size:** `bixelWidth × bixelWidth` mm² at isocenter (e.g. 5×5 mm²)
- **Weight `w[j]`:** fluence (intensity) assigned to bixel j by the optimizer
- **Dose contribution:** column j of D gives the 3D dose distribution from bixel j at unit weight
- **Physical meaning:** one infinitesimally thin pencil beam (photons) or spot (protons/carbon)

For photon IMRT with 1 energy per ray: `nBixels_per_beam = nRays_per_beam`

---

## resultGUI — Optimization / Forward Dose Result

Output of `matRad_fluenceOptimization` or forward dose calculation.

| Field | Type | Description |
|---|---|---|
| `physicalDose` | `(Ny,Nx,Nz)` float | Physical dose per fraction [Gy/fx] on CT grid |
| `w` | `(nBixels,)` float | Optimized bixel weight vector |
| `wInit` | `(nBixels,)` float | Initial weights (before optimization) |
| `qi` | struct array | Quality indicators per structure (DVH metrics) |

**Quality indicators (`qi`) fields:**

| Field | Description |
|---|---|
| `D_mean` | Mean dose [Gy] |
| `D_min`, `D_max` | Min/max dose [Gy] |
| `D_2`, `D_50`, `D_95`, `D_98` | Dose at 2/50/95/98th percentile [Gy] |
| `V_95` | Volume receiving ≥ 95% of prescription [%] |
| `EUD` | Equivalent uniform dose [Gy] |
| `TCP` | Tumor control probability |
| `NTCP` | Normal tissue complication probability |

---

## Coordinate Systems

### World (Patient) Coordinates
- Standard LPS or RAS depending on DICOM origin
- Default origin: geometric center of CT volume
- Units: mm

### BEV (Beam's-Eye-View) Coordinates
- Source at `[0, −SAD, 0]`
- Beam travels in `+y` direction toward isocenter
- `x` = lateral (left-right in beam view), `z` = vertical (up-down)
- Transform from world to BEV:
  ```
  R = R_couch(y-axis) @ R_gantry(z-axis)
  pos_bev = R @ pos_world
  ```

### Isocenter Projection (perspective)
```
iso_lat_x = x_bev × SAD / (SAD + y_bev)
iso_lat_z = z_bev × SAD / (SAD + y_bev)
```
Used to project patient voxels onto the beam's isocenter plane for STF generation.

---

## Object Relationships

```
PLN
 ├── defines beam angles → STF (one beam struct per angle)
 │     └── ray[] (one per bixel grid cell that hits target)
 │
CT + CST
 ├── CT: HU volume → density cube (via HLUT)
 ├── CST: structure voxel indices + dose objectives
 │
 ↓  calc_dose_influence(ct, cst, stf, pln)
 │
DIJ
 └── physicalDose: sparse (nVoxels × nBixels) matrix
       ↓  D @ w
     dose_flat → reshape → dose_3d [Gy/fx]
       ↑
       w (bixel weights) — found by optimizer minimizing
         Σ_structures penalty × obj(dose_in_structure, d_ref)
```
