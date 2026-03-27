# Photon Dose Calculation Engines in pyMatRad

This document describes the three photon dose calculation engines implemented in pyMatRad.  All three engines share the same upstream infrastructure (CT geometry, STF beam steering, Siddon ray tracing, DIJ sparse matrix assembly) but differ fundamentally in their physical model, speed/accuracy trade-off, and computational backend.

---

## Table of Contents

1. [Common Infrastructure](#1-common-infrastructure)
2. [Engine 1 — SVD Pencil Beam (SVDPB)](#2-engine-1--svd-pencil-beam-svdpb)
3. [Engine 2 — ompMC Analytical](#3-engine-2--ompc-analytical)
4. [Engine 3 — TOPAS Monte Carlo](#4-engine-3--topas-monte-carlo)
5. [Comparison Summary](#5-comparison-summary)
6. [Selecting an Engine](#6-selecting-an-engine)

---

## 1. Common Infrastructure

All three engines extend `DoseEngineBase` (in `dose_engine_base.py`) and use the following shared components.

### 1.1 Coordinate System

| Symbol | Description |
|--------|-------------|
| `cubeDim = [Ny, Nx, Nz]` | CT array shape (MATLAB convention: row=y, col=x, slice=z) |
| World frame | right-handed, mm; origin at CT corner |
| BEV frame | Beam's-Eye-View; source at `[0, -SAD, 0]`, beam travels in +y |
| Rotation | `R = R_Couch(y-axis) @ R_Gantry(z-axis)` |
| Fortran order | CT stored column-major (z slowest); 1-based linear indices like MATLAB |

### 1.2 Ray Tracing — Siddon Algorithm

Radiological depth is computed for each voxel using the Siddon ray-tracing algorithm (`rayTracing/siddon.py`).  The algorithm:

1. Finds all CT voxel boundary crossings along the source→voxel ray.
2. Accumulates `ρ_water × Δℓ` (relative electron density × path length) to give radiological depth in mm of water equivalent.
3. Returns geometric distance from source, for inverse-square correction.

The `ray_tracing_fast` dispatcher (`rayTracing/dispatch.py`) runs Siddon for all voxels in the dose grid within the lateral cutoff, returning a flat `rad_depths` array aligned with the dose voxel index array `V_dose_grid`.

### 1.3 DIJ Sparse Matrix

The dose influence matrix `dij["physicalDose"][0]` has shape `(N_voxels, N_bixels)`.  Each column holds the dose (Gy per monitor unit) deposited in every voxel by a single beamlet.  The matrix is assembled in COO format from per-beam worker results, then converted to CSC for efficient matrix–vector products during optimization.

### 1.4 HU → Relative Electron Density

All engines convert Hounsfield Units to water-equivalent density using the same linear approximation:

```
ρ_rel = 0           for HU ≤ −1000  (air)
ρ_rel = 1 + HU/1000 otherwise        (water ≡ 1.0)
ρ_rel = clamp(ρ_rel, 0, 3)
```

TOPAS additionally maps HU to discrete material tags (air / lung / water / bone) for Geant4 geometry.

### 1.5 Parallelization Pattern

The SVPB and ompMC engines use `ProcessPoolExecutor` for beam-level parallelism:

- **Main process** (sequential): SSD computation, FFT convolution, geometry rotation, Siddon ray tracing (these touch the large CT array and cannot be pickled efficiently).
- **Worker processes** (parallel, one per beam): dose math using pre-computed geometry arrays. Return COO sparse data.
- **Main process** (sequential): assembles COO arrays → CSC matrix.

Number of workers is controlled by the `PYMATRAD_WORKERS` environment variable (default: `os.cpu_count()`).

---

## 2. Engine 1 — SVD Pencil Beam (SVDPB)

**Source:** `DoseEngines/photon_svd_engine.py`
**Class:** `PhotonPencilBeamSVDEngine`
**Short name:** `SVDPB`
**Reference:** Scholz et al., 1994, *Phys. Med. Biol.* (PMID: 8497215)

### 2.1 Physical Model

The SVDPB engine decomposes the photon dose kernel into a sum of analytical components using Singular Value Decomposition (SVD).  A Gaussian penumbra filter accounts for the finite source size.

#### Depth–dose (Scholz 1994, Eq. 17)

The primary photon beam is attenuated with coefficient `m`.  Each scatter kernel component `i` has its own exponential decay constant `β_i`:

```
D_i(z) = β_i / (β_i − m) × [exp(−m·z) − exp(−β_i·z)]
```

Special case when `β_i ≈ m`:

```
D_i(z) = m · z · exp(−m · z)
```

Parameters `m` and `β_i` are stored in the machine data file (`photons_Generic.mat`), calibrated to match measured depth–dose curves for a specific linac energy (typically 6 MV).

#### Lateral kernel convolution (Eq. 19)

For each scatter component, the lateral kernel `K_i(r)` is a radially-symmetric function tabulated at positions `kernelPos` in the machine file.  It is evaluated on a 2D grid by interpolation, then convolved with the primary photon fluence `F(x, z)` using 2D FFT:

```
Ĥ_i(x, z) = IFFT2[ FFT2(F) × FFT2(K_i) ]
```

The resulting 2D dose distribution for each scatter component is sampled at voxel positions using a `RegularGridInterpolator`.

#### Fluence model

For uniform bixel weight (non-field-based):
- A square aperture of size `bixelWidth × bixelWidth` defines the primary fluence `F`.
- A Gaussian filter with `σ = penumbraFWHM / (2√(2 ln 2))` is convolved with `F` to account for finite source size (penumbra).
- The kernel convolution is pre-computed once per beam and cached (`_interp_kernel_cache`).

#### Total bixel dose (Eq. 17–19 combined)

```
D(x, y, z) = Σ_i [ D_i(z_rad) × Ĥ_i(x_iso, z_iso) ] × (SAD / r_geo)²
```

where:
- `z_rad` = radiological depth along the beam axis
- `x_iso, z_iso` = lateral distances at the isocenter plane (projected)
- `r_geo` = geometric distance from source (for inverse-square law)

### 2.2 Convolution Grids

Three nested grids are set up in `_setup_convolution_grids`:

| Grid | Size | Purpose |
|------|------|---------|
| Field grid (`F_x`, `F_z`) | `bixelWidth / convRes` | Uniform fluence aperture |
| Gaussian filter | `5σ / convRes` each side | Penumbra smoothing |
| Kernel grid (`kernel_x`, `kernel_z`) | `kernelCutoff / convRes` | Lateral kernel support |
| Convolution output | sum of all above | Result of F * G * K |

Default `intConvResolution = 0.5 mm`.  The convolution output grid spans the sum of all three half-extents.

### 2.3 DIJ Sampling

To keep the DIJ matrix sparse, the engine applies probabilistic importance sampling (`_sample_dij`) — a port of the MATLAB method:

- **Core region** (`r < latCutOff + bixelWidth/√2`): all voxels kept as-is.
- **Tail region**: voxels clustered by radiological depth in bins of `deltaRadDepth` mm.  Within each cluster, each voxel is sampled with probability `dose / max_dose_in_cluster`.  Accepted voxels are assigned the cluster maximum dose (unbiased estimator).

This reduces DIJ storage by ~10–50× while preserving dose statistics for optimization.

### 2.4 Workflow Summary

```
_init_dose_calc()
  └─ load machine file (SVD kernels, m, betas, penumbraFWHM)
  └─ set up convolution grids
  └─ pre-compute Gaussian-convolved fluence (Fpre)

_calc_dose()
  ├─ convert HU → RED (water-equivalent density)
  ├─ compute SSD via Siddon ray trace (all beams, sequential)
  ├─ for each beam (sequential setup):
  │    ├─ rotate voxel coordinates to BEV frame
  │    ├─ Siddon ray trace → rad_depths, geo_dists
  │    ├─ project voxels to iso plane → iso_lat_x, iso_lat_z
  │    └─ pre-compute kernel interpolators (FFT convolution)
  ├─ dispatch bundles to ProcessPoolExecutor (parallel, one per beam)
  │    └─ _calc_beam_worker():
  │         ├─ for each bixel: apply cutoff, sample valid voxels
  │         ├─ vectorised: depth-dose × lateral kernel × ISL correction
  │         ├─ DIJ sampling (importance sampling of tail voxels)
  │         └─ return COO sparse data
  └─ assemble COO → CSC → dij["physicalDose"][0]
```

### 2.5 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernelCutoff` | 20 mm | Lateral kernel support radius |
| `intConvResolution` | 0.5 mm | Convolution grid resolution |
| `enableDijSampling` | True | Enable tail voxel importance sampling |
| `dij_sampling.relDoseThreshold` | 0.01 | Core dose threshold (fraction of max) |
| `dij_sampling.latCutOff` | 20 mm | Core radius for sampling |
| `dij_sampling.deltaRadDepth` | 5 mm | Cluster size for tail sampling |

---

## 3. Engine 2 — ompMC Analytical

**Source:** `DoseEngines/photon_ompc_engine.py`
**Class:** `PhotonOmpMCEngine`
**Short name:** `ompMC`
**Reference:** Analytic port of `matRad_PhotonOmpMCEngine.m`; physics from ompMC (open Monte Carlo library)

### 3.1 Physical Model

The ompMC engine implements a TERMA-based (Total Energy Released per unit MAss) analytical approximation.  It uses NIST XCOM tabulated attenuation coefficients for a 6 MV (effective ~2 MeV) photon beam, without the SVD kernel decomposition.

#### Physical constants (NIST XCOM at 2.0 MeV, water)

| Quantity | Value | Units |
|----------|-------|-------|
| `μ_total / ρ` | 0.0497 | cm²/g |
| `μ_en / ρ` | 0.0270 | cm²/g |
| `μ_total` | 0.00497 | mm⁻¹ (at unit density) |
| `μ_en` | 0.00270 | mm⁻¹ (at unit density) |

The attenuation at density `ρ(x)` along the ray is accumulated as the radiological depth `z_rad = ∫ ρ(s) ds`.

#### Step 1 — Primary fluence

Exponential attenuation with inverse-square law:

```
Φ_primary(z_rad, r_geo) = (SAD / r_geo)² × exp(−μ_total × z_rad)
```

#### Step 2 — Primary dose (TERMA)

Energy deposited per unit mass by primary photons:

```
D_primary = Φ_primary × μ_en
```

#### Step 3 — Lateral profile

The bixel aperture (width `w`) convolved with a Gaussian penumbra (σ from machine FWHM) gives an analytic erf-product profile:

```
P(lx, lz) = [erf((w/2 + lx) / (√2 σ)) − erf((−w/2 + lx) / (√2 σ))]
           × [erf((w/2 + lz) / (√2 σ)) − erf((−w/2 + lz) / (√2 σ))]
           / centre_value
```

where `centre_value = [erf(w/(2√2 σ))]² × 4` normalizes the profile to 1 on the beam axis.  For zero penumbra, a hard rectangular aperture is used instead.

`lx, lz` are the lateral distances of the voxel (projected to the isocenter plane) from the bixel center.

#### Step 4 — Scatter correction

Compton scatter builds up with depth.  A first-order exponential model adds a depth-dependent fraction to the primary dose:

```
C(z_rad) = f_scatter × [1 − exp(−z_rad / d_buildup)]
```

Default values: `f_scatter = 0.28` (28% scatter fraction at large depth), `d_buildup = 80 mm`.

#### Step 5 — Total dose

```
D = D_primary × P(lx, lz) × [1 + C(z_rad)] × k_calib
```

The calibration factor `k_calib = 23220 × (bixelWidth / 50 mm)²` scales dose so that the ompMC result matches SVPB at the reference point (5 cm depth, 5×5 cm² open field, SSD = 900 mm).

### 3.2 Workflow Summary

```
_init_dose_calc()
  └─ load machine (penumbraFWHM, bixelWidth)
  └─ compute penumbra sigma: σ = FWHM / (2√(2 ln 2))
  └─ set effective lateral cutoff

_calc_dose()
  ├─ convert HU → RED
  ├─ compute SSD via Siddon (sequential)
  ├─ for each beam (sequential setup):
  │    ├─ rotate voxel coords to BEV frame
  │    ├─ Siddon ray trace → rad_depths, geo_dists
  │    └─ project → iso_lat_x, iso_lat_z
  ├─ dispatch to ProcessPoolExecutor (parallel, one per beam)
  │    └─ _ompc_beam_worker():
  │         ├─ for each bixel: lateral cutoff mask
  │         ├─ primary fluence: (SAD/r)² × exp(−μ_total × z_rad)
  │         ├─ primary dose: fluence × μ_en
  │         ├─ lateral profile: erf product (or hard rect if σ≈0)
  │         ├─ scatter correction: f × (1 − exp(−z/d_buildup))
  │         ├─ total dose × calibration
  │         └─ return COO sparse data
  └─ assemble COO → CSC → dij["physicalDose"][0]
```

### 3.3 Differences from SVDPB

| Aspect | SVDPB | ompMC |
|--------|-------|-------|
| Depth–dose | SVD multi-exponential (Scholz Eq. 17) | Simple exponential (μ_total, NIST) |
| Lateral model | Convolved kernel (machine-calibrated) | Analytic erf × erf (penumbra σ only) |
| Scatter | Implicit in SVD kernels | Explicit depth-dependent correction |
| Machine data | Requires full SVD kernel data in .mat | Only needs `penumbraFWHMatIso` |
| DIJ sampling | Probabilistic tail sampling | No tail sampling (all voxels kept) |
| Speed | Moderate (FFT convolution per beam) | Fast (no FFT; fully vectorized) |
| Accuracy | High (machine-calibrated kernels) | Approximate (fixed μ coefficients) |

---

## 4. Engine 3 — TOPAS Monte Carlo

**Source:** `DoseEngines/topas_mc_engine.py`
**Class:** `TopasMCEngine`
**Short name:** `TOPAS`
**Backend:** OpenTOPAS 4.x (Geant4-based) via local binary or REST API

### 4.1 Physical Model

Unlike the two analytical engines, TOPAS runs a full Geant4 Monte Carlo simulation.  Every photon history is tracked through the CT geometry using the Standard Electromagnetic physics list, including:

- Photoelectric effect
- Compton scattering
- Pair production
- Rayleigh scattering
- Secondary electron transport (delta rays, bremsstrahlung)

This makes TOPAS the most physically accurate of the three engines, at the cost of much longer computation time and statistical noise (which decreases as 1/√N with number of histories).

### 4.2 CT Geometry

The HU cube is written to a binary file (`matRad_cube.dat`) as a flat `int16` array in Fortran (column-major, z-fastest) order.  TOPAS reads it via the `TsImageCube` geometry component with the `ByTagNumber` converter:

| HU range | Tag | Material |
|----------|-----|----------|
| HU < −950 | 0 | Air (`G4_AIR`) |
| −950 ≤ HU < −700 | 1 | Lung (80% air / 20% water mix) |
| −700 ≤ HU < 101 | 2 | Water / soft tissue (`G4_WATER`) |
| HU ≥ 101 | 3 | Bone (`G4_BONE_COMPACT_ICRU`) |

The CT voxel size and origin are taken directly from `ct["resolution"]` and `ct["x/y/z"]` arrays.

### 4.3 Beam Source (OpenTOPAS 4.x)

OpenTOPAS 4.x changed the beam source API significantly from TOPAS 3.x.  The engine generates the following key parameters in the `.txt` parameter file:

#### Nozzle orientation

In 4.x, a `TsVRTSource` fires photons along the component's **−Z axis**.  The nozzle must therefore be rotated so that −Z points from the source toward the isocenter.  Given the source position vector `src_world` (in world coordinates, relative to iso):

```python
d = -src_world / |src_world|          # unit vector: source → isocenter
rot_y = arctan2(−dx, √(dy² + dz²))    # rotation around Y
rot_x = arctan2(dy, −dz)              # rotation around X
```

These Euler angles are written as `d:Ge/Nozzle/RotX` and `d:Ge/Nozzle/RotY`.

#### Beam distribution (required in 4.x)

```
s:So/Photon/BeamPositionDistribution = "None"     # point source
s:So/Photon/BeamAngularDistribution  = "Flat"     # uniform divergence
s:So/Photon/BeamAngularCutoffShape   = "Rectangle"
d:So/Photon/BeamAngularCutoffX      = <half_angle> deg
d:So/Photon/BeamAngularCutoffY      = <half_angle> deg
```

The half-angle is computed from the total field size at the isocenter:

```python
field_half = n_bixels × 5 mm / 2
half_angle = arctan(field_half / SAD)
```

> **Note:** In TOPAS 3.x, `s:So/Photon/BeamShape = "Rectangle"` was used with `BeamFlatteningHalfWidth/Height`.  This syntax is removed in 4.x and will cause an error.

#### Dose scorer

```
s:Sc/DoseAtPhantom/Quantity                  = "DoseToMedium"
s:Sc/DoseAtPhantom/Component                 = "Patient"
s:Sc/DoseAtPhantom/OutputType                = "Binary"
s:Sc/DoseAtPhantom/IfOutputFileAlreadyExists = "Overwrite"
b:Sc/DoseAtPhantom/OutputToConsole           = "False"
```

The scorer writes `<prefix>.bin` and `<prefix>.binheader`.

### 4.4 Execution Modes

The engine supports three execution modes, selected via `pln["propDoseCalc"]["externalCalculation"]`:

| Mode | Value | Description |
|------|-------|-------------|
| Local | `"off"` (default) | Write files and run TOPAS binary locally |
| Write-only | `"write"` | Write input files only (for HPC cluster submission) |
| Read-only | `"<folder>"` | Read previously computed results from folder |

In addition, the engine can use the **OpenTOPAS REST API** instead of a local binary.

### 4.5 REST API Mode

When `pln["propDoseCalc"]["topasApiUrl"]` is set, the engine delegates each beam simulation to the OpenTOPAS API server:

```python
pln["propDoseCalc"]["topasApiUrl"]   = "http://localhost:7778"
pln["propDoseCalc"]["topasApiToken"] = "topas-dev-..."
```

The API workflow (`_run_beam_via_api`):

1. **Upload**: multipart POST to `/jobs` with the `.txt` parameter file as `param_file` and the CT binary as one of the `input_files`.
2. **Poll**: GET `/jobs/{job_id}` every 5 seconds until status is `done` or `failed`.
3. **Download**: GET `/jobs/{job_id}/results` → receives a zip archive.
4. **Extract**: unzip `.bin`, `.binheader`, and `.log` files into the local working directory.
5. **Parse**: read dose binary with `_read_topas_dose_binary`.

The API server (FastAPI + uvicorn) runs inside the Docker container, allowing TOPAS to be offloaded to a dedicated server or HPC node while pyMatRad runs on a client workstation.

### 4.6 Binary Dose Reader

TOPAS writes dose results as:
- `<prefix>.binheader` — ASCII file with grid dimensions and voxel size
- `<prefix>.bin` — raw `float32` array

**OpenTOPAS 4.x header format** (changed from 3.x):

```
# X in 200 bins of 0.2 cm
# Y in 200 bins of 0.2 cm
# Z in 100 bins of 0.3 cm
```

> **Note:** TOPAS 3.x used `Bins In X: 200` syntax.  The current `_read_topas_dose_binary` parser handles the 3.x format.  OpenTOPAS 4.x files require parsing `# X in N bins` lines instead.

**Binary layout:** TOPAS 4.x writes two `float32` values per voxel — Sum and SumSquared — so the binary file is 2× the expected size.  Only the first half (Sum) contains the dose.

**Axis order:** TOPAS writes in C order with X fastest → reshape as `(Nz, Ny, Nx)`, then transpose to matRad order `(Ny, Nx, Nz)`.

### 4.7 DIJ Assembly

By default (`calcDij = False`), one TOPAS simulation is run per beam with all bixels at uniform weight 1.  The resulting dose cube is distributed uniformly across all bixels in that beam (approximation).  Each column of the DIJ corresponding to a bixel of beam `b` receives `dose_cube / n_bixels_in_beam`.

For accurate per-bixel DIJ (`calcDij = True`), a separate TOPAS simulation is run per beamlet — much slower but physically correct.

### 4.8 Workflow Summary

```
_init_dose_calc()
  └─ create/set working directory
  └─ load machine (SAD, SCD)

_calc_dose()
  ├─ check TOPAS availability (local binary or API)
  ├─ write CT binary: matRad_cube.dat
  ├─ for each beam:
  │    ├─ write TOPAS .txt parameter file (_write_beam_file)
  │    │    ├─ World + Patient (TsImageCube, ByTagNumber)
  │    │    ├─ Nozzle (RotX/RotY from source direction vector)
  │    │    ├─ Source (BeamPositionDistribution/BeamAngularDistribution)
  │    │    └─ Scorer (DoseToMedium, Binary output)
  │    └─ run simulation:
  │         ├─ Local: subprocess.run(topas, param.txt)
  │         └─ API: upload → poll → download → extract
  └─ assemble dij from dose cubes
```

### 4.9 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `topasExec` | `"topas"` | Path to TOPAS binary (or `$TOPAS_EXEC`) |
| `numHistories` | 100,000 | Particle histories per beam (↑ = ↓ noise) |
| `numThreads` | 0 (auto) | CPU threads for TOPAS |
| `calcDij` | False | Per-bixel DIJ (slow but accurate) |
| `externalCalculation` | `"off"` | Execution mode |
| `topasApiUrl` | `""` | OpenTOPAS API server URL |
| `topasApiToken` | `""` | Bearer token for API auth |
| `workingDir` | temp dir | Directory for TOPAS input/output files |

---

## 5. Comparison Summary

| Criterion | SVDPB | ompMC | TOPAS MC |
|-----------|-------|-------|----------|
| **Physics model** | SVD multi-exponential kernel (Scholz 1994) | TERMA + exponential (NIST μ) + scatter corr. | Full Geant4 MC (photoelectric, Compton, pair production, e⁻ transport) |
| **Accuracy** | High (machine-calibrated) | Moderate (approximate scatter) | Highest (reference standard) |
| **Speed** | Fast (seconds–minutes) | Very fast (seconds) | Slow (minutes–hours) |
| **Statistical noise** | None (deterministic) | None (deterministic) | Yes (1/√N; ~2% at 100k histories) |
| **Machine data** | Requires `.mat` with SVD kernels | Only `penumbraFWHM` from `.mat` | No machine data needed |
| **Heterogeneity** | Via radiological depth (1D) | Via radiological depth (1D) | Full 3D simulation |
| **Lateral scatter** | Machine-measured kernel | erf penumbra + fixed scatter fraction | Physical (Compton geometry) |
| **Beam penumbra** | Gaussian filter (penumbraFWHM) | erf convolution (penumbraFWHM) | Geometric (finite source not modeled) |
| **Parallelism** | ProcessPoolExecutor (beam-level) | ProcessPoolExecutor (beam-level) | TOPAS built-in threading |
| **External compute** | No | No | Yes (REST API / cluster) |
| **DIJ sparsity** | Importance sampling | All computed voxels | Beam-averaged (uniform dist.) |
| **Typical use** | Clinical planning, optimization | Quick checks / validation | Reference / QA |
| **Config key** | `"engine": "SVDPB"` | `"engine": "ompMC"` | `"engine": "TOPAS"` |

---

## 6. Selecting an Engine

```python
pln["propDoseCalc"] = {
    # ── Option A: SVD Pencil Beam (recommended for planning) ──
    "engine": "SVDPB",

    # ── Option B: ompMC analytical (fast checks) ──
    "engine": "ompMC",

    # ── Option C: TOPAS MC — local binary ──
    "engine": "TOPAS",
    "topasExec": "/path/to/topas",
    "numHistories": 500_000,

    # ── Option C: TOPAS MC — remote API ──
    "engine": "TOPAS",
    "topasApiUrl": "http://localhost:7778",
    "topasApiToken": "topas-dev-a3f8c2d1-4b7e-4f9a-8c3d-2e1f5a6b7c8d",
    "numHistories": 100_000,

    # Shared grid option (all engines)
    "doseGrid": {"resolution": {"x": 3, "y": 3, "z": 3}},
}
```

### Guidance

- **Optimization / plan exploration**: use **SVDPB**.  It is fast, uses machine-measured SVD kernels, and produces the most clinically accurate analytical result.
- **Quick sanity checks**: use **ompMC**.  No machine kernel data required; runs in seconds.
- **Reference dosimetry / commissioning / research**: use **TOPAS**.  Physically exact but requires significant computation time.  Use the REST API mode to offload computation to a dedicated server.
- **HPC cluster runs**: use `externalCalculation = "write"` to generate TOPAS input files, submit them to the cluster, then re-run with `externalCalculation = "<result_folder>"` to load the output.
