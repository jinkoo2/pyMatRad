# matRad Dose Calculation Algorithms

Reference for the photon SVD pencil beam dose engine used in matRad and pyMatRad.

---

## Overview: Full Pipeline

```
CT + CST + PLN
      │
      ▼
1. STF Generation          — ray/bixel geometry per beam
      │
      ▼
2. CT Processing           — HU → water-equivalent density (RED)
      │
      ▼
3. Machine Data Loading    — SVD kernels, attenuation params
      │
      ▼
4. [for each beam]
   4a. Coordinate Transform — world → BEV
   4b. Ray Tracing          — radiological depth per voxel (Siddon)
   4c. Kernel Convolution   — lateral dose profile (2D FFT)
   4d. Dose Assembly        — bixel dose → sparse column of D
      │
      ▼
5. Dose Influence Matrix D  — sparse (nVoxels × nBixels)
      │
      ▼
6. Optimization             — find w* = argmin Σ objectives(D·w)
      │
      ▼
7. Forward Dose             — dose_3d = reshape(D @ w*, dims)
```

---

## Step 1 — STF Generation

**Source:** `matRad_generateStf.m` / `matRad/steering/stf_generator.py`

For each beam (gantry/couch angle pair):

1. **Rotate target voxel positions into BEV** using:
   ```
   R = R_couch(couch_angle, y-axis) @ R_gantry(gantry_angle, z-axis)
   pos_bev = R @ pos_world
   ```

2. **Project onto isocenter plane** (perspective):
   ```
   lat_x = pos_bev.x × SAD / (SAD + pos_bev.y)
   lat_z = pos_bev.z × SAD / (SAD + pos_bev.y)
   ```

3. **Quantize to bixel grid** (spacing = `bixelWidth`, e.g. 5 mm):
   ```
   ray_x = round(lat_x / bixelWidth) × bixelWidth
   ray_z = round(lat_z / bixelWidth) × bixelWidth
   ```

4. **Add margin** around target projection (`= bixelWidth` for photons)

5. **Discard empty rays** (no target voxels behind them)

**Result:** STF with ray positions, source points, and SSD per ray.

---

## Step 2 — CT Processing (HU → RED)

**Source:** `photon_svd_engine.py:_calc_water_eq_density`

Convert Hounsfield units to relative electron density (water = 1.0):

**Preferred (MATLAB):** piecewise-linear HLUT interpolation
```
RED = interp1(hlut(:,1), hlut(:,2), HU, 'linear', 'extrap')
```

**Fallback (Python, water phantom):** linear approximation
```
RED = max(0, 1 + HU / 1000)
  → HU = -1000 (air)   → RED = 0
  → HU =     0 (water) → RED = 1
  → HU =  1000 (bone)  → RED ≈ 2  (overestimates; HLUT gives ~1.8)
```

**Outside-density masking:** voxels outside all CST structures are set to RED=0, so ray tracing does not accumulate radiological depth in air outside the patient.

---

## Step 3 — Machine Data

**Source:** `Generic.mat` (or other machine file)

Key parameters extracted per SSD lookup table entry:

| Parameter | Symbol | Typical value | Description |
|---|---|---|---|
| `m` | m | ~0.03 mm⁻¹ | Primary photon attenuation coefficient |
| `betas` | β₁,β₂,β₃ | [0.04, 0.15, 0.60] mm⁻¹ | Scatter kernel decay rates (3 SVD components) |
| `kernelPos` | r | [0…200] mm | Radial positions for lateral kernel samples |
| `kernel1/2/3` | K₁,K₂,K₃ | arrays | Lateral kernel values at each radius, per SSD |
| `SAD` | SAD | 1000 mm | Source-to-axis distance |
| `penumbraFWHM` | σ | ~5 mm | Geometric penumbra (beam hardening + source size) |
| `primaryFluence` | Φ | spectrum | Primary photon fluence spectrum |

The three kernel components are the result of **SVD decomposition** of the full phase-space kernel:
```
K_full(r, z) ≈ Σᵢ₌₁³  βᵢ/(βᵢ−m) × (e^{−m·z} − e^{−βᵢ·z}) × Kᵢ(r)
```

---

## Step 4 — Per-Beam Dose Calculation

### 4a. Coordinate Transform (World → BEV)

```
pos_bev = R @ (pos_world − sourcePoint_world)
```

From BEV coordinates, compute:
- **Geometric distance** from source: `d_geo = ||pos_bev||`
- **Lateral distances** from beam axis at isocenter plane:
  ```
  iso_lat_x = pos_bev.x × SAD / d_geo_y
  iso_lat_z = pos_bev.z × SAD / d_geo_y
  ```

### 4b. Ray Tracing — Radiological Depth (Siddon Algorithm)

**Source:** `matRad/rayTracing/siddon.py:ray_tracing_fast`

The Siddon algorithm traces a line from source to each voxel through the CT density grid, computing the water-equivalent path length:

```
rad_depth = Σ_segments  RED(voxel) × path_length_in_voxel

where path_length_in_voxel = ||entry_point − exit_point||  [mm]
```

**Steps:**
1. Find all CT voxel boundary crossings along the ray
2. For each segment between crossings: multiply voxel RED × segment length
3. Accumulate to get cumulative radiological depth at each voxel

**Also computed:** `d_geo` (geometric distance, for inverse-square correction) and `SSD` (depth of first non-air voxel, for kernel SSD lookup).

**Lateral cutoff:** voxels farther than `geometricLateralCutOff` (default 50 mm) from the ray axis are skipped — their dose contribution is negligible.

### 4c. Kernel Convolution (2D FFT)

For each ray, the lateral dose profile is computed by convolving the bixel's fluence aperture with the lateral kernel.

**Fluence model:**
```
Φ(x, z) = rect(x/bixelWidth) ⊗ rect(z/bixelWidth) ⊗ G(σ_penumbra)
```
where `G(σ)` is a Gaussian (geometric penumbra, source size + beam hardening).

**Convolution** (done once per bixel via 2D FFT):
```
K̃ᵢ(x, z) = Φ(x, z) ⊛ Kᵢ(√(x²+z²))   for i = 1, 2, 3
```

The result `K̃ᵢ` is stored as a 2D interpolator sampled at each voxel's lateral position.

### 4d. Photon Dose Formula (Scholz 1994, PMB)

**Source:** `photon_svd_engine.py:_calc_single_bixel`
**Reference:** Scholz et al., Phys. Med. Biol. 39 (1994) 731–746, Eq. 17–19

The dose at a voxel with radiological depth `z_rad` and lateral offset `(x, z)` from the ray:

```
D(x, z, z_rad) = Σᵢ₌₁³  [βᵢ/(βᵢ−m)] × [e^{−m·z_rad} − e^{−βᵢ·z_rad}]
                          ×  K̃ᵢ(x, z)
                          ×  (SAD / d_geo)²
```

**Term-by-term interpretation:**

| Term | Role |
|---|---|
| `βᵢ/(βᵢ−m)` | Normalization for SVD component i |
| `e^{−m·z_rad} − e^{−βᵢ·z_rad}` | Depth-dose shape: builds up then falls off |
| `K̃ᵢ(x, z)` | Lateral spread (scatter kernel convolved with fluence) |
| `(SAD/d_geo)²` | Inverse-square law correction (geometric divergence) |

**Special case** (when βᵢ = m, degenerate):
```
contribution_i = m × z_rad × e^{−m·z_rad} × K̃ᵢ(x, z) × (SAD/d_geo)²
```

**Implementation (Python):**
```python
for i in range(3):
    beta = betas[i]
    depth_dose = beta / (beta - m) * (np.exp(-m * z_rad) - np.exp(-beta * z_rad))
    dose[:, i] = depth_dose * kernel_vals[:, i]

bixel_dose = dose.sum(axis=1) * (SAD / d_geo)**2
bixel_dose = np.maximum(bixel_dose, 0.0)
```

---

## Step 5 — Dose Influence Matrix D

After computing bixel doses for all bixels:

```
D[voxel_idx, bixel_idx] = dose at voxel from bixel at unit weight
```

D is stored as a **sparse matrix** (scipy.sparse.csc in Python, MATLAB sparse).
Typical sparsity: 0.1%–1% non-zero entries.

---

## Step 6 — Fluence Optimization

**Source:** `matRad/optimization/fluence_optimization.py`

Find bixel weights `w` minimizing the total objective:

```
min_{w ≥ 0}  f(w) = Σ_structures  Σ_objectives  penalty × obj(dose_in_structure)
```

where `dose = D @ w` (matrix-vector product).

**Gradient** (computed analytically):
```
∇f(w) = D^T @ (∂f/∂dose)
```

This allows efficient gradient computation:
1. Forward: `dose = D @ w`  — one sparse matrix-vector multiply
2. Objective gradient w.r.t. dose: `∂f/∂dose` — per-voxel scalar, cheap
3. Backward: `∇f = D^T @ (∂f/∂dose)` — one sparse matrix-vector multiply

**Solver:** L-BFGS-B (scipy) or IPOPT, with bound constraint `w ≥ 0`.

**Objective function examples:**

| Objective | f(d) | ∂f/∂d |
|---|---|---|
| SquaredDeviation | `p × (d − d_ref)²` | `2p × (d − d_ref)` |
| SquaredOverdosing | `p × max(0, d − d_ref)²` | `2p × max(0, d − d_ref)` |
| SquaredUnderdosing | `p × max(0, d_ref − d)²` | `−2p × max(0, d_ref − d)` |
| MeanDose | `p × (mean(d) − d_ref)²` | `2p × (mean(d) − d_ref) / N` |

Note: `d_ref` in the optimizer is per-fraction [Gy/fx]; divide total prescription by `numOfFractions`.

---

## Step 7 — Forward Dose

Given optimized weights `w*`:

```python
dose_flat = D @ w_opt          # shape: (nVoxels,)  [Gy/fx]
dose_3d   = dose_flat.reshape(doseGrid.dimensions, order='F')  # [Gy/fx]
dose_total = dose_3d * numOfFractions                           # [Gy]
```

---

## SVD Kernel Decomposition

The SVD (Singular Value Decomposition) of the photon pencil beam kernel separates the full 3D dose kernel into depth-dose and lateral components:

```
K_full(r, z) ≈ Σᵢ₌₁³  fᵢ(z) × gᵢ(r)
```

where:
- `fᵢ(z) = βᵢ/(βᵢ−m) × (e^{−m·z} − e^{−βᵢ·z})` — depth-dose shape
- `gᵢ(r) = Kᵢ(r)` — lateral spread kernel (sampled at discrete radii)

**Why SVD?** Computing the full 3D convolution directly is expensive (O(N³)). The separable decomposition allows:
1. Pre-compute 2D lateral convolution `Φ ⊛ Kᵢ(r)` once per bixel per SSD → O(N² log N) via FFT
2. Multiply by depth-dose `fᵢ(z_rad)` per voxel → O(N)

This reduces dose calc from hours to minutes for clinical plans.

---

## Coordinate System Summary

```
World (patient):            BEV (beam's-eye-view):
  x: left → right            x: lateral (same as gantry x at 0°)
  y: ant → post              y: depth (source→isocenter direction)
  z: inf → sup               z: vertical

Source position in BEV:  [0, −SAD, 0]
Isocenter in BEV:        [0,    0, 0]
Beam travels:            +y direction
```

**Rotation from world to BEV:**
```
R = R_couch(couch_angle, around y-axis) @ R_gantry(gantry_angle, around z-axis)
pos_bev = R @ pos_world
```

---

## Key Numerical Parameters (Generic 6 MV photon)

| Parameter | Value | Notes |
|---|---|---|
| SAD | 1000 mm | Source-to-axis distance |
| Primary attenuation `m` | ~0.030 mm⁻¹ | |
| Scatter decay `β₁` | ~0.040 mm⁻¹ | Broad scatter component |
| Scatter decay `β₂` | ~0.150 mm⁻¹ | Medium scatter |
| Scatter decay `β₃` | ~0.600 mm⁻¹ | Narrow scatter |
| Penumbra FWHM | ~5 mm | At isocenter |
| Kernel radial cutoff | 200 mm | Beyond this, dose ≈ 0 |
| Lateral geometric cutoff | 50 mm | Per-ray, voxels beyond skipped |
| Typical bixelWidth | 5 mm | Pencil beam spacing |

---

## Limitations of the SVD Pencil Beam Model

| Limitation | Impact | When it matters |
|---|---|---|
| Lateral kernel does not vary with depth | Overestimates scatter at deep depths | Thick patients, large fields |
| Radiological depth scaling only (no explicit scatter transport) | Inaccurate in sharp density interfaces | Lung-tissue boundaries, air cavities |
| Kernel pre-computed at discrete SSDs | Interpolation error at non-standard SSDs | Non-standard beam geometries |
| Ignores beam hardening variation | Small error in depth-dose tail | High-Z inserts, prosthetics |

For high-accuracy heterogeneous cases, Monte Carlo (e.g. matRad_example12) should be used instead.

---

## Files Reference

| File | Role |
|---|---|
| `matRad/matRad_generateStf.m` | STF generation (MATLAB) |
| `matRad/steering/stf_generator.py` | STF generation (Python) |
| `matRad/matRad_calcDoseInfluence.m` | Dose calc entry point (MATLAB) |
| `matRad/doseCalc/calc_dose_influence.py` | Dose calc entry point (Python) |
| `matRad/doseCalc/DoseEngines/photon_svd_engine.py` | SVD photon engine (Python) |
| `matRad/rayTracing/siddon.py` | Siddon ray tracer (Python) |
| `matRad/geometry/geometry.py` | Coordinate transforms, rotation matrices |
| `matRad/optimization/fluence_optimization.py` | L-BFGS-B optimizer (Python) |
| `matRad/basedata/Generic.mat` | 6 MV photon machine kernels |
