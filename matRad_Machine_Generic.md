# Generic 6 MV Photon Machine — Data Reference

**File:** `matRad/basedata/photons_Generic.mat`
**Loaded by:** `matRad/basedata/load_machine.py` (`load_machine('photons', 'Generic')`)

---

## Metadata

| Field | Value |
|---|---|
| `machine` | Generic |
| `radiationMode` | photons |
| `energy` | 6 MV |
| `SAD` | 1000 mm (source-to-axis distance) |
| `SCD` | 500 mm (source-to-collimator distance) |
| `penumbraFWHMatIso` | 5 mm (geometric penumbra FWHM at isocenter) |
| `description` | Photon pencil beam kernels for a 6 MV machine |
| `created_on` | 27-Oct-2015 |
| `created_by` | wieserh (DKFZ) |

---

## Physical Parameters

### Primary Photon Attenuation: `m`

```
m = 0.005066 mm⁻¹
```

Used in the depth-dose formula. Physically corresponds to the primary photon beam attenuation in water:

```
HVL = ln(2) / m = 136.8 mm water   (half-value layer)
TVL = ln(10) / m = 454.8 mm water  (tenth-value layer)
```

This is the effective linear attenuation coefficient for the 6 MV beam after beam hardening.

---

### SVD Scatter Kernel Decay Rates: `betas`

Three exponential decay parameters representing the three SVD components of the scatter kernel:

| Component | β [mm⁻¹] | 1/β [mm] | Physical interpretation |
|---|---|---|---|
| β₁ | 0.3252 | 3.1 mm | Narrow scatter — electrons near primary ray |
| β₂ | 0.0160 | 62.5 mm | Medium scatter — intermediate range photons |
| β₃ | 0.0051 | 196.1 mm | Broad scatter — long-range low-energy photons |

Each `βᵢ` controls how fast scatter component `i` builds up and falls off with depth.

---

### Depth-Dose Shape (per SVD component)

The depth-dose contribution of component `i` at water-equivalent depth `z` [mm]:

```
fᵢ(z) = βᵢ / (βᵢ − m) × (e^{−m·z} − e^{−βᵢ·z})
```

Evaluated at key depths:

| Depth z [mm] | f₁ (β₁=0.3252) | f₂ (β₂=0.0160) | f₃ (β₃=0.0051) | Physical region |
|---|---|---|---|---|
| 0 | 0.000 | 0.000 | 0.000 | Surface |
| 10 | 0.926 | 0.144 | 0.049 | Build-up |
| 30 | 0.873 | 0.352 | 0.131 | Near build-up max |
| 50 | 0.789 | 0.478 | 0.198 | Dmax region |
| 100 | 0.612 | 0.586 | 0.307 | Falling dose |
| 150 | 0.475 | 0.552 | 0.357 | Deep dose |
| 200 | 0.369 | 0.472 | 0.369 | Very deep |

**Key observations:**
- Component 1 (β₁ large) peaks early (~10 mm) and falls off rapidly — captures the primary beam and narrow-angle scatter
- Component 2 (β₂ medium) rises slowly and peaks around 100–150 mm — broad-angle Compton scatter
- Component 3 (β₃ small, ~m) rises very slowly — low-energy scattered photons, long range

---

## Lateral Scatter Kernels: `kernel`

The kernel array stores one entry per SSD (source-to-surface distance), sampled in 1 mm steps.

| Property | Value |
|---|---|
| SSD range | 500–1000 mm (501 entries, 1 mm step) |
| Radial positions | `kernelPos`: 0–179.5 mm (360 points, 0.5 mm step) |
| Components per SSD | 3 lateral kernels: `kernel1`, `kernel2`, `kernel3` |

Each `kernelN` is a 1D array of 360 values giving the lateral dose profile of SVD component N at radii 0–179.5 mm.

**Kernel values at SSD=750 mm:**

| Radius r | kernel1 | kernel2 | kernel3 |
|---|---|---|---|
| 0 mm | 0.8035 | −0.0717 | 0.0643 |
| 10 mm | ~0.0000 | ~0.0000 | ~0.0000 |
| 50 mm | ~0.0000 | ~0.0000 | ~0.0000 |

The kernels are highly concentrated at r=0 — the pencil beam deposits almost all its dose within ~5 mm of the beam axis. Scatter tails are captured by the broader β₂ and β₃ components.

**Physical meaning of each component's lateral kernel:**
- `kernel1` (narrow, β₁=0.3252): primary beam + electron scatter, sharp peak at r=0
- `kernel2` (medium, β₂=0.0160): Compton-scattered photons, wider spread but low amplitude
- `kernel3` (broad, β₃=0.0051): very low-energy scattered photons, long-range low-dose tail

---

## Primary Fluence Spectrum: `primaryFluence`

The energy fluence spectrum of the unattenuated primary beam at the isocenter plane.
Shape: `(38, 2)` — columns are `[energy (MeV), relative_fluence]`.

| Energy [MeV] | Rel. fluence | Notes |
|---|---|---|
| 0 | 1.000 | Reference |
| 14.1 | 1.000 | |
| 28.3 | 1.019 | |
| 56.6 | 1.045 | |
| … | … | |
| 325.3 | (max) | High-energy tail |

Energy range: 0–325 MeV, fluence range: 0.018–1.082.

This spectrum is used to weight the depth-dose convolution. A 6 MV linac produces a bremsstrahlung spectrum with most energy below 6 MeV effective, but the spectrum extends higher.

---

## How These Parameters Are Used in Dose Calculation

### Full dose formula (Scholz 1994):

```
D(r, z_rad) = Σᵢ₌₁³  fᵢ(z_rad) × K̃ᵢ(r) × (SAD / d_geo)²
```

where:

| Symbol | Source in machine file | Role |
|---|---|---|
| `fᵢ(z_rad)` | `m`, `betas[i]` | Depth-dose shape for component i |
| `K̃ᵢ(r)` | `kernel[SSD_idx].kernelN` | Lateral spread for component i, convolved with fluence |
| `SAD / d_geo` | `meta.SAD = 1000 mm` | Inverse-square law correction |
| SSD lookup | `kernel[SSD_idx].SSD` | Select correct lateral kernel for beam geometry |
| Penumbra | `penumbraFWHMatIso = 5 mm` | Gaussian blur applied to fluence before convolution |

### Workflow:

```
1. Compute SSD for each ray (ray tracing surface detection)
2. Look up kernel[SSD] by nearest-mm SSD → kernel1, kernel2, kernel3 at 360 radii
3. For each bixel: convolve fluence aperture with each kernel (2D FFT)
   → K̃₁(x,z), K̃₂(x,z), K̃₃(x,z) as 2D interpolators
4. For each voxel: evaluate f₁(z_rad), f₂(z_rad), f₃(z_rad) using m and betas
5. Dose = Σᵢ fᵢ × K̃ᵢ × (SAD/d_geo)²
```

---

## Machine File Structure (Python dict / MATLAB struct)

```
machine
├── meta
│   ├── machine          "Generic"
│   ├── radiationMode    "photons"
│   ├── SAD              1000.0    [mm]
│   ├── SCD              500.0     [mm]
│   ├── description      "photon pencil beam kernels for a 6MV machine"
│   ├── created_on       "27-Oct-2015"
│   └── created_by       "wieserh"
│
└── data
    ├── energy           6         [MV]
    ├── m                0.005066  [mm⁻¹]  primary attenuation
    ├── betas            [0.3252, 0.0160, 0.0051]  [mm⁻¹]  SVD decay rates
    ├── penumbraFWHMatIso 5        [mm]  geometric penumbra at isocenter
    ├── kernelPos        [0, 0.5, 1.0, ..., 179.5]  (360,) [mm]  radial positions
    ├── primaryFluence   (38, 2)  [MeV, rel]  beam energy spectrum
    └── kernel           (501,)   one struct per SSD (500–1000 mm, 1 mm step)
          └── kernel[i]
                ├── SSD      [mm]  source-to-surface distance
                ├── kernel1  (360,)  lateral kernel, SVD component 1
                ├── kernel2  (360,)  lateral kernel, SVD component 2
                └── kernel3  (360,)  lateral kernel, SVD component 3
```

---

## Limitations and Assumptions

| Assumption | Impact |
|---|---|
| Single energy (6 MV nominal) | No energy-dependent kernel variation within the beam |
| Kernels pre-computed at discrete SSDs (1 mm step) | Nearest-SSD lookup; small interpolation error at non-integer SSDs |
| Lateral kernel sampled to 179.5 mm radius | Voxels beyond ~180 mm from ray axis receive zero dose from this model |
| Penumbra modeled as fixed Gaussian (5 mm FWHM) | Does not account for MLC transmission, field-size-dependent hardening |
| No output factor correction | Absolute dose calibration handled externally (monitor units) |
| Generic (not patient-specific) | No actual commissioning measurements; suitable for research, not clinical use |
