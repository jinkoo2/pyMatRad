# Machine Parameter Tuning Guide

This document describes the tunable parameters in the TrueBeam machine files
(`userdata/machines/photons_TrueBeam_*.npy`) and how to adjust them to improve
agreement between pyMatRad and measured GBD data.

The machine files are built by `matRad/machineBuilder/kernel_calc.py` from the
raw GBD CSV files. Edit the builder and re-run `matRad/machineBuilder/build_truebeam.py`
to regenerate.

---

## Current Accuracy Summary

| Metric | 3×3 | 10×10 | 20×20 |
|--------|-----|-------|-------|
| PDD vs GBD | < 1% | < 1% | **3–4% undershoot** |
| Profile FWHM vs GBD (5–30 cm) | good | good | good |
| Profile FWHM vs GBD (shallow) | acceptable (6X, 6XFFF) | — | **under-penumbra (10XFFF, 15X)** |

pyMatRad and MATLAB matRad agree within 0.5% on all cases, so all remaining errors
are model/kernel errors, not code errors.

---

## Parameter Reference

### `m` — primary photon attenuation coefficient (mm⁻¹)

```
6X:     m = 0.004693
6XFFF:  m ≈ 0.00452
10XFFF: m ≈ 0.00387
15X:    m ≈ 0.00396
```

Controls the overall exponential fall-off of the primary photon fluence with depth:

```
primary_dose ∝ exp(-m · rad_depth)
```

- Increasing `m` steepens the PDD at all depths and field sizes equally.
- **Do not change `m` to fix a field-size-dependent error** — it affects all fields
  by the same factor.
- Fit `m` to the slope of the 10×10 PDD at depths > 5 cm.

---

### `betas` — scatter kernel decay constants (mm⁻¹)

Three values `[β₁, β₂, β₃]` control the depth distribution of scatter:

| Term | Typical value (6X) | Peak depth | Role |
|------|--------------------|-----------|------|
| β₁ | 0.254 mm⁻¹ | ~16 mm | Narrow primary component; sets dmax |
| β₂ | 0.0146 mm⁻¹ | ~115 mm | Medium-range scatter; drives PDD 5–20 cm |
| β₃ | 0.00469 mm⁻¹ | ~213 mm | Long-range scatter; drives PDD 20–30 cm |

The depth-dose for each component is (Scholz 1994, Eq. 17):

```
D_i(rd) = β_i / (β_i − m) · [exp(−m·rd) − exp(−β_i·rd)]
```

Peak location: `rd_max = ln(β_i / m) / (β_i − m)`

**To fix the 20×20 PDD undershoot (3–4% at 10–30 cm):**

The error occurs because the scatter at large field sizes is underestimated.
Re-fitting `β₂` (and its associated lateral kernel) with a multi-field-size
optimisation — using 3×3, 10×10, and 20×20 PDDs simultaneously — will yield
a value that balances accuracy across field sizes. Lowering `β₂` slightly
(flatter scatter depth-dose) with a wider lateral kernel tail achieves this.

---

### `kernel1/2/3` — lateral scatter kernels

Stored as `data["kernel"][ssd_idx]["kernel1/2/3"]`, tabulated at radii
`r = 0 … 179.5 mm` in 0.5 mm steps, for SSDs from 500 mm to 1000 mm.

The total dose at a voxel is:

```
D(r, rd) = Σ_i  D_depth_i(rd) · kernel_i(r)
```

where `r` is the lateral distance from the bixel to the voxel in the
isocenter plane.

#### Effect on field-size dependence

The PDD for an open field equals the sum of all bixel contributions.
For a large field, the CAX voxel receives scatter from bixels far off-axis.
The amplitude of `kernel2` and `kernel3` at large `r` (> 50 mm) directly
controls how much scatter arrives from the outer annulus of the field:

- **Wider kernel2/kernel3 tails** → more scatter at large fields → raises
  20×20 PDD at mid-to-deep depths → reduces the 3–4% undershoot.
- Has negligible effect on 3×3 and 10×10 (outer annulus is empty for those).

#### Effect on profiles

- `kernel1` (on-axis, narrow) dominates the dose near the beam axis and
  drives the shape of the dose peak.
- Normalisation: kernels are normalised so that the 10×10 open-field PDD
  matches GBD. Changing kernel tails requires re-normalisation.

---

### `penumbraFWHMatIso` — primary fluence Gaussian width (mm)

```
Current: 6.0 mm  (all four energies)
```

A Gaussian of this FWHM is convolved with the bixel fluence pattern before
the lateral kernel is applied. This sets the dose falloff at field edges.

**To calibrate:** extract the 20–80% penumbra width from the GBD profile CSV
at a shallow depth (e.g. 1.5 cm) and match:

```
penumbra_20_80 ≈ penumbraFWHMatIso * 0.85   (empirical rule)
```

Reducing `penumbraFWHMatIso` sharpens the penumbra; increasing it broadens it.
Typical values: 4–8 mm depending on energy and collimator setting.

---

### `primaryFluence` — off-axis fluence profile

```
shape: (N, 2)   columns: [off-axis radius mm, relative fluence]
Used when: pln["propDoseCalc"]["useCustomPrimaryPhotonFluence"] = True
```

For **flat-field (FF) beams** (6X, 15X) this is nearly constant = 1.
For **FFF beams** (6XFFF, 10XFFF) this represents the horn-shaped profile
that rises off-axis.

**To calibrate:** use the GBD shallow-depth profiles (at dmax) as a direct
measurement of the primary fluence shape, since scatter buildup is minimal
there.

**Relation to the shallow-depth narrow-profile issue (10XFFF, 15X):**
At dmax, the measured FWHM is 3–21 cm (varying with field size), but pyMatRad
predicts 0.3–0.8 cm. This is unlikely to be a `primaryFluence` issue (the
fluence is broad); the root cause may be in how the near-equal `β₃ ≈ m`
case is handled in the depth-dose formula. Under investigation.

---

### `surfaceDose` — surface dose fraction

```
6X: 0.451  (45.1% of dmax)
```

Scales the electron-contamination contribution visible as enhanced dose near
the surface. Affects the buildup region (0 – dmax).

### `electronRangeIntensity` — electron contamination intensity

```
Current: 0.001  (all energies)
```

Relative intensity of the electron contamination component. Increasing it
raises the surface dose and modifies the buildup curve shape.

---

## Recommended Tuning Workflow

For the **3–4% large-field PDD error**:

1. Fix `m` and `β₁` to the 10×10 PDD (they are already correct).
2. Jointly optimise `β₂`, `β₃`, and the tails of `kernel2`, `kernel3`
   by minimising the residual between predicted and measured PDD for
   3×3, 10×10, and 20×20 simultaneously.
3. Re-normalise the kernels so the 10×10 PDD is still matched.
4. Rebuild the machine files and re-run `examples/validate_truebeam.py`.

For the **penumbra width**:

1. Read the 20–80% penumbra from the GBD 1.5 cm (or dmax) profile.
2. Adjust `penumbraFWHMatIso` to match, rebuild, and validate.

For **FFF shallow-depth profiles (10XFFF, 15X)**:

1. Print `β₃ − m` for those energies and verify the near-equality threshold
   in the SVD engine (`abs(beta - m) < 1e-10`). The threshold may need to
   be widened (e.g. `1e-6 * max(beta, m)`) to catch the near-singular case.
2. Validate against the GBD dmax profiles.
