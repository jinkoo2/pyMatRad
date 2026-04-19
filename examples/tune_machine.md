# Machine Tuning Results

Automated parameter tuning for TrueBeam photon machine files.
Parameters (m, betas, penumbraFWHMatIso) were optimised to minimise
PDD and lateral-profile error vs GBD reference data.

Date run: 2026-04-19

---

## Tuned Parameters

| Energy | fwhm [mm] | m [mm⁻¹] | β₁ [mm⁻¹] | β₂ [mm⁻¹] | β₃ [mm⁻¹] | err before | err after | abs_calib [cGy/MU] | d_max [mm] |
|--------|-----------|----------|----------|----------|----------|-----------|-----------|-------------------|-----------|
| TrueBeam_6X | 6.881 | 0.004817 | 0.21457 | 0.01689 | 0.00542 | 2.2489 | 2.2489 | nan | nan |
| TrueBeam_6XFFF | 6.628 | 0.005322 | 0.22292 | 0.01526 | 0.00594 | 1.8239 | 1.8239 | nan | nan |
| TrueBeam_10XFFF | 8.015 | 0.003956 | 0.13986 | 0.01267 | 0.00462 | 1.3903 | 1.3903 | nan | nan |
| TrueBeam_15X | 7.961 | 0.003019 | 0.11293 | 0.01164 | 0.00248 | 1.3270 | 1.3270 | nan | nan |

---

## Notes

- **fwhm**: `penumbraFWHMatIso` — Gaussian source FWHM controlling lateral penumbra.
- **m**: primary photon attenuation coefficient. Governs exponential depth-dose tail.
- **β₁, β₂, β₃**: SVD scatter-kernel decay constants controlling build-up region
  and depth-scatter contributions.
- **err**: weighted RMSE (PDD at 5/10/20/30 cm + in-field profile), lower is better.
- Lateral kernel weights (kernel1–3) were **not** rebuilt; those encode field-size-
  specific scatter from the original GBD TPR table. To improve 3×3 vs 20×20
  PDD differences, rebuild kernels via `machineBuilder/build_truebeam.py`.
- **abs_calib**: TG-51 calibration factor written to `machine["meta"]["tg51"]`.

## Workflow

```
python examples/tune_machine.py              # tune all energies
python examples/tune_machine.py --dry-run    # baseline only
python examples/calibrate_machine.py --machine TrueBeam_6X  # re-calibrate only
```
