"""
Example 1 — Water phantom, no optimisation.

Compares SVPB, ompMC, and TOPAS dose engines on a simple water phantom
with uniform beamlet weights (w = 1 for all bixels).

Results reported:
  * Max / mean dose per engine
  * Wall-clock time per engine
  * Cross-engine dose ratio (ompMC / SVPB, TOPAS / SVPB)

Usage
-----
  python examples/example1_no_opti.py

TOPAS is skipped automatically if the binary is not found.
"""

import os
import sys
import shutil
import time
import numpy as np

PYMATRAD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PYMATRAD_ROOT)


def main():
    from matRad.phantoms.builder import PhantomBuilder
    from matRad.geometry.geometry import get_world_axes
    from matRad.steering.stf_generator import generate_stf
    from matRad.doseCalc.calc_dose_influence import calc_dose_influence
    from matRad.optimization.DoseObjectives import SquaredDeviation, SquaredOverdosing

    # -----------------------------------------------------------------------
    # Phantom setup
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("Example 1 — Water phantom, no optimisation: engine comparison")
    print("=" * 65)

    CT_DIM = [200, 200, 100]
    CT_RES = [2, 2, 3]

    builder = PhantomBuilder(CT_DIM, CT_RES, num_of_ct_scen=1)
    builder.add_spherical_target(
        "PTV", radius=20,
        objectives=[SquaredDeviation(penalty=800, d_ref=45).to_dict()], HU=0,
    )
    builder.add_box_oar(
        "OAR1", [60, 30, 60], offset=[0, -15, 0],
        objectives=[SquaredOverdosing(penalty=400, d_ref=0).to_dict()], HU=0,
    )
    builder.add_box_oar(
        "OAR2", [60, 30, 60], offset=[0, 15, 0],
        objectives=[SquaredOverdosing(penalty=10, d_ref=0).to_dict()], HU=0,
    )
    ct, cst = builder.get_ct_cst()
    ct = get_world_axes(ct)

    base_pln = {
        "radiationMode":  "photons",
        "machine":        "Generic",
        "bioModel":       "none",
        "multScen":       "nomScen",
        "numOfFractions": 30,
        "propStf": {
            "gantryAngles": [0, 72, 144, 216, 288],
            "couchAngles":  [0, 0, 0, 0, 0],
            "bixelWidth":   5,
            "addMargin":    True,
        },
        "propOpt":      {"runDAO": False, "runSequencing": False},
        "propDoseCalc": {"doseGrid": {"resolution": {"x": 3, "y": 3, "z": 3}}},
    }

    print(f"\nCT: {CT_DIM} voxels @ {CT_RES} mm")
    print(f"Beams: {base_pln['propStf']['gantryAngles']}°")

    print("\nGenerating beam geometry (STF)...")
    stf = generate_stf(ct, cst, base_pln)
    n_bixels = sum(b["totalNumOfBixels"] for b in stf)
    print(f"  Total bixels: {n_bixels}")

    # -----------------------------------------------------------------------
    # Helper: run one engine
    # -----------------------------------------------------------------------
    def run_engine(engine_name):
        pln = dict(base_pln)
        pln["propDoseCalc"] = dict(base_pln["propDoseCalc"])
        pln["propDoseCalc"]["engine"] = engine_name
        t0      = time.perf_counter()
        dij     = calc_dose_influence(ct, cst, stf, pln)
        elapsed = time.perf_counter() - t0
        D         = dij["physicalDose"][0].tocsc()
        dose_flat = np.asarray(D @ np.ones(D.shape[1])).ravel()
        return dij, dose_flat, elapsed, D

    # -----------------------------------------------------------------------
    # SVPB
    # -----------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("Engine 1: SVPB (SVD Pencil Beam)  [default]")
    print("-" * 65)
    dij_svpb, dose_svpb, t_svpb, D_svpb = run_engine("SVPB")
    print(f"  Elapsed : {t_svpb:.2f} s")
    print(f"  DIJ     : {D_svpb.shape}  nnz={D_svpb.nnz}")
    print(f"  Max dose: {dose_svpb.max():.4f} Gy/fx")
    print(f"  Mean>0  : {dose_svpb[dose_svpb > 0].mean():.4f} Gy/fx")

    # -----------------------------------------------------------------------
    # ompMC
    # -----------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("Engine 2: ompMC (TERMA + scatter)")
    print("-" * 65)
    dij_ompc, dose_ompc, t_ompc, D_ompc = run_engine("ompMC")
    print(f"  Elapsed : {t_ompc:.2f} s")
    print(f"  DIJ     : {D_ompc.shape}  nnz={D_ompc.nnz}")
    print(f"  Max dose: {dose_ompc.max():.4f} Gy/fx")
    print(f"  Mean>0  : {dose_ompc[dose_ompc > 0].mean():.4f} Gy/fx")

    # -----------------------------------------------------------------------
    # TOPAS (skipped if binary not found)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("Engine 3: TOPAS (Geant4 MC)")
    print("-" * 65)
    dose_topas = t_topas = D_topas = None

    topas_exec = os.environ.get("TOPAS_EXEC", "topas")
    if shutil.which(topas_exec) is None:
        print(f"  TOPAS binary not found ('{topas_exec}').")
        print("  Set env TOPAS_EXEC or pln['propDoseCalc']['topasExec'].")
        print("  Skipping TOPAS — install from https://topas.mgh.harvard.edu/")
    else:
        try:
            _, dose_topas, t_topas, D_topas = run_engine("TOPAS")
            print(f"  Elapsed : {t_topas:.2f} s")
            print(f"  DIJ     : {D_topas.shape}  nnz={D_topas.nnz}")
            print(f"  Max dose: {dose_topas.max():.4f} Gy/fx")
            print(f"  Mean>0  : {dose_topas[dose_topas > 0].mean():.4f} Gy/fx")
        except Exception as exc:
            print(f"  TOPAS failed: {exc}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SUMMARY — Water Phantom")
    print("=" * 65)
    print(f"  {'Engine':<10} {'Time (s)':>10}  {'Max dose':>12}  "
          f"{'Mean>0':>12}  {'Ratio/SVPB':>12}")
    print("  " + "-" * 60)

    ref_max = dose_svpb.max()

    def _row(name, t, dose):
        r = dose.max() / ref_max if ref_max > 0 else float("nan")
        print(f"  {name:<10} {t:>10.2f}  {dose.max():>12.4f}  "
              f"{dose[dose > 0].mean():>12.4f}  {r:>12.4f}")

    _row("SVPB",  t_svpb, dose_svpb)
    _row("ompMC", t_ompc, dose_ompc)
    if dose_topas is not None:
        _row("TOPAS", t_topas, dose_topas)
    else:
        print(f"  {'TOPAS':<10} {'N/A':>10}  {'N/A':>12}  {'N/A':>12}  {'N/A':>12}")

    print()
    print(f"  Speed factor ompMC / SVPB : {t_ompc / t_svpb:.2f}×")
    if t_topas is not None:
        print(f"  Speed factor TOPAS / SVPB : {t_topas / t_svpb:.2f}×")

    # -----------------------------------------------------------------------
    # Cross-engine comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Cross-engine dose comparison (ompMC vs SVPB)")
    print("=" * 65)

    both = (dose_svpb > 0) & (dose_ompc > 0)
    if both.sum() > 0:
        ratio = dose_ompc[both] / dose_svpb[both]
        print(f"  Voxels with dose>0 in both : {both.sum()}")
        print(f"  ompMC/SVPB ratio  "
              f"mean={ratio.mean():.3f}  std={ratio.std():.3f}  "
              f"median={np.median(ratio):.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
