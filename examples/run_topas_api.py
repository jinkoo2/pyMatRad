"""
Run Example 1 (water phantom) with the TOPAS MC engine via the REST API.

No GUI / no matplotlib display needed — prints dose statistics only.

Usage:
    conda run -n pyMatRad python examples/run_topas_api.py
"""

import os
import sys
import time
import numpy as np

# headless matplotlib (must be set before any matRad import that touches mpl)
import matplotlib
matplotlib.use("Agg")

PYMATRAD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PYMATRAD_ROOT)

TOPAS_API_URL   = "http://localhost:7778"
TOPAS_API_TOKEN = "topas-dev-a3f8c2d1-4b7e-4f9a-8c3d-2e1f5a6b7c8d"


def main():
    from matRad.phantoms.builder import PhantomBuilder
    from matRad.geometry.geometry import get_world_axes
    from matRad.steering.stf_generator import generate_stf
    from matRad.doseCalc.calc_dose_influence import calc_dose_influence
    from matRad.optimization.DoseObjectives import SquaredDeviation, SquaredOverdosing

    print("=" * 65)
    print("Example 1 — Water phantom, TOPAS via REST API")
    print("=" * 65)
    print(f"  API URL : {TOPAS_API_URL}")

    # ── Phantom ─────────────────────────────────────────────────────────
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

    pln = {
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
        "propDoseCalc": {
            "engine":        "TOPAS",
            "doseGrid":      {"resolution": {"x": 3, "y": 3, "z": 3}},
            "numHistories":  100_000,
            "topasApiUrl":   TOPAS_API_URL,
            "topasApiToken": TOPAS_API_TOKEN,
        },
    }

    print(f"\nCT: {CT_DIM} voxels @ {CT_RES} mm")
    print(f"Beams: {pln['propStf']['gantryAngles']}°")

    print("\nGenerating STF...")
    stf = generate_stf(ct, cst, pln)
    n_bixels = sum(b["totalNumOfBixels"] for b in stf)
    print(f"  Total bixels: {n_bixels}")

    print("\nRunning TOPAS via API...")
    t0  = time.perf_counter()
    dij = calc_dose_influence(ct, cst, stf, pln)
    elapsed = time.perf_counter() - t0

    D         = dij["physicalDose"][0].tocsc()
    dose_flat = np.asarray(D @ np.ones(D.shape[1])).ravel()

    print("\n" + "=" * 65)
    print("Results")
    print("=" * 65)
    print(f"  Elapsed : {elapsed:.1f} s")
    print(f"  DIJ     : {D.shape}  nnz={D.nnz}")
    print(f"  Max dose: {dose_flat.max():.6f} Gy/fx")
    nz = dose_flat > 0
    print(f"  Mean>0  : {dose_flat[nz].mean():.6f} Gy/fx  ({nz.sum()} voxels)")
    print("\nDone.")


if __name__ == "__main__":
    main()
