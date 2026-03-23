"""
Example 2 — TG119 phantom, no optimisation.

Compares SVPB, ompMC, and TOPAS dose engines on the TG119 photon phantom
with uniform beamlet weights (w = 1 for all bixels).

Results reported:
  * Max / mean dose per engine
  * Per-beam DIJ column sums
  * Wall-clock time per engine
  * Cross-engine dose ratio (ompMC / SVPB, TOPAS / SVPB)

Usage
-----
  python examples/example2_no_opti.py

TOPAS is skipped automatically if the binary is not found.
"""

import os
import sys
import shutil
import time
import numpy as np

PYMATRAD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TG119_MAT = r"C:\Users\jkim20\Desktop\projects\tps\matRad\matRad\phantoms\TG119.mat"
sys.path.insert(0, PYMATRAD_ROOT)


def _load_tg119(path: str):
    """Load TG119 CT and CST from MATLAB .mat file."""
    import scipy.io as sio

    def _s(x):
        return float(np.array(x).ravel()[0])

    try:
        mat  = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        ct_m = mat["ct"]
        cst_m = mat["cst"]

        dims  = np.array(ct_m.cubeDim).ravel().astype(int)
        res_x = _s(ct_m.resolution.x)
        res_y = _s(ct_m.resolution.y)
        res_z = _s(ct_m.resolution.z)

        # cubeHU is a MATLAB cell{1} — squeeze_me turns it into the array
        cube_raw = np.array(ct_m.cubeHU)
        if cube_raw.dtype == object:
            cube_hu = np.array(cube_raw.flat[0]).astype(float)
        else:
            cube_hu = cube_raw.astype(float)

        Ny, Nx, Nz = int(dims[0]), int(dims[1]), int(dims[2])
        cube_hu = cube_hu.reshape((Ny, Nx, Nz), order="F")

        ct = {
            "cubeDim":     np.array([Ny, Nx, Nz]),
            "resolution":  {"x": res_x, "y": res_y, "z": res_z},
            "cubeHU":      [cube_hu],
            "numOfCtScen": 1,
        }

        # Rebuild CST — cst_m is a (N,6) numpy object array; iterate rows
        cst = []
        n_struct = cst_m.shape[0] if hasattr(cst_m, "shape") and cst_m.ndim == 2 else len(cst_m)
        for i in range(n_struct):
            row_m = cst_m[i]  # shape (6,) — the 6 columns of this row
            voxels = np.array(row_m[3]).ravel().astype(np.int64)
            cst.append([
                int(row_m[0]), str(row_m[1]), str(row_m[2]),
                [voxels], {}, {},
            ])

        return ct, cst, Ny, Nx, Nz, res_x, res_y, res_z

    except Exception as exc_scipy:
        # Try h5py (MATLAB v7.3)
        import h5py
        f = h5py.File(path, "r")

        dims  = np.array(f["ct/cubeDim"]).ravel().astype(int)
        res_x = float(np.array(f["ct/resolution/x"]).ravel()[0])
        res_y = float(np.array(f["ct/resolution/y"]).ravel()[0])
        res_z = float(np.array(f["ct/resolution/z"]).ravel()[0])

        Ny, Nx, Nz = int(dims[0]), int(dims[1]), int(dims[2])
        hu_raw = np.array(f["ct/cubeHU"]).T.astype(float)
        cube_hu = hu_raw.reshape((Ny, Nx, Nz), order="F")

        ct = {
            "cubeDim":     np.array([Ny, Nx, Nz]),
            "resolution":  {"x": res_x, "y": res_y, "z": res_z},
            "cubeHU":      [cube_hu],
            "numOfCtScen": 1,
        }
        f.close()
        return ct, [], Ny, Nx, Nz, res_x, res_y, res_z


def main():
    from matRad.geometry.geometry import get_world_axes
    from matRad.steering.stf_generator import generate_stf
    from matRad.doseCalc.calc_dose_influence import calc_dose_influence

    print("=" * 65)
    print("Example 2 — TG119 phantom, no optimisation: engine comparison")
    print("=" * 65)

    if not os.path.exists(TG119_MAT):
        print(f"ERROR: TG119.mat not found at:\n  {TG119_MAT}")
        sys.exit(1)

    print(f"\nLoading TG119.mat...")
    ct, cst, Ny, Nx, Nz, rx, ry, rz = _load_tg119(TG119_MAT)
    ct = get_world_axes(ct)
    print(f"CT: {Ny}×{Nx}×{Nz} voxels @ ({rx},{ry},{rz}) mm")
    print(f"CST: {len(cst)} structures")

    base_pln = {
        "radiationMode":  "photons",
        "machine":        "Generic",
        "bioModel":       "none",
        "multScen":       "nomScen",
        "numOfFractions": 30,
        "propStf": {
            "gantryAngles": [0, 51, 102, 153, 204, 255, 306],
            "couchAngles":  [0, 0, 0, 0, 0, 0, 0],
            "bixelWidth":   5,
            "addMargin":    True,
        },
        "propOpt":      {"runDAO": False, "runSequencing": False},
        "propDoseCalc": {"doseGrid": {"resolution": {"x": 3, "y": 3, "z": 3}}},
    }

    print(f"\nBeams: {base_pln['propStf']['gantryAngles']}°")
    print("\nGenerating beam geometry (STF)...")
    stf = generate_stf(ct, cst, base_pln)
    n_bixels_total = sum(b["totalNumOfBixels"] for b in stf)
    print(f"  Total bixels: {n_bixels_total}")
    for ib, b in enumerate(stf):
        print(f"    beam[{ib}] gantry={b['gantryAngle']:6.1f}°  "
              f"bixels={b['totalNumOfBixels']}")

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

    def _beam_sums(D):
        off, sums = 0, []
        for b in stf:
            nb  = b["totalNumOfBixels"]
            sl  = D[:, off:off+nb]
            sums.append(float(sl.data.sum()) if sl.nnz > 0 else 0.0)
            off += nb
        return sums

    # -----------------------------------------------------------------------
    # SVPB
    # -----------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("Engine 1: SVPB (SVD Pencil Beam)  [default]")
    print("-" * 65)
    _, dose_svpb, t_svpb, D_svpb = run_engine("SVPB")
    print(f"  Elapsed : {t_svpb:.2f} s")
    print(f"  DIJ     : {D_svpb.shape}  nnz={D_svpb.nnz}")
    print(f"  Max dose: {dose_svpb.max():.4f} Gy/fx")
    print(f"  Mean>0  : {dose_svpb[dose_svpb > 0].mean():.4f} Gy/fx")
    s_svpb = _beam_sums(D_svpb)
    for ib, (b, s) in enumerate(zip(stf, s_svpb)):
        print(f"    beam[{ib}] g={b['gantryAngle']:6.1f}°  col-sum={s:.4f}")

    # -----------------------------------------------------------------------
    # ompMC
    # -----------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("Engine 2: ompMC (TERMA + scatter)")
    print("-" * 65)
    _, dose_ompc, t_ompc, D_ompc = run_engine("ompMC")
    print(f"  Elapsed : {t_ompc:.2f} s")
    print(f"  DIJ     : {D_ompc.shape}  nnz={D_ompc.nnz}")
    print(f"  Max dose: {dose_ompc.max():.4f} Gy/fx")
    print(f"  Mean>0  : {dose_ompc[dose_ompc > 0].mean():.4f} Gy/fx")
    s_ompc = _beam_sums(D_ompc)
    for ib, (b, s) in enumerate(zip(stf, s_ompc)):
        print(f"    beam[{ib}] g={b['gantryAngle']:6.1f}°  col-sum={s:.4f}")

    # -----------------------------------------------------------------------
    # TOPAS
    # -----------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("Engine 3: TOPAS (Geant4 MC)")
    print("-" * 65)
    dose_topas = t_topas = D_topas = None

    topas_exec = os.environ.get("TOPAS_EXEC", "topas")
    if shutil.which(topas_exec) is None:
        print(f"  TOPAS binary not found ('{topas_exec}').")
        print("  Set TOPAS_EXEC env var or pln['propDoseCalc']['topasExec'].")
        print("  Skipping TOPAS.")
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
    print("SUMMARY — TG119 Phantom")
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
