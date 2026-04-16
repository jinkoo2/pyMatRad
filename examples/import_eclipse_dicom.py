"""
Import Eclipse DICOM plans and run pyMatRad dose calculations.

Usage
-----
    python examples/import_eclipse_dicom.py
    python examples/import_eclipse_dicom.py --plan 7beam_IMRT
    python examples/import_eclipse_dicom.py --plan ap_IMRT --ct-dir ../ap_sMLC
    python examples/import_eclipse_dicom.py --plan 7beam_IMRT --no-dose-calc

Plans available (in ../_sample_plans/eclipse_tps/):
    Plan          CT     Struct  Dose
    7beam_IMRT    yes    yes     yes   — prostate IMRT, 7 beams 6 MV
    6x_10x10      yes    yes     yes   — open 10x10 field
    ap_sMLC       yes    yes     yes   — AP plan with static MLC
    ap_IMRT       no*    no      yes   — AP IMRT, needs --ct-dir ../ap_sMLC
    ap_VMAT       no*    no      yes   — AP VMAT, needs --ct-dir ../ap_sMLC
    stereophan_ap_IMRT    yes  no  no  — stereo phantom (CT only)
    stereophan_ap_VMAT    yes  no  no  — stereo phantom (CT only)
    stereophan_IMRT_7beams yes no  no  — stereo phantom (CT only)

*ap_IMRT and ap_VMAT share the CT from ap_sMLC.
"""

import os
import sys
import argparse
import pickle
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SAMPLE_PLANS_ROOT = os.path.join(ROOT, "..", "_sample_plans", "eclipse_tps")

from matRad.dicom import import_dicom
import matRad

# Default cache directory — one folder per plan, inside examples/cache/
DEFAULT_CACHE_ROOT = os.path.join(ROOT, "examples", "cache")


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def _cache_dir(plan_name: str, cache_root: str) -> str:
    d = os.path.join(cache_root, plan_name)
    os.makedirs(d, exist_ok=True)
    return d


def save_import(plan_name: str, ct: dict, cst: list, pln: dict,
                dose_eclipse, dose_grid: dict, cache_root: str):
    """Save DICOM import result to cache.

    Large arrays (CT cubes, Eclipse dose, structure voxel indices) are stored
    in a compressed .npz; small metadata and pln go into a .pkl.
    """
    d = _cache_dir(plan_name, cache_root)

    # ── Arrays: compressed npz ───────────────────────────────────────────
    arrays = {
        "ct_cube":   ct["cube"][0].astype(np.float32),
        "ct_cubeHU": ct["cubeHU"][0].astype(np.float32),
        "ct_x": ct["x"], "ct_y": ct["y"], "ct_z": ct["z"],
    }
    if dose_eclipse is not None:
        arrays["dose_eclipse"] = dose_eclipse.astype(np.float32)
    # Structure voxel indices
    for i, row in enumerate(cst):
        vox = row[3][0] if isinstance(row[3], list) else row[3]
        arrays[f"cst_vox_{i}"] = np.asarray(vox, dtype=np.int64)

    np.savez_compressed(os.path.join(d, "import_arrays.npz"), **arrays)

    # ── Metadata: pickle (small) ─────────────────────────────────────────
    ct_meta = {k: v for k, v in ct.items() if k not in ("cube", "cubeHU")}
    cst_meta = [[row[0], row[1], row[2], None, row[4], row[5]] for row in cst]
    meta = {"ct_meta": ct_meta, "cst_meta": cst_meta, "pln": pln,
            "dose_grid": dose_grid, "n_cst": len(cst),
            "rtplan_file": pln.get("_rtplan_file", None)}
    with open(os.path.join(d, "import_meta.pkl"), "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = (os.path.getsize(os.path.join(d, "import_arrays.npz")) +
               os.path.getsize(os.path.join(d, "import_meta.pkl"))) / 1e6
    print(f"  Saved import → {d}/  ({size_mb:.0f} MB)")


def load_import(plan_name: str, cache_root: str):
    """Load DICOM import result from cache.  Returns None if not found."""
    d = os.path.join(cache_root, plan_name)
    arr_path  = os.path.join(d, "import_arrays.npz")
    meta_path = os.path.join(d, "import_meta.pkl")
    if not (os.path.exists(arr_path) and os.path.exists(meta_path)):
        return None

    arrays = np.load(arr_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    ct = meta["ct_meta"].copy()
    ct["cube"]   = [arrays["ct_cube"]]
    ct["cubeHU"] = [arrays["ct_cubeHU"]]

    cst = []
    for i, row in enumerate(meta["cst_meta"]):
        vox = arrays[f"cst_vox_{i}"]
        cst.append([row[0], row[1], row[2], [vox], row[4], row[5]])

    dose = arrays["dose_eclipse"].astype(np.float32) if "dose_eclipse" in arrays else None

    size_mb = (os.path.getsize(arr_path) + os.path.getsize(meta_path)) / 1e6
    print(f"  Loaded import ← {d}/  ({size_mb:.0f} MB)")
    return {"ct": ct, "cst": cst, "pln": meta["pln"],
            "dose": dose, "dose_grid": meta["dose_grid"],
            "rtplan_file": meta.get("rtplan_file", None)}


def _grid_tag(dose_grid_mm: float, bixel_width_mm: float = 5.0) -> str:
    """Compact tag encoding dose grid and bixel width, e.g. 'dg5.0mm_bw5.0mm'."""
    return f"dg{dose_grid_mm:.1f}mm_bw{bixel_width_mm:.1f}mm"


def save_dij(plan_name: str, dij: dict, stf: list, cache_root: str,
             dose_grid_mm: float = 5.0, bixel_width_mm: float = 5.0,
             field_stf: bool = False):
    """Save dij sparse matrix + metadata and stf to cache."""
    d   = _cache_dir(plan_name, cache_root)
    tag = _grid_tag(dose_grid_mm, bixel_width_mm) + ("_field" if field_stf else "")
    mat_path  = os.path.join(d, f"dij_matrix_{tag}.npz")
    meta_path = os.path.join(d, f"dij_meta_{tag}.pkl")
    sp.save_npz(mat_path, dij["physicalDose"][0].tocsc())
    dij_meta = {k: v for k, v in dij.items() if k != "physicalDose"}
    with open(meta_path, "wb") as f:
        pickle.dump({"dij_meta": dij_meta, "stf": stf}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved dij → {os.path.basename(mat_path)} + {os.path.basename(meta_path)}")


def load_dij(plan_name: str, cache_root: str, dose_grid_mm: float = 5.0,
             bixel_width_mm: float = 5.0, field_stf: bool = False):
    """Load dij and stf from cache.  Returns (dij, stf) or (None, None)."""
    d   = os.path.join(cache_root, plan_name)
    tag = _grid_tag(dose_grid_mm, bixel_width_mm) + ("_field" if field_stf else "")
    mat_path  = os.path.join(d, f"dij_matrix_{tag}.npz")
    meta_path = os.path.join(d, f"dij_meta_{tag}.pkl")
    if not (os.path.exists(mat_path) and os.path.exists(meta_path)):
        return None, None
    matrix = sp.load_npz(mat_path)
    with open(meta_path, "rb") as f:
        data = pickle.load(f)
    dij = data["dij_meta"]
    dij["physicalDose"] = [matrix]
    stf = data["stf"]
    print(f"  Loaded dij ← {os.path.basename(mat_path)}  shape={matrix.shape}")
    return dij, stf


def save_result(plan_name: str, mode: str, dose: np.ndarray,
                w: np.ndarray, cache_root: str, dose_grid_mm: float = 5.0,
                bixel_width_mm: float = 5.0):
    """Save dose result (physicalDose cube + weight vector) to cache."""
    d    = _cache_dir(plan_name, cache_root)
    tag  = _grid_tag(dose_grid_mm, bixel_width_mm)
    path = os.path.join(d, f"result_{mode}_{tag}.npz")
    np.savez_compressed(path, physicalDose=dose, w=w)
    print(f"  Saved result → {os.path.basename(path)}")


def load_result(plan_name: str, mode: str, cache_root: str,
                dose_grid_mm: float = 5.0, bixel_width_mm: float = 5.0):
    """Load dose result from cache.  Returns (dose, w) or (None, None)."""
    tag  = _grid_tag(dose_grid_mm, bixel_width_mm)
    path = os.path.join(cache_root, plan_name, f"result_{mode}_{tag}.npz")
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    print(f"  Loaded result ← {os.path.basename(path)}")
    return data["physicalDose"], data["w"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dose_comparison_plots(dose_eclipse: np.ndarray,
                          dose_matrad: np.ndarray,
                          ct: dict,
                          out_dir: str,
                          plan_name: str,
                          iso_mm: np.ndarray = None):
    """Save axial / coronal / sagittal dose comparison figures.

    Slices pass through the isocenter (iso_mm [x,y,z] in mm).
    Falls back to the dose-weighted centroid if iso_mm is None.
    Each figure has three panels: Eclipse | pyMatRad | Difference.
    """
    os.makedirs(out_dir, exist_ok=True)
    x, y, z = ct["x"], ct["y"], ct["z"]

    # Determine slice indices — prefer isocenter, fall back to dose centroid
    if iso_mm is not None:
        ix0 = int(np.argmin(np.abs(x - iso_mm[0])))
        iy0 = int(np.argmin(np.abs(y - iso_mm[1])))
        iz0 = int(np.argmin(np.abs(z - iso_mm[2])))
        slice_label = f"iso ({iso_mm[0]:.0f}, {iso_mm[1]:.0f}, {iso_mm[2]:.0f}) mm"
    else:
        # Dose-weighted centroid of Eclipse dose
        ref = dose_eclipse if dose_eclipse is not None else dose_matrad
        total = ref.sum()
        if total > 0:
            Yg, Xg, Zg = np.meshgrid(y, x, z, indexing="ij")
            iy0 = int(np.argmin(np.abs(y - (ref * Yg).sum() / total)))
            ix0 = int(np.argmin(np.abs(x - (ref * Xg).sum() / total)))
            iz0 = int(np.argmin(np.abs(z - (ref * Zg).sum() / total)))
        else:
            Ny, Nx, Nz = ct["cubeDim"]
            iy0, ix0, iz0 = Ny // 2, Nx // 2, Nz // 2
        slice_label = f"dose centroid ({x[ix0]:.0f}, {y[iy0]:.0f}, {z[iz0]:.0f}) mm"

    print(f"  Slice planes: {slice_label}")

    vmax = max(dose_eclipse.max() if dose_eclipse is not None else 0,
               dose_matrad.max())

    views = [
        ("axial",
         dose_eclipse[:, :, iz0] if dose_eclipse is not None else np.zeros_like(dose_matrad[:, :, iz0]),
         dose_matrad[:, :, iz0],
         x, y, "x [mm]", "y [mm]", f"axial  z = {z[iz0]:.0f} mm"),
        ("coronal",
         dose_eclipse[iy0, :, :].T if dose_eclipse is not None else np.zeros_like(dose_matrad[iy0, :, :].T),
         dose_matrad[iy0, :, :].T,
         x, z, "x [mm]", "z [mm]", f"coronal  y = {y[iy0]:.0f} mm"),
        ("sagittal",
         dose_eclipse[:, ix0, :].T if dose_eclipse is not None else np.zeros_like(dose_matrad[:, ix0, :].T),
         dose_matrad[:, ix0, :].T,
         y, z, "y [mm]", "z [mm]", f"sagittal  x = {x[ix0]:.0f} mm"),
    ]

    for view_name, ec_slice, mr_slice, ax1, ax2, xlabel, ylabel, view_label in views:
        diff = mr_slice - ec_slice
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        im0 = axes[0].imshow(ec_slice, origin="lower", vmin=0, vmax=vmax, cmap="jet",
                             extent=[ax1[0], ax1[-1], ax2[0], ax2[-1]], aspect="auto")
        axes[0].set_title("Eclipse RTDose")
        axes[0].set_xlabel(xlabel); axes[0].set_ylabel(ylabel)
        plt.colorbar(im0, ax=axes[0], label="Gy")

        im1 = axes[1].imshow(mr_slice, origin="lower", vmin=0, vmax=vmax, cmap="jet",
                             extent=[ax1[0], ax1[-1], ax2[0], ax2[-1]], aspect="auto")
        axes[1].set_title("pyMatRad dose")
        axes[1].set_xlabel(xlabel)
        plt.colorbar(im1, ax=axes[1], label="Gy")

        im2 = axes[2].imshow(diff, origin="lower", cmap="bwr",
                             vmin=-vmax * 0.1, vmax=vmax * 0.1,
                             extent=[ax1[0], ax1[-1], ax2[0], ax2[-1]], aspect="auto")
        axes[2].set_title("Difference (matRad − Eclipse)")
        axes[2].set_xlabel(xlabel)
        plt.colorbar(im2, ax=axes[2], label="Gy")

        fig.suptitle(f"{plan_name} — {view_label}")
        fig.tight_layout()
        out = os.path.join(out_dir, f"{plan_name}_{view_name}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Line profile comparison
# ---------------------------------------------------------------------------

def dose_line_profiles(dose_eclipse: np.ndarray,
                       dose_matrad: np.ndarray,
                       dg: dict,
                       out_dir: str,
                       plan_name: str,
                       iso_mm: np.ndarray = None):
    """Save line-profile comparison figure along x, y, z axes through iso.

    Three panels (one per axis), each showing Eclipse and matRad dose [Gy]
    and their difference.  The profiles are extracted on the matRad dose grid.
    """
    os.makedirs(out_dir, exist_ok=True)

    x = np.asarray(dg["x"]).ravel()
    y = np.asarray(dg["y"]).ravel()
    z = np.asarray(dg["z"]).ravel()

    if iso_mm is not None:
        ix0 = int(np.argmin(np.abs(x - iso_mm[0])))
        iy0 = int(np.argmin(np.abs(y - iso_mm[1])))
        iz0 = int(np.argmin(np.abs(z - iso_mm[2])))
    else:
        iy0, ix0, iz0 = (s // 2 for s in dose_matrad.shape)

    # profiles through iso: matRad array shape is (Ny, Nx, Nz)
    profiles = [
        ("x", x, "x [mm]", f"y={y[iy0]:.0f}, z={z[iz0]:.0f} mm",
         dose_eclipse[:, :, iz0][iy0, :],   # (Nx,)
         dose_matrad [:, :, iz0][iy0, :]),
        ("y", y, "y [mm]", f"x={x[ix0]:.0f}, z={z[iz0]:.0f} mm",
         dose_eclipse[:, ix0, iz0],          # (Ny,)
         dose_matrad [:, ix0, iz0]),
        ("z", z, "z [mm]", f"x={x[ix0]:.0f}, y={y[iy0]:.0f} mm",
         dose_eclipse[iy0, ix0, :],          # (Nz,)
         dose_matrad [iy0, ix0, :]),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    for row, (axis_name, coords, xlabel, fixed_label, ec_prof, mr_prof) in enumerate(profiles):
        ax_dose = axes[row, 0]
        ax_diff = axes[row, 1]

        ax_dose.plot(coords, ec_prof, "b-",  lw=1.5, label="Eclipse")
        ax_dose.plot(coords, mr_prof, "r--", lw=1.5, label="pyMatRad")
        ax_dose.set_xlabel(xlabel)
        ax_dose.set_ylabel("Dose [Gy]")
        ax_dose.set_title(f"{axis_name}-profile  ({fixed_label} fixed)")
        ax_dose.legend(fontsize=8)
        ax_dose.grid(True)

        diff = mr_prof - ec_prof
        ax_diff.plot(coords, diff, "k-", lw=1.5)
        ax_diff.axhline(0, color="gray", lw=0.8, ls="--")
        ax_diff.set_xlabel(xlabel)
        ax_diff.set_ylabel("Diff (matRad − Eclipse) [Gy]")
        ax_diff.set_title(f"{axis_name}-profile difference  ({fixed_label} fixed)")
        ax_diff.grid(True)

    iso_str = (f"iso ({iso_mm[0]:.0f}, {iso_mm[1]:.0f}, {iso_mm[2]:.0f}) mm"
               if iso_mm is not None else "centre")
    fig.suptitle(f"{plan_name}  —  line profiles through {iso_str}", fontsize=12)
    fig.tight_layout()

    out = os.path.join(out_dir, f"{plan_name}_line_profiles.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(plan_name: str, ct_dir: str = None, calc_dose: bool = True,
        eclipse_fluence: bool = False, cache_root: str = DEFAULT_CACHE_ROOT,
        force: bool = False, dose_grid_mm: float = 5.0,
        bixel_width_mm: float = 5.0):
    plan_dir = os.path.join(SAMPLE_PLANS_ROOT, plan_name)
    if not os.path.isdir(plan_dir):
        print(f"ERROR: plan directory not found: {plan_dir}")
        print("Available plans:")
        for d in sorted(os.listdir(SAMPLE_PLANS_ROOT)):
            if os.path.isdir(os.path.join(SAMPLE_PLANS_ROOT, d)):
                print(f"  {d}")
        sys.exit(1)

    # Resolve optional separate CT directory
    ct_dir_abs = None
    if ct_dir is not None:
        ct_dir_abs = (os.path.join(SAMPLE_PLANS_ROOT, ct_dir)
                      if not os.path.isabs(ct_dir) else ct_dir)

    print(f"\n{'='*60}")
    print(f"  Plan: {plan_name}")
    print(f"{'='*60}")

    # ── 1. Import DICOM ──────────────────────────────────────────────────
    cached = None if force else load_import(plan_name, cache_root)
    if cached is not None:
        ct, cst, pln, dose_eclipse, dose_eclipse_grid, rtplan_file = (
            cached["ct"], cached["cst"], cached["pln"],
            cached["dose"], cached["dose_grid"], cached.get("rtplan_file"))
    else:
        result = import_dicom(plan_dir, ct_dir=ct_dir_abs)
        ct, cst, pln, dose_eclipse, dose_eclipse_grid = (
            result["ct"], result["cst"], result["pln"],
            result["dose"], result["dose_grid"])
        rtplan_file = pln.get("_rtplan_file")
        save_import(plan_name, ct, cst, pln, dose_eclipse,
                    dose_eclipse_grid, cache_root)

    print(f"\nStructures ({len(cst)}):")
    for row in cst:
        vox = row[3][0] if isinstance(row[3], list) else row[3]
        print(f"  [{row[2]:6s}] {row[1]:30s}  {len(vox)} voxels")

    if pln is None:
        print("\nNo RTPlan found — skipping dose calculation.")
        return

    print(f"\nBeams ({len(pln['propStf']['gantryAngles'])}):")
    for i, (g, c, mu) in enumerate(zip(
            pln["propStf"]["gantryAngles"],
            pln["propStf"]["couchAngles"],
            pln["propStf"]["beamMU"])):
        print(f"  {i+1}: gantry={g:.1f}°  couch={c:.1f}°  MU={mu:.1f}")

    if not calc_dose:
        print("\n--no-dose-calc specified, skipping.")
        return

    # ── 2. Set dose-calc options ─────────────────────────────────────────
    # Use 5 mm dose grid and single-process execution to avoid OOM.
    # A full clinical CT at 3 mm would be ~4 M voxels × 2694 bixels and
    # will exhaust RAM in subprocesses.  numWorkers=1 runs everything in
    # the main process so the OS cannot silently kill a child.
    r = float(dose_grid_mm)
    bw = float(bixel_width_mm)
    print(f"  Dose grid: {r:.1f} mm isotropic  |  Bixel width: {bw:.1f} mm")
    pln["propDoseCalc"].update({
        "doseGrid":              {"resolution": {"x": r, "y": r, "z": r}},
        "ignoreOutsideDensities": False,
        "numWorkers":            1,
    })
    pln["propStf"]["bixelWidth"] = bw

    # ── 3. Resolve RTPLAN file path ──────────────────────────────────────
    if eclipse_fluence:
        if rtplan_file and os.path.isfile(rtplan_file):
            plan_file = rtplan_file
        else:
            import glob as _glob, pydicom as _pydicom
            candidates = _glob.glob(os.path.join(plan_dir, "**", "*.dcm"),
                                    recursive=True)
            plan_file = next(
                (p for p in candidates
                 if getattr(_pydicom.dcmread(p, stop_before_pixels=True),
                            "Modality", "").upper() == "RTPLAN"),
                None
            )
            if plan_file is None:
                raise FileNotFoundError(f"No RTPLAN .dcm found in {plan_dir}")

        # Load machine early — needed by both stf_from_rtplan_aperture and
        # import_rtplan_fluence (TG-51 calibration factor).
        from matRad.basedata import load_machine
        machine = load_machine(pln)

    # ── 4. Generate STF + dij (load from cache if available) ─────────────
    # eclipse-fluence mode uses a field-aperture STF (bixels over the full
    # MLC jaw opening) instead of the PTV-projected STF.  The two are cached
    # under different tags so they don't collide.
    field_stf = eclipse_fluence
    dij, stf = (None, None) if force else load_dij(
        plan_name, cache_root, dose_grid_mm, bixel_width_mm, field_stf)
    if dij is None:
        if eclipse_fluence:
            print("\nGenerating field-aperture beam geometry (STF from MLC jaws) ...")
            stf = matRad.dicom.stf_from_rtplan_aperture(
                plan_file, pln, bixel_width_mm, machine=machine)
        else:
            print("\nGenerating beam geometry (STF) ...")
            stf = matRad.generate_stf(ct, cst, pln)
        total_bixels = sum(b["totalNumOfBixels"] for b in stf)
        print(f"  {len(stf)} beams, {total_bixels} bixels total")

        print("\nCalculating dose influence matrix ...")
        dij = matRad.calc_dose_influence(ct, cst, stf, pln)
        save_dij(plan_name, dij, stf, cache_root, dose_grid_mm, bixel_width_mm, field_stf)
    else:
        total_bixels = sum(b["totalNumOfBixels"] for b in stf)
        print(f"  {len(stf)} beams, {total_bixels} bixels total (from cache)")

    mode = "eclipse_fluence" if eclipse_fluence else "reoptimised"
    dose_matrad, w_cached = (None, None) if force else load_result(plan_name, mode, cache_root, dose_grid_mm, bixel_width_mm)

    if dose_matrad is None:
        if eclipse_fluence:
            # ── 5a. Reproduce Eclipse dose from MLC leaf sequences ────────
            print("\nImporting Eclipse MLC fluence ...")
            w = matRad.dicom.import_rtplan_fluence(
                plan_file, stf, machine=machine,
                num_fractions=pln.get("numOfFractions", 1))
            beam_mus = pln["propStf"]["beamMU"]
            print(f"  Weight vector: {len(w)} bixels  non-zero: {np.sum(w>0)}")
            print(f"  Total MU: {beam_mus.sum():.1f}  "
                  f"Total w: {w.sum():.3f}")
            print("\nComputing dose from Eclipse fluence ...")
            result_matrad = matRad.calc_dose_direct(dij, w)
        else:
            # ── 5b. Independent matRad fluence optimisation ───────────────
            print("\nOptimising fluence ...")
            result_matrad = matRad.fluence_optimization(dij, cst, pln)
            w = result_matrad["w"]

        dose_matrad = result_matrad["physicalDose"]
        save_result(plan_name, mode, dose_matrad, w, cache_root, dose_grid_mm, bixel_width_mm)
    else:
        w = w_cached

    label = "matRad (Eclipse fluence)" if eclipse_fluence else "matRad (re-optimised)"
    print(f"  {label} max dose: {dose_matrad.max():.3f} Gy")

    # ── 6. Comparison ────────────────────────────────────────────────────
    if dose_eclipse is not None:
        # Interpolate Eclipse dose from its native DICOM grid directly onto
        # the matRad dose calc grid — one hop, no CT grid intermediate.
        from scipy.interpolate import RegularGridInterpolator
        dg = dij["doseGrid"]
        interp = RegularGridInterpolator(
            (dose_eclipse_grid["y"], dose_eclipse_grid["x"], dose_eclipse_grid["z"]),
            dose_eclipse,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        DGy, DGx, DGz = np.meshgrid(dg["y"], dg["x"], dg["z"], indexing="ij")
        dose_eclipse_dg = interp(
            np.stack([DGy.ravel(), DGx.ravel(), DGz.ravel()], axis=1)
        ).reshape(dose_matrad.shape)

        diff = dose_matrad - dose_eclipse_dg
        mask = dose_eclipse_dg > 0.05 * dose_eclipse_dg.max()
        print(f"\nDose comparison (voxels with Eclipse dose > 5% max):")
        print(f"  Eclipse max  : {dose_eclipse_dg.max():.3f} Gy")
        print(f"  {label:30s} max: {dose_matrad.max():.3f} Gy")
        print(f"  Mean |diff|  : {np.abs(diff[mask]).mean():.3f} Gy  "
              f"({np.abs(diff[mask]).mean() / dose_eclipse_dg[mask].mean() * 100:.1f}%)")
        print(f"  Max  |diff|  : {np.abs(diff[mask]).max():.3f} Gy")

        suffix = "_eclipse_fluence" if eclipse_fluence else "_reoptimised"
        out_dir = os.path.join(ROOT, "examples", "dicom_comparison_plots")
        iso = pln["propStf"]["isoCenter"]
        iso_mm = np.atleast_2d(iso)[0] if iso.ndim > 1 else iso
        dose_grid_dict = {"x": dg["x"], "y": dg["y"], "z": dg["z"],
                          "cubeDim": list(dose_matrad.shape)}
        dose_comparison_plots(dose_eclipse_dg, dose_matrad, dose_grid_dict, out_dir,
                              plan_name + suffix, iso_mm=iso_mm)
        dose_line_profiles(dose_eclipse_dg, dose_matrad, dg, out_dir,
                           plan_name + suffix, iso_mm=iso_mm)
    else:
        print("\nNo RTDose found — skipping comparison plots.")

    print(f"\nDone: {plan_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Eclipse DICOM and run pyMatRad")
    parser.add_argument("--plan", default="7beam_IMRT",
                        help="Subdirectory name inside _sample_plans/eclipse_tps/")
    parser.add_argument("--ct-dir", default=None,
                        help="Separate CT directory (relative to eclipse_tps/) "
                             "for plans that share a CT, e.g. --ct-dir ap_sMLC")
    parser.add_argument("--no-dose-calc", action="store_true",
                        help="Only import DICOM, skip dose calculation")
    parser.add_argument("--eclipse-fluence", action="store_true",
                        help="Use Eclipse MLC leaf sequences to reproduce Eclipse dose "
                             "(default: re-optimise fluence from scratch)")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_ROOT,
                        help=f"Directory for cached import/dij/result files "
                             f"(default: {DEFAULT_CACHE_ROOT})")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cache and recompute everything from scratch")
    parser.add_argument("--dose-grid", type=float, default=5.0, metavar="MM",
                        help="Isotropic dose grid resolution in mm (default: 5.0). "
                             "Each grid size gets its own dij/result cache entry.")
    parser.add_argument("--bixel-width", type=float, default=5.0, metavar="MM",
                        help="Bixel (pencil beam) width in mm at isocenter (default: 5.0). "
                             "Controls the lateral BEV ray spacing during STF generation. "
                             "Each bixel width gets its own dij/result cache entry.")
    args = parser.parse_args()

    run(args.plan, ct_dir=args.ct_dir, calc_dose=not args.no_dose_calc,
        eclipse_fluence=args.eclipse_fluence,
        cache_root=args.cache_dir, force=args.force,
        dose_grid_mm=args.dose_grid, bixel_width_mm=args.bixel_width)
