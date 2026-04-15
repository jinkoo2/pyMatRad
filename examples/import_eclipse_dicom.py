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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SAMPLE_PLANS_ROOT = os.path.join(ROOT, "..", "_sample_plans", "eclipse_tps")

from matRad.dicom import import_dicom
import matRad


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dose_comparison_plots(dose_eclipse: np.ndarray,
                          dose_matrad: np.ndarray,
                          ct: dict,
                          out_dir: str,
                          plan_name: str):
    """Save axial / coronal / sagittal dose comparison figures."""
    os.makedirs(out_dir, exist_ok=True)
    Ny, Nx, Nz = ct["cubeDim"]
    x, y, z = ct["x"], ct["y"], ct["z"]

    iy0 = Ny // 2
    ix0 = Nx // 2
    iz0 = Nz // 2

    vmax = max(dose_eclipse.max(), dose_matrad.max())

    views = [
        ("axial",    dose_eclipse[:, :, iz0],  dose_matrad[:, :, iz0],   x,  y, "x [mm]", "y [mm]"),
        ("coronal",  dose_eclipse[iy0, :, :].T, dose_matrad[iy0, :, :].T, x,  z, "x [mm]", "z [mm]"),
        ("sagittal", dose_eclipse[:, ix0, :].T, dose_matrad[:, ix0, :].T, y,  z, "y [mm]", "z [mm]"),
    ]

    for name, ec_slice, mr_slice, ax1, ax2, xlabel, ylabel in views:
        diff = mr_slice - ec_slice
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        im0 = axes[0].imshow(ec_slice, origin="lower", vmin=0, vmax=vmax, cmap="jet",
                             extent=[ax1[0], ax1[-1], ax2[0], ax2[-1]], aspect="auto")
        axes[0].set_title("Eclipse RTDose"); axes[0].set_xlabel(xlabel); axes[0].set_ylabel(ylabel)
        plt.colorbar(im0, ax=axes[0], label="Gy")

        im1 = axes[1].imshow(mr_slice, origin="lower", vmin=0, vmax=vmax, cmap="jet",
                             extent=[ax1[0], ax1[-1], ax2[0], ax2[-1]], aspect="auto")
        axes[1].set_title("pyMatRad dose"); axes[1].set_xlabel(xlabel)
        plt.colorbar(im1, ax=axes[1], label="Gy")

        im2 = axes[2].imshow(diff, origin="lower", cmap="bwr",
                             vmin=-vmax * 0.1, vmax=vmax * 0.1,
                             extent=[ax1[0], ax1[-1], ax2[0], ax2[-1]], aspect="auto")
        axes[2].set_title("Difference (matRad − Eclipse)"); axes[2].set_xlabel(xlabel)
        plt.colorbar(im2, ax=axes[2], label="Gy")

        fig.suptitle(f"{plan_name} — {name}")
        fig.tight_layout()
        out = os.path.join(out_dir, f"{plan_name}_{name}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(plan_name: str, ct_dir: str = None, calc_dose: bool = True,
        eclipse_fluence: bool = False):
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
    result = import_dicom(plan_dir, ct_dir=ct_dir_abs)
    ct          = result["ct"]
    cst         = result["cst"]
    pln         = result["pln"]
    dose_eclipse = result["dose"]

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
    pln["propDoseCalc"].update({
        "doseGrid":              {"resolution": {"x": 5.0, "y": 5.0, "z": 5.0}},
        "ignoreOutsideDensities": False,
        "numWorkers":            1,
    })

    # ── 3. Generate STF ──────────────────────────────────────────────────
    print("\nGenerating beam geometry (STF) ...")
    stf = matRad.generate_stf(ct, cst, pln)
    total_bixels = sum(b["totalNumOfBixels"] for b in stf)
    print(f"  {len(stf)} beams, {total_bixels} bixels total")

    # ── 4. Dose influence matrix ─────────────────────────────────────────
    print("\nCalculating dose influence matrix ...")
    dij = matRad.calc_dose_influence(ct, cst, stf, pln)

    if eclipse_fluence:
        # ── 5a. Reproduce Eclipse dose using imported MLC leaf sequences ──
        print("\nImporting Eclipse MLC fluence ...")
        plan_file = os.path.join(plan_dir, "plan.dcm")
        w = matRad.dicom.import_rtplan_fluence(plan_file, stf)
        print(f"  Weight vector: {len(w)} bixels  non-zero: {np.sum(w>0)}")
        print(f"  Total MU in w: {w.sum():.1f}")

        print("\nComputing dose from Eclipse fluence ...")
        result_matrad = matRad.calc_dose_direct(dij, w)
        label = "matRad (Eclipse fluence)"
    else:
        # ── 5b. Independent matRad fluence optimisation ───────────────────
        print("\nOptimising fluence ...")
        result_matrad = matRad.fluence_optimization(dij, cst, pln)
        label = "matRad (re-optimised)"

    dose_matrad = result_matrad["physicalDose"]
    print(f"  {label} max dose: {dose_matrad.max():.3f} Gy")

    # ── 6. Comparison ────────────────────────────────────────────────────
    if dose_eclipse is not None:
        diff = dose_matrad - dose_eclipse
        mask = dose_eclipse > 0.05 * dose_eclipse.max()
        print(f"\nDose comparison (voxels with Eclipse dose > 5% max):")
        print(f"  Eclipse max  : {dose_eclipse.max():.3f} Gy")
        print(f"  {label:30s} max: {dose_matrad.max():.3f} Gy")
        print(f"  Mean |diff|  : {np.abs(diff[mask]).mean():.3f} Gy  "
              f"({np.abs(diff[mask]).mean() / dose_eclipse[mask].mean() * 100:.1f}%)")
        print(f"  Max  |diff|  : {np.abs(diff[mask]).max():.3f} Gy")

        suffix = "_eclipse_fluence" if eclipse_fluence else "_reoptimised"
        out_dir = os.path.join(ROOT, "examples", "dicom_comparison_plots")
        dose_comparison_plots(dose_eclipse, dose_matrad, ct, out_dir,
                              plan_name + suffix)
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
    args = parser.parse_args()

    run(args.plan, ct_dir=args.ct_dir, calc_dose=not args.no_dose_calc,
        eclipse_fluence=args.eclipse_fluence)
