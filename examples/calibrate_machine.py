"""
calibrate_machine.py — Compute and store the TG-51 absolute calibration factor.

TG-51 reference conditions
---------------------------
    SSD   = 100 cm (1000 mm)
    Field = 10×10 cm (100×100 mm) at isocenter
    Depth = d_max (depth of dose maximum, beam-energy dependent)
    Dose  = 1 cGy per 1 MU

The script runs a forward dose calculation on a virtual water phantom and
derives ``abs_calib`` [Gy/MU] such that the engine produces exactly 1 cGy at
d_max for 1 MU delivered with the 10×10 field at SSD = 100 cm.

Usage
-----
    python examples/calibrate_machine.py --machine TrueBeam_6X
    python examples/calibrate_machine.py --machine TrueBeam_15X --radiation-mode photons
    python examples/calibrate_machine.py --machine TrueBeam_6X --dry-run
    python examples/calibrate_machine.py --machine TrueBeam_6X --plot-dir examples/cache/calib_plots
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

import matRad
from matRad.basedata import load_machine

# ---------------------------------------------------------------------------
# TG-51 reference constants
# ---------------------------------------------------------------------------
TG51_SSD_MM   = 1000.0   # reference SSD [mm]  (100 cm)
TG51_FIELD_MM = 100.0    # square field side    (10 cm)
TG51_DOSE_CGY = 1.0      # 1 cGy per MU at d_max
TG51_MU_REF   = 1.0      # 1 MU reference delivery

# Phantom geometry constants
_PHANTOM_RES_MM  = 2.0   # voxel side [mm]
_PHANTOM_DEPTH_N = 150   # voxels in depth (y) → 300 mm
_PHANTOM_WIDTH_N = 100   # voxels in x and z  → 200 mm  (±100 mm)
# Target VOI covers a 160×300×160 mm box to ensure full field coverage
_TARGET_WIDTH_N  = 80    # voxels in x and z  → 160 mm  (±80 mm)

USER_MACHINE_DIR = os.path.join(ROOT, "userdata", "machines")


# ---------------------------------------------------------------------------
# Phantom builder
# ---------------------------------------------------------------------------

def _build_water_phantom(sad_mm: float):
    """
    Build a 200×300×200 mm water phantom suitable for TG-51 calibration.

    For a single AP beam at gantry=0 the y-axis is the depth axis.  The
    isocenter is placed on the phantom entrance surface so that

        SSD = TG51_SSD_MM = 1000 mm   (for any machine SAD).

    Returns (ct, cst, iso, y_surface_mm)
    """
    from matRad.phantoms.builder import PhantomBuilder

    res  = _PHANTOM_RES_MM
    Nx   = _PHANTOM_WIDTH_N   # 100 vox → 200 mm
    Ny   = _PHANTOM_DEPTH_N   # 150 vox → 300 mm
    Nz   = _PHANTOM_WIDTH_N

    builder = PhantomBuilder([Nx, Ny, Nz], [res, res, res])
    builder.add_box_target("CalibField", [_TARGET_WIDTH_N, Ny, _TARGET_WIDTH_N],
                           HU=0)
    ct, cst = builder.get_ct_cst()

    # Override all voxels to water (builder initialises to −1000 HU / air)
    ct["cubeHU"][0][:] = 0.0

    # World axes (get_world_axes would compute these on first use; add them now)
    ct["x"] = -Nx / 2 * res + res * np.arange(Nx)  # −100 … +98 mm
    ct["y"] = -Ny / 2 * res + res * np.arange(Ny)  # −150 … +148 mm
    ct["z"] = -Nz / 2 * res + res * np.arange(Nz)  # −100 … +98 mm

    # Phantom entrance surface (first voxel centre in y, approximately)
    y_surface = float(ct["y"][0])   # ≈ −150 mm

    # Place isocenter so that  SSD = TG51_SSD_MM for the given machine SAD:
    #   source_y  = iso_y − SAD
    #   SSD       = y_surface − source_y = y_surface − iso_y + SAD = 1000
    #   → iso_y   = y_surface + SAD − 1000
    iso_y = y_surface + (sad_mm - TG51_SSD_MM)
    iso   = np.array([0.0, iso_y, 0.0])

    return ct, cst, iso, y_surface


# ---------------------------------------------------------------------------
# Main calibration routine
# ---------------------------------------------------------------------------

def calibrate(radiation_mode: str, machine_name: str,
              dry_run: bool = False, plot_dir: str = None,
              force: bool = False):
    """
    Compute TG-51 abs_calib and store it in the .npy machine file.

    Parameters
    ----------
    radiation_mode : str   e.g. "photons"
    machine_name   : str   e.g. "TrueBeam_6X"
    dry_run        : bool  If True, report result without modifying any file.
    plot_dir       : str   Directory for PDD plot (None → no plot).
    force          : bool  Overwrite existing tg51 entry without prompting.
    """
    # ── 1. Load machine ─────────────────────────────────────────────────
    pln_probe = {"radiationMode": radiation_mode, "machine": machine_name}
    machine   = load_machine(pln_probe)
    meta      = machine.get("meta", {})
    SAD       = float(meta.get("SAD", 1000.0))
    print(f"\nMachine : {radiation_mode}_{machine_name}")
    print(f"SAD     : {SAD:.0f} mm")

    existing_tg51 = meta.get("tg51", None)
    if existing_tg51 and not force:
        print(f"\nExisting tg51 entry found: {existing_tg51}")
        ans = input("Overwrite? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return None

    # ── 2. Water phantom ─────────────────────────────────────────────────
    ct, cst, iso, y_surface = _build_water_phantom(SAD)
    Ny, Nx, Nz = ct["cubeDim"]
    print(f"\nPhantom : {Nx}×{Ny}×{Nz} vox  res={_PHANTOM_RES_MM} mm")
    print(f"y-range : {ct['y'][0]:.0f} … {ct['y'][-1]:.0f} mm  (depth axis)")
    print(f"SSD     : {y_surface - (iso[1] - SAD):.0f} mm  "
          f"iso_y={iso[1]:.0f} mm  source_y={iso[1]-SAD:.0f} mm")

    # ── 3. Plan ──────────────────────────────────────────────────────────
    bixel_width = 5.0   # mm  (standard bixel size)
    pln = {
        "radiationMode":  radiation_mode,
        "machine":        machine_name,
        "bioModel":       "none",
        "multScen":       "nomScen",
        "numOfFractions": 1,
        "propStf": {
            "gantryAngles":     [0.0],
            "couchAngles":      [0.0],
            "bixelWidth":       bixel_width,
            "isoCenter":        iso.tolist(),
            "addMargin":        False,
            "fillEmptyBixels":  False,
            "visMode":          0,
        },
        "propDoseCalc": {
            "doseGrid": {
                "resolution": {"x": _PHANTOM_RES_MM,
                               "y": _PHANTOM_RES_MM,
                               "z": _PHANTOM_RES_MM}
            },
            "numWorkers":            1,
            "ignoreOutsideDensities": False,
        },
        "propOpt": {"runDAO": False, "runSequencing": False},
    }

    # ── 4. STF ───────────────────────────────────────────────────────────
    print("\nGenerating STF ...")
    stf = matRad.generate_stf(ct, cst, pln)
    n_bixels_total = sum(b["totalNumOfBixels"] for b in stf)
    print(f"  {len(stf)} beam(s)  {n_bixels_total} bixels total")

    # ── 5. dij ───────────────────────────────────────────────────────────
    print("\nComputing dose influence matrix ...")
    dij = matRad.calc_dose_influence(ct, cst, stf, pln)
    mat = dij["physicalDose"][0]
    print(f"  dij shape={mat.shape}  nnz={mat.nnz:,}")

    # ── 6. Weight vector: 1 for bixels inside 10×10 cm field, 0 elsewhere
    half_field = TG51_FIELD_MM / 2.0   # ±50 mm
    w = np.zeros(n_bixels_total, dtype=np.float64)
    col = 0
    for beam in stf:
        for ray in beam["rays"]:
            rp = np.asarray(ray["rayPos_bev"])
            if abs(rp[0]) <= half_field and abs(rp[2]) <= half_field:
                w[col] = 1.0
            col += 1
    n_in_field = int(np.sum(w > 0))
    print(f"\n  Bixels in {TG51_FIELD_MM:.0f}×{TG51_FIELD_MM:.0f} mm field: "
          f"{n_in_field} / {n_bixels_total}")
    if n_in_field == 0:
        raise RuntimeError(
            "No bixels found inside the 10×10 cm reference field.  "
            "Check phantom dimensions and isocenter placement."
        )

    # ── 7. Forward dose ──────────────────────────────────────────────────
    print("\nComputing forward dose ...")
    result = matRad.calc_dose_direct(dij, w)
    dose   = result["physicalDose"]   # (Ny_dg, Nx_dg, Nz_dg)

    # ── 8. Central-axis depth-dose profile ──────────────────────────────
    dg    = dij["doseGrid"]
    dg_x  = np.asarray(dg["x"]).ravel()
    dg_y  = np.asarray(dg["y"]).ravel()
    dg_z  = np.asarray(dg["z"]).ravel()

    ix_iso = int(np.argmin(np.abs(dg_x - iso[0])))
    iz_iso = int(np.argmin(np.abs(dg_z - iso[2])))

    # Depth measured from phantom surface along beam axis (+y for gantry=0)
    depths    = dg_y - y_surface          # mm from entrance surface
    axis_dose = dose[:, ix_iso, iz_iso]   # central-axis dose vs depth

    # Keep only positive depths (inside phantom)
    inside    = depths >= 0.0
    depths_in = depths[inside]
    dose_in   = axis_dose[inside]

    if dose_in.size == 0 or dose_in.max() == 0.0:
        raise RuntimeError(
            "Central-axis dose is zero.  "
            "Verify the phantom is filled with water and the isocenter is set correctly."
        )

    # ── 9. d_max and abs_calib ───────────────────────────────────────────
    i_max      = int(np.argmax(dose_in))
    d_max_mm   = float(depths_in[i_max])
    dose_dmax  = float(dose_in[i_max])
    abs_calib  = (TG51_DOSE_CGY / 100.0) / dose_dmax   # 1 cGy = 0.01 Gy

    print(f"\n{'─'*55}")
    print(f"  TG-51 calibration results")
    print(f"{'─'*55}")
    print(f"  d_max            = {d_max_mm:.1f} mm")
    print(f"  dose(d_max, w=1) = {dose_dmax:.6e}  (engine units)")
    print(f"  abs_calib        = {abs_calib:.8f} Gy/MU")
    print(f"                   = {abs_calib * 100:.4f} cGy/MU")
    print(f"{'─'*55}")

    # ── 10. PDD plot ─────────────────────────────────────────────────────
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: depth-dose curve
        ax = axes[0]
        ax.plot(depths_in, dose_in / dose_dmax * 100, "b-", lw=1.8)
        ax.axvline(d_max_mm, color="r", ls="--", lw=1.2,
                   label=f"d_max = {d_max_mm:.1f} mm")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("Relative dose [%]")
        ax.set_title(f"Central-axis PDD — {radiation_mode}_{machine_name}\n"
                     f"10×10 cm  SSD = {TG51_SSD_MM/10:.0f} cm")
        ax.legend(fontsize=9)
        ax.grid(True)

        # Right: lateral profile at d_max
        iy_dmax = int(np.where(inside)[0][i_max])
        lat_x   = dg_x - iso[0]
        lat_prof = dose[iy_dmax, :, iz_iso]
        lat_prof = lat_prof / (dose_dmax if dose_dmax > 0 else 1.0) * 100
        ax2 = axes[1]
        ax2.plot(lat_x, lat_prof, "g-", lw=1.8)
        ax2.axvline(-half_field, color="r", ls="--", lw=1.0, label="±50 mm field edge")
        ax2.axvline( half_field, color="r", ls="--", lw=1.0)
        ax2.set_xlabel("Lateral x [mm]")
        ax2.set_ylabel("Relative dose [%]")
        ax2.set_title(f"Lateral profile at d_max = {d_max_mm:.1f} mm")
        ax2.legend(fontsize=9)
        ax2.grid(True)

        fig.suptitle(
            f"{radiation_mode}_{machine_name}  |  "
            f"abs_calib = {abs_calib*100:.4f} cGy/MU  |  "
            f"d_max = {d_max_mm:.1f} mm",
            fontsize=11
        )
        fig.tight_layout()
        plot_path = os.path.join(plot_dir,
                                 f"pdd_{radiation_mode}_{machine_name}.png")
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"\n  Saved PDD/profile plot → {plot_path}")

    # ── 11. Save to machine file ─────────────────────────────────────────
    tg51_entry = {
        "abs_calib":  abs_calib,
        "d_max_mm":   d_max_mm,
        "ssd_ref_mm": TG51_SSD_MM,
        "field_mm":   TG51_FIELD_MM,
        "mu_ref":     TG51_MU_REF,
    }

    if dry_run:
        print(f"\n--dry-run: would write machine[\"meta\"][\"tg51\"] = {tg51_entry}")
        print("  (no file modified)")
        return abs_calib, d_max_mm

    machine_npy = os.path.join(USER_MACHINE_DIR,
                               f"{radiation_mode}_{machine_name}.npy")
    if os.path.isfile(machine_npy):
        # Re-load the stored dict (may have been loaded from .mat earlier)
        stored = np.load(machine_npy, allow_pickle=True).item()
    else:
        # Machine was loaded from a .mat file in matRad/basedata/;
        # save a new .npy copy in userdata/machines/ so future loads prefer it.
        print(f"\n  No .npy in {USER_MACHINE_DIR}; creating one ...")
        os.makedirs(USER_MACHINE_DIR, exist_ok=True)
        stored = machine

    stored["meta"]["tg51"] = tg51_entry
    np.save(machine_npy, stored, allow_pickle=True)
    print(f"\n  Updated  → {machine_npy}")
    print(f"  tg51     = {tg51_entry}")

    return abs_calib, d_max_mm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute TG-51 abs_calib for a pyMatRad machine file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Calibrate TrueBeam 6X (photons) and update the .npy file
  python examples/calibrate_machine.py --machine TrueBeam_6X

  # Preview only (no file write)
  python examples/calibrate_machine.py --machine TrueBeam_6X --dry-run

  # Save depth-dose plot as well
  python examples/calibrate_machine.py --machine TrueBeam_6X \\
      --plot-dir examples/cache/calib_plots

  # Calibrate a custom machine for electrons
  python examples/calibrate_machine.py \\
      --machine MyLinac_6MeV --radiation-mode electrons
""",
    )
    parser.add_argument("--machine", required=True, metavar="NAME",
                        help="Machine name, e.g. TrueBeam_6X  "
                             "(file: {radiation-mode}_{NAME}.npy/.mat)")
    parser.add_argument("--radiation-mode", default="photons", metavar="MODE",
                        help="Radiation mode (default: photons)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute but do not write the result to any file.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing tg51 entry without prompting.")
    parser.add_argument("--plot-dir", default=None, metavar="DIR",
                        help="Directory to save PDD + lateral profile PNG. "
                             "Default: no plot.")
    args = parser.parse_args()

    calibrate(
        radiation_mode=args.radiation_mode,
        machine_name=args.machine,
        dry_run=args.dry_run,
        plot_dir=args.plot_dir,
        force=args.force,
    )
