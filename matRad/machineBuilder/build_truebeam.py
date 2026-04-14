"""
Build matRad photon machine files from Varian TrueBeam Golden Beam Data.

Python port of my_scripts/truebeam_xxx/s5_build_machine.m

Usage
-----
# Build all four energies (6X, 6XFFF, 10XFFF, 15X):
from matRad.machineBuilder.build_truebeam import build_all_truebeam

GBD_ROOT   = "/gpfs/projects/KimGroup/projects/tps/matRad/my_scripts/TrueBeamGBD"
OUTPUT_DIR = "/gpfs/projects/KimGroup/projects/tps/pyMatRad/userdata/machines"
build_all_truebeam(GBD_ROOT, OUTPUT_DIR)

# Or build a single energy:
build_truebeam_machine("TrueBeam_6X", GBD_ROOT, OUTPUT_DIR)
"""

import os
import numpy as np
from .read_gbd_data import read_output_factors, read_depth_dose_tpr, read_primary_fluence
from .kernel_calc import generate_machine, save_machine


# ---------------------------------------------------------------------------
# Per-energy configuration  (mirrors s1–s4 MATLAB scripts)
# ---------------------------------------------------------------------------

_CONFIGS = {
    "TrueBeam_6X": {
        "gbd_subdir":   "6MV Beam Data",
        "profile_file": "Open Field Profiles at 1.5cm.csv",
        "params": {
            "SAD": 1000.0, "photon_energy": 6.0, "fwhm_gauss": 6.0,
            "electron_range_intensity": 0.001,
            "source_collimator_distance": 345.0,
            "source_tray_distance": 565.0,
            "dose_reference_ssd": 950.0, "dose_reference_depth": 50.0,
        },
    },
    "TrueBeam_6XFFF": {
        "gbd_subdir":   "6FFF Beam Data",
        "profile_file": "Open Field Profiles at 1.5cm.csv",
        "params": {
            "SAD": 1000.0, "photon_energy": 6.0, "fwhm_gauss": 6.0,
            "electron_range_intensity": 0.001,
            "source_collimator_distance": 345.0,
            "source_tray_distance": 565.0,
            "dose_reference_ssd": 950.0, "dose_reference_depth": 50.0,
        },
    },
    "TrueBeam_10XFFF": {
        "gbd_subdir":   "10FFF Beam Data",
        "profile_file": "Open Field Profiles at 2.4cm.csv",
        "params": {
            "SAD": 1000.0, "photon_energy": 10.0, "fwhm_gauss": 6.0,
            "electron_range_intensity": 0.001,
            "source_collimator_distance": 345.0,
            "source_tray_distance": 565.0,
            "dose_reference_ssd": 950.0, "dose_reference_depth": 50.0,
        },
    },
    "TrueBeam_15X": {
        "gbd_subdir":   "15MV Beam Data",
        "profile_file": "Open Field Profiles at 3cm.csv",
        "params": {
            "SAD": 1000.0, "photon_energy": 15.0, "fwhm_gauss": 6.0,
            "electron_range_intensity": 0.001,
            "source_collimator_distance": 345.0,
            "source_tray_distance": 565.0,
            "dose_reference_ssd": 950.0, "dose_reference_depth": 50.0,
        },
    },
}


# ---------------------------------------------------------------------------
# Per-machine builder
# ---------------------------------------------------------------------------

def build_truebeam_machine(
    machine_name: str,
    gbd_root: str,
    output_dir: str,
    verbose: bool = True,
) -> dict:
    """
    Build one TrueBeam photon machine and save as a .npy file.

    Parameters
    ----------
    machine_name : "TrueBeam_6X" | "TrueBeam_6XFFF" | "TrueBeam_10XFFF" | "TrueBeam_15X"
    gbd_root     : root directory of GBD CSV data (contains sub-dirs like "6MV Beam Data")
    output_dir   : where to write the .npy file
    verbose      : print progress

    Returns
    -------
    machine : dict  (also saved to disk as photons_{machine_name}.npy)
    """
    if machine_name not in _CONFIGS:
        raise ValueError(
            f"Unknown machine '{machine_name}'. "
            f"Available: {sorted(_CONFIGS.keys())}"
        )

    cfg     = _CONFIGS[machine_name]
    bd_dir  = os.path.join(gbd_root, cfg["gbd_subdir"])
    ssd_mm  = float(cfg["params"]["SAD"])

    if verbose:
        print(f"\n=== Building {machine_name} ===")
        print(f"  Source: {bd_dir}")

    # 1. Output factors
    of_mm, of_vals = read_output_factors(
        os.path.join(bd_dir, "Open field Output Factors.csv")
    )
    if verbose:
        print(f"  OF:      {len(of_mm)} square field sizes "
              f"[{of_mm[0]:.0f}–{of_mm[-1]:.0f} mm]")

    # 2. PDD → TPR
    fs_mm, d_mm, tpr = read_depth_dose_tpr(
        os.path.join(bd_dir, "Open Field Depth Dose.csv"),
        ssd_mm=ssd_mm,
    )
    if verbose:
        print(f"  TPR:     {len(fs_mm)} field sizes, {len(d_mm)} depth points")

    # 3. Primary fluence
    pf_r, pf_vals = read_primary_fluence(
        os.path.join(bd_dir, cfg["profile_file"])
    )
    if verbose:
        print(f"  PrimFlu: {len(pf_r)} points, r=[{pf_r[0]:.1f}–{pf_r[-1]:.1f}] mm")

    # 4. Generate kernel data
    if verbose:
        print(f"  Generating kernels for 501 SSDs (500–1000 mm) …")
    machine = generate_machine(
        name=machine_name,
        params=cfg["params"],
        tpr_field_sizes_mm=fs_mm,
        tpr_depths_mm=d_mm,
        tpr=tpr,
        of_mm=of_mm,
        of_vals=of_vals,
        pf_r=pf_r,
        pf_vals=pf_vals,
    )

    # 5. Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"photons_{machine_name}.npy")
    save_machine(machine, out_path)
    return machine


# ---------------------------------------------------------------------------
# Build all four energies
# ---------------------------------------------------------------------------

def build_all_truebeam(
    gbd_root: str,
    output_dir: str,
    verbose: bool = True,
) -> dict:
    """
    Build machine files for all four TrueBeam energies.

    Returns
    -------
    machines : dict  mapping machine_name → machine dict
    """
    machines = {}
    for name in _CONFIGS:
        machines[name] = build_truebeam_machine(name, gbd_root, output_dir,
                                                verbose=verbose)
    return machines


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    _HERE = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Build TrueBeam photon machine files from GBD CSV data"
    )
    parser.add_argument(
        "--gbd-root",
        default=os.path.abspath(
            os.path.join(_HERE, "../../../../matRad/my_scripts/TrueBeamGBD")
        ),
        help="Root directory of TrueBeamGBD CSV data",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.abspath(os.path.join(_HERE, "../../../userdata/machines")),
        help="Directory for output .npy files",
    )
    parser.add_argument(
        "--machine",
        default="all",
        choices=["all"] + sorted(_CONFIGS.keys()),
        help="Which machine to build (default: all)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.machine == "all":
        build_all_truebeam(args.gbd_root, args.output_dir, verbose=not args.quiet)
    else:
        build_truebeam_machine(args.machine, args.gbd_root, args.output_dir,
                               verbose=not args.quiet)
