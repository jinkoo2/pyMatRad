"""
Validate TrueBeam photon machine files in pyMatRad.

Python port of my_scripts/truebeam_xxx/s7_validate_pdd_profiles_3x3.m  (3×3 cm²)
                                     /s8_validate_pdd_profiles_10x10.m (10×10 cm²)
                                     /s9_validate_pdd_profiles_20x20.m (20×20 cm²)

For each of the 12 cases (4 energies × 3 field sizes) the script:
  1. Builds a compact water-phantom CT
  2. Runs the pyMatRad SVD photon dose engine with uniform bixel weights
  3. Extracts the central-axis PDD and crossline profiles
  4. Loads the corresponding GBD reference curves
  5. Loads the MATLAB s7/s8/s9 result .mat files (when available)
  6. Saves comparison PNGs to  examples/validation_plots/

Usage
-----
  conda activate scikit-learn
  cd /path/to/pyMatRad
  python examples/validate_truebeam.py
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GBD_ROOT    = "/gpfs/projects/KimGroup/projects/tps/matRad/my_scripts/TrueBeamGBD"
MATRAD_DIR  = "/gpfs/projects/KimGroup/projects/tps/matRad"
PLOT_DIR    = os.path.join(ROOT, "examples", "validation_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

RES = 2   # mm isotropic voxel size


# ============================================================
# Case definitions
# ============================================================

ENERGY_CONFIGS = {
    "TrueBeam_6X": {
        "gbd_subdir":       "6MV Beam Data",
        "dd_col":           "10x10cm2",        # depth-dose column; overridden per case
        "prof_col":         "Field Size: 10x10 cm2",
        "shallow_prof_cm":  1.5,
        "shallow_prof_file":"Open Field Profiles at 1.5cm.csv",
    },
    "TrueBeam_6XFFF": {
        "gbd_subdir":       "6FFF Beam Data",
        "dd_col":           "10x10cm2",
        "prof_col":         "Field Size: 10x10 cm2",
        "shallow_prof_cm":  1.5,
        "shallow_prof_file":"Open Field Profiles at 1.5cm.csv",
    },
    "TrueBeam_10XFFF": {
        "gbd_subdir":       "10FFF Beam Data",
        "dd_col":           "10x10cm2",
        "prof_col":         "Field Size: 10x10 cm2",
        "shallow_prof_cm":  2.4,
        "shallow_prof_file":"Open Field Profiles at 2.4cm.csv",
    },
    "TrueBeam_15X": {
        "gbd_subdir":       "15MV Beam Data",
        "dd_col":           "10x10cm2",
        "prof_col":         "Field Size: 10x10 cm2",
        "shallow_prof_cm":  3.0,
        "shallow_prof_file":"Open Field Profiles at 3cm.csv",
    },
}

FIELD_CONFIGS = {
    "3x3": {
        "Nx": 60, "Ny": 160, "Nz": 60,
        "target_half_mm": 15,
        "bixelWidth": 2,
        "dd_col_tag":   "3x3cm2",
        "prof_col_tag": "Field Size: 3x3 cm2",
        "mat_prefix":   "s7",
        "mat_suffix":   "3x3",
    },
    "10x10": {
        "Nx": 130, "Ny": 160, "Nz": 130,
        "target_half_mm": 50,
        "bixelWidth": 5,
        "dd_col_tag":   "10x10cm2",
        "prof_col_tag": "Field Size: 10x10 cm2",
        "mat_prefix":   "s8",
        "mat_suffix":   "10x10",
    },
    
    # commened out because killed from OOM.
    # "20x20": {
    #     "Nx": 200, "Ny": 160, "Nz": 200,
    #     "target_half_mm": 100,
    #     "bixelWidth": 5,
    #     "dd_col_tag":   "20x20cm2",
    #     "prof_col_tag": "Field Size: 20x20 cm2",
    #     "mat_prefix":   "s9",
    #     "mat_suffix":   "20x20",
    # },
    
}

# Profile depths [cm]: first depth is energy-specific, rest are common
_COMMON_DEPTHS_CM = [5.0, 10.0, 20.0, 30.0]
PROFILE_FILES_CM = {   # (depth_cm, filename) for each energy
    "TrueBeam_6X":    [(1.5,  "Open Field Profiles at 1.5cm.csv"),
                       (5.0,  "Open Field Profiles at 5cm.csv"),
                       (10.0, "Open Field Profiles at 10cm.csv"),
                       (20.0, "Open Field Profiles at 20cm.csv"),
                       (30.0, "Open Field Profiles at 30cm.csv")],
    "TrueBeam_6XFFF": [(1.5,  "Open Field Profiles at 1.5cm.csv"),
                       (5.0,  "Open Field Profiles at 5cm.csv"),
                       (10.0, "Open Field Profiles at 10cm.csv"),
                       (20.0, "Open Field Profiles at 20cm.csv"),
                       (30.0, "Open Field Profiles at 30cm.csv")],
    "TrueBeam_10XFFF":[(2.4,  "Open Field Profiles at 2.4cm.csv"),
                       (5.0,  "Open Field Profiles at 5cm.csv"),
                       (10.0, "Open Field Profiles at 10cm.csv"),
                       (20.0, "Open Field Profiles at 20cm.csv"),
                       (30.0, "Open Field Profiles at 30cm.csv")],
    "TrueBeam_15X":   [(3.0,  "Open Field Profiles at 3cm.csv"),
                       (5.0,  "Open Field Profiles at 5cm.csv"),
                       (10.0, "Open Field Profiles at 10cm.csv"),
                       (20.0, "Open Field Profiles at 20cm.csv"),
                       (30.0, "Open Field Profiles at 30cm.csv")],
}

# Energy abbreviation for MATLAB result filenames
_ENERGY_ABBREV = {
    "TrueBeam_6X":     "6x",
    "TrueBeam_6XFFF":  "6xfff",
    "TrueBeam_10XFFF": "10xfff",
    "TrueBeam_15X":    "15x",
}


# ============================================================
# GBD reference loading
# ============================================================

def load_gbd_pdd(gbd_dir, col_tag, ssd_mm=1000.0):
    """Load PDD from GBD depth-dose CSV. Returns (depths_mm, pdd_norm_pct)."""
    fpath = os.path.join(gbd_dir, "Open Field Depth Dose.csv")
    df    = pd.read_csv(fpath, skiprows=5)
    depths_mm = df.iloc[:, 0].values.astype(float) * 10.0
    if col_tag not in df.columns:
        raise ValueError(f"Column '{col_tag}' not found in {fpath}. "
                         f"Available: {list(df.columns)}")
    pdd = df[col_tag].values.astype(float)
    pdd_norm = 100.0 * pdd / np.nanmax(pdd)
    return depths_mm, pdd_norm


def load_gbd_profile(fpath, col_tag):
    """
    Load one lateral profile from a GBD profiles CSV.
    Returns (x_cm, dose_norm_pct) normalised so that CAX=100%.
    """
    df = pd.read_csv(fpath, skiprows=7)
    if col_tag not in df.columns:
        raise ValueError(f"Column '{col_tag}' not found. "
                         f"Available: {list(df.columns)}")
    x_cm = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
    dose = pd.to_numeric(df[col_tag], errors="coerce").values
    ok   = ~np.isnan(x_cm) & ~np.isnan(dose)
    x_cm, dose = x_cm[ok], dose[ok]
    cax = float(np.interp(0.0, x_cm, dose))
    return x_cm, 100.0 * dose / cax


# ============================================================
# MATLAB results loader
# ============================================================

def load_matlab_results(mat_path):
    """Load a MATLAB s7/s8/s9 result file. Returns dict or None."""
    if not os.path.isfile(mat_path):
        return None
    try:
        raw = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        result = {}
        for k in ["depth_calc_mm", "pdd_calc_norm",
                  "x_calc_cm", "prof_calc_norm",
                  "depth_meas_mm", "pdd_meas_norm",
                  "depth_used_mm", "profile_depths_cm"]:
            if k in raw:
                result[k] = np.asarray(raw[k])
        # prof_calc_norm is a cell array → list of arrays
        if "prof_calc_norm" in result:
            pcn = result["prof_calc_norm"]
            if pcn.dtype == object:
                result["prof_calc_norm"] = [np.asarray(pcn[i]).ravel()
                                            for i in range(len(pcn))]
            else:
                result["prof_calc_norm"] = [pcn.ravel()]
        return result
    except Exception as e:
        print(f"  Warning: could not load {mat_path}: {e}")
        return None


# ============================================================
# pyMatRad dose calculation
# ============================================================

def build_water_phantom(Nx, Ny, Nz, res, target_half_mm, profile_depths_mm=None):
    """
    Build a water phantom CT matching the MATLAB s7/s8/s9 setup.

    CT axes:
      x: [-Nx/2 .. Nx/2-1] * res   (laterally centred)
      y: [0 .. Ny-1] * res          (depth; surface at y=0)
      z: [-Nz/2 .. Nz/2-1] * res   (laterally centred)
    cubeDim = [Ny, Nx, Nz]  (MATLAB/matRad convention)

    Full VOI: all voxels are included in the dose grid for accurate
    profile and PDD extraction at any depth.
    """
    x = (np.arange(Nx) - Nx // 2) * float(res)
    y = np.arange(Ny) * float(res)
    z = (np.arange(Nz) - Nz // 2) * float(res)

    ct = {
        "x": x, "y": y, "z": z,
        "resolution": {"x": float(res), "y": float(res), "z": float(res)},
        "cubeDim":    [Ny, Nx, Nz],
        "numOfCtScen": 1,
        "cubeHU": [np.zeros((Ny, Nx, Nz), dtype=np.float32)],
        "cube":   [np.ones( (Ny, Nx, Nz), dtype=np.float32)],
    }

    # ----- PTV: full 3D box from surface to 160mm depth, field footprint -----
    ix_tgt = np.where((x >= -target_half_mm) & (x <= target_half_mm))[0]
    iz_tgt = np.where((z >= -target_half_mm) & (z <= target_half_mm))[0]
    iy_tgt = np.arange(len(y))   # all depths
    mask_ptv = np.zeros((Ny, Nx, Nz), dtype=bool)
    iy_g, ix_g, iz_g = np.meshgrid(iy_tgt, ix_tgt, iz_tgt, indexing="ij")
    mask_ptv[iy_g, ix_g, iz_g] = True
    V_target = np.where(mask_ptv.ravel(order="F"))[0] + 1   # 1-based Fortran

    # ----- OAR (Water): full phantom volume -----
    V_oar = np.arange(1, Ny * Nx * Nz + 1, dtype=np.int64)

    meta_oar = {"Priority": 2, "Visible": 1, "visibleColor": [0, 0.5, 0],
                "alphaX": 0.1, "betaX": 0.05, "TissueClass": 1}
    meta_ptv = {"Priority": 1, "Visible": 1, "visibleColor": [1, 0, 0],
                "alphaX": 0.1, "betaX": 0.05, "TissueClass": 1}

    cst = [
        [0, "Water",  "OAR",    [V_oar],    meta_oar, []],
        [1, "PTV",    "TARGET", [V_target], meta_ptv, []],
    ]
    return ct, cst


def run_pymatrad(energy_name, field_cfg, verbose=True):
    """
    Run pyMatRad SVD dose calculation for one energy/field-size case.
    Returns dict with depth_mm, pdd_norm, x_cm, profiles_norm, depth_used_mm.
    """
    from matRad.geometry.geometry import get_world_axes
    from matRad.steering.stf_generator import generate_stf
    from matRad.doseCalc.calc_dose_influence import calc_dose_influence

    Nx = field_cfg["Nx"]
    Ny = field_cfg["Ny"]
    Nz = field_cfg["Nz"]
    bw = field_cfg["bixelWidth"]
    th = field_cfg["target_half_mm"]

    ct, cst = build_water_phantom(Nx, Ny, Nz, RES, th)
    ct = get_world_axes(ct)

    pln = {
        "radiationMode":  "photons",
        "machine":        energy_name,
        "bioModel":       "none",
        "multScen":       "nomScen",
        "numOfFractions": 1,
        "propStf": {
            "gantryAngles": [0],
            "couchAngles":  [0],
            "bixelWidth":   bw,
            "isoCenter":    [[0.0, 0.0, 0.0]],
            "addMargin":    False,
        },
        "propDoseCalc": {
            "doseGrid": {"resolution": {"x": RES, "y": RES, "z": RES}},
            "useCustomPrimaryPhotonFluence": True,
            "enableDijSampling":            False,
        },
    }

    if verbose:
        print(f"    Generating STF …")
    stf = generate_stf(ct, cst, pln)
    n_bixels = sum(b["totalNumOfBixels"] for b in stf)
    if verbose:
        print(f"    STF: {stf[0]['numOfRays']} rays, {n_bixels} bixels")

    if verbose:
        print(f"    Calculating dose …")
    t0  = time.perf_counter()
    dij = calc_dose_influence(ct, cst, stf, pln)
    dt  = time.perf_counter() - t0
    if verbose:
        print(f"    Done in {dt:.1f}s")

    # Uniform weights → open field dose
    D        = dij["physicalDose"][0].tocsc()
    w        = np.ones(D.shape[1])
    dose_flat = np.asarray(D @ w).ravel()   # (n_dose_voxels,)

    # Reshape to dose grid
    dg   = dij["doseGrid"]
    dims = dg["dimensions"]   # [Ny, Nx, Nz]
    dose_cube = dose_flat.reshape(dims[0], dims[1], dims[2], order="F")

    x_dg = np.asarray(dg["x"]).ravel()
    y_dg = np.asarray(dg["y"]).ravel()
    z_dg = np.asarray(dg["z"]).ravel()

    # Isocenter index
    ix0 = int(np.argmin(np.abs(x_dg)))
    iy0 = int(np.argmin(np.abs(y_dg)))
    iz0 = int(np.argmin(np.abs(z_dg)))

    # PDD: central axis along y
    pdd_raw  = dose_cube[:, ix0, iz0].astype(float)
    pdd_max  = float(np.max(pdd_raw))
    pdd_norm = 100.0 * pdd_raw / pdd_max if pdd_max > 0 else pdd_raw

    # Profiles at specified depths
    depth_targets_cm = [d for d, _ in PROFILE_FILES_CM[energy_name]]
    profiles_norm  = []
    depth_used_mm  = []
    for d_cm in depth_targets_cm:
        iy_d = int(np.argmin(np.abs(y_dg - d_cm * 10.0)))
        depth_used_mm.append(float(y_dg[iy_d]))
        prof_raw = dose_cube[iy_d, :, iz0].astype(float)
        cax      = float(dose_cube[iy_d, ix0, iz0])
        profiles_norm.append(100.0 * prof_raw / cax if cax > 0 else prof_raw)

    return {
        "depth_mm":     y_dg,
        "pdd_norm":     pdd_norm,
        "x_cm":         x_dg / 10.0,
        "profiles_norm":profiles_norm,
        "depth_used_mm":np.array(depth_used_mm),
        "profile_depths_cm": np.array(depth_targets_cm),
    }


# ============================================================
# Plotting
# ============================================================

def _fwhm(x, y):
    """Full-width at half-maximum of a profile (x monotone, CAX=100%)."""
    x, y = np.asarray(x), np.asarray(y)
    above = y >= 50
    tr = np.diff(above.astype(int))
    rise = np.where(tr == 1)[0]
    fall = np.where(tr == -1)[0]
    if len(rise) == 0 or len(fall) == 0:
        return np.nan
    r = rise[0];  f = fall[-1]
    xl = float(np.interp(50.0, [y[r], y[r+1]], [x[r], x[r+1]]))
    xr = float(np.interp(50.0, [y[f+1], y[f]], [x[f+1], x[f]]))
    return xr - xl


def save_comparison_plots(case_name, py_res, gbd_ref, matlab_res, plot_dir):
    """
    Save two PNG files per case:
      case_name_pdd.png     – PDD comparison
      case_name_profiles.png – all 5 profile depths on one figure
    """
    # ---- PDD ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gbd_ref["depths_mm"], gbd_ref["pdd_norm"],
            "b-",  lw=2,    label="GBD measured")
    ax.plot(py_res["depth_mm"],  py_res["pdd_norm"],
            "r--", lw=2,    label="pyMatRad")
    if matlab_res is not None:
        ax.plot(matlab_res["depth_calc_mm"], matlab_res["pdd_calc_norm"],
                "g:",  lw=2, label="MATLAB matRad")
    ax.set_xlabel("Depth [mm]");  ax.set_ylabel("Relative Dose [%]")
    ax.set_title(f"PDD – {case_name}")
    ax.set_xlim(0, 320);  ax.set_ylim(0, 110)
    ax.legend();  ax.grid(True)
    pdd_path = os.path.join(plot_dir, f"{case_name}_pdd.png")
    fig.savefig(pdd_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- Profiles ----
    n_depths = len(py_res["profile_depths_cm"])
    fig, axes = plt.subplots(1, n_depths, figsize=(4 * n_depths, 4), sharey=True)
    if n_depths == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        d_cm  = float(py_res["profile_depths_cm"][k])
        d_mm  = float(py_res["depth_used_mm"][k])

        ax.plot(gbd_ref["x_cm"][k], gbd_ref["profiles_norm"][k],
                "b-",  lw=1.5, label="GBD" if k == 0 else "")
        ax.plot(py_res["x_cm"], py_res["profiles_norm"][k],
                "r--", lw=1.5, label="pyMatRad" if k == 0 else "")
        if matlab_res is not None and k < len(matlab_res.get("prof_calc_norm", [])):
            ax.plot(matlab_res["x_calc_cm"], matlab_res["prof_calc_norm"][k],
                    "g:",  lw=1.5, label="MATLAB" if k == 0 else "")

        # field-edge markers
        half_cm = float(case_name.split("_")[2].split("x")[0]) / 2.0
        ax.axvline(-half_cm, color="k", lw=0.7, ls=":")
        ax.axvline( half_cm, color="k", lw=0.7, ls=":")

        ax.set_title(f"{d_cm:.1f} cm\n(grid {d_mm:.0f} mm)")
        ax.set_xlabel("Off-axis [cm]")
        ax.set_xlim(-max(8.0, half_cm * 2.5), max(8.0, half_cm * 2.5))
        ax.set_ylim(0, 120)
        ax.grid(True)

    axes[0].set_ylabel("Relative Dose [%]")
    # Global legend in first subplot
    handles = [
        plt.Line2D([0],[0], color="b", lw=1.5, label="GBD"),
        plt.Line2D([0],[0], color="r", lw=1.5, ls="--", label="pyMatRad"),
    ]
    if matlab_res is not None:
        handles.append(plt.Line2D([0],[0], color="g", lw=1.5, ls=":", label="MATLAB"))
    axes[0].legend(handles=handles, fontsize=7)

    fig.suptitle(f"Crossline profiles – {case_name}", fontsize=11)
    fig.tight_layout()
    prof_path = os.path.join(plot_dir, f"{case_name}_profiles.png")
    fig.savefig(prof_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return pdd_path, prof_path


def print_summary(case_name, py_res, gbd_ref, matlab_res):
    """Print PDD and profile quantitative summary."""
    print(f"\n  === {case_name} summary ===")

    # PDD
    check_depths_cm = [5, 10, 20, 30]
    print("  PDD (depth → pyMatRad vs GBD" +
          (" vs MATLAB)" if matlab_res else ")"))
    for d_cm in check_depths_cm:
        pv = float(np.interp(d_cm*10, py_res["depth_mm"],  py_res["pdd_norm"]))
        gv = float(np.interp(d_cm*10, gbd_ref["depths_mm"], gbd_ref["pdd_norm"]))
        if matlab_res is not None:
            mv = float(np.interp(d_cm*10, matlab_res["depth_calc_mm"],
                                 matlab_res["pdd_calc_norm"]))
            print(f"    {d_cm:2d}cm: py={pv:5.1f}%  GBD={gv:5.1f}%  "
                  f"Δ(py-GBD)={pv-gv:+.1f}%  "
                  f"MATLAB={mv:5.1f}%  Δ(py-MAT)={pv-mv:+.1f}%")
        else:
            print(f"    {d_cm:2d}cm: py={pv:5.1f}%  GBD={gv:5.1f}%  Δ={pv-gv:+.1f}%")

    # FWHM per profile depth
    print("  Profiles (FWHM: pyMatRad vs GBD" +
          (" vs MATLAB)" if matlab_res else ")"))
    for k, d_cm in enumerate(py_res["profile_depths_cm"]):
        fw_py  = _fwhm(py_res["x_cm"], py_res["profiles_norm"][k])
        fw_gbd = _fwhm(gbd_ref["x_cm"][k], gbd_ref["profiles_norm"][k])
        if matlab_res is not None and k < len(matlab_res.get("prof_calc_norm", [])):
            fw_mat = _fwhm(matlab_res["x_calc_cm"], matlab_res["prof_calc_norm"][k])
            print(f"    {d_cm:.1f}cm: FWHM py={fw_py:.2f}cm  GBD={fw_gbd:.2f}cm  "
                  f"Δ(py-GBD)={fw_py-fw_gbd:+.2f}cm  "
                  f"MATLAB={fw_mat:.2f}cm  Δ(py-MAT)={fw_py-fw_mat:+.2f}cm")
        else:
            print(f"    {d_cm:.1f}cm: FWHM py={fw_py:.2f}cm  GBD={fw_gbd:.2f}cm  "
                  f"Δ={fw_py-fw_gbd:+.2f}cm")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 65)
    print("  TrueBeam machine validation  —  pyMatRad")
    print("=" * 65)
    print(f"  Plots → {PLOT_DIR}")

    total_cases = len(ENERGY_CONFIGS) * len(FIELD_CONFIGS)
    done = 0
    t_start = time.perf_counter()

    for energy_name in ENERGY_CONFIGS:
        ecfg = ENERGY_CONFIGS[energy_name]
        abbrev = _ENERGY_ABBREV[energy_name]
        gbd_dir = os.path.join(GBD_ROOT, ecfg["gbd_subdir"])

        for field_size, fcfg in FIELD_CONFIGS.items():
            done += 1
            case_name = f"{energy_name}_{field_size}"
            print(f"\n[{done}/{total_cases}] {case_name}")

            # ---- Load MATLAB result (may not exist yet) ----
            mat_fname = (f"{fcfg['mat_prefix']}_water_phantom_{fcfg['mat_suffix']}"
                         f"_{abbrev}_results.mat")
            mat_path  = os.path.join(MATRAD_DIR, mat_fname)
            matlab_res = load_matlab_results(mat_path)
            if matlab_res:
                print(f"  MATLAB result: {mat_fname} ✓")
            else:
                print(f"  MATLAB result: {mat_fname} (not found – skipped in plot)")

            # ---- Load GBD reference ----
            print(f"  Loading GBD reference …")
            depths_mm, pdd_norm = load_gbd_pdd(gbd_dir, fcfg["dd_col_tag"])
            profiles_norm_gbd = []
            x_cm_gbd          = []
            for d_cm, fname in PROFILE_FILES_CM[energy_name]:
                fpath = os.path.join(gbd_dir, fname)
                xc, pn = load_gbd_profile(fpath, fcfg["prof_col_tag"])
                x_cm_gbd.append(xc)
                profiles_norm_gbd.append(pn)
            gbd_ref = {
                "depths_mm":     depths_mm,
                "pdd_norm":      pdd_norm,
                "x_cm":          x_cm_gbd,
                "profiles_norm": profiles_norm_gbd,
            }

            # ---- pyMatRad calculation ----
            print(f"  Running pyMatRad …")
            py_res = run_pymatrad(energy_name, fcfg, verbose=True)

            # ---- Plots ----
            pdd_png, prof_png = save_comparison_plots(
                case_name, py_res, gbd_ref, matlab_res, PLOT_DIR
            )
            print(f"  Saved: {os.path.basename(pdd_png)}")
            print(f"  Saved: {os.path.basename(prof_png)}")

            # ---- Summary ----
            print_summary(case_name, py_res, gbd_ref, matlab_res)

    print(f"\n{'='*65}")
    print(f"  All {total_cases} cases done in "
          f"{(time.perf_counter()-t_start)/60:.1f} min")
    print(f"  Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
