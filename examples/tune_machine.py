"""
tune_machine.py — Iterative parameter tuning for TrueBeam photon machine files.

Tunes five machine parameters to minimise PDD and lateral-profile differences
vs Varian Golden Beam Data (GBD) for 3×3, 10×10, and 20×20 cm² open fields.
After tuning, performs TG-51 absolute calibration (1 cGy/MU at d_max,
10×10 cm, SSD = 100 cm).

Tunable parameters
------------------
  penumbraFWHMatIso  [mm]    Gaussian source FWHM at isocenter (lateral penumbra)
  m                  [mm⁻¹]  Primary photon attenuation coefficient
  beta1              [mm⁻¹]  First scatter-kernel decay constant  (fast build-up)
  beta2              [mm⁻¹]  Second scatter-kernel decay constant (mid scatter)
  beta3              [mm⁻¹]  Third scatter-kernel decay constant  (tail scatter)

Note on betas and kernels
--------------------------
The depth-dose component in the SVD engine is evaluated at run-time using the
stored m and betas; the lateral kernel weights (kernel1–3) were pre-computed
from the TPR data and are NOT rebuilt by this script.  Tuning m/betas therefore
adjusts the depth-dose envelope while leaving the relative lateral spread intact.
To also change lateral kernel shapes, re-run machineBuilder/build_truebeam.py
with adjusted TPR data and a new fwhm_gauss.

Usage
-----
    # Tune all four TrueBeam energies (sequential, may take several hours)
    python examples/tune_machine.py

    # Tune one energy
    python examples/tune_machine.py --machine TrueBeam_6X

    # Only use 10×10 field for the optimizer (fastest, recommended first pass)
    python examples/tune_machine.py --machine TrueBeam_6X --field 10x10

    # Dry run: compute baseline error, print it, do not optimise or save
    python examples/tune_machine.py --machine TrueBeam_6X --dry-run

    # Cap number of optimizer iterations (default 60)
    python examples/tune_machine.py --machine TrueBeam_6X --max-iter 30

    # Save comparison plots after tuning
    python examples/tune_machine.py --plot-dir examples/cache/tune_plots
"""

import os
import sys
import copy
import time
import argparse
import textwrap
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Pull the shared validation infrastructure from validate_truebeam
# ---------------------------------------------------------------------------
import importlib.util as _ilu
_vt_path = os.path.join(ROOT, "examples", "validate_truebeam.py")
_spec = _ilu.spec_from_file_location("validate_truebeam", _vt_path)
_vt = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_vt)

ENERGY_CONFIGS   = _vt.ENERGY_CONFIGS
FIELD_CONFIGS    = _vt.FIELD_CONFIGS
PROFILE_FILES_CM = _vt.PROFILE_FILES_CM
build_water_phantom  = _vt.build_water_phantom
load_gbd_pdd         = _vt.load_gbd_pdd
load_gbd_profile     = _vt.load_gbd_profile

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GBD_ROOT         = os.path.join(ROOT, "..", "matRad", "my_scripts", "TrueBeamGBD")
USER_MACHINE_DIR = os.path.join(ROOT, "userdata", "machines")
TUNE_CACHE_DIR   = os.path.join(ROOT, "examples", "cache", "tune_cache")
_TEMP_MACHINE    = "TUNING_TEMP"     # temporary machine name (no energy in name —
                                      # we always delete before re-use)
RES = 2   # mm  (must match validate_truebeam.py)

# ---------------------------------------------------------------------------
# Default optimisation bounds
# ---------------------------------------------------------------------------
_PARAM_BOUNDS = {
    "fwhm":  (3.0,   15.0),    # penumbraFWHMatIso [mm]
    "m":     (0.001,  0.08),   # attenuation [mm⁻¹]
    "beta1": (0.02,   5.0),    # build-up decay [mm⁻¹]
    "beta2": (0.005,  2.0),    # mid-scatter decay
    "beta3": (0.001,  1.0),    # tail-scatter decay
}

# Depths at which PDD errors are evaluated [cm]
_PDD_CHECK_DEPTHS_CM = [5.0, 10.0, 20.0, 30.0]

# Field sizes included in the objective (key names from FIELD_CONFIGS)
_ALL_FIELDS = ["3x3", "10x10", "20x20"]

# ---------------------------------------------------------------------------
# Machine patching helpers
# ---------------------------------------------------------------------------

def _patch_machine(base_machine: dict,
                   fwhm: float, m: float, betas: np.ndarray) -> dict:
    """Return a deep copy of *base_machine* with the five tunable parameters replaced."""
    machine = copy.deepcopy(base_machine)
    machine["data"]["penumbraFWHMatIso"] = float(fwhm)
    machine["data"]["fwhm"]             = float(fwhm)   # some readers use this key
    machine["data"]["m"]                = float(m)
    machine["data"]["betas"]            = np.asarray(betas, dtype=float).ravel()
    return machine


def _save_temp_machine(machine: dict) -> str:
    """
    Save *machine* as ``photons_TUNING_TEMP.npy`` in userdata/machines/.
    Returns the file path.
    """
    os.makedirs(USER_MACHINE_DIR, exist_ok=True)
    path = os.path.join(USER_MACHINE_DIR, f"photons_{_TEMP_MACHINE}.npy")
    np.save(path, machine, allow_pickle=True)
    return path


def _delete_temp_machine():
    path = os.path.join(USER_MACHINE_DIR, f"photons_{_TEMP_MACHINE}.npy")
    if os.path.isfile(path):
        os.remove(path)


# ---------------------------------------------------------------------------
# Dose calculation with a given machine dict
# ---------------------------------------------------------------------------

def _run_dose(machine_dict: dict, field_cfg: dict,
              depth_targets_cm: list, verbose: bool = False) -> dict:
    """
    Run pyMatRad SVD dose for one field-size case using *machine_dict*.

    Saves the patched machine to a temp .npy, points the plan at it,
    runs dij, computes open-field dose, and extracts PDD + profiles.

    Returns
    -------
    dict with keys:
        depth_mm        : (Ny,) depth axis [mm]
        pdd_norm        : (Ny,) normalised PDD [%]
        x_cm            : (Nx,) lateral axis [cm]
        profiles_norm   : list of (Nx,) arrays, one per depth
        depth_used_mm   : (n_depths,) actual grid depths used
    """
    from matRad.geometry.geometry import get_world_axes
    from matRad.steering.stf_generator import generate_stf
    from matRad.doseCalc.calc_dose_influence import calc_dose_influence

    Nx = field_cfg["Nx"]
    Ny = field_cfg["Ny"]
    Nz = field_cfg["Nz"]
    bw = field_cfg["bixelWidth"]
    th = field_cfg["target_half_mm"]

    profile_depths_mm = [d * 10.0 for d in depth_targets_cm]

    ct, cst = build_water_phantom(Nx, Ny, Nz, RES, th,
                                  profile_depths_mm=profile_depths_mm)
    ct = get_world_axes(ct)

    # Save patched machine, plan pointing to it
    _save_temp_machine(machine_dict)

    pln = {
        "radiationMode":  "photons",
        "machine":        _TEMP_MACHINE,
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
            "ignoreOutsideDensities":       False,
            "numWorkers":                   1,
        },
    }

    try:
        stf  = generate_stf(ct, cst, pln)
        dij  = calc_dose_influence(ct, cst, stf, pln)
    finally:
        _delete_temp_machine()

    # Forward dose with uniform weights
    D        = dij["physicalDose"][0].tocsc()
    w        = np.ones(D.shape[1])
    dose_flat = np.asarray(D @ w).ravel()

    dg   = dij["doseGrid"]
    dims = dg["dimensions"]
    dose_cube = dose_flat.reshape(dims[0], dims[1], dims[2], order="F")

    x_dg = np.asarray(dg["x"]).ravel()
    y_dg = np.asarray(dg["y"]).ravel()
    z_dg = np.asarray(dg["z"]).ravel()

    ix0 = int(np.argmin(np.abs(x_dg)))
    iz0 = int(np.argmin(np.abs(z_dg)))

    pdd_raw  = dose_cube[:, ix0, iz0].astype(float)
    pdd_max  = float(np.max(pdd_raw))
    pdd_norm = 100.0 * pdd_raw / pdd_max if pdd_max > 0 else pdd_raw

    profiles_norm = []
    depth_used_mm = []
    for d_cm in depth_targets_cm:
        iy_d = int(np.argmin(np.abs(y_dg - d_cm * 10.0)))
        depth_used_mm.append(float(y_dg[iy_d]))
        prof_raw = dose_cube[iy_d, :, iz0].astype(float)
        cax      = float(dose_cube[iy_d, ix0, iz0])
        profiles_norm.append(100.0 * prof_raw / cax if cax > 0 else prof_raw)

    return {
        "depth_mm":      y_dg,
        "pdd_norm":      pdd_norm,
        "x_cm":          x_dg / 10.0,
        "profiles_norm": profiles_norm,
        "depth_used_mm": np.array(depth_used_mm),
    }


# ---------------------------------------------------------------------------
# Error metric
# ---------------------------------------------------------------------------

def _compute_error(py_res: dict, gbd_pdd_depths: np.ndarray,
                   gbd_pdd: np.ndarray, gbd_profiles: list,
                   gbd_x_cm: list, field_half_cm: float,
                   depth_targets_cm: list,
                   pdd_weight: float = 1.0,
                   profile_weight: float = 0.5) -> float:
    """
    Compute weighted RMSE between pyMatRad output and GBD reference.

    PDD error: RMSE at _PDD_CHECK_DEPTHS_CM (in %), deeper depths weighted 2×.
    Profile error: RMSE of within-field dose relative to GBD (in %).

    Returns a single non-negative float (lower is better).
    """
    # ---- PDD ----
    pdd_errors = []
    for i, d_cm in enumerate(_PDD_CHECK_DEPTHS_CM):
        pv = float(np.interp(d_cm * 10, py_res["depth_mm"], py_res["pdd_norm"]))
        gv = float(np.interp(d_cm * 10, gbd_pdd_depths, gbd_pdd))
        # Weight deeper depths (≥ 20 cm) twice as much
        w  = 2.0 if d_cm >= 20.0 else 1.0
        pdd_errors.append(w * (pv - gv) ** 2)
    pdd_rmse = float(np.sqrt(np.mean(pdd_errors)))

    # ---- Profiles ----
    prof_errors = []
    for k, d_cm in enumerate(depth_targets_cm):
        if k >= len(py_res["profiles_norm"]) or k >= len(gbd_profiles):
            continue
        py_prof  = py_res["profiles_norm"][k]
        py_x     = py_res["x_cm"]
        gbd_prof = gbd_profiles[k]
        gbd_x    = gbd_x_cm[k]

        # Evaluate both at common x-positions within the field (±field_half_cm)
        x_eval = np.linspace(-field_half_cm * 0.9, field_half_cm * 0.9, 20)
        py_interp  = np.interp(x_eval, py_x,  py_prof,  left=np.nan, right=np.nan)
        gbd_interp = np.interp(x_eval, gbd_x, gbd_prof, left=np.nan, right=np.nan)
        ok = np.isfinite(py_interp) & np.isfinite(gbd_interp)
        if ok.sum() > 0:
            prof_errors.append(float(np.mean((py_interp[ok] - gbd_interp[ok]) ** 2)))

    profile_rmse = float(np.sqrt(np.mean(prof_errors))) if prof_errors else 0.0

    return pdd_weight * pdd_rmse + profile_weight * profile_rmse


# ---------------------------------------------------------------------------
# Load GBD reference data for all three fields
# ---------------------------------------------------------------------------

def _load_gbd_all_fields(energy_name: str, fields: list) -> dict:
    """
    Returns dict keyed by field name (e.g. "10x10") with GBD PDD and profiles.
    """
    ecfg    = ENERGY_CONFIGS[energy_name]
    gbd_dir = os.path.join(GBD_ROOT, ecfg["gbd_subdir"])
    result  = {}

    for field in fields:
        fcfg = FIELD_CONFIGS[field]
        depths_mm, pdd_norm = load_gbd_pdd(gbd_dir, fcfg["dd_col_tag"])

        profiles_norm = []
        x_cm_list     = []
        for d_cm, fname in PROFILE_FILES_CM[energy_name]:
            fpath = os.path.join(gbd_dir, fname)
            xc, pn = load_gbd_profile(fpath, fcfg["prof_col_tag"])
            x_cm_list.append(xc)
            profiles_norm.append(pn)

        result[field] = {
            "depths_mm": depths_mm,
            "pdd_norm":  pdd_norm,
            "x_cm":      x_cm_list,
            "profiles_norm": profiles_norm,
            "depth_targets_cm": [d for d, _ in PROFILE_FILES_CM[energy_name]],
        }

    return result


# ---------------------------------------------------------------------------
# Objective function factory
# ---------------------------------------------------------------------------

def _make_objective(energy_name: str, base_machine: dict, gbd_data: dict,
                    opt_fields: list, verbose: bool = False):
    """
    Return a callable that computes the error metric for a given parameter vector.

    Parameter vector: [log(fwhm), log(m), log(beta1), log(beta2), log(beta3)]
    (log-transformed for unconstrained optimisation; ensures positivity).
    """
    eval_count = [0]
    best_error = [np.inf]
    best_params = [None]
    call_log    = []   # [(params_phys, error)]

    def objective(x_log: np.ndarray) -> float:
        fwhm  = float(np.exp(x_log[0]))
        m     = float(np.exp(x_log[1]))
        beta1 = float(np.exp(x_log[2]))
        beta2 = float(np.exp(x_log[3]))
        beta3 = float(np.exp(x_log[4]))
        betas = np.array([beta1, beta2, beta3])

        # Physical validity checks — penalise invalid configurations
        lb = _PARAM_BOUNDS
        if not (lb["fwhm"][0]  <= fwhm  <= lb["fwhm"][1]):
            return 1e6
        if not (lb["m"][0]     <= m     <= lb["m"][1]):
            return 1e6
        if not (lb["beta1"][0] <= beta1 <= lb["beta1"][1]):
            return 1e6
        if not (lb["beta2"][0] <= beta2 <= lb["beta2"][1]):
            return 1e6
        if not (lb["beta3"][0] <= beta3 <= lb["beta3"][1]):
            return 1e6

        machine = _patch_machine(base_machine, fwhm, m, betas)
        eval_count[0] += 1
        t0 = time.perf_counter()

        total_error = 0.0
        for field in opt_fields:
            fcfg = FIELD_CONFIGS[field]
            gbd  = gbd_data[field]
            half_cm = float(field.split("x")[0]) / 2.0
            try:
                py_res = _run_dose(machine, fcfg, gbd["depth_targets_cm"],
                                   verbose=False)
                err = _compute_error(
                    py_res,
                    gbd["depths_mm"], gbd["pdd_norm"],
                    gbd["profiles_norm"], gbd["x_cm"],
                    half_cm, gbd["depth_targets_cm"],
                )
            except Exception as e:
                print(f"    [eval {eval_count[0]}] dose calc failed: {e}")
                return 1e6

            total_error += err

        total_error /= len(opt_fields)
        dt = time.perf_counter() - t0

        call_log.append({"params": dict(fwhm=fwhm, m=m,
                                        beta1=beta1, beta2=beta2, beta3=beta3),
                         "error": total_error})

        if total_error < best_error[0]:
            best_error[0]  = total_error
            best_params[0] = dict(fwhm=fwhm, m=m,
                                  beta1=beta1, beta2=beta2, beta3=beta3)
            marker = " *** NEW BEST"
        else:
            marker = ""

        print(f"  [{eval_count[0]:3d}] err={total_error:8.4f}  "
              f"fwhm={fwhm:.2f}  m={m:.5f}  "
              f"b=({beta1:.4f},{beta2:.4f},{beta3:.4f})  "
              f"{dt:.0f}s{marker}")

        return total_error

    objective.eval_count  = eval_count
    objective.best_error  = best_error
    objective.best_params = best_params
    objective.call_log    = call_log

    return objective


# ---------------------------------------------------------------------------
# Checkpoint helpers (resume after interruption)
# ---------------------------------------------------------------------------

def _checkpoint_path(energy_name: str, fields_tag: str) -> str:
    os.makedirs(TUNE_CACHE_DIR, exist_ok=True)
    return os.path.join(TUNE_CACHE_DIR, f"{energy_name}_{fields_tag}_ckpt.npy")


def _save_checkpoint(path: str, best_params: dict, best_error: float):
    np.save(path, {"params": best_params, "error": best_error}, allow_pickle=True)


def _load_checkpoint(path: str):
    if not os.path.isfile(path):
        return None, None
    data = np.load(path, allow_pickle=True).item()
    return data.get("params"), data.get("error")


# ---------------------------------------------------------------------------
# Full single-energy tuning
# ---------------------------------------------------------------------------

def tune_energy(energy_name: str, opt_fields: list,
                max_iter: int = 60, dry_run: bool = False,
                force: bool = False, verbose: bool = True) -> dict:
    """
    Tune machine parameters for one TrueBeam energy.

    Returns dict with best_params (physical values) and baseline/tuned errors.
    """
    from scipy.optimize import minimize

    print(f"\n{'='*65}")
    print(f"  Tuning: {energy_name}   (fields: {', '.join(opt_fields)})")
    print(f"{'='*65}")

    # ---- Load base machine ----
    from matRad.basedata import load_machine
    pln_probe = {"radiationMode": "photons", "machine": energy_name}
    base_machine = load_machine(pln_probe)
    data = base_machine["data"]

    m0    = float(data.get("m", 0.03))
    b0    = np.asarray(data.get("betas", [0.5, 0.02, 0.005])).ravel()
    fwhm0 = float(data.get("penumbraFWHMatIso", 6.0))
    print(f"  Initial: fwhm={fwhm0:.2f}  m={m0:.5f}  "
          f"betas=({b0[0]:.4f},{b0[1]:.4f},{b0[2]:.4f})")

    # ---- Load GBD data ----
    print(f"  Loading GBD reference data …")
    gbd_data = _load_gbd_all_fields(energy_name, opt_fields)

    # ---- Checkpoint: check for prior result ----
    fields_tag  = "_".join(opt_fields)
    ckpt_path   = _checkpoint_path(energy_name, fields_tag)
    ckpt_params, ckpt_error = _load_checkpoint(ckpt_path)
    if ckpt_params is not None and not force:
        print(f"\n  Checkpoint found (error={ckpt_error:.4f}). "
              f"Pass --force to re-run tuning.")

    # ---- Baseline error ----
    print(f"\n  Computing baseline error …")
    obj_fn = _make_objective(energy_name, base_machine, gbd_data,
                             opt_fields, verbose=verbose)
    x0_log = np.array([np.log(fwhm0), np.log(m0),
                       np.log(b0[0]), np.log(b0[1]), np.log(b0[2])])
    baseline_error = obj_fn(x0_log)
    print(f"  Baseline error: {baseline_error:.4f}")

    if dry_run:
        print("  --dry-run: skipping optimisation.")
        return {
            "energy":         energy_name,
            "baseline_error": baseline_error,
            "tuned_error":    None,
            "best_params":    dict(fwhm=fwhm0, m=m0,
                                   beta1=b0[0], beta2=b0[1], beta3=b0[2]),
            "improved":       False,
        }

    # ---- Use checkpoint if available ----
    if ckpt_params is not None and not force:
        print(f"  Using checkpoint (not re-running optimiser).")
        best_params = ckpt_params
        best_error  = ckpt_error
    else:
        # ---- Nelder-Mead optimisation ----
        print(f"\n  Starting Nelder-Mead (max_iter={max_iter}) …\n")
        opt_result = minimize(
            obj_fn,
            x0_log,
            method="Nelder-Mead",
            options={
                "maxiter":   max_iter,
                "xatol":     0.005,
                "fatol":     0.01,
                "adaptive":  True,
            },
        )
        best_params = obj_fn.best_params[0]
        best_error  = obj_fn.best_error[0]
        print(f"\n  Optimisation finished: {opt_result.message}")
        _save_checkpoint(ckpt_path, best_params, best_error)

    print(f"\n  Best error  : {best_error:.4f}  (baseline: {baseline_error:.4f})")
    print(f"  Best params :")
    print(f"    penumbraFWHMatIso = {best_params['fwhm']:.3f} mm  (was {fwhm0:.3f})")
    print(f"    m                = {best_params['m']:.6f} mm⁻¹  (was {m0:.6f})")
    print(f"    betas            = ({best_params['beta1']:.5f}, "
          f"{best_params['beta2']:.5f}, {best_params['beta3']:.5f})")
    print(f"    betas (was)      = ({b0[0]:.5f}, {b0[1]:.5f}, {b0[2]:.5f})")

    return {
        "energy":         energy_name,
        "baseline_error": baseline_error,
        "tuned_error":    best_error,
        "best_params":    best_params,
        "improved":       best_error < baseline_error,
    }


# ---------------------------------------------------------------------------
# Save tuned machine + TG-51 calibration
# ---------------------------------------------------------------------------

def apply_and_calibrate(energy_name: str, best_params: dict,
                        plot_dir: str = None) -> dict:
    """
    Patch the machine file with the tuned parameters and re-run TG-51
    calibration.  Overwrites (or creates) ``userdata/machines/photons_{energy}.npy``.

    Returns the tg51 dict written to the machine.
    """
    from matRad.basedata import load_machine

    # Import calibrate_machine.py dynamically (it lives next to this script)
    _cm_path = os.path.join(ROOT, "examples", "calibrate_machine.py")
    _cm_spec = _ilu.spec_from_file_location("calibrate_machine", _cm_path)
    _cm = _ilu.module_from_spec(_cm_spec)
    _cm_spec.loader.exec_module(_cm)
    calibrate = _cm.calibrate

    print(f"\n  Applying tuned parameters to {energy_name} …")

    # Load original
    pln_probe = {"radiationMode": "photons", "machine": energy_name}
    machine   = load_machine(pln_probe)

    # Patch
    machine = _patch_machine(
        machine,
        fwhm  = best_params["fwhm"],
        m     = best_params["m"],
        betas = np.array([best_params["beta1"],
                          best_params["beta2"],
                          best_params["beta3"]]),
    )

    # Save to userdata/machines/
    os.makedirs(USER_MACHINE_DIR, exist_ok=True)
    out_path = os.path.join(USER_MACHINE_DIR, f"photons_{energy_name}.npy")
    np.save(out_path, machine, allow_pickle=True)
    print(f"  Saved → {out_path}")

    # TG-51 calibration
    print(f"  Running TG-51 calibration …")
    abs_calib, d_max = calibrate(
        radiation_mode="photons",
        machine_name=energy_name,
        dry_run=False,
        force=True,
        plot_dir=plot_dir,
    )
    return {"abs_calib": abs_calib, "d_max_mm": d_max}


# ---------------------------------------------------------------------------
# Comparison plots (tuned vs baseline vs GBD)
# ---------------------------------------------------------------------------

def _save_tuning_plots(energy_name: str, field: str,
                       baseline_res: dict, tuned_res: dict,
                       gbd_data: dict, plot_dir: str):
    os.makedirs(plot_dir, exist_ok=True)
    fcfg    = FIELD_CONFIGS[field]
    gbd     = gbd_data[field]
    n_depths = len(gbd["depth_targets_cm"])
    half_cm  = float(field.split("x")[0]) / 2.0

    # PDD
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gbd["depths_mm"],          gbd["pdd_norm"],
            "b-",  lw=2, label="GBD measured")
    ax.plot(baseline_res["depth_mm"],  baseline_res["pdd_norm"],
            "k:",  lw=2, label="Before tuning")
    ax.plot(tuned_res["depth_mm"],     tuned_res["pdd_norm"],
            "r--", lw=2, label="After tuning")
    ax.set_xlabel("Depth [mm]"); ax.set_ylabel("Relative Dose [%]")
    ax.set_title(f"PDD – {energy_name}  {field}")
    ax.set_xlim(0, 320); ax.set_ylim(0, 110)
    ax.legend(); ax.grid(True)
    fig.savefig(os.path.join(plot_dir,
                f"{energy_name}_{field}_tuning_pdd.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Profiles
    fig, axes = plt.subplots(1, n_depths,
                             figsize=(4 * n_depths, 4), sharey=True)
    if n_depths == 1:
        axes = [axes]
    for k, ax in enumerate(axes):
        d_cm = gbd["depth_targets_cm"][k]
        ax.plot(gbd["x_cm"][k],            gbd["profiles_norm"][k],
                "b-",  lw=1.5, label="GBD" if k == 0 else "")
        ax.plot(baseline_res["x_cm"],      baseline_res["profiles_norm"][k],
                "k:",  lw=1.5, label="Before" if k == 0 else "")
        ax.plot(tuned_res["x_cm"],         tuned_res["profiles_norm"][k],
                "r--", lw=1.5, label="After" if k == 0 else "")
        ax.axvline(-half_cm, color="k", lw=0.7, ls=":")
        ax.axvline( half_cm, color="k", lw=0.7, ls=":")
        ax.set_title(f"{d_cm:.1f} cm")
        ax.set_xlabel("Off-axis [cm]")
        ax.set_xlim(-max(8.0, half_cm * 2.5), max(8.0, half_cm * 2.5))
        ax.set_ylim(0, 120)
        ax.grid(True)
    axes[0].set_ylabel("Relative Dose [%]")
    handles = [
        plt.Line2D([0], [0], color="b",  lw=1.5,       label="GBD"),
        plt.Line2D([0], [0], color="k",  lw=1.5, ls=":",  label="Before"),
        plt.Line2D([0], [0], color="r",  lw=1.5, ls="--", label="After"),
    ]
    axes[0].legend(handles=handles, fontsize=7)
    fig.suptitle(f"Profiles – {energy_name}  {field}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir,
                f"{energy_name}_{field}_tuning_profiles.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown summary writer
# ---------------------------------------------------------------------------

def _write_markdown(results: list, out_path: str):
    lines = [
        "# Machine Tuning Results",
        "",
        "Automated parameter tuning for TrueBeam photon machine files.",
        "Parameters (m, betas, penumbraFWHMatIso) were optimised to minimise",
        "PDD and lateral-profile error vs GBD reference data.",
        "",
        f"Date run: {time.strftime('%Y-%m-%d')}",
        "",
        "---",
        "",
        "## Tuned Parameters",
        "",
        "| Energy | fwhm [mm] | m [mm⁻¹] | β₁ [mm⁻¹] | β₂ [mm⁻¹] | β₃ [mm⁻¹] | "
        "err before | err after | abs_calib [cGy/MU] | d_max [mm] |",
        "|--------|-----------|----------|----------|----------|----------|"
        "-----------|-----------|-------------------|-----------|",
    ]
    for r in results:
        p = r["best_params"]
        tg51 = r.get("tg51", {})
        ac   = tg51.get("abs_calib", float("nan"))
        dm   = tg51.get("d_max_mm",  float("nan"))
        te   = r["tuned_error"]
        te_s = f"{te:.4f}" if te is not None else "—"
        lines.append(
            f"| {r['energy']} "
            f"| {p['fwhm']:.3f} "
            f"| {p['m']:.6f} "
            f"| {p['beta1']:.5f} "
            f"| {p['beta2']:.5f} "
            f"| {p['beta3']:.5f} "
            f"| {r['baseline_error']:.4f} "
            f"| {te_s} "
            f"| {ac*100:.4f} "
            f"| {dm:.1f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Notes",
        "",
        "- **fwhm**: `penumbraFWHMatIso` — Gaussian source FWHM controlling lateral penumbra.",
        "- **m**: primary photon attenuation coefficient. Governs exponential depth-dose tail.",
        "- **β₁, β₂, β₃**: SVD scatter-kernel decay constants controlling build-up region",
        "  and depth-scatter contributions.",
        "- **err**: weighted RMSE (PDD at 5/10/20/30 cm + in-field profile), lower is better.",
        "- Lateral kernel weights (kernel1–3) were **not** rebuilt; those encode field-size-",
        "  specific scatter from the original GBD TPR table. To improve 3×3 vs 20×20",
        "  PDD differences, rebuild kernels via `machineBuilder/build_truebeam.py`.",
        "- **abs_calib**: TG-51 calibration factor written to `machine[\"meta\"][\"tg51\"]`.",
        "",
        "## Workflow",
        "",
        "```",
        "python examples/tune_machine.py              # tune all energies",
        "python examples/tune_machine.py --dry-run    # baseline only",
        "python examples/calibrate_machine.py --machine TrueBeam_6X  # re-calibrate only",
        "```",
    ]

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n  Markdown summary → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description="Tune TrueBeam machine parameters to minimise GBD error.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples
        --------
          # Tune all four energies (hours; run overnight or on cluster)
          python examples/tune_machine.py

          # Quick single-energy dry run (baseline error only, no optimisation)
          python examples/tune_machine.py --machine TrueBeam_6X --dry-run

          # Use only 10×10 field for the optimizer (~3× faster than all fields)
          python examples/tune_machine.py --machine TrueBeam_6X --field 10x10

          # Resume an interrupted run (checkpoint files live in examples/cache/tune_cache/)
          python examples/tune_machine.py --machine TrueBeam_6X

          # Force re-run ignoring checkpoints
          python examples/tune_machine.py --machine TrueBeam_6X --force

          # Save comparison plots before/after tuning
          python examples/tune_machine.py --plot-dir examples/cache/tune_plots
        """),
    )
    p.add_argument("--machine",   default=None, metavar="NAME",
                   help="Machine name (e.g. TrueBeam_6X). Default: tune all 4.")
    p.add_argument("--field",     default=None,
                   choices=list(FIELD_CONFIGS.keys()),
                   help="Restrict optimizer to one field size. Default: all three.")
    p.add_argument("--max-iter",  default=60, type=int, metavar="N",
                   help="Maximum Nelder-Mead iterations per energy (default: 60).")
    p.add_argument("--dry-run",   action="store_true",
                   help="Compute baseline error only; do not optimise or save.")
    p.add_argument("--force",     action="store_true",
                   help="Ignore existing checkpoints; re-run from scratch.")
    p.add_argument("--no-calibrate", action="store_true",
                   help="Skip TG-51 calibration step after tuning.")
    p.add_argument("--plot-dir",  default=None, metavar="DIR",
                   help="Directory for before/after comparison plots.")
    p.add_argument("--output-md", default=os.path.join(ROOT, "examples", "machine_tuning.md"),
                   metavar="FILE",
                   help="Path for the markdown summary (default: examples/machine_tuning.md).")
    return p


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    energies   = ([args.machine] if args.machine
                  else list(ENERGY_CONFIGS.keys()))
    opt_fields = ([args.field] if args.field else _ALL_FIELDS)

    print("=" * 65)
    print("  TrueBeam machine parameter tuning")
    print("=" * 65)
    print(f"  Energies    : {', '.join(energies)}")
    print(f"  Opt. fields : {', '.join(opt_fields)}")
    print(f"  Max iter    : {args.max_iter}")
    print(f"  Dry run     : {args.dry_run}")
    print(f"  Force       : {args.force}")
    print(f"  Plot dir    : {args.plot_dir or '(none)'}")

    t_total = time.perf_counter()
    all_results = []

    for energy_name in energies:
        t_e = time.perf_counter()

        # ---- Tune ----
        result = tune_energy(
            energy_name  = energy_name,
            opt_fields   = opt_fields,
            max_iter     = args.max_iter,
            dry_run      = args.dry_run,
            force        = args.force,
            verbose      = True,
        )

        # ---- Plots before / after (skip if dry-run) ----
        if args.plot_dir and not args.dry_run:
            print(f"\n  Computing before/after plots for all fields …")
            from matRad.basedata import load_machine as _lm
            gbd_all = _load_gbd_all_fields(energy_name, _ALL_FIELDS)

            base_machine = _lm({"radiationMode": "photons", "machine": energy_name})
            tuned_machine = _patch_machine(
                base_machine,
                fwhm  = result["best_params"]["fwhm"],
                m     = result["best_params"]["m"],
                betas = np.array([result["best_params"]["beta1"],
                                   result["best_params"]["beta2"],
                                   result["best_params"]["beta3"]]),
            )
            for field in _ALL_FIELDS:
                fcfg = FIELD_CONFIGS[field]
                gbd  = gbd_all[field]
                dt_cm = gbd["depth_targets_cm"]
                try:
                    print(f"    {energy_name} {field}: baseline …", end="", flush=True)
                    base_res  = _run_dose(base_machine,  fcfg, dt_cm)
                    print(" tuned …", end="", flush=True)
                    tuned_res = _run_dose(tuned_machine, fcfg, dt_cm)
                    print(" done.")
                    _save_tuning_plots(energy_name, field,
                                       base_res, tuned_res, gbd_all,
                                       args.plot_dir)
                except Exception as e:
                    print(f"\n    Warning: plot failed for {field}: {e}")

        # ---- Apply + calibrate (skip if dry-run) ----
        tg51 = {}
        if not args.dry_run and result["improved"]:
            if not args.no_calibrate:
                try:
                    tg51 = apply_and_calibrate(
                        energy_name  = energy_name,
                        best_params  = result["best_params"],
                        plot_dir     = args.plot_dir,
                    )
                except Exception as e:
                    print(f"\n  Warning: calibration failed: {e}")
            else:
                # Still save the tuned parameters even if not calibrating
                from matRad.basedata import load_machine as _lm
                _m = _patch_machine(
                    _lm({"radiationMode": "photons", "machine": energy_name}),
                    fwhm  = result["best_params"]["fwhm"],
                    m     = result["best_params"]["m"],
                    betas = np.array([result["best_params"]["beta1"],
                                       result["best_params"]["beta2"],
                                       result["best_params"]["beta3"]]),
                )
                out_path = os.path.join(USER_MACHINE_DIR,
                                        f"photons_{energy_name}.npy")
                np.save(out_path, _m, allow_pickle=True)
                print(f"  Saved tuned machine → {out_path}  (no TG-51 calibration)")
        elif not args.dry_run and not result["improved"]:
            print(f"\n  Tuning did not improve error for {energy_name}; "
                  f"original machine kept.")

        result["tg51"] = tg51
        all_results.append(result)

        dt_e = time.perf_counter() - t_e
        print(f"\n  {energy_name} finished in {dt_e/60:.1f} min")

    # ---- Markdown summary ----
    if not args.dry_run:
        _write_markdown(all_results, args.output_md)

    print(f"\n{'='*65}")
    print(f"  All done in {(time.perf_counter()-t_total)/60:.1f} min total")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
