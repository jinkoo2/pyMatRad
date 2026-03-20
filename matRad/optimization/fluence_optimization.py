"""
Fluence optimization for IMRT/IMPT.

Python port of matRad_fluenceOptimization.m
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from typing import Optional, List, Dict


def fluence_optimization(dij: dict, cst: list, pln: dict) -> dict:
    """
    Inverse fluence optimization.

    Port of matRad_fluenceOptimization.m

    Parameters
    ----------
    dij : dict
        Dose influence matrix
    cst : list
        Structure set
    pln : dict
        Plan struct

    Returns
    -------
    dict
        Result with physicalDose field
    """
    from ..config import MatRad_Config
    from ..geometry import set_overlap_priorities, resize_cst_to_grid
    cfg = MatRad_Config.instance()

    cfg.disp_info("Starting fluence optimization...\n")

    num_fractions = pln.get("numOfFractions", 1)

    # Set overlap priorities
    cst = set_overlap_priorities(cst)

    # Resize CST to dose grid
    dose_grid = dij["doseGrid"]
    ct_grid = dij["ctGrid"]

    # For each structure, get voxel indices in dose grid
    # First ensure cst has dose grid indices (already done in initDoseCalc for stf-based)
    # Here we do it again from scratch in case cst isn't resized yet
    from ..geometry.geometry import (
        linear_index_to_subscript, subscript_to_linear_index
    )

    cst_dose = resize_cst_to_grid(cst, ct_grid, dose_grid)

    n_bixels = dij["totalNumOfBixels"]
    n_voxels_dose = dij["doseGrid"]["numOfVoxels"]

    # Extract dose matrix
    D_mat = dij["physicalDose"][0]  # Sparse (n_voxels, n_bixels)
    if not sp.issparse(D_mat):
        D_mat = sp.csc_matrix(D_mat)

    # Initialize weights
    w0 = _initialize_weights(cst_dose, D_mat, n_bixels, num_fractions)
    w0 = np.maximum(w0, 0.0)

    cfg.disp_info(f"  Number of bixels: {n_bixels}\n")
    cfg.disp_info(f"  Optimization starting...\n")

    # Build optimization problem
    objectives = _collect_objectives(cst_dose, num_fractions)

    if not objectives:
        cfg.disp_warning("No objectives defined. Using uniform dose distribution.")
        w_opt = w0
    else:
        w_opt = _run_optimization(D_mat, objectives, cst_dose, w0, n_bixels, n_voxels_dose, cfg)

    # Compute result
    result = _compute_result_cubes(w_opt, D_mat, dij, pln)
    result["w"] = w_opt

    cfg.disp_info("Fluence optimization complete.\n")
    return result


def _initialize_weights(
    cst_dose: list,
    D_mat: sp.spmatrix,
    n_bixels: int,
    num_fractions: int,
) -> np.ndarray:
    """
    Initialize bixel weights.
    Port of weight initialization in matRad_fluenceOptimization.m
    """
    # Find reference doses from objectives
    ref_doses = []
    for row in cst_dose:
        if len(row) < 6 or not row[5]:
            continue
        for obj in row[5]:
            if isinstance(obj, dict):
                params = obj.get("parameters", [])
                if params and params[0] > 0:
                    ref_doses.append(float(params[0]) / num_fractions)
            elif hasattr(obj, "parameters"):
                params = obj.parameters
                if params and params[0] > 0:
                    ref_doses.append(float(params[0]) / num_fractions)

    if ref_doses:
        target_dose = np.mean(ref_doses)
    else:
        target_dose = 1.0

    # Simple initialization: uniform weights
    if n_bixels > 0 and D_mat.shape[0] > 0:
        # Estimate: w = target_dose / mean_dose_per_unit_weight
        mean_dose = np.asarray(D_mat.sum(axis=1)).ravel()
        non_zero = mean_dose > 0
        if np.any(non_zero):
            mean_influence = np.mean(mean_dose[non_zero]) / n_bixels
            w0 = target_dose / max(mean_influence * n_bixels, 1e-10) / n_bixels * np.ones(n_bixels)
        else:
            w0 = np.ones(n_bixels) * 0.01
    else:
        w0 = np.ones(n_bixels) * 0.01

    return w0


def _collect_objectives(
    cst_dose: list,
    num_fractions: int,
) -> list:
    """Collect all objectives from CST."""
    from .DoseObjectives.objectives import DoseObjective

    objectives = []
    for row_idx, row in enumerate(cst_dose):
        if len(row) < 6 or not row[5]:
            continue

        vox_list = row[3]
        if isinstance(vox_list, list) and len(vox_list) > 0:
            vox_ix = np.asarray(vox_list[0], dtype=np.int64) - 1  # 0-based
        else:
            vox_ix = np.asarray(vox_list, dtype=np.int64) - 1

        for obj in row[5]:
            if obj is None:
                continue
            # Convert to DoseObjective if it's a dict
            if isinstance(obj, dict):
                obj = _dict_to_objective(obj, num_fractions)

            if obj is not None:
                objectives.append({
                    "struct_idx": row_idx,
                    "struct_name": row[1],
                    "struct_type": row[2],
                    "vox_ix": vox_ix,
                    "objective": obj,
                })

    return objectives


def _dict_to_objective(obj_dict: dict, num_fractions: int):
    """Convert objective dict to DoseObjective instance."""
    from .DoseObjectives.objectives import (
        SquaredDeviation, SquaredOverdosing, SquaredUnderdosing, MeanDose
    )

    class_map = {
        "matRad_SquaredDeviation": SquaredDeviation,
        "SquaredDeviation": SquaredDeviation,
        "matRad_SquaredOverdosing": SquaredOverdosing,
        "SquaredOverdosing": SquaredOverdosing,
        "matRad_SquaredUnderdosing": SquaredUnderdosing,
        "SquaredUnderdosing": SquaredUnderdosing,
        "matRad_MeanDose": MeanDose,
        "MeanDose": MeanDose,
    }

    class_name = obj_dict.get("className", obj_dict.get("class", "SquaredDeviation"))
    ObjClass = class_map.get(class_name, SquaredDeviation)

    penalty = float(obj_dict.get("penalty", 1.0))
    params = obj_dict.get("parameters", [60.0])
    if not params:
        params = [60.0]
    d_ref = float(params[0]) / num_fractions

    return ObjClass(penalty=penalty, d_ref=d_ref)


def _run_optimization(
    D_mat: sp.spmatrix,
    objectives: list,
    cst_dose: list,
    w0: np.ndarray,
    n_bixels: int,
    n_voxels_dose: int,
    cfg,
) -> np.ndarray:
    """
    Run scipy L-BFGS-B optimization.

    Port of IPOPT/fmincon optimization in matRad.
    """

    iteration = [0]
    obj_history = []

    def objective_and_gradient(w):
        """Compute total objective and gradient w.r.t. weights."""
        # Compute dose: d = D @ w
        dose = D_mat @ w  # (n_voxels,)

        total_obj = 0.0
        total_grad_dose = np.zeros(n_voxels_dose)

        for obj_info in objectives:
            vox_ix = obj_info["vox_ix"]
            obj = obj_info["objective"]

            if len(vox_ix) == 0:
                continue

            dose_struct = dose[vox_ix]

            # Compute objective
            f = obj.compute_dose_objective_function(dose_struct)
            total_obj += obj.penalty * f

            # Compute gradient w.r.t. dose in structure
            grad_dose_struct = obj.compute_dose_objective_gradient(dose_struct)
            total_grad_dose[vox_ix] += obj.penalty * grad_dose_struct

        # Chain rule: grad_w = D.T @ grad_dose
        grad_w = D_mat.T @ total_grad_dose

        return total_obj, grad_w

    def callback(w):
        iteration[0] += 1
        if iteration[0] % 10 == 0:
            obj_val, _ = objective_and_gradient(w)
            obj_history.append(obj_val)
            cfg.disp_info(f"\r  Iteration {iteration[0]}: obj = {obj_val:.6f}")

    # Bounds: weights >= 0
    from scipy.optimize import Bounds
    bounds = Bounds(lb=0.0, ub=np.inf)

    result = minimize(
        fun=objective_and_gradient,
        x0=w0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        callback=callback,
        options={
            "maxiter": 500,
            "ftol": 1e-9,
            "gtol": 1e-6,
            "disp": False,
        },
    )

    cfg.disp_info(f"\n  Optimization converged: {result.success}, message: {result.message}\n")
    return result.x


def _compute_result_cubes(
    w: np.ndarray,
    D_mat: sp.spmatrix,
    dij: dict,
    pln: dict,
) -> dict:
    """
    Compute dose cubes from optimized weights.
    Port of matRad_calcCubes.m
    """
    dose_dims = dij["doseGrid"]["dimensions"]
    n_voxels_dose = dij["doseGrid"]["numOfVoxels"]

    # Compute dose vector
    dose_vec = D_mat @ w

    # Reshape to 3D cube (Fortran order to match MATLAB)
    dose_cube = np.zeros(int(np.prod(dose_dims)))
    dose_cube[:len(dose_vec)] = dose_vec
    dose_cube = dose_cube.reshape(dose_dims, order="F")

    result = {
        "physicalDose": dose_cube,
        "w": w,
        "doseGrid": dij["doseGrid"],
    }

    return result
