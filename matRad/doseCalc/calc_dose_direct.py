"""
Direct dose calculation from pre-computed dij and bixel weights.
Port of matRad_calcDoseDirect.m
"""

import numpy as np
import scipy.sparse as sp


def calc_dose_direct(dij: dict, w: np.ndarray) -> dict:
    """
    Compute dose from a pre-computed dij matrix and bixel weight vector.

    This is a simple matrix-vector product: dose = dij @ w.
    Use this to forward-project Eclipse MU weights (from import_rtplan_fluence)
    through the matRad dose engine instead of running a new optimisation.

    Parameters
    ----------
    dij : dict
        Dose influence matrix from calc_dose_influence().
        Must contain 'physicalDose' (list with one sparse matrix of shape
        (n_voxels_dose, n_bixels)).
    w : ndarray, shape (n_bixels,)
        Per-bixel weights.  For Eclipse reproduction, use the output of
        import_rtplan_fluence().

    Returns
    -------
    result : dict with keys:
        'physicalDose'   — ndarray (Ny, Nx, Nz) [Gy]
        'w'              — the weight vector (copy)
        'doseGrid'       — dose grid dict from dij
    """
    w = np.asarray(w, dtype=float).ravel()

    D_sparse = dij["physicalDose"][0]  # sparse (n_vox_dose, n_bix)
    if D_sparse is None:
        raise ValueError("dij['physicalDose'][0] is None — run calc_dose_influence first.")

    dose_vec = np.asarray(D_sparse @ w).ravel()

    dose_grid = dij["doseGrid"]
    dims = dose_grid["dimensions"]   # [Ny, Nx, Nz]
    dose_cube = dose_vec.reshape(dims, order="F")

    return {
        "physicalDose": dose_cube,
        "w":            w.copy(),
        "doseGrid":     dose_grid,
    }
