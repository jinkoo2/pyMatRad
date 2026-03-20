"""
Plan analysis utilities.

Python port of matRad_planAnalysis.m and matRad_calcDVH.m
"""

import numpy as np
from typing import Optional, Dict, List, Tuple


def plan_analysis(
    result: dict,
    ct: dict,
    cst: list,
    stf: Optional[list] = None,
    pln: Optional[dict] = None,
) -> dict:
    """
    Perform plan analysis: compute DVH and quality indicators.

    Port of matRad_planAnalysis.m

    Parameters
    ----------
    result : dict
        Result from fluence optimization with physicalDose
    ct : dict
        CT struct
    cst : list
        Structure set
    stf : list, optional
    pln : dict, optional

    Returns
    -------
    dict
        Updated result with dvh and qi (quality indicators)
    """
    from ..config import MatRad_Config
    cfg = MatRad_Config.instance()

    cfg.disp_info("Computing plan analysis...\n")

    dose_cube = result.get("physicalDose", None)
    if dose_cube is None:
        cfg.disp_error("Result does not contain physicalDose!")

    # Compute DVH and quality indicators for each structure
    dvh_list = []
    qi_list = []

    # Check if dose is in dose grid space
    dose_grid = result.get("doseGrid", None)

    for row in cst:
        struct_name = row[1]
        struct_type = row[2]
        vox_list = row[3]

        if isinstance(vox_list, list) and len(vox_list) > 0:
            vox_ix = np.asarray(vox_list[0], dtype=np.int64) - 1  # 0-based
        else:
            vox_ix = np.asarray(vox_list, dtype=np.int64) - 1

        if len(vox_ix) == 0:
            dvh_list.append({"name": struct_name, "dvh": None, "doseValues": [], "volumePoints": []})
            qi_list.append({"name": struct_name, "type": struct_type})
            continue

        # Extract dose in structure
        dose_flat = dose_cube.ravel(order="F")
        valid_ix = vox_ix[(vox_ix >= 0) & (vox_ix < len(dose_flat))]

        if len(valid_ix) == 0:
            dvh_list.append({"name": struct_name, "dvh": None, "doseValues": [], "volumePoints": []})
            qi_list.append({"name": struct_name, "type": struct_type})
            continue

        dose_struct = dose_flat[valid_ix]

        # Compute DVH
        dvh = calc_dvh(dose_struct)
        dvh["name"] = struct_name
        dvh_list.append(dvh)

        # Compute quality indicators
        qi = calc_quality_indicators(dose_struct, struct_type)
        qi["name"] = struct_name
        qi["type"] = struct_type
        qi_list.append(qi)

    result["dvh"] = dvh_list
    result["qi"] = qi_list

    return result


def calc_dvh(dose_in_structure: np.ndarray, n_points: int = 1000) -> dict:
    """
    Calculate dose-volume histogram.

    Port of matRad_calcDVH.m

    Parameters
    ----------
    dose_in_structure : np.ndarray
        Dose values in the structure voxels
    n_points : int
        Number of DVH sampling points

    Returns
    -------
    dict with:
        - doseValues: dose axis values
        - volumePoints: cumulative volume fraction (%) at each dose value
        - diffVolumePoints: differential DVH volume fractions
    """
    if len(dose_in_structure) == 0:
        return {"doseValues": np.array([]), "volumePoints": np.array([]), "diffVolumePoints": np.array([])}

    max_dose = np.max(dose_in_structure) * 1.05
    if max_dose <= 0:
        max_dose = 1.0

    dose_grid = np.linspace(0, max_dose, n_points)
    n_vox = len(dose_in_structure)

    # Cumulative DVH: V(d) = fraction of volume receiving >= d
    cum_vol = np.array([
        np.sum(dose_in_structure >= d) / n_vox * 100.0
        for d in dose_grid
    ])

    # Differential DVH
    bin_width = dose_grid[1] - dose_grid[0] if len(dose_grid) > 1 else 1.0
    diff_vol = np.array([
        np.sum(
            (dose_in_structure > d - bin_width / 2) &
            (dose_in_structure <= d + bin_width / 2)
        ) / n_vox * 100.0
        for d in dose_grid
    ])

    return {
        "doseValues": dose_grid,
        "volumePoints": cum_vol,
        "diffVolumePoints": diff_vol,
    }


def calc_quality_indicators(
    dose_in_structure: np.ndarray,
    struct_type: str = "OAR",
) -> dict:
    """
    Calculate dose quality indicators.

    Parameters
    ----------
    dose_in_structure : np.ndarray
    struct_type : str

    Returns
    -------
    dict
        Quality indicators: mean, min, max, D95, D98, D5, V_xxx
    """
    if len(dose_in_structure) == 0:
        return {
            "D_mean": 0.0, "D_min": 0.0, "D_max": 0.0,
            "D_95": 0.0, "D_98": 0.0, "D_5": 0.0,
        }

    qi = {
        "D_mean": float(np.mean(dose_in_structure)),
        "D_min": float(np.min(dose_in_structure)),
        "D_max": float(np.max(dose_in_structure)),
        "D_std": float(np.std(dose_in_structure)),
    }

    # Dose to X% of volume: D_X% = dose such that X% of volume receives >= that dose
    for pct in [2, 5, 50, 95, 98]:
        # D_95 means 95% of volume receives at least this dose
        # -> sort ascending, take (1-0.95)=0.05 percentile
        qi[f"D_{pct}"] = float(np.percentile(dose_in_structure, 100 - pct))

    return qi


def calc_dvh_metric(dvh: dict, dose_or_volume: str, value: float) -> float:
    """
    Extract DVH metric from precomputed DVH.

    Parameters
    ----------
    dvh : dict
        DVH dict with doseValues and volumePoints
    dose_or_volume : str
        'dose' to get dose at given volume, 'volume' to get volume at given dose
    value : float
        Volume (%) or dose (Gy) depending on query type

    Returns
    -------
    float
        Requested metric
    """
    dose_vals = dvh.get("doseValues", np.array([]))
    vol_pts = dvh.get("volumePoints", np.array([]))

    if len(dose_vals) == 0:
        return 0.0

    if dose_or_volume == "volume":
        # V(dose) = volume% receiving >= dose
        idx = np.searchsorted(-dose_vals, -value)
        if idx >= len(vol_pts):
            return 0.0
        return float(vol_pts[idx])
    else:
        # D(volume%) = dose received by volume% of structure
        # Find dose where cumulative DVH = volume%
        idx = np.searchsorted(-vol_pts, -value)
        if idx >= len(dose_vals):
            return float(dose_vals[-1]) if len(dose_vals) > 0 else 0.0
        if idx == 0:
            return float(dose_vals[0])
        return float(np.interp(value, vol_pts[::-1], dose_vals[::-1]))
