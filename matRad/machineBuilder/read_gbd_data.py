"""
Read Varian Golden Beam Data (GBD) CSV files.

Python port of the s1/s2/s3 MATLAB scripts in my_scripts/truebeam_xxx/.

All outputs use mm for distances and dimensionless fractions (0–1) for doses.
"""

import numpy as np


def read_output_factors(csv_file: str):
    """
    Read square-field output factors from a GBD output-factor CSV.

    The CSV is a 2-D OF matrix (Y-field rows × X-field cols).  Square-field
    OFs sit on the diagonal where Y size == X size.

    Layout (MATLAB 1-indexed):
      rows 1–6  metadata
      row  7    "Field Size Y [cm]", "", X₁, X₂, ...
      rows 8+   "", Y_i, OF(Y_i,X₁), OF(Y_i,X₂), ...

    Returns
    -------
    field_mm : (N,) array  – square field sizes [mm]
    of_vals  : (N,) array  – output factor values (dimensionless)
    """
    import pandas as pd

    raw = pd.read_csv(csv_file, header=None, dtype=str)

    # Row index 6 (0-based): X sizes start at column 2
    x_sizes = np.array([
        float(v) for v in raw.iloc[6, 2:].values
        if str(v).strip() not in ("", "nan")
    ])

    # Rows 7+ (0-based): Y size in col 1, OF values in cols 2+
    field_cm_list, of_list = [], []
    for i in range(7, len(raw)):
        y_str = str(raw.iloc[i, 1]).strip()
        if y_str in ("", "nan"):
            continue
        try:
            y = float(y_str)
        except ValueError:
            continue
        # diagonal entry: find X col where X == Y
        matches = np.where(np.isclose(x_sizes, y))[0]
        if len(matches) == 1:
            val_str = str(raw.iloc[i, 2 + matches[0]]).strip()
            if val_str not in ("", "nan"):
                field_cm_list.append(y)
                of_list.append(float(val_str))

    return np.array(field_cm_list) * 10.0, np.array(of_list)   # cm → mm


def read_depth_dose_tpr(csv_file: str, ssd_mm: float = 1000.0):
    """
    Read PDD from a GBD depth-dose CSV and convert to TPR.

    Layout:
      rows 1–5  metadata
      row  6    column headers  "Depth [cm]", "3x3cm2", ...
      rows 7+   data (depth in cm, dose in %)

    Conversion  PDD → TPR:
        TPR(d) = PDD(d) × [(SSD + d) / (SSD + d_ref)]²
    where d_ref = depth of dose maximum for each field size.

    Returns
    -------
    field_sizes_mm : (M,) array  – field sizes [mm]
    depths_mm      : (N,) array  – depths [mm]
    tpr            : (N, M) array – Tissue Phantom Ratio (0–1 range)
    """
    import pandas as pd

    df = pd.read_csv(csv_file, skiprows=5)       # row 5 (0-based) = header

    depths_mm = df.iloc[:, 0].values.astype(float) * 10.0

    headers = list(df.columns[1:])
    field_sizes_mm = []
    for h in headers:
        try:
            field_sizes_mm.append(float(str(h).strip().split("x")[0]) * 10.0)
        except (ValueError, IndexError):
            field_sizes_mm.append(np.nan)
    field_sizes_mm = np.array(field_sizes_mm)

    pdd = df.iloc[:, 1:].values.astype(float) / 100.0   # % → fraction

    tpr = np.zeros_like(pdd)
    for col in range(pdd.shape[1]):
        dmax_idx = int(np.argmax(pdd[:, col]))
        d_ref    = depths_mm[dmax_idx]
        isl      = ((ssd_mm + depths_mm) / (ssd_mm + d_ref)) ** 2
        tpr[:, col] = pdd[:, col] * isl

    return field_sizes_mm, depths_mm, tpr


def read_primary_fluence(csv_file: str):
    """
    Read primary fluence from a shallow-depth lateral profile CSV.

    The 40×40 cm² column at the shallowest available depth is used as the
    primary fluence estimate.  Off-axis positions are collapsed to radius,
    symmetric halves are averaged, and the result is normalised to 1 at r=0.

    Layout:
      rows 1–7  metadata
      row  8    column headers  "Off axis position [cm]", "Field Size: 3x3 cm2", ...
      rows 9+   data (empty cells where field not measured at this depth)

    Returns
    -------
    r_mm    : (K,) array – radial off-axis distance [mm], starts at 0
    fluence : (K,) array – normalised primary fluence (1.0 at r = 0)
    """
    import pandas as pd

    df = pd.read_csv(csv_file, skiprows=7)    # row 7 (0-based) = header

    # Find the "40x40" column
    col_idx = None
    for j, col in enumerate(df.columns):
        if "40x40" in str(col):
            col_idx = j
            break
    if col_idx is None:
        raise ValueError(f"Cannot find 40x40 column in {csv_file}")

    x_cm   = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
    dose   = pd.to_numeric(df.iloc[:, col_idx], errors="coerce").values
    valid  = ~np.isnan(x_cm) & ~np.isnan(dose)
    x_cm, dose = x_cm[valid], dose[valid]

    # Collapse to radius and average symmetric halves
    r_cm = np.abs(x_cm)
    r_unique, inv_idx = np.unique(r_cm, return_inverse=True)
    dose_avg = np.array([dose[inv_idx == k].mean() for k in range(len(r_unique))])

    r_mm    = r_unique * 10.0
    fluence = dose_avg / 100.0
    order   = np.argsort(r_mm)
    r_mm    = r_mm[order]
    fluence = fluence[order]
    fluence = fluence / fluence[0]   # normalise centre to 1.0

    return r_mm, fluence
