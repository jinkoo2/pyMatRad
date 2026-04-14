"""
Photon pencil-beam kernel calculation engine.

Python port of:
  ppbkc_calcKernelNorm.m         → calc_kernel_norm()
  ppbkc_outputFactorCorrection.m → output_factor_correction()
  ppbkc_generateBaseData.m       → generate_machine()

All distances in mm throughout.
"""

import numpy as np
from datetime import date
from scipy.interpolate import CubicSpline, RectBivariateSpline, interp1d
from scipy.fft import fft2, ifft2, fftshift
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# Kernel normalisation  (port of ppbkc_calcKernelNorm.m – radialInt branch)
# ---------------------------------------------------------------------------

def calc_kernel_norm(
    kernel_extension: int,
    kernel_resolution: float,
    pf_r: np.ndarray,
    pf_vals: np.ndarray,
) -> np.ndarray:
    """
    Compute radial kernel normalisation vector.

    Parameters
    ----------
    kernel_extension  : int   – number of kernel pixels, e.g. 720
    kernel_resolution : float – pixel size [mm], e.g. 0.5
    pf_r   : (P,) – radial positions of primary fluence [mm]
    pf_vals: (P,) – primary fluence (normalised to 1.0 at r=0)

    Returns
    -------
    f_r_norm : (kernel_extension//2,) array
    """
    dr = kernel_resolution
    n  = kernel_extension // 2
    r     = dr * np.arange(n)            # [0, 0.5, ..., 179.5]
    r_mid = (r[:-1] + r[1:]) / 2        # midpoints

    fluence_mid = np.interp(r_mid, pf_r, pf_vals, left=pf_vals[0], right=0.0)
    ring_int    = dr * 2.0 * np.pi * r_mid * fluence_mid

    # Spline from midpoints back to full grid (MATLAB interp1 'spline')
    cs       = CubicSpline(r_mid, ring_int, extrapolate=True)
    f_r_norm = 4.0 * cs(r)

    # r=0 override (matches MATLAB: fRNorm(1) = primaryFluence(1,2))
    f_r_norm[0] = float(pf_vals[0])
    return f_r_norm


# ---------------------------------------------------------------------------
# Output-factor correction  (port of ppbkc_outputFactorCorrection.m)
# ---------------------------------------------------------------------------

def output_factor_correction(
    of_mm: np.ndarray,
    of_vals: np.ndarray,
    pf_r: np.ndarray,
    pf_vals: np.ndarray,
    kernel_extension: int,
    kernel_resolution: float,
    fwhm_gauss: float,
) -> tuple:
    """
    Small-field correction for output factors.

    Corrects fields smaller than 5.4·√2·σ by computing the effective
    fluence×Gaussian convolution at the detector centre.

    Returns
    -------
    of_mm      : unchanged input
    of_vals_out: corrected output factors
    """
    sigma     = fwhm_gauss / np.sqrt(8.0 * np.log(2.0))   # mm
    sigma_vox = sigma / kernel_resolution                  # pixels
    i_center  = kernel_extension // 2                      # e.g. 360

    # 2-D pixel index grid  [-n/2+1 .. n/2]  (MATLAB meshgrid convention)
    n   = kernel_extension
    idx = np.arange(-n // 2 + 1, n // 2 + 1)
    X, Y = np.meshgrid(idx, idx)   # X varies along columns

    # Radial distance in mm
    Z      = np.sqrt(X ** 2 + Y ** 2) * kernel_resolution
    primflu = np.interp(Z.ravel(), pf_r, pf_vals,
                        left=pf_vals[0], right=0.0).reshape(n, n)

    # 2-D separable Gaussian in pixel space
    gauss_filter = (
        (1.0 / (np.sqrt(2 * np.pi) * sigma_vox)) *
        np.exp(-X ** 2 / (2 * sigma_vox ** 2)) *
        (1.0 / (np.sqrt(2 * np.pi) * sigma_vox)) *
        np.exp(-Y ** 2 / (2 * sigma_vox ** 2))
    )
    fft_gauss = fft2(gauss_filter, s=(n, n))

    threshold_mm = 5.4 * np.sqrt(2.0) * sigma
    of_vals_out  = of_vals.copy()

    for i, fs_mm in enumerate(of_mm):
        if fs_mm >= threshold_mm:
            continue

        # Square field mask (MATLAB 1-indexed lower/upper → Python 0-indexed)
        half  = fs_mm / 2.0 / kernel_resolution
        lower = int(round(i_center - half + 1)) - 1   # → 0-based
        upper = int(np.floor(i_center + half))         # Python slice exclusive

        field_shape          = np.zeros((n, n))
        field_shape[lower:upper, lower:upper] = 1.0

        conv_res = fftshift(np.real(ifft2(
            fft2(field_shape * primflu, s=(n, n)) * fft_gauss
        )))

        # MATLAB: convRes(iCenter-1, iCenter-1) with 1-based iCenter=360
        #       → Python:  [358, 358]
        center_val = conv_res[i_center - 2, i_center - 2]
        if center_val != 0.0:
            of_vals_out[i] = of_vals[i] / center_val

    return of_mm, of_vals_out


# ---------------------------------------------------------------------------
# Main generator  (port of ppbkc_generateBaseData.m)
# ---------------------------------------------------------------------------

def generate_machine(
    name: str,
    params: dict,
    tpr_field_sizes_mm: np.ndarray,
    tpr_depths_mm: np.ndarray,
    tpr: np.ndarray,
    of_mm: np.ndarray,
    of_vals: np.ndarray,
    pf_r: np.ndarray,
    pf_vals: np.ndarray,
) -> dict:
    """
    Generate a matRad-compatible photon machine dict from commissioning data.

    Port of ppbkc_generateBaseData.m.

    Parameters
    ----------
    name    : machine name, e.g. "TrueBeam_6X"
    params  : scalar parameters dict – required keys:
                'SAD'                        [mm]
                'photon_energy'              [MV]
                'fwhm_gauss'                 [mm]
                'electron_range_intensity'
                'source_collimator_distance' [mm]
    tpr_field_sizes_mm : (M,)     – field sizes used in TPR measurement
    tpr_depths_mm      : (N,)     – depths used in TPR measurement
    tpr                : (N, M)   – Tissue Phantom Ratio values
    of_mm    : (K,)  – square field sizes for output factor
    of_vals  : (K,)  – measured output factors
    pf_r     : (P,)  – primary fluence radial positions [mm]
    pf_vals  : (P,)  – primary fluence values (normalised to 1 at r=0)

    Returns
    -------
    machine : dict  compatible with pyMatRad load_machine / photon_svd_engine
    """
    SAD  = float(params["SAD"])
    fwhm = float(params["fwhm_gauss"])

    # ------------------------------------------------------------------
    # Insert field-size 0 column into TPR (linear extrapolation)
    # ------------------------------------------------------------------
    if tpr_field_sizes_mm[0] != 0.0:
        tpr_zero = np.array([
            float(interp1d(tpr_field_sizes_mm[:2], tpr[d, :2],
                           fill_value="extrapolate")(0.0))
            for d in range(len(tpr_depths_mm))
        ])
        tpr_field_sizes_mm = np.concatenate([[0.0], tpr_field_sizes_mm])
        tpr = np.column_stack([tpr_zero, tpr])

    # ------------------------------------------------------------------
    # Fit attenuation coefficient µ (exponential behind build-up)
    # ------------------------------------------------------------------
    tpr_max     = np.max(tpr, axis=0)
    tpr_max_idx = np.argmax(tpr, axis=0)
    mean_max_mm = float(np.ceil(np.mean(tpr_depths_mm[tpr_max_idx])))

    tpr_0 = tpr[:, 0] / tpr_max[0]          # depth-dose for fs=0, normalised

    ix     = int(np.argmin(np.abs(tpr_depths_mm - mean_max_mm)))
    t_post = tpr_depths_mm[ix + 1:]
    log_t  = -np.log(tpr_0[ix + 1:])
    n_pts  = len(t_post)

    fSx  = t_post.sum()
    fSxx = (t_post ** 2).sum()
    fSy  = log_t.sum()
    fSxy = (log_t * t_post).sum()
    mu   = (fSxy - fSx * fSy / n_pts) / (fSxx - fSx ** 2 / n_pts)

    # ------------------------------------------------------------------
    # Compute betas (Batho / Scholz exponential decomposition)
    # ------------------------------------------------------------------
    def _max_pos(x, mu):
        """Depth of maximum of  β/(β-µ) · (e^{-µd} - e^{-βd})."""
        if abs(x - mu) < 1e-12:
            return 1.0 / mu
        return (np.log(mu) - np.log(x)) / (mu - x)

    targets = [
        mean_max_mm,
        (mean_max_mm + 1.0 / mu) / 2.0,
        1.0 / mu,
    ]
    betas = []
    for tgt in targets:
        res = minimize_scalar(
            lambda x: (_max_pos(x, mu) - tgt) ** 2,
            bounds=(1e-6, 1000.0),
            method="bounded",
        )
        betas.append(float(res.x))
    betas = np.array(betas)

    # ------------------------------------------------------------------
    # Kernel normalisation
    # ------------------------------------------------------------------
    kernel_extension  = 720
    kernel_resolution = 0.5   # mm
    kernel_norm = calc_kernel_norm(kernel_extension, kernel_resolution,
                                   pf_r, pf_vals)

    # ------------------------------------------------------------------
    # Output-factor correction (small-field)
    # ------------------------------------------------------------------
    _, of_corrected = output_factor_correction(
        of_mm, of_vals, pf_r, pf_vals,
        kernel_extension, kernel_resolution, fwhm
    )

    # ------------------------------------------------------------------
    # Equivalent field sizes and OF interpolated onto them
    # ------------------------------------------------------------------
    n_half = kernel_extension // 2   # 360
    equiv_fs = np.arange(1, n_half + 1) * kernel_resolution * np.sqrt(np.pi)
    # [0.886, 1.772, ..., 318.5] mm

    # MATLAB: interp1(..., 'linear', 'extrap') → linear extrapolation
    of_interp    = interp1d(of_mm, of_corrected, kind="linear",
                            fill_value="extrapolate")
    cor_of_equi  = of_interp(equiv_fs)

    # ------------------------------------------------------------------
    # 2-D spline for scaled TPR (MATLAB interp2 'spline')
    # ------------------------------------------------------------------
    tpr_spl = RectBivariateSpline(
        tpr_depths_mm, tpr_field_sizes_mm, tpr, kx=3, ky=3
    )

    # ------------------------------------------------------------------
    # Assemble machine dict
    # ------------------------------------------------------------------
    machine = {
        "meta": {
            "radiationMode": "photons",
            "machine":       name,
            "dataType":      "-",
            "created_on":    str(date.today()),
            "created_by":    "pyMatRad.machineBuilder.kernel_calc",
            "description":   f"photon pencil beam kernels for {name}",
            "name":          name,
            "SAD":           SAD,
            "SCD":           float(params.get("source_collimator_distance", 345.0)),
        },
        "data": {
            "energy":                 float(params["photon_energy"]),
            "kernelPos":              kernel_resolution * np.arange(n_half),
            "penumbraFWHMatIso":      fwhm,
            "fwhm":                   fwhm,
            "electronRangeIntensity": float(params.get("electron_range_intensity", 0.001)),
            "primaryFluence":         np.column_stack([pf_r, pf_vals]),
            "m":                      float(mu),
            "betas":                  betas,
            "surfaceDose":            float(tpr[0, 0]),
        },
    }

    # ------------------------------------------------------------------
    # Kernel loop  –  501 SSDs from 500 to 1000 mm
    # ------------------------------------------------------------------
    kernels = []
    for i_ssd in range(501):
        ssd   = float(i_ssd + 500)
        scale = (ssd + tpr_depths_mm) / SAD   # (N,) depth-dependent scale

        # Scaled TPR:  scaledTpr[d,j] = TPR(depth=d, fs=scale[d]·fs[j])
        depth_pts = np.repeat(tpr_depths_mm, len(tpr_field_sizes_mm))
        fs_pts    = (scale[:, None] * tpr_field_sizes_mm[None, :]).ravel()
        fs_pts    = np.clip(fs_pts, tpr_field_sizes_mm[0], tpr_field_sizes_mm[-1])
        scaled_tpr = tpr_spl.ev(depth_pts, fs_pts).reshape(
            len(tpr_depths_mm), len(tpr_field_sizes_mm)
        )

        # Basis depth-dose functions (post-build-up only)
        D_mat = np.column_stack([
            (betas[k] / (betas[k] - mu)) *
            (np.exp(-mu * t_post) - np.exp(-betas[k] * t_post))
            for k in range(3)
        ])   # (n_post, 3)

        mx1  = D_mat.T @ D_mat                          # (3, 3)
        mx2  = D_mat.T @ scaled_tpr[ix + 1:, :]        # (3, M)
        W_ri = np.linalg.solve(mx1, mx2).T              # (M, 3)

        # Append surface-dose column (kernel4)
        W_ri4 = tpr[0, :] / tpr[0, 0]
        W_ri  = np.column_stack([W_ri, W_ri4])          # (M, 4)

        # Interpolate weights to equivalent field sizes (MATLAB interp1 'spline')
        kentry = {"SSD": ssd}
        for col, kname in enumerate(["kernel1", "kernel2", "kernel3", "kernel4"]):
            cs      = CubicSpline(tpr_field_sizes_mm, W_ri[:, col], extrapolate=True)
            D_spl   = cs(equiv_fs)
            prod    = cor_of_equi * D_spl
            grad    = np.concatenate([[cor_of_equi[0] * D_spl[0]], np.diff(prod)])
            kentry[kname] = grad / kernel_norm

        kernels.append(kentry)

    machine["data"]["kernel"] = kernels
    return machine


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_machine(machine: dict, filepath: str) -> None:
    """Save machine dict to a .npy file (numpy pickle)."""
    np.save(filepath, machine, allow_pickle=True)
    print(f"Saved: {filepath}")


def load_machine_npy(filepath: str) -> dict:
    """Load a machine dict from a .npy file saved by save_machine()."""
    return np.load(filepath, allow_pickle=True).item()
