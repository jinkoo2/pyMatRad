"""
Compare dose profiles from four engines at the isocenter.

Loads matRad (MATLAB) and pyMatRad (SVPB) doses from MHD files,
runs ompMC and TOPAS (via REST API) on the same phantom, then plots
X / Y / Z dose profiles through the isocenter for all four engines.

Usage:
    conda run -n pyMatRad python examples/compare_dose_profiles.py

TOPAS is run via the OpenTOPAS REST API at TOPAS_API_URL.
Set TOPAS_API_URL / TOPAS_API_TOKEN env vars to override defaults.
"""

import os
import sys
import time
import zlib
import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PYMATRAD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PYMATRAD_ROOT)

DATA_DIR       = os.path.join(PYMATRAD_ROOT, "_data", "example1_no_opti")
OUTPUT_DIR     = DATA_DIR
TOPAS_API_URL  = os.environ.get("TOPAS_API_URL",   "http://localhost:7778")
TOPAS_API_TOKEN = os.environ.get("TOPAS_API_TOKEN",
                                  "topas-dev-a3f8c2d1-4b7e-4f9a-8c3d-2e1f5a6b7c8d")
N_HISTORIES    = 50_000_000   # per beam


# ---------------------------------------------------------------------------
# MHD reader (no SimpleITK required)
# ---------------------------------------------------------------------------

def read_mhd(mhd_path: str) -> np.ndarray:
    """Read a .mhd / .zraw image pair and return a float32 ndarray (Ny, Nx, Nz)."""
    header = {}
    mhd_dir = os.path.dirname(mhd_path)
    with open(mhd_path) as fh:
        for line in fh:
            if "=" in line:
                key, val = line.split("=", 1)
                header[key.strip()] = val.strip()

    dim_size   = [int(v) for v in header["DimSize"].split()]     # Nx, Ny, Nz
    spacing    = [float(v) for v in header["ElementSpacing"].split()]
    offset     = [float(v) for v in header.get("Offset", "0 0 0").split()]
    data_file  = os.path.join(mhd_dir, header["ElementDataFile"])
    compressed = header.get("CompressedData", "False").lower() == "true"
    comp_size  = int(header.get("CompressedDataSize", "0"))

    nx, ny, nz = dim_size

    with open(data_file, "rb") as fh:
        raw = fh.read(comp_size if compressed and comp_size > 0 else -1)

    if compressed:
        raw = zlib.decompress(raw)

    arr = np.frombuffer(raw, dtype=np.float32).reshape((nz, ny, nx), order="C")
    # MHD stores (Nz, Ny, Nx) in C-order; reorder to matRad (Ny, Nx, Nz)
    arr = arr.transpose(1, 2, 0)   # → (Ny, Nx, Nz)

    return arr, np.array(spacing), np.array(offset), np.array(dim_size)


# ---------------------------------------------------------------------------
# Phantom + plan setup (identical to example1_no_opti)
# ---------------------------------------------------------------------------

def build_ct_cst_pln():
    from matRad.phantoms.builder import PhantomBuilder
    from matRad.geometry.geometry import get_world_axes
    from matRad.optimization.DoseObjectives import SquaredDeviation, SquaredOverdosing

    CT_DIM = [200, 200, 100]
    CT_RES = [2, 2, 3]

    builder = PhantomBuilder(CT_DIM, CT_RES, num_of_ct_scen=1)
    builder.add_spherical_target(
        "PTV", radius=20,
        objectives=[SquaredDeviation(penalty=800, d_ref=45).to_dict()], HU=0,
    )
    builder.add_box_oar(
        "OAR1", [60, 30, 60], offset=[0, -15, 0],
        objectives=[SquaredOverdosing(penalty=400, d_ref=0).to_dict()], HU=0,
    )
    builder.add_box_oar(
        "OAR2", [60, 30, 60], offset=[0, 15, 0],
        objectives=[SquaredOverdosing(penalty=10, d_ref=0).to_dict()], HU=0,
    )
    ct, cst = builder.get_ct_cst()
    ct = get_world_axes(ct)

    pln = {
        "radiationMode":  "photons",
        "machine":        "Generic",
        "bioModel":       "none",
        "multScen":       "nomScen",
        "numOfFractions": 30,
        "propStf": {
            "gantryAngles": [0, 72, 144, 216, 288],
            "couchAngles":  [0, 0, 0, 0, 0],
            "bixelWidth":   5,
            "addMargin":    True,
        },
        "propOpt":      {"runDAO": False, "runSequencing": False},
        "propDoseCalc": {"doseGrid": {"resolution": {"x": 3, "y": 3, "z": 3}}},
    }
    return ct, cst, pln


# ---------------------------------------------------------------------------
# Run a pyMatRad engine and return flat total-dose + dij meta
# ---------------------------------------------------------------------------

def run_engine(ct, cst, stf, pln, engine_name, extra_props=None):
    from matRad.doseCalc.calc_dose_influence import calc_dose_influence

    p = dict(pln)
    p["propDoseCalc"] = dict(pln["propDoseCalc"])
    p["propDoseCalc"]["engine"] = engine_name
    if extra_props:
        p["propDoseCalc"].update(extra_props)

    t0  = time.perf_counter()
    dij = calc_dose_influence(ct, cst, stf, p)
    elapsed = time.perf_counter() - t0

    D = dij["physicalDose"][0].tocsc()
    w = np.ones(D.shape[1])
    dose_flat = np.asarray(D @ w).ravel()
    return dij, dose_flat, elapsed


# ---------------------------------------------------------------------------
# Reshape flat dose → 3D and extract isocenter profiles
# ---------------------------------------------------------------------------

def flat_to_3d(dose_flat, dij):
    """Reshape flat dose vector to (Ny, Nx, Nz) using dij grid info."""
    grid = dij["doseGrid"]
    ny = grid["dimensions"][0]
    nx = grid["dimensions"][1]
    nz = grid["dimensions"][2]
    cube = np.zeros(ny * nx * nz)
    # dij voxels are Fortran-indexed into the dose grid
    cube[dij["voxelIndices"] - 1] = dose_flat
    return cube.reshape((ny, nx, nz), order="F")


def iso_profiles(dose3d, spacing, offset):
    """
    Return X, Y, Z profiles through the isocenter voxel.

    Returns (ix, iy, iz, prof_x, prof_y, prof_z, coords_x, coords_y, coords_z).
    """
    ny, nx, nz = dose3d.shape
    # Isocenter is at world (0, 0, 0); find nearest voxel
    x_coords = offset[0] + np.arange(nx) * spacing[0]
    y_coords = offset[1] + np.arange(ny) * spacing[1]
    z_coords = offset[2] + np.arange(nz) * spacing[2]
    ix = int(np.argmin(np.abs(x_coords)))
    iy = int(np.argmin(np.abs(y_coords)))
    iz = int(np.argmin(np.abs(z_coords)))
    return (
        ix, iy, iz,
        dose3d[iy, :, iz],   # X profile (vary ix, fix iy, iz)
        dose3d[:, ix, iz],   # Y profile (vary iy, fix ix, iz)
        dose3d[iy, ix, :],   # Z profile (vary iz, fix iy, ix)
        x_coords, y_coords, z_coords,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Dose engine comparison — water phantom isocenter profiles")
    print("=" * 70)

    # ---- Load matRad (MATLAB) and pyMatRad (SVPB) doses from MHD ----------
    print("\n[1] Loading MHD dose files...")
    dose_matrad_3d, sp_mat, off_mat, dim_mat = read_mhd(
        os.path.join(DATA_DIR, "dose_matrad.mhd"))
    dose_pymatrad_3d, sp_py,  off_py,  dim_py  = read_mhd(
        os.path.join(DATA_DIR, "dose_pymatrad.mhd"))

    print(f"    matRad dose    : {dose_matrad_3d.shape}  spacing={sp_mat}  max={dose_matrad_3d.max():.4f}")
    print(f"    pyMatRad dose  : {dose_pymatrad_3d.shape}  spacing={sp_py}   max={dose_pymatrad_3d.max():.4f}")

    # ---- Build phantom & STF (deterministic) ------------------------------
    print("\n[2] Building phantom and beam geometry...")
    ct, cst, pln = build_ct_cst_pln()

    from matRad.steering.stf_generator import generate_stf
    stf = generate_stf(ct, cst, pln)
    n_bixels = sum(b["totalNumOfBixels"] for b in stf)
    print(f"    Beams: {[b['gantryAngle'] for b in stf]}°   total bixels: {n_bixels}")

    # ---- ompMC ------------------------------------------------------------
    print("\n[3] Running ompMC...")
    dij_ompc, dose_ompc_flat, t_ompc = run_engine(ct, cst, stf, pln, "ompMC")
    print(f"    Elapsed: {t_ompc:.1f} s   max dose: {dose_ompc_flat.max():.4f}")

    # ---- TOPAS via API ----------------------------------------------------
    print(f"\n[4] Running TOPAS via API ({N_HISTORIES:,} histories/beam)...")
    dij_topas, dose_topas_flat, t_topas = run_engine(
        ct, cst, stf, pln, "TOPAS",
        extra_props={
            "topasApiUrl":   TOPAS_API_URL,
            "topasApiToken": TOPAS_API_TOKEN,
            "numHistories":  N_HISTORIES,
        },
    )
    topas_max = dose_topas_flat.max()
    print(f"    Elapsed: {t_topas:.1f} s   max dose (raw): {topas_max:.4e}")

    # ---- Reshape ompMC + TOPAS to 3D first (needed for normalization) ------
    print("\n[5a] Reshaping dose cubes to 3D for normalization...")
    grid_info = dij_ompc["doseGrid"]
    dose_spacing = np.array([
        grid_info["resolution"]["x"],
        grid_info["resolution"]["y"],
        grid_info["resolution"]["z"],
    ])
    dose_offset = np.array([
        grid_info["x"][0], grid_info["y"][0], grid_info["z"][0],
    ])

    def flat_to_3d_local(flat, dij_local):
        ny_, nx_, nz_ = [int(v) for v in dij_local["doseGrid"]["dimensions"]]
        return np.asarray(flat).reshape((ny_, nx_, nz_), order="F")

    dose_ompc_3d_raw  = flat_to_3d_local(dose_ompc_flat,  dij_ompc)
    dose_topas_3d_raw = flat_to_3d_local(dose_topas_flat, dij_topas)

    # ---- Normalize ompMC and TOPAS to matRad central mean dose ------------
    # Reference: mean dose in a 20x20x20 mm cube centred at the isocenter
    HALF_BOX = 10.0   # mm half-side → 20x20x20 mm box

    def central_mean(dose3d, spacing, offset):
        ny, nx, nz = dose3d.shape
        x_c = offset[0] + np.arange(nx) * spacing[0]
        y_c = offset[1] + np.arange(ny) * spacing[1]
        z_c = offset[2] + np.arange(nz) * spacing[2]
        mask = (
            np.abs(x_c[np.newaxis, :, np.newaxis]) <= HALF_BOX,
            np.abs(y_c[:, np.newaxis, np.newaxis]) <= HALF_BOX,
            np.abs(z_c[np.newaxis, np.newaxis, :]) <= HALF_BOX,
        )
        box = mask[0] & mask[1] & mask[2]
        vals = dose3d[box]
        return float(vals.mean()) if len(vals) > 0 else 1.0

    matrad_central = central_mean(dose_matrad_3d,  sp_mat, off_mat)
    ompc_central   = central_mean(dose_ompc_3d_raw, dose_spacing, dose_offset)
    topas_central  = central_mean(dose_topas_3d_raw, dose_spacing, dose_offset)

    scale_ompc  = matrad_central / ompc_central  if ompc_central  > 0 else 1.0
    scale_topas = matrad_central / topas_central if topas_central > 0 else 1.0

    print(f"    matRad central mean : {matrad_central:.4f} Gy/fx")
    print(f"    ompMC  central mean : {ompc_central:.4e}  → scale {scale_ompc:.4e}")
    print(f"    TOPAS  central mean : {topas_central:.4e}  → scale {scale_topas:.4e}")

    dose_ompc_3d  = dose_ompc_3d_raw  * scale_ompc
    dose_topas_3d = dose_topas_3d_raw * scale_topas

    print(f"    ompMC  3D: {dose_ompc_3d.shape}  max={dose_ompc_3d.max():.4f}")
    print(f"    TOPAS  3D: {dose_topas_3d.shape}  max={dose_topas_3d.max():.4f}")

    # ---- Extract isocenter profiles ----------------------------------------
    print("\n[6] Extracting isocenter profiles...")

    # Use the actual axis arrays from each grid for accurate coordinates
    def dose_grid_axes(dij_local):
        dg = dij_local["doseGrid"]
        sp = np.array([dg["resolution"]["x"], dg["resolution"]["y"], dg["resolution"]["z"]])
        off = np.array([dg["x"][0], dg["y"][0], dg["z"][0]])
        return sp, off

    sp_o, off_o = dose_grid_axes(dij_ompc)
    sp_t, off_t = dose_grid_axes(dij_topas)

    ix_m,  iy_m,  iz_m,  px_m,  py_m,  pz_m,  cx_m,  cy_m,  cz_m  = \
        iso_profiles(dose_matrad_3d,   sp_mat, off_mat)
    ix_py, iy_py, iz_py, px_py, py_py, pz_py, cx_py, cy_py, cz_py  = \
        iso_profiles(dose_pymatrad_3d, sp_py,  off_py)
    ix_o,  iy_o,  iz_o,  px_o,  py_o,  pz_o,  cx_o,  cy_o,  cz_o  = \
        iso_profiles(dose_ompc_3d,  sp_o, off_o)
    ix_t,  iy_t,  iz_t,  px_t,  py_t,  pz_t,  cx_t,  cy_t,  cz_t  = \
        iso_profiles(dose_topas_3d, sp_t, off_t)

    print(f"    matRad   isocenter voxel: ({iy_m},{ix_m},{iz_m})  dose={dose_matrad_3d[iy_m,ix_m,iz_m]:.4f}")
    print(f"    pyMatRad isocenter voxel: ({iy_py},{ix_py},{iz_py}) dose={dose_pymatrad_3d[iy_py,ix_py,iz_py]:.4f}")
    print(f"    ompMC    isocenter voxel: ({iy_o},{ix_o},{iz_o})  dose={dose_ompc_3d[iy_o,ix_o,iz_o]:.4f}")
    print(f"    TOPAS    isocenter voxel: ({iy_t},{ix_t},{iz_t})  dose={dose_topas_3d[iy_t,ix_t,iz_t]:.4f}")

    # ---- Plot --------------------------------------------------------------
    print("\n[7] Plotting profiles...")
    PLOT_RANGE = (-150, 150)   # mm, applied to all three axes

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Photon dose profiles at isocenter — 5-beam water phantom\n"
        f"(TOPAS: {N_HISTORIES:,} histories/beam, ompMC & TOPAS normalized to matRad central mean)",
        fontsize=12,
    )

    profiles = [
        ("X (mm)", [cx_m, cx_py, cx_o, cx_t], [px_m, px_py, px_o, px_t]),
        ("Y (mm)", [cy_m, cy_py, cy_o, cy_t], [py_m, py_py, py_o, py_t]),
        ("Z (mm)", [cz_m, cz_py, cz_o, cz_t], [pz_m, pz_py, pz_o, pz_t]),
    ]
    labels = ["matRad (MATLAB)", "pyMatRad (SVPB)", "ompMC", "TOPAS MC"]
    colors = ["black", "steelblue", "tomato", "seagreen"]
    styles = ["-", "--", "-.", ":"]
    lws    = [2.0, 1.8, 1.8, 1.8]

    for ax, (xlabel, coords_list, profs_list) in zip(axes, profiles):
        for coords, prof, label, color, ls, lw in zip(
            coords_list, profs_list, labels, colors, styles, lws
        ):
            # Clip to plot range
            mask = (coords >= PLOT_RANGE[0]) & (coords <= PLOT_RANGE[1])
            ax.plot(coords[mask], prof[mask], ls, color=color, lw=lw, label=label)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Dose (Gy/fx)", fontsize=12)
        ax.set_xlim(PLOT_RANGE)
        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5, label="isocenter")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Profile along {xlabel[:1]}-axis")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "dose_profiles.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"    Saved → {out_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
