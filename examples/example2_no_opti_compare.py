"""
Example 2 No-Opti: pyMatRad dose calculation with uniform weights,
comparison against MATLAB matRad (TG119 phantom), and OpenTPS GUI visualization.

Pipeline:
  1. Load MATLAB no-opti results  (STF + physicalDose as ground-truth reference)
  2. Load TG119 CT/CST from TG119.mat
  3. Run pyMatRad dose calc using MATLAB's exact STF + uniform weights (w = 1)
  4. Compare dose statistics (max, mean, per-beam DIJ column sums)
  5. Export CT + both doses to MHD  →  _data/example2_no_opti/
  6. Launch OpenTPS GUI

Reference file:
  examples/_matRad_ref_outputs/example2_no_opti/example2_results.mat
  Supports both MATLAB v5 (scipy.io) and v7.3 / HDF5 (h5py) formats.
"""

import os, sys, logging
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import h5py

# ------------------------------------------------------------------ paths ---
PYMATRAD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENTPS_CORE  = r'C:\Users\jkim20\Desktop\projects\tps\opentps2\opentps_core'
OPENTPS_GUI   = r'C:\Users\jkim20\Desktop\projects\tps\opentps2\opentps_gui'
TG119_MAT     = r'C:\Users\jkim20\Desktop\projects\tps\matRad\matRad\phantoms\TG119.mat'

REF_MAT  = os.path.join(PYMATRAD_ROOT,
    r'examples\_matRad_ref_outputs\example2_no_opti\example2_results.mat')
DATA_DIR = os.path.join(PYMATRAD_ROOT, '_data', 'example2_no_opti')

sys.path.insert(0, PYMATRAD_ROOT)
sys.path.insert(0, OPENTPS_CORE)
sys.path.insert(0, OPENTPS_GUI)

os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------- helpers ---
def h5val(f, ref):
    return np.array(f[ref]).ravel()

def h5scalar(f, path):
    return float(np.array(f[path]).ravel()[0])

def _s(x):
    return float(np.array(x).ravel()[0])

def _v(x):
    return np.array(x).ravel().astype(np.float64)

def to_opentps(cube_ny_nx_nz):
    """Convert matRad (Ny,Nx,Nz) → OpenTPS (Nx,Ny,Nz)."""
    return cube_ny_nx_nz.transpose(1, 0, 2).copy()

def flat_dose_to_opentps(flat, ny, nx, nz):
    """Reshape flat matRad-indexed dose → OpenTPS (Nx,Ny,Nz)."""
    return flat.reshape((ny, nx, nz), order='F').transpose(1, 0, 2).copy()


# ---------------------------------------------------------------- loaders ---
def _load_scipy(path):
    """Load from MATLAB v5 format via scipy.io."""
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)

    dij_m    = mat['dij']
    result_m = mat['resultGUI']
    stf_m    = mat['stf']

    # dose grid
    dg = dij_m.doseGrid
    dg_dims = np.array(dg.dimensions).ravel().astype(int)
    Ny_dg, Nx_dg, Nz_dg = int(dg_dims[0]), int(dg_dims[1]), int(dg_dims[2])
    dg_res_x = _s(dg.resolution.x)
    dg_res_y = _s(dg.resolution.y)
    dg_res_z = _s(dg.resolution.z)
    dg_x = _v(dg.x);  dg_y = _v(dg.y);  dg_z = _v(dg.z)

    # resultGUI physicalDose — scipy.io returns MATLAB [Ny,Nx,Nz] as (Ny,Nx,Nz)
    ml_dose_mat = np.array(result_m.physicalDose, dtype=np.float64)
    if ml_dose_mat.ndim == 1:
        ml_dose_mat = ml_dose_mat.reshape((Ny_dg, Nx_dg, Nz_dg), order='F')

    # weights
    w_saved = _v(result_m.w)

    # DIJ sparse matrix — dij.physicalDose is cell{1,1} of sparse
    pd_raw = dij_m.physicalDose
    if sp.issparse(pd_raw):
        D_ml = pd_raw.tocsc()
    else:
        D_ml = pd_raw.flat[0].tocsc()

    # STF struct array
    if stf_m.ndim == 0:
        stf_m = np.array([stf_m.item()])
    stf_list = []
    for ib in range(len(stf_m)):
        bm  = stf_m[ib]
        ga  = _s(bm.gantryAngle)
        ca  = _s(bm.couchAngle)
        sad = _s(bm.SAD)
        bw  = _s(bm.bixelWidth)
        nr  = int(_s(bm.numOfRays))
        nb  = int(_s(bm.totalNumOfBixels))
        iso  = _v(bm.isoCenter)
        sp_w = _v(bm.sourcePoint)
        sp_b = _v(bm.sourcePoint_bev)

        rays_m = bm.ray
        if rays_m.ndim == 0:
            rays_m = np.array([rays_m.item()])
        rays = []
        for ir in range(len(rays_m)):
            ray  = rays_m[ir]
            rbev = _v(ray.rayPos_bev)
            tp   = _v(ray.targetPoint)
            rp   = _v(ray.rayPos) if hasattr(ray, 'rayPos') else rbev.copy()
            rays.append({
                'rayPos_bev':      rbev,
                'targetPoint_bev': np.array([2.0*rbev[0], sad, 2.0*rbev[2]]),
                'rayPos':          rp,
                'targetPoint':     tp,
                'energy':          np.array([6.0]),
            })
        stf_list.append({
            'gantryAngle': ga, 'couchAngle': ca, 'SAD': sad, 'bixelWidth': bw,
            'isoCenter': iso, 'sourcePoint': sp_w, 'sourcePoint_bev': sp_b,
            'numOfRays': nr, 'totalNumOfBixels': nb, 'ray': rays,
            'radiationMode': 'photons', 'machine': 'Generic',
        })

    return (Ny_dg, Nx_dg, Nz_dg, dg_res_x, dg_res_y, dg_res_z,
            dg_x, dg_y, dg_z, ml_dose_mat, w_saved, D_ml, stf_list)


def _load_h5(path):
    """Load from MATLAB v7.3 (HDF5) format via h5py."""
    f = h5py.File(path, 'r')

    dg_dims = np.array(f['dij/doseGrid/dimensions']).ravel().astype(int)
    Ny_dg, Nx_dg, Nz_dg = int(dg_dims[0]), int(dg_dims[1]), int(dg_dims[2])
    dg_res_x = h5scalar(f, 'dij/doseGrid/resolution/x')
    dg_res_y = h5scalar(f, 'dij/doseGrid/resolution/y')
    dg_res_z = h5scalar(f, 'dij/doseGrid/resolution/z')
    dg_x = h5val(f, 'dij/doseGrid/x')
    dg_y = h5val(f, 'dij/doseGrid/y')
    dg_z = h5val(f, 'dij/doseGrid/z')

    ml_dose_mat = np.array(f['resultGUI/physicalDose']).T   # (Ny,Nx,Nz)
    w_saved = np.array(f['resultGUI/w']).ravel() if 'resultGUI/w' in f else None

    ref_pd = f['dij/physicalDose'][0, 0]
    sp_grp = f[ref_pd]
    ir_idx = np.array(sp_grp['ir'],   dtype=np.int64)
    jc_arr = np.array(sp_grp['jc'],   dtype=np.int64)
    data   = np.array(sp_grp['data'], dtype=np.float64)
    n_vox  = Ny_dg * Nx_dg * Nz_dg
    D_ml   = sp.csc_matrix((data, ir_idx, jc_arr), shape=(n_vox, len(jc_arr) - 1))

    stf_h5  = f['stf']
    n_beams = stf_h5['gantryAngle'].shape[0]
    stf_list = []
    for ib in range(n_beams):
        ga   = float(h5val(f, stf_h5['gantryAngle'][ib, 0])[0])
        ca   = float(h5val(f, stf_h5['couchAngle'][ib, 0])[0])
        sad  = float(h5val(f, stf_h5['SAD'][ib, 0])[0])
        bw   = float(h5val(f, stf_h5['bixelWidth'][ib, 0])[0])
        nr   = int(h5val(f, stf_h5['numOfRays'][ib, 0])[0])
        nb   = int(h5val(f, stf_h5['totalNumOfBixels'][ib, 0])[0])
        iso  = h5val(f, stf_h5['isoCenter'][ib, 0])
        sp_b = h5val(f, stf_h5['sourcePoint_bev'][ib, 0])
        sp_w = h5val(f, stf_h5['sourcePoint'][ib, 0])
        ray_grp = f[stf_h5['ray'][ib, 0]]
        rays = []
        for ir in range(nr):
            rbev = h5val(f, ray_grp['rayPos_bev'][ir, 0]).astype(float)
            tp   = h5val(f, ray_grp['targetPoint'][ir, 0]).astype(float)
            rp   = h5val(f, ray_grp['rayPos'][ir, 0]).astype(float) \
                   if 'rayPos' in ray_grp else rbev.copy()
            rays.append({
                'rayPos_bev':      rbev,
                'targetPoint_bev': np.array([2.0*rbev[0], sad, 2.0*rbev[2]]),
                'rayPos':          rp,
                'targetPoint':     tp,
                'energy':          np.array([6.0]),
            })
        stf_list.append({
            'gantryAngle': ga, 'couchAngle': ca, 'SAD': sad, 'bixelWidth': bw,
            'isoCenter': iso, 'sourcePoint': sp_w, 'sourcePoint_bev': sp_b,
            'numOfRays': nr, 'totalNumOfBixels': nb, 'ray': rays,
            'radiationMode': 'photons', 'machine': 'Generic',
        })
    f.close()
    return (Ny_dg, Nx_dg, Nz_dg, dg_res_x, dg_res_y, dg_res_z,
            dg_x, dg_y, dg_z, ml_dose_mat, w_saved, D_ml, stf_list)


# ==========================================================================
# 1.  Load MATLAB no-opti results  (auto-detect v5 vs v7.3)
# ==========================================================================
print("=" * 65)
print("STEP 1 — Load MATLAB no-opti results (TG119)")
print("=" * 65)

try:
    with h5py.File(REF_MAT, 'r'):
        pass
    print("  Detected: MATLAB v7.3 (HDF5) — using h5py")
    (Ny_dg, Nx_dg, Nz_dg, dg_res_x, dg_res_y, dg_res_z,
     dg_x, dg_y, dg_z, ml_dose_mat, w_saved, D_ml, stf_ml) = _load_h5(REF_MAT)
except OSError:
    print("  Detected: MATLAB v5 — using scipy.io")
    (Ny_dg, Nx_dg, Nz_dg, dg_res_x, dg_res_y, dg_res_z,
     dg_x, dg_y, dg_z, ml_dose_mat, w_saved, D_ml, stf_ml) = _load_scipy(REF_MAT)

n_beams = len(stf_ml)
print(f"  MATLAB dose grid : {Ny_dg}×{Nx_dg}×{Nz_dg}  res=({dg_res_x},{dg_res_y},{dg_res_z}) mm")
print(f"    x: [{dg_x[0]:.1f}, {dg_x[-1]:.1f}]  y: [{dg_y[0]:.1f}, {dg_y[-1]:.1f}]  z: [{dg_z[0]:.1f}, {dg_z[-1]:.1f}]")
print(f"\n  MATLAB physicalDose shape (Ny,Nx,Nz) : {ml_dose_mat.shape}")
print(f"  MATLAB max dose : {ml_dose_mat.max():.4f} Gy/fx")
print(f"\n  MATLAB DIJ : {D_ml.shape}  nnz={D_ml.nnz}")

print(f"\n  STF beams ({n_beams} total):")
for ib, bm in enumerate(stf_ml):
    print(f"    beam[{ib}] gantry={bm['gantryAngle']:6.1f}°  "
          f"rays={bm['numOfRays']:4d}  bixels={bm['totalNumOfBixels']:4d}")
total_bixels_ml = sum(b['totalNumOfBixels'] for b in stf_ml)
print(f"  Total bixels (MATLAB) : {total_bixels_ml}")

if w_saved is not None:
    is_uniform = np.allclose(w_saved, 1.0, atol=1e-6)
    print(f"\n  resultGUI/w : n={len(w_saved)}  min={w_saved.min():.4f}  "
          f"max={w_saved.max():.4f}  mean={w_saved.mean():.4f}")
    print(f"  weights are {'uniform (w=1) ✓' if is_uniform else 'NOT uniform — WARNING: optimized result!'}")


# ==========================================================================
# 2.  Load TG119 CT/CST
# ==========================================================================
print("\n" + "=" * 65)
print("STEP 2 — Load TG119 CT/CST")
print("=" * 65)

raw     = sio.loadmat(TG119_MAT, squeeze_me=True, struct_as_record=False)
ct_raw  = raw['ct']
cst_raw = raw['cst']

ct = {
    'cubeDim':    [int(ct_raw.cubeDim[0]), int(ct_raw.cubeDim[1]), int(ct_raw.cubeDim[2])],
    'resolution': {'x': float(ct_raw.resolution.x),
                   'y': float(ct_raw.resolution.y),
                   'z': float(ct_raw.resolution.z)},
    'x':    np.asarray(ct_raw.x).ravel(),
    'y':    np.asarray(ct_raw.y).ravel(),
    'z':    np.asarray(ct_raw.z).ravel(),
    'cubeHU':      [np.asarray(ct_raw.cubeHU)],
    'cube':        [np.asarray(ct_raw.cube)],
    'numOfCtScen': 1,
    'hlut':        ct_raw.hlut,
}
cst = []
for i in range(cst_raw.shape[0]):
    row = cst_raw[i]
    vox = np.asarray(row[3], dtype=np.int64).ravel()
    cst.append([int(row[0]), str(row[1]), str(row[2]), [vox], {}, row[5]])

Ny_ct, Nx_ct, Nz_ct = ct['cubeDim']
ct_origin  = (float(ct['x'][0]), float(ct['y'][0]), float(ct['z'][0]))
ct_spacing = (ct['resolution']['x'], ct['resolution']['y'], ct['resolution']['z'])
print(f"  TG119 CT : {Ny_ct}×{Nx_ct}×{Nz_ct}  res={ct_spacing}  origin={ct_origin}")
print(f"  CST rows : {len(cst)}")
for row in cst:
    print(f"    [{row[0]}] {row[1]:<20} type={row[2]}  voxels={len(row[3][0])}")


# ==========================================================================
# 3.  Run pyMatRad dose calc (MATLAB STF + uniform weights)
# ==========================================================================
print("\n" + "=" * 65)
print("STEP 3 — pyMatRad dose calculation (uniform weights)")
print("=" * 65)

from matRad.doseCalc.calc_dose_influence import calc_dose_influence

pln = {
    "radiationMode": "photons",
    "machine":       "Generic",
    "bioModel":      "none",
    "multScen":      "nomScen",
    "numOfFractions": 30,
    "propStf": {
        "gantryAngles": [b['gantryAngle'] for b in stf_ml],
        "couchAngles":  [0] * n_beams,
        "bixelWidth":   5,
    },
    "propOpt": {"runDAO": False, "runSequencing": False},
    "propDoseCalc": {"doseGrid": {"resolution": {
        "x": dg_res_x, "y": dg_res_y, "z": dg_res_z,
    }}},
}

dij = calc_dose_influence(ct, cst, stf_ml, pln)
D_py = dij['physicalDose'][0].tocsc()

total_bixels_py = D_py.shape[1]
print(f"  pyMatRad DIJ : {D_py.shape}  nnz={D_py.nnz}")
print(f"  Total bixels : pyMatRad={total_bixels_py}  MATLAB={total_bixels_ml}")
if total_bixels_py != total_bixels_ml:
    print(f"  WARNING: bixel count mismatch!")

w_uniform    = np.ones(total_bixels_py)
dose_py_flat = np.asarray(D_py @ w_uniform).ravel()

dg_py      = dij['doseGrid']
Ny_py      = int(dg_py['dimensions'][0])
Nx_py      = int(dg_py['dimensions'][1])
Nz_py      = int(dg_py['dimensions'][2])
dg_py_origin  = (float(dg_py['x'][0]), float(dg_py['y'][0]), float(dg_py['z'][0]))
dg_py_spacing = (dg_py['resolution']['x'], dg_py['resolution']['y'], dg_py['resolution']['z'])
print(f"\n  pyMatRad dose grid : {Ny_py}×{Nx_py}×{Nz_py}  res={dg_py_spacing}")
print(f"  pyMatRad max dose  : {dose_py_flat.max():.4f} Gy/fx")


# ==========================================================================
# 4.  Dose comparison
# ==========================================================================
print("\n" + "=" * 65)
print("STEP 4 — Dose comparison (uniform fluence, w = 1)")
print("=" * 65)

w_ml_uniform = np.ones(D_ml.shape[1])
dose_ml_flat = np.asarray(D_ml @ w_ml_uniform).ravel()

def compare(label, v_py, v_ml):
    err  = (v_py - v_ml) / v_ml * 100 if v_ml != 0 else float('nan')
    flag = "  <<<" if abs(err) > 5 else ""
    print(f"  {label:<35} {v_py:>12.4f}  {v_ml:>12.4f}  {err:>8.2f}%{flag}")

print(f"\n  {'Metric':<35} {'pyMatRad':>12}  {'MATLAB':>12}  {'Err%':>8}")
print("  " + "-" * 72)
compare("Max dose (Gy/fx)",      dose_py_flat.max(),                    dose_ml_flat.max())
compare("Mean dose > 0 (Gy/fx)", dose_py_flat[dose_py_flat>0].mean(),   dose_ml_flat[dose_ml_flat>0].mean())
compare("DIJ total sum",         D_py.data.sum(),                       D_ml.data.sum())
compare("DIJ nnz",               float(D_py.nnz),                       float(D_ml.nnz))
compare("DIJ max entry",         D_py.data.max(),                       D_ml.data.max())

print(f"\n  Per-beam DIJ column-sum + SSD comparison:")
print(f"  {'Beam':>5} {'Gantry':>7}  {'py_sum':>13}  {'ml_sum':>13}  {'Err%':>7}  {'py_SSD':>8}  {'ml_SSD':>8}")
print("  " + "-" * 75)
ML_SSDS = [939, 920, 849, 828, 902, 902, 826, 847, 920]   # from console log
bixel_offset = 0
for ib, beam in enumerate(stf_ml):
    nb   = beam['totalNumOfBixels']
    cols = slice(bixel_offset, bixel_offset + nb)
    py_s = float(D_py[:, cols].data.sum()) if D_py[:, cols].nnz > 0 else 0.0
    ml_s = float(D_ml[:, cols].data.sum()) if D_ml[:, cols].nnz > 0 else 0.0
    err  = (py_s - ml_s) / ml_s * 100 if ml_s != 0 else float('nan')
    flag = "  <<<" if abs(err) > 5 else ""
    # pyMatRad SSD: use center-most ray (closest to isocenter in BEV)
    rays_bev = np.array([r['rayPos_bev'] for r in beam['ray']])
    ctr = int(np.argmin(np.sum(rays_bev**2, axis=1)))
    py_ssd = float(beam['ray'][ctr].get('SSD', float('nan')))
    ml_ssd = ML_SSDS[ib] if ib < len(ML_SSDS) else float('nan')
    print(f"  {ib+1:>5} {beam['gantryAngle']:>7.1f}°  {py_s:>13.4f}  {ml_s:>13.4f}  "
          f"{err:>7.2f}%  {py_ssd:>8.1f}  {ml_ssd:>8.1f}{flag}")
    bixel_offset += nb


# ==========================================================================
# 4b.  HU → density diagnostics
# ==========================================================================
print("\n" + "=" * 65)
print("STEP 4b — HU → density diagnostics")
print("=" * 65)

hlut = np.asarray(ct_raw.hlut)          # (N,2): col0=HU, col1=RED
hu_test = np.array([-1000, -500, -100, 0, 100, 500, 700, 1000, 1040])

red_hlut   = np.interp(hu_test, hlut[:, 0], hlut[:, 1])
red_linear = np.where(hu_test <= -1000, 0.0, np.clip(1.0 + hu_test/1000.0, 0.0, 3.0))

print(f"  HLUT table: {hlut.shape[0]} points,  "
      f"HU range [{hlut[:,0].min():.0f}, {hlut[:,0].max():.0f}],  "
      f"RED range [{hlut[:,1].min():.4f}, {hlut[:,1].max():.4f}]")
print(f"\n  {'HU':>6}  {'RED (HLUT)':>12}  {'RED (linear)':>14}  {'Diff':>8}")
print("  " + "-" * 48)
for hu, r_h, r_l in zip(hu_test, red_hlut, red_linear):
    print(f"  {hu:>6.0f}  {r_h:>12.4f}  {r_l:>14.4f}  {r_h-r_l:>8.4f}")

# Check which density cube pyMatRad will actually use
if 'cube' in ct and ct['cube'] is not None:
    cube_vals = ct['cube'][0].ravel()
    print(f"\n  ct['cube'] exists → pyMatRad will use pre-computed density directly")
    print(f"  density range: [{cube_vals.min():.4f}, {cube_vals.max():.4f}]  "
          f"mean={cube_vals.mean():.4f}")
    # Compare a sample of HU→density via HLUT vs ct.cube
    hu_flat  = ct['cubeHU'][0].ravel()
    red_from_hlut = np.interp(hu_flat, hlut[:, 0], hlut[:, 1])
    diff = cube_vals - red_from_hlut
    print(f"  ct.cube vs HLUT(cubeHU): max_diff={np.abs(diff).max():.6f}  "
          f"mean_diff={diff.mean():.6f}")
    if np.allclose(cube_vals, red_from_hlut, atol=1e-4):
        print("  → ct.cube matches HLUT conversion ✓")
    else:
        print("  → ct.cube DIFFERS from HLUT conversion — investigate!")
else:
    print("\n  ct['cube'] not found → pyMatRad will use linear HU→RED fallback (ISSUE!)")

# Count voxels with non-zero dose in each
nnz_py = int(np.sum(dose_py_flat > 0))
nnz_ml = int(np.sum(dose_ml_flat > 0))
print(f"\n  Non-zero dose voxels:  pyMatRad={nnz_py:,}  MATLAB={nnz_ml:,}  "
      f"diff={nnz_py-nnz_ml:+,}")
print(f"  → pyMatRad spreads dose to {abs(nnz_py-nnz_ml):,} {'more' if nnz_py>nnz_ml else 'fewer'} voxels")
print(f"  → This {'lowers' if nnz_py>nnz_ml else 'raises'} the mean without affecting the peak")
print(f"  → Likely cause: different kernel lateral cutoff extent")


# ==========================================================================
# 4c.  Dose profiles at isocenter  (x, y, z)
# ==========================================================================
print("\n" + "=" * 65)
print("STEP 4c — Dose profiles at isocenter")
print("=" * 65)

import matplotlib.pyplot as plt

# Dose grid coordinates (both doses on same 167×167×107 grid)
dose_x = dg_x   # (Nx_dg,) but stored as x-coords of dose grid
dose_y = dg_y
dose_z = dg_z

# Isocenter: use first beam's isoCenter
iso_world = stf_ml[0]['isoCenter']   # (x, y, z) in mm
print(f"  Isocenter (world) : {iso_world}")

# Find nearest dose-grid voxel indices
# Dose grid dims: (Ny_dg, Nx_dg, Nz_dg) with axes dg_y, dg_x, dg_z
iso_iy = int(np.argmin(np.abs(dg_y - iso_world[1])))
iso_ix = int(np.argmin(np.abs(dg_x - iso_world[0])))
iso_iz = int(np.argmin(np.abs(dg_z - iso_world[2])))
print(f"  Nearest dose voxel: iy={iso_iy} (y={dg_y[iso_iy]:.1f}mm)  "
      f"ix={iso_ix} (x={dg_x[iso_ix]:.1f}mm)  iz={iso_iz} (z={dg_z[iso_iz]:.1f}mm)")

# Reshape flat doses → 3D (Ny, Nx, Nz) on dose grid
dose_py_3d = dose_py_flat.reshape((Ny_py, Nx_py, Nz_py), order='F')
dose_ml_3d = dose_ml_flat.reshape((Ny_dg, Nx_dg, Nz_dg), order='F')

# Extract profiles (converting to total dose: × NUM_FX)
NUM_FX = 30
prof_x_py = dose_py_3d[iso_iy, :, iso_iz] * NUM_FX   # along x: vary ix
prof_x_ml = dose_ml_3d[iso_iy, :, iso_iz] * NUM_FX
prof_y_py = dose_py_3d[:, iso_ix, iso_iz] * NUM_FX   # along y: vary iy
prof_y_ml = dose_ml_3d[:, iso_ix, iso_iz] * NUM_FX
prof_z_py = dose_py_3d[iso_iy, iso_ix, :] * NUM_FX   # along z: vary iz
prof_z_ml = dose_ml_3d[iso_iy, iso_ix, :] * NUM_FX

# Save profiles as CSV
import csv
PROFILE_DIR = DATA_DIR
profiles = [
    ('x', dg_x, prof_x_py, prof_x_ml),
    ('y', dg_y, prof_y_py, prof_y_ml),
    ('z', dg_z, prof_z_py, prof_z_ml),
]
for axis, coords, py_prof, ml_prof in profiles:
    csv_path = os.path.join(PROFILE_DIR, f'profile_{axis}.csv')
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow([f'{axis}_mm', 'dose_matrad_Gy', 'dose_pymatrad_Gy', 'diff_Gy', 'err_pct'])
        for coord, d_ml, d_py in zip(coords, ml_prof, py_prof):
            diff = d_py - d_ml
            err  = (diff / d_ml * 100) if d_ml > 0.01 else float('nan')
            writer.writerow([f'{coord:.1f}', f'{d_ml:.6f}', f'{d_py:.6f}',
                             f'{diff:.6f}', f'{err:.2f}' if not np.isnan(err) else 'nan'])
    print(f"  Saved profile_{axis}.csv  ({len(coords)} points,  "
          f"py_max={py_prof.max():.2f} Gy  ml_max={ml_prof.max():.2f} Gy)")

# Plot all three profiles
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Dose profiles at isocenter — matRad vs pyMatRad (uniform fluence, 30 fx)', fontsize=12)

for row, (axis, coords, py_prof, ml_prof) in enumerate(profiles):
    ax_dose = axes[row, 0]
    ax_diff = axes[row, 1]

    ax_dose.plot(coords, ml_prof, 'b-',  linewidth=1.5, label='matRad')
    ax_dose.plot(coords, py_prof, 'r--', linewidth=1.5, label='pyMatRad')
    ax_dose.set_xlabel(f'{axis} [mm]')
    ax_dose.set_ylabel('Dose [Gy]')
    ax_dose.set_title(f'{axis.upper()}-profile (through isocenter)')
    ax_dose.legend()
    ax_dose.grid(True, alpha=0.3)

    diff = py_prof - ml_prof
    ax_diff.plot(coords, diff, 'g-', linewidth=1.0)
    ax_diff.axhline(0, color='k', linewidth=0.5)
    ax_diff.set_xlabel(f'{axis} [mm]')
    ax_diff.set_ylabel('pyMatRad − matRad [Gy]')
    ax_diff.set_title(f'{axis.upper()}-profile difference')
    ax_diff.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(PROFILE_DIR, 'dose_profiles.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\n  Saved dose_profiles.png  →  {fig_path}")
plt.show(block=True)   # display interactively; close window to continue to OpenTPS GUI


# ==========================================================================
# 5.  Export to MHD for OpenTPS GUI
# ==========================================================================
print("\n" + "=" * 65)
print("STEP 5 — Export MHD files  →  " + DATA_DIR)
print("=" * 65)

from opentps.core.data.images._ctImage import CTImage
from opentps.core.data.images._doseImage import DoseImage
from opentps.core.io.mhdIO import exportImageMHD

NUM_FX = 30

# CT
ct_arr = to_opentps(ct['cubeHU'][0]).astype(np.float32)
ct_img = CTImage(imageArray=ct_arr, name="TG119_CT",
                 origin=ct_origin, spacing=ct_spacing)
ct_path = os.path.join(DATA_DIR, 'ct.mhd')
exportImageMHD(ct_path, ct_img)
print(f"  Saved CT           : {ct_path}")

# matRad dose — D_ml @ ones on dose grid
ml_dose_opentps = flat_dose_to_opentps(dose_ml_flat, Ny_dg, Nx_dg, Nz_dg).astype(np.float32) * NUM_FX
ml_origin  = (float(dg_x[0]), float(dg_y[0]), float(dg_z[0]))
ml_spacing = (dg_res_x, dg_res_y, dg_res_z)
ml_dose_img = DoseImage(imageArray=ml_dose_opentps, name="matRad_uniform_dose",
                        origin=ml_origin, spacing=ml_spacing)
ml_path = os.path.join(DATA_DIR, 'dose_matrad.mhd')
exportImageMHD(ml_path, ml_dose_img)
print(f"  Saved matRad dose  : {ml_path}  (max={ml_dose_opentps.max():.2f} Gy, ×{NUM_FX} fx)")

# pyMatRad dose — D_py @ ones on pyMatRad dose grid
py_dose_opentps = flat_dose_to_opentps(dose_py_flat, Ny_py, Nx_py, Nz_py).astype(np.float32) * NUM_FX
py_dose_img = DoseImage(imageArray=py_dose_opentps, name="pyMatRad_uniform_dose",
                        origin=dg_py_origin, spacing=dg_py_spacing)
py_path = os.path.join(DATA_DIR, 'dose_pymatrad.mhd')
exportImageMHD(py_path, py_dose_img)
print(f"  Saved pyMatRad dose: {py_path}  (max={py_dose_opentps.max():.2f} Gy, ×{NUM_FX} fx)")

print(f"\n  MHD export complete → {DATA_DIR}")


# ==========================================================================
# 6.  Launch OpenTPS GUI
# ==========================================================================
print("\n" + "=" * 65)
print("STEP 6 — Launch OpenTPS GUI")
print("=" * 65)

logging.basicConfig(level=logging.WARNING)

from PyQt5.QtWidgets import QApplication
from opentps.core.data import PatientList
from opentps.core.data._patient import Patient
from opentps.core.io.mhdIO import importImageMHD
from opentps.core.utils.programSettings import ProgramSettings
from opentps.gui.viewController import ViewController

ct_loaded   = importImageMHD(ct_path)
d_ml_loaded = importImageMHD(ml_path)
d_py_loaded = importImageMHD(py_path)

ct_image = CTImage(imageArray=ct_loaded._imageArray,   name="CT (TG119)",
                   origin=ct_loaded._origin,            spacing=ct_loaded._spacing)
dose_matrad   = DoseImage(imageArray=d_ml_loaded._imageArray, name="matRad  (uniform, Gy)",
                          origin=d_ml_loaded._origin,         spacing=d_ml_loaded._spacing)
dose_pymatrad = DoseImage(imageArray=d_py_loaded._imageArray, name="pyMatRad (uniform, Gy)",
                          origin=d_py_loaded._origin,         spacing=d_py_loaded._spacing)

print(f"  CT           : {ct_image._imageArray.shape}  max HU={ct_image._imageArray.max():.0f}")
print(f"  matRad dose  : {dose_matrad._imageArray.shape}  max={dose_matrad._imageArray.max():.2f} Gy")
print(f"  pyMatRad dose: {dose_pymatrad._imageArray.shape}  max={dose_pymatrad._imageArray.max():.2f} Gy")

patient = Patient(name="example2_no_opti")
patient.appendPatientData(ct_image)
patient.appendPatientData(dose_matrad)
patient.appendPatientData(dose_pymatrad)

patientList = PatientList()
patientList.append(patient)

app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

mainConfig = ProgramSettings()
vc = ViewController(patientList)
vc.mainConfig = mainConfig
vc.dose1 = dose_matrad
vc.dose2 = dose_pymatrad

print("\n  Starting OpenTPS GUI ...")
print("  dose1 = matRad  uniform dose (MATLAB reference)")
print("  dose2 = pyMatRad uniform dose")

vc.mainWindow.show()
app.exec_()
