"""
Dose Engine Comparison: pyMatRad vs MATLAB matRad

Method:
  1. Read MATLAB's STF (exact ray positions) from matRad_example2_ref.mat
  2. Run pyMatRad dose calc with MATLAB's exact STF
  3. Apply uniform fluence (w=1) to both DIJs: dose = D @ ones
  4. Compare voxel-by-voxel and per-structure

This isolates dose engine differences from any optimizer effects.
"""

import os, sys
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REF_FILE  = r'C:\Users\jkim20\Desktop\projects\tps\pyMatRad\examples\_matRad_ref_outputs\example2\matRad_example2_ref.mat'
TG119_MAT = r'C:\Users\jkim20\Desktop\projects\tps\matRad\matRad\phantoms\TG119.mat'

# =========================================================================
# Helper: dereference a scalar/vector HDF5 object ref
# =========================================================================
def h5val(f, ref):
    return np.array(f[ref]).ravel()

def h5scalar(f, path):
    return float(np.array(f[path]).ravel()[0])

# =========================================================================
# 1. Load MATLAB DIJ (stored as CSC sparse: ir, jc, data)
# =========================================================================
print("=" * 60)
print("Loading MATLAB DIJ...")
print("=" * 60)

f = h5py.File(REF_FILE, 'r')

ref_pd = f['dij/physicalDose'][0, 0]
sp_grp = f[ref_pd]
ir   = np.array(sp_grp['ir'],   dtype=np.int64)   # row indices (0-based)
jc   = np.array(sp_grp['jc'],   dtype=np.int64)   # col pointers
data = np.array(sp_grp['data'], dtype=np.float64)

n_bixels_ml = len(jc) - 1

# Dose grid dims first (needed for correct matrix shape)
dg_dims  = np.array(f['dij/doseGrid/dimensions']).ravel().astype(int)  # [Ny, Nx, Nz]
n_vox_dose_ml = int(np.prod(dg_dims))   # true dose grid size
D_matlab = sp.csc_matrix((data, ir, jc), shape=(n_vox_dose_ml, n_bixels_ml))
print(f"  MATLAB DIJ: {D_matlab.shape}  nnz={D_matlab.nnz}")

dg_res_x = h5scalar(f, 'dij/doseGrid/resolution/x')
dg_res_y = h5scalar(f, 'dij/doseGrid/resolution/y')
dg_res_z = h5scalar(f, 'dij/doseGrid/resolution/z')
dg_x     = h5val(f, 'dij/doseGrid/x')
dg_y     = h5val(f, 'dij/doseGrid/y')
dg_z     = h5val(f, 'dij/doseGrid/z')
print(f"  MATLAB dose grid: {dg_dims}  res=({dg_res_x},{dg_res_y},{dg_res_z}) mm")
print(f"  n_voxels_dose from DIJ: {n_vox_dose_ml}")

num_fractions = int(round(h5scalar(f, 'pln/numOfFractions')))

# =========================================================================
# 2. Reconstruct MATLAB STF in pyMatRad format
# =========================================================================
print("\nReading MATLAB STF...")
stf_h5 = f['stf']
n_beams = stf_h5['gantryAngle'].shape[0]

stf = []
for ib in range(n_beams):
    ga   = float(h5val(f, stf_h5['gantryAngle'][ib, 0])[0])
    ca   = float(h5val(f, stf_h5['couchAngle'][ib, 0])[0])
    sad  = float(h5val(f, stf_h5['SAD'][ib, 0])[0])
    bw   = float(h5val(f, stf_h5['bixelWidth'][ib, 0])[0])
    nr   = int(h5val(f, stf_h5['numOfRays'][ib, 0])[0])
    nb   = int(h5val(f, stf_h5['totalNumOfBixels'][ib, 0])[0])
    iso  = h5val(f, stf_h5['isoCenter'][ib, 0])
    sp_b = h5val(f, stf_h5['sourcePoint_bev'][ib, 0])
    sp   = h5val(f, stf_h5['sourcePoint'][ib, 0])
    nbpr = h5val(f, stf_h5['numOfBixelsPerRay'][ib, 0]).astype(int)

    # Read each ray
    ray_grp = f[stf_h5['ray'][ib, 0]]
    rays = []
    for ir_idx in range(nr):
        rpb  = h5val(f, ray_grp['rayPos_bev'][ir_idx, 0])
        tpb  = h5val(f, ray_grp['targetPoint_bev'][ir_idx, 0])
        rp   = h5val(f, ray_grp['rayPos'][ir_idx, 0])
        tp   = h5val(f, ray_grp['targetPoint'][ir_idx, 0])
        rays.append({
            'rayPos_bev':      rpb.astype(float),
            'targetPoint_bev': tpb.astype(float),
            'rayPos':          rp.astype(float),
            'targetPoint':     tp.astype(float),
            'energy':          np.array([6.0]),
        })

    stf.append({
        'gantryAngle':       ga,
        'couchAngle':        ca,
        'SAD':               sad,
        'bixelWidth':        bw,
        'numOfRays':         nr,
        'totalNumOfBixels':  nb,
        'isoCenter':         iso.astype(float),
        'sourcePoint_bev':   sp_b.astype(float),
        'sourcePoint':       sp.astype(float),
        'numOfBixelsPerRay': nbpr.tolist(),
        'radiationMode':     'photons',
        'machine':           'Generic',
        'ray':               rays,
    })
    print(f"  beam[{ib}]: gantry={ga:.0f}°  rays={nr}  bixels={nb}")

f.close()

# =========================================================================
# 3. Load CT from TG119.mat
# =========================================================================
print("\nLoading CT from TG119.mat...")
raw = sio.loadmat(TG119_MAT, squeeze_me=True, struct_as_record=False)
ct_raw = raw['ct']
cst_raw = raw['cst']

ct = {
    'cubeDim':    [int(ct_raw.cubeDim[0]), int(ct_raw.cubeDim[1]), int(ct_raw.cubeDim[2])],
    'resolution': {'x': float(ct_raw.resolution.x),
                   'y': float(ct_raw.resolution.y),
                   'z': float(ct_raw.resolution.z)},
    'x':          np.asarray(ct_raw.x).ravel(),
    'y':          np.asarray(ct_raw.y).ravel(),
    'z':          np.asarray(ct_raw.z).ravel(),
    'cubeHU':     [np.asarray(ct_raw.cubeHU)],
    'cube':       [np.asarray(ct_raw.cube)],
    'numOfCtScen': 1,
    'hlut':       ct_raw.hlut,
}

cst = []
for i in range(cst_raw.shape[0]):
    row = cst_raw[i]
    vox = np.asarray(row[3], dtype=np.int64).ravel()
    cst.append([int(row[0]), str(row[1]), str(row[2]), vox, {}, row[5]])

# =========================================================================
# 4. Build PLN and run pyMatRad dose calc with MATLAB's STF
# =========================================================================
pln = {
    'radiationMode': 'photons',
    'machine':       'Generic',
    'numOfFractions': num_fractions,
    'propStf': {'gantryAngles': [b['gantryAngle'] for b in stf],
                'couchAngles':  [b['couchAngle']  for b in stf],
                'bixelWidth':   stf[0]['bixelWidth']},
    'propOpt': {},
    'propDoseCalc': {
        'doseGrid': {'resolution': {'x': dg_res_x, 'y': dg_res_y, 'z': dg_res_z}}
    },
}

# MATLAB SSDs from console.log (coarse plan, 8 beams, 50° spacing)
matlab_ssd = {0: 939, 50: 905, 100: 849, 150: 894,
              200: 902, 250: 840, 300: 878, 350: 938}

print("\nRunning pyMatRad dose calc with MATLAB's exact STF...")
from matRad.doseCalc.calc_dose_influence import calc_dose_influence
dij_py = calc_dose_influence(ct, cst, stf, pln)
D_py = dij_py['physicalDose'][0]
print(f"  pyMatRad DIJ: {D_py.shape}  nnz={D_py.nnz}")

# =========================================================================
# SSD comparison (after dose calc, stf rays have 'SSD' field set)
# =========================================================================
print("\n" + "=" * 60)
print("SSD COMPARISON (center ray, per beam)")
print("=" * 60)
print(f"\n  {'Beam':>5} {'Gantry':>8} {'pyMatRad SSD':>14} {'MATLAB SSD':>12} {'Diff mm':>9} {'Diff%':>7}")
print("  " + "-" * 55)
for ib, beam in enumerate(stf):
    ga = int(round(beam['gantryAngle']))
    ray_pos = np.array([np.asarray(r['rayPos_bev']).ravel()
                        for r in beam['ray']])
    center_idx = int(np.argmin(np.sum(ray_pos**2, axis=1)))
    py_ssd = beam['ray'][center_idx].get('SSD', None)
    ml_ssd = matlab_ssd.get(ga, None)
    if py_ssd is not None and ml_ssd is not None:
        diff_mm = py_ssd - ml_ssd
        diff_pct = diff_mm / ml_ssd * 100
        flag = "  <<" if abs(diff_pct) > 3 else ""
        print(f"  {ib+1:>5} {ga:>7}°  {py_ssd:>12.1f} {ml_ssd:>12.0f} "
              f"{diff_mm:>+8.1f} {diff_pct:>+6.1f}%{flag}")
    else:
        print(f"  {ib+1:>5} {ga:>7}°  py={py_ssd}  ml={ml_ssd}")

# =========================================================================
# 5. Uniform-fluence dose comparison
# =========================================================================
print("\n" + "=" * 60)
print("DOSE ENGINE COMPARISON (uniform fluence, w=1 per bixel)")
print("=" * 60)

w_ml = np.ones(n_bixels_ml)
w_py = np.ones(D_py.shape[1])

dose_ml_flat = np.asarray(D_matlab.dot(w_ml)).ravel()   # Gy/fraction
dose_py_flat = np.asarray(D_py.dot(w_py)).ravel()

# Reshape to 3D — MATLAB dose grid [Ny, Nx, Nz]
Ny, Nx, Nz_ml = int(dg_dims[0]), int(dg_dims[1]), int(dg_dims[2])
dose_ml_3d = dose_ml_flat.reshape((Ny, Nx, Nz_ml), order='F')

# pyMatRad dose grid might have different Nz — get from dij
py_dims = dij_py['doseGrid']['dimensions']
dose_py_3d = dose_py_flat.reshape(py_dims, order='F')

print(f"\n  MATLAB dose grid:   {dose_ml_3d.shape}")
print(f"  pyMatRad dose grid: {dose_py_3d.shape}")

# Overall stats
print(f"\n  {'Metric':<35} {'pyMatRad':>12} {'MATLAB':>12} {'Err%':>8}")
print("  " + "-" * 71)

def prow(name, py_v, ml_v):
    err = (py_v - ml_v) / (ml_v + 1e-12) * 100
    flag = "  <<" if abs(err) > 10 else ""
    print(f"  {name:<35} {py_v:>12.4f} {ml_v:>12.4f} {err:>7.2f}%{flag}")

prow("Max dose (Gy/fx)",    dose_py_3d.max(),               dose_ml_3d.max())
prow("Mean dose >0 (Gy/fx)",
     dose_py_flat[dose_py_flat>1e-6].mean(),
     dose_ml_flat[dose_ml_flat>1e-6].mean())
prow("DIJ total sum",       float(D_py.sum()),               float(D_matlab.sum()))
prow("DIJ nnz",             float(D_py.nnz),                 float(D_matlab.nnz))

# Per-structure stats (on CT grid using CST voxel indices)
# Need to remap CST to pyMatRad dose grid first
from matRad.geometry.geometry import resize_cst_to_grid

py_dg = dij_py['doseGrid']
cst_py_dg = resize_cst_to_grid(cst, ct, py_dg)

matlab_dg_info = {
    'dimensions': list(dg_dims),
    'numOfVoxels': int(np.prod(dg_dims)),
    'resolution': {'x': dg_res_x, 'y': dg_res_y, 'z': dg_res_z},
    'x': dg_x, 'y': dg_y, 'z': dg_z,
}
cst_ml_dg = resize_cst_to_grid(cst, ct, matlab_dg_info)

print(f"\n  {'Structure':<18} {'D_mean py':>10} {'D_mean ml':>10} {'Err%':>7} "
      f"  {'D_max py':>10} {'D_max ml':>10} {'Err%':>7}")
print("  " + "-" * 80)

for (row_py, row_ml) in zip(cst_py_dg, cst_ml_dg):
    name = row_py[1]
    vtype = row_py[2]

    vox_py = np.asarray(row_py[3], dtype=np.int64).ravel() - 1
    vox_ml = np.asarray(row_ml[3], dtype=np.int64).ravel() - 1

    valid_py = vox_py[(vox_py >= 0) & (vox_py < len(dose_py_flat))]
    valid_ml = vox_ml[(vox_ml >= 0) & (vox_ml < len(dose_ml_flat))]

    if len(valid_py) == 0 or len(valid_ml) == 0:
        print(f"  {name:<18}  (no valid voxels)")
        continue

    d_py = dose_py_flat[valid_py]
    d_ml = dose_ml_flat[valid_ml]

    mean_py, mean_ml = d_py.mean(), d_ml.mean()
    max_py,  max_ml  = d_py.max(),  d_ml.max()

    err_mean = (mean_py - mean_ml) / (mean_ml + 1e-12) * 100
    err_max  = (max_py  - max_ml ) / (max_ml  + 1e-12) * 100
    flag = "  <<" if abs(err_mean) > 10 else ""

    print(f"  {name:<18} {mean_py*num_fractions:>10.2f} {mean_ml*num_fractions:>10.2f} "
          f"{err_mean:>6.1f}%{flag}  {max_py*num_fractions:>10.2f} "
          f"{max_ml*num_fractions:>10.2f} {err_max:>6.1f}%")
    print(f"  {'':18} {'(n='+str(len(valid_py))+')':>10} {'(n='+str(len(valid_ml))+')':>10}")

# Voxel-level correlation (on common voxels mapped to CT grid)
# Use MATLAB physicalDose (already computed optimal dose) for reference shape
print("\n  Checking voxel-level correlation (uniform dose, dose grid overlap)...")
# Both grids have same Ny, Nx; differ in Nz — compare the overlapping z-slices
Nz_min = min(dose_py_3d.shape[2], Nz_ml)
py_clip = dose_py_3d[:, :, :Nz_min].ravel()
ml_clip = dose_ml_3d[:, :, :Nz_min].ravel()

mask = (ml_clip > 1e-6) | (py_clip > 1e-6)
if mask.sum() > 0:
    corr = np.corrcoef(py_clip[mask], ml_clip[mask])[0, 1]
    rmse = np.sqrt(np.mean((py_clip[mask] - ml_clip[mask])**2))
    print(f"  Pearson correlation (dose>0 voxels): {corr:.6f}")
    print(f"  RMSE (Gy/fx): {rmse:.6f}  ({rmse*num_fractions:.4f} Gy total)")
    print(f"  Mean abs error: {np.mean(np.abs(py_clip[mask]-ml_clip[mask]))*num_fractions:.4f} Gy")

print("\nDone.")
