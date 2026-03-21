"""
Debug example2 (TG119) dose calculation.
Compares per-beam DIJ column sums between pyMatRad and MATLAB.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import h5py

REF = r'C:\Users\jkim20\Desktop\projects\tps\pyMatRad\examples\_matRad_ref_outputs\example2\matRad_example2_ref.mat'
TG119_MAT = r'C:\Users\jkim20\Desktop\projects\tps\matRad\matRad\phantoms\TG119.mat'

# ============================================================
# Load MATLAB DIJ and STF
# ============================================================
print("Loading MATLAB reference...")
f = h5py.File(REF, 'r')

ref_pd = f['dij/physicalDose'][0, 0]
sp_grp = f[ref_pd]
ir   = np.array(sp_grp['ir'],   dtype=np.int64)
jc   = np.array(sp_grp['jc'],   dtype=np.int64)
data_ml = np.array(sp_grp['data'], dtype=np.float64)

n_vox_dose = int(float(np.array(f['dij/doseGrid/numOfVoxels']).ravel()[0]))
D_matlab = sp.csc_matrix((data_ml, ir, jc), shape=(n_vox_dose, len(jc)-1))
print(f"MATLAB DIJ: {D_matlab.shape}, nnz={D_matlab.nnz}")

def h5val(f, ref):
    return np.array(f[ref]).ravel()

stf_h5 = f['stf']
n_beams = stf_h5['gantryAngle'].shape[0]

stf_py = []
for b in range(n_beams):
    ga  = float(h5val(f, stf_h5['gantryAngle'][b, 0])[0])
    ca  = float(h5val(f, stf_h5['couchAngle'][b, 0])[0])
    sad = float(h5val(f, stf_h5['SAD'][b, 0])[0])
    iso = h5val(f, stf_h5['isoCenter'][b, 0])
    sp_w  = h5val(f, stf_h5['sourcePoint'][b, 0])
    sp_bev = h5val(f, stf_h5['sourcePoint_bev'][b, 0])
    nr  = int(h5val(f, stf_h5['numOfRays'][b, 0])[0])
    bw  = float(h5val(f, stf_h5['bixelWidth'][b, 0])[0])
    n_tot = int(h5val(f, stf_h5['totalNumOfBixels'][b, 0])[0])
    ray_grp = f[stf_h5['ray'][b, 0]]
    rays = []
    for r in range(nr):
        tp  = h5val(f, ray_grp['targetPoint'][r, 0])
        rbev = h5val(f, ray_grp['rayPos_bev'][r, 0])
        tp_bev = np.array([2.0 * rbev[0], sad, 2.0 * rbev[2]])
        ray_d = {'targetPoint': tp, 'targetPoint_bev': tp_bev,
                 'rayPos_bev': rbev, 'numOfBixels': 1}
        if 'SSD' in ray_grp:
            try:
                ray_d['SSD'] = float(h5val(f, ray_grp['SSD'][r, 0])[0])
            except: pass
        rays.append(ray_d)
    stf_py.append({'gantryAngle': ga, 'couchAngle': ca, 'SAD': sad,
                   'isoCenter': iso, 'sourcePoint': sp_w, 'sourcePoint_bev': sp_bev,
                   'numOfRays': nr, 'ray': rays, 'bixelWidth': bw,
                   'totalNumOfBixels': n_tot})
f.close()
print(f"STF loaded: {n_beams} beams, {sum(b['totalNumOfBixels'] for b in stf_py)} total bixels")

# ============================================================
# Load TG119 CT/CST
# ============================================================
print("Loading TG119.mat...")
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
    cst.append([int(row[0]), str(row[1]), str(row[2]), [vox], {}, row[5]])

# ============================================================
# Run Python dose calc
# ============================================================
from matRad.doseCalc.calc_dose_influence import calc_dose_influence

pln = {
    "radiationMode": "photons", "machine": "Generic",
    "bioModel": "none", "multScen": "nomScen", "numOfFractions": 30,
    "propStf": {"gantryAngles": [b['gantryAngle'] for b in stf_py],
                "couchAngles": [0]*n_beams, "bixelWidth": 5,
                "isoCenter": None, "visMode": 0,
                "addMargin": True, "fillEmptyBixels": False},
    "propOpt": {"runDAO": False, "runSequencing": False},
    "propDoseCalc": {"doseGrid": {"resolution": {"x": 3, "y": 3, "z": 3}}},
}

dij = calc_dose_influence(ct, cst, stf_py, pln)
D_python = dij["physicalDose"][0].tocsc()
print(f"Python DIJ: {D_python.shape}, nnz={D_python.nnz}")

# ============================================================
# Per-beam comparison
# ============================================================
print("\n" + "=" * 70)
print("PER-BEAM FULL DIJ COLUMN SUM")
print("=" * 70)
print(f"  {'Beam':>4} {'GA':>6} {'Bixels':>7} | {'py_colsum':>12} {'ml_colsum':>12} {'Err%':>8}")
print("  " + "-" * 60)

bixel_counter = 0
for b, beam in enumerate(stf_py):
    nb = beam['totalNumOfBixels']
    cols = slice(bixel_counter, bixel_counter + nb)
    py_sum = D_python[:, cols].data.sum() if D_python[:, cols].nnz > 0 else 0.0
    ml_sum = D_matlab[:, cols].data.sum() if D_matlab[:, cols].nnz > 0 else 0.0
    err = (py_sum - ml_sum) / ml_sum * 100 if ml_sum != 0 else float('nan')
    flag = "  <<" if abs(err) > 5 else ""
    print(f"  {b+1:>4} {beam['gantryAngle']:>6.1f}° {nb:>7} | {py_sum:>12.4f} {ml_sum:>12.4f} {err:>8.2f}%{flag}")
    bixel_counter += nb

print("\n" + "=" * 70)
print("PER-BEAM CENTER-BIXEL COMPARISON")
print("=" * 70)
print(f"  {'Beam':>4} {'GA':>6} {'Rays':>5} | {'py_max':>10} {'ml_max':>10} {'maxRatio':>9} | {'py_sum':>10} {'ml_sum':>10} {'sumRatio':>9}")
print("  " + "-" * 90)

bixel_counter = 0
for b, beam in enumerate(stf_py):
    nb = beam['totalNumOfBixels']
    rays_bev = np.array([r['rayPos_bev'] for r in beam['ray']])
    center_idx = int(np.argmin(np.sum(rays_bev**2, axis=1)))
    center_bixel = bixel_counter + center_idx

    py_col = np.asarray(D_python[:, center_bixel].todense()).ravel()
    ml_col = np.asarray(D_matlab[:, center_bixel].todense()).ravel()

    py_max = py_col.max(); ml_max = ml_col.max()
    py_sum = py_col.sum(); ml_sum = ml_col.sum()
    max_ratio = py_max / ml_max if ml_max > 0 else float('nan')
    sum_ratio = py_sum / ml_sum if ml_sum > 0 else float('nan')

    print(f"  {b+1:>4} {beam['gantryAngle']:>6.1f}° {beam['numOfRays']:>5} | "
          f"{py_max:>10.6f} {ml_max:>10.6f} {max_ratio:>9.4f} | "
          f"{py_sum:>10.4f} {ml_sum:>10.4f} {sum_ratio:>9.4f}")
    bixel_counter += nb

print("\n" + "=" * 70)
print("PER-BEAM NNZ COMPARISON")
print("=" * 70)
bixel_counter = 0
for b, beam in enumerate(stf_py):
    nb = beam['totalNumOfBixels']
    cols = slice(bixel_counter, bixel_counter + nb)
    py_nnz = D_python[:, cols].nnz
    ml_nnz = D_matlab[:, cols].nnz
    err = (py_nnz - ml_nnz) / ml_nnz * 100 if ml_nnz != 0 else float('nan')
    print(f"  Beam {b+1} ({beam['gantryAngle']}°): py_nnz={py_nnz}  ml_nnz={ml_nnz}  diff={err:.1f}%")
    bixel_counter += nb

print("\nDone.")
