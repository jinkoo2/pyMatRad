"""
Debug angle-dependent dose error.
Compares center-bixel depth dose for EVERY beam between pyMatRad and MATLAB.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scipy.sparse as sp
import h5py

REF = r'C:\Users\jkim20\Desktop\projects\tps\pyMatRad\examples\_matRad_ref_outputs\example1\matRad_example1_ref.mat'

# ============================================================
# Load MATLAB DIJ and STF
# ============================================================
f = h5py.File(REF, 'r')

ref_pd = f['dij/physicalDose'][0, 0]
sp_grp = f[ref_pd]
ir   = np.array(sp_grp['ir'],   dtype=np.int64)
jc   = np.array(sp_grp['jc'],   dtype=np.int64)
data_ml = np.array(sp_grp['data'], dtype=np.float64)

dg_dims = np.array(f['dij/doseGrid/dimensions']).ravel().astype(int)
n_vox_dose = int(np.prod(dg_dims))
D_matlab = sp.csc_matrix((data_ml, ir, jc), shape=(n_vox_dose, len(jc)-1))

stf_h5 = f['stf']
n_beams = stf_h5['gantryAngle'].shape[0]

def h5val(f, ref):
    return np.array(f[ref]).ravel()

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
        rbev= h5val(f, ray_grp['rayPos_bev'][r, 0])
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

# ============================================================
# Run Python dose calc
# ============================================================
from matRad.phantoms.builder import PhantomBuilder
from matRad.optimization.DoseObjectives import SquaredDeviation, SquaredOverdosing
from matRad.doseCalc.calc_dose_influence import calc_dose_influence

builder = PhantomBuilder([200, 200, 100], [2, 2, 3], num_of_ct_scen=1)
builder.add_spherical_target("Volume1", radius=20,
    objectives=[SquaredDeviation(penalty=800, d_ref=45).to_dict()], HU=0)
builder.add_box_oar("Volume2", [60, 30, 60], offset=[0, -15, 0],
    objectives=[SquaredOverdosing(penalty=400, d_ref=0).to_dict()], HU=0)
builder.add_box_oar("Volume3", [60, 30, 60], offset=[0, 15, 0],
    objectives=[SquaredOverdosing(penalty=10, d_ref=0).to_dict()], HU=0)
ct, cst = builder.get_ct_cst()

pln = {
    "radiationMode": "photons", "machine": "Generic",
    "bioModel": "none", "multScen": "nomScen", "numOfFractions": 30,
    "propStf": {"gantryAngles": [0,70,140,210,280,350], "couchAngles": [0]*6,
                "bixelWidth": 5, "isoCenter": None, "visMode": 0,
                "addMargin": True, "fillEmptyBixels": False},
    "propOpt": {"runDAO": False, "runSequencing": False},
    "propDoseCalc": {"doseGrid": {"resolution": {"x": 3, "y": 3, "z": 3}}},
}

dij = calc_dose_influence(ct, cst, stf_py, pln)
D_python = dij["physicalDose"][0].tocsc()

# ============================================================
# Per-beam center-bixel comparison
# ============================================================
print("\n" + "=" * 70)
print("PER-BEAM CENTER-BIXEL DEPTH DOSE COMPARISON")
print("=" * 70)
print(f"  {'Beam':>4} {'GA':>6} {'Rays':>5} | {'py_max':>10} {'ml_max':>10} {'maxRatio':>9} | {'py_sum':>10} {'ml_sum':>10} {'sumRatio':>9}")
print("  " + "-" * 80)

bixel_counter = 0
for b, beam in enumerate(stf_py):
    nb = beam['totalNumOfBixels']
    rays_bev = np.array([r['rayPos_bev'] for r in beam['ray']])
    center_idx = int(np.argmin(np.sum(rays_bev**2, axis=1)))
    center_bixel = bixel_counter + center_idx

    py_col = np.asarray(D_python[:, center_bixel].todense()).ravel()
    ml_col = np.asarray(D_matlab[:, center_bixel].todense()).ravel()

    py_max = py_col.max()
    ml_max = ml_col.max()
    py_sum = py_col.sum()
    ml_sum = ml_col.sum()

    max_ratio = py_max / ml_max if ml_max > 0 else float('nan')
    sum_ratio = py_sum / ml_sum if ml_sum > 0 else float('nan')

    print(f"  {b+1:>4} {beam['gantryAngle']:>6.1f}° {beam['numOfRays']:>5} | "
          f"{py_max:>10.6f} {ml_max:>10.6f} {max_ratio:>9.4f} | "
          f"{py_sum:>10.4f} {ml_sum:>10.4f} {sum_ratio:>9.4f}")
    bixel_counter += nb

# ============================================================
# Per-beam ALL-bixel comparison
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
    flag = "  <<"  if abs(err) > 3 else ""
    print(f"  {b+1:>4} {beam['gantryAngle']:>6.1f}° {nb:>7} | {py_sum:>12.4f} {ml_sum:>12.4f} {err:>8.2f}%{flag}")
    bixel_counter += nb

# ============================================================
# Center bixel detailed voxel comparison for beam 1 vs beam 3
# ============================================================
print("\n" + "=" * 70)
print("CENTER-BIXEL TOP VOXELS: BEAM 1 (0°) vs BEAM 3 (140°)")
print("=" * 70)

dg_dims_str = f"dose grid: {dg_dims}"
print(f"  {dg_dims_str}")

bixel_counter = 0
for b in [0, 2]:  # beam 1 and beam 3
    beam = stf_py[b]
    rays_bev = np.array([r['rayPos_bev'] for r in beam['ray']])
    center_idx = int(np.argmin(np.sum(rays_bev**2, axis=1)))
    center_bixel = bixel_counter + center_idx if b == 0 else \
        sum(s['totalNumOfBixels'] for s in stf_py[:b]) + center_idx

    py_col = np.asarray(D_python[:, center_bixel].todense()).ravel()
    ml_col = np.asarray(D_matlab[:, center_bixel].todense()).ravel()

    top = np.argsort(ml_col)[::-1][:10]
    print(f"\n  Beam {b+1} (gantry={beam['gantryAngle']}°), center_bixel={center_bixel}, center_ray rayPos_bev={rays_bev[center_idx]}")
    print(f"  {'vox_idx':>8} {'py_dose':>12} {'ml_dose':>12} {'ratio':>8}")
    for vox in top:
        ratio = py_col[vox] / ml_col[vox] if ml_col[vox] > 0 else float('nan')
        print(f"  {vox:>8} {py_col[vox]:>12.6f} {ml_col[vox]:>12.6f} {ratio:>8.4f}")

# Also check nnz comparison per beam
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
