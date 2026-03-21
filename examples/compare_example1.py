"""
Dose Engine Comparison: pyMatRad vs MATLAB matRad — Example 1 (water phantom)

Uses MATLAB's exact STF so any differences are purely in the dose engine:
kernel computation, depth-dose, normalization, lateral spread.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scipy.sparse as sp
import h5py

REF = r'C:\Users\jkim20\Desktop\projects\tps\pyMatRad\examples\_matRad_ref_outputs\example1\matRad_example1_ref.mat'

# ============================================================
# Helper
# ============================================================
def h5val(f, ref):
    return np.array(f[ref]).ravel()

def h5scalar(f, path):
    return float(np.array(f[path]).ravel()[0])

# ============================================================
# 1. Load MATLAB DIJ
# ============================================================
print("=" * 60)
print("Loading MATLAB DIJ (example1)...")
print("=" * 60)

f = h5py.File(REF, 'r')

ref_pd = f['dij/physicalDose'][0, 0]
sp_grp = f[ref_pd]
ir   = np.array(sp_grp['ir'],   dtype=np.int64)
jc   = np.array(sp_grp['jc'],   dtype=np.int64)
data = np.array(sp_grp['data'], dtype=np.float64)

n_bixels_ml = len(jc) - 1
dg_dims = np.array(f['dij/doseGrid/dimensions']).ravel().astype(int)
n_vox_dose = int(np.prod(dg_dims))
D_matlab = sp.csc_matrix((data, ir, jc), shape=(n_vox_dose, n_bixels_ml))
print(f"  MATLAB DIJ: {D_matlab.shape}  nnz={D_matlab.nnz}")
print(f"  data range: [{data.min():.6f}, {data.max():.6f}] Gy/fx per bixel")

dg_res_x = h5scalar(f, 'dij/doseGrid/resolution/x')
dg_res_y = h5scalar(f, 'dij/doseGrid/resolution/y')
dg_res_z = h5scalar(f, 'dij/doseGrid/resolution/z')
dg_x = h5val(f, 'dij/doseGrid/x')
dg_y = h5val(f, 'dij/doseGrid/y')
dg_z = h5val(f, 'dij/doseGrid/z')
print(f"  Dose grid: {dg_dims}  res=({dg_res_x},{dg_res_y},{dg_res_z}) mm")
print(f"  x: [{dg_x[0]:.1f}, {dg_x[-1]:.1f}]  y: [{dg_y[0]:.1f}, {dg_y[-1]:.1f}]  z: [{dg_z[0]:.1f}, {dg_z[-1]:.1f}]")

# ============================================================
# 2. Load MATLAB STF
# ============================================================
print("\nReading MATLAB STF...")
stf_h5 = f['stf']
n_beams = stf_h5['gantryAngle'].shape[0]

stf_py = []
for b in range(n_beams):
    ga  = float(h5val(f, stf_h5['gantryAngle'][b, 0])[0])
    ca  = float(h5val(f, stf_h5['couchAngle'][b, 0])[0])
    sad = float(h5val(f, stf_h5['SAD'][b, 0])[0])
    iso = h5val(f, stf_h5['isoCenter'][b, 0])
    sp_w= h5val(f, stf_h5['sourcePoint'][b, 0])
    sp_bev = h5val(f, stf_h5['sourcePoint_bev'][b, 0])
    nr  = int(h5val(f, stf_h5['numOfRays'][b, 0])[0])
    bw  = float(h5val(f, stf_h5['bixelWidth'][b, 0])[0])
    n_tot = int(h5val(f, stf_h5['totalNumOfBixels'][b, 0])[0])

    ray_grp = f[stf_h5['ray'][b, 0]]
    rays = []
    for r in range(nr):
        tp  = h5val(f, ray_grp['targetPoint'][r, 0])
        rbev= h5val(f, ray_grp['rayPos_bev'][r, 0])
        n_bix_r = 1  # single bixel per ray in SVD engine
        # targetPoint_bev = [2*rayPos_bev[0], SAD, 2*rayPos_bev[2]] (see stf_generator.py)
        tp_bev = np.array([2.0 * rbev[0], sad, 2.0 * rbev[2]])
        ray_d = {
            'targetPoint': tp,
            'targetPoint_bev': tp_bev,
            'rayPos_bev': rbev,
            'numOfBixels': n_bix_r,
        }
        # Load SSD if stored
        if 'SSD' in ray_grp:
            try:
                ssd_val = float(h5val(f, ray_grp['SSD'][r, 0])[0])
                ray_d['SSD'] = ssd_val
            except:
                pass
        rays.append(ray_d)

    beam = {
        'gantryAngle': ga, 'couchAngle': ca, 'SAD': sad,
        'isoCenter': iso, 'sourcePoint': sp_w, 'sourcePoint_bev': sp_bev,
        'numOfRays': nr, 'ray': rays, 'bixelWidth': bw,
        'totalNumOfBixels': n_tot,
    }
    stf_py.append(beam)
    print(f"  beam[{b}]: gantry={ga}°  rays={nr}  bixels={n_tot}")

# Load per-beam MATLAB doses for detailed comparison
print("\nLoading per-beam MATLAB doses...")
ml_beam_doses = []
for b in range(1, n_beams + 1):
    key = f'resultGUI/physicalDose_beam{b}'
    if key in f:
        d = np.array(f[key])  # shape (Nz, Ny, Nx) in MATLAB HDF5 = (Nz, Ny, Nx)
        ml_beam_doses.append(d)
        print(f"  beam{b}: shape={d.shape}  max={d.max():.4f}  sum={d.sum():.2f}")
    else:
        ml_beam_doses.append(None)

# MATLAB resultGUI total dose
ml_total = np.array(f['resultGUI/physicalDose'])
print(f"  total: shape={ml_total.shape}  max={ml_total.max():.4f}  sum={ml_total.sum():.2f}")
f.close()

# ============================================================
# 3. Build CT from scratch (same as Python example1 would)
# ============================================================
print("\n" + "=" * 60)
print("Building CT and running pyMatRad dose calc...")
print("=" * 60)

from matRad.phantoms.builder import PhantomBuilder
from matRad.optimization.DoseObjectives import SquaredDeviation, SquaredOverdosing
from matRad.steering.stf_generator import generate_stf
from matRad.doseCalc.calc_dose_influence import calc_dose_influence

ct_dim = [200, 200, 100]
ct_resolution = [2, 2, 3]
builder = PhantomBuilder(ct_dim, ct_resolution, num_of_ct_scen=1)

objective1 = SquaredDeviation(penalty=800, d_ref=45)
objective2 = SquaredOverdosing(penalty=400, d_ref=0)
objective3 = SquaredOverdosing(penalty=10, d_ref=0)

builder.add_spherical_target("Volume1", radius=20, objectives=[objective1.to_dict()], HU=0)
builder.add_box_oar("Volume2", [60, 30, 60], offset=[0, -15, 0],
                     objectives=[objective2.to_dict()], HU=0)
builder.add_box_oar("Volume3", [60, 30, 60], offset=[0, 15, 0],
                     objectives=[objective3.to_dict()], HU=0)
ct, cst = builder.get_ct_cst()

pln = {
    "radiationMode": "photons", "machine": "Generic",
    "bioModel": "none", "multScen": "nomScen", "numOfFractions": 30,
    "propStf": {
        "gantryAngles": [0, 70, 140, 210, 280, 350],
        "couchAngles": [0]*6, "bixelWidth": 5,
        "isoCenter": None, "visMode": 0, "addMargin": True, "fillEmptyBixels": False,
    },
    "propOpt": {"runDAO": False, "runSequencing": False},
    "propDoseCalc": {"doseGrid": {"resolution": {"x": 3, "y": 3, "z": 3}}},
}

# Use MATLAB STF instead of generating our own
dij = calc_dose_influence(ct, cst, stf_py, pln)
D_python = dij["physicalDose"][0].tocsc()

print(f"\n  pyMatRad DIJ: {D_python.shape}  nnz={D_python.nnz}")

# ============================================================
# 4. SSD comparison
# ============================================================
print("\n" + "=" * 60)
print("SSD COMPARISON")
print("=" * 60)
print(f"  {'Beam':>5} {'Gantry':>8} {'py SSD':>10} {'ml SSD':>10} {'Diff':>8}")
print("  " + "-" * 45)
for b, beam in enumerate(stf_py):
    ga = beam['gantryAngle']
    rays_bev = np.array([r['rayPos_bev'] for r in beam['ray']])
    center_idx = int(np.argmin(np.sum(rays_bev**2, axis=1)))
    py_ssd = float(beam['ray'][center_idx].get('SSD', float('nan')))
    ml_ssd = float(beam['ray'][center_idx].get('SSD', float('nan')))  # same source
    print(f"  {b+1:>5} {ga:>8.1f}° {py_ssd:>10.1f}")

# ============================================================
# 5. DIJ comparison (uniform fluence w=1)
# ============================================================
print("\n" + "=" * 60)
print("DOSE ENGINE COMPARISON (uniform fluence, w=1)")
print("=" * 60)

n_bix_py = D_python.shape[1]
n_bix_ml = D_matlab.shape[1]
print(f"  n_bixels:  pyMatRad={n_bix_py}  MATLAB={n_bix_ml}")

# Apply uniform fluence: dose = D @ ones
w = np.ones(n_bix_py)
dose_py = np.asarray(D_python @ w).ravel()

w_ml = np.ones(n_bix_ml)
dose_ml_flat = np.asarray(D_matlab @ w_ml).ravel()

print(f"\n  Metric                        pyMatRad       MATLAB     Err%")
print("  " + "-" * 65)

def compare(name, v_py, v_ml):
    err = (v_py - v_ml) / v_ml * 100 if v_ml != 0 else float('nan')
    flag = "  <<" if abs(err) > 5 else ""
    print(f"  {name:<28} {v_py:>12.4f} {v_ml:>12.4f} {err:>8.2f}%{flag}")

compare("Max dose (Gy/fx)", dose_py.max(), dose_ml_flat.max())
compare("Mean dose >0 (Gy/fx)",
        dose_py[dose_py > 0].mean(), dose_ml_flat[dose_ml_flat > 0].mean())
compare("DIJ total sum", D_python.data.sum(), D_matlab.data.sum())
compare("DIJ nnz", float(D_python.nnz), float(D_matlab.nnz))
compare("DIJ max entry", D_python.data.max(), D_matlab.data.max())
compare("DIJ mean entry (>0)", D_python.data.mean(), D_matlab.data.mean())

# ============================================================
# 6. Per-beam analysis
# ============================================================
print("\n" + "=" * 60)
print("PER-BEAM DIJ SUM (column sums per beam block)")
print("=" * 60)

# Identify bixel column ranges per beam
bixel_counter = 0
print(f"  {'Beam':>5} {'Gantry':>8} {'py_colsum':>14} {'ml_colsum':>14} {'Err%':>8}")
print("  " + "-" * 55)
for b, beam in enumerate(stf_py):
    nb = beam['totalNumOfBixels']
    cols = slice(bixel_counter, bixel_counter + nb)
    py_sum = D_python[:, cols].data.sum() if D_python[:, cols].nnz > 0 else 0.0
    ml_sum = D_matlab[:, cols].data.sum() if D_matlab[:, cols].nnz > 0 else 0.0
    err = (py_sum - ml_sum) / ml_sum * 100 if ml_sum != 0 else float('nan')
    flag = "  <<" if abs(err) > 5 else ""
    print(f"  {b+1:>5} {beam['gantryAngle']:>8.1f}° {py_sum:>14.4f} {ml_sum:>14.4f} {err:>8.2f}%{flag}")
    bixel_counter += nb

# ============================================================
# 7. Center-bixel depth dose (first beam, center ray)
# ============================================================
print("\n" + "=" * 60)
print("CENTER BIXEL DEPTH DOSE PROFILE (beam 1, center ray)")
print("=" * 60)

# Find center bixel of beam 1
rays_bev_b0 = np.array([r['rayPos_bev'] for r in stf_py[0]['ray']])
center_ray = int(np.argmin(np.sum(rays_bev_b0**2, axis=1)))
center_bixel = center_ray  # 1 bixel per ray

# Get column from DIJ
py_col = np.asarray(D_python[:, center_bixel].todense()).ravel()
ml_col = np.asarray(D_matlab[:, center_bixel].todense()).ravel()

print(f"  Center bixel index: {center_bixel}")
print(f"  py max: {py_col.max():.6f}  at vox {py_col.argmax()}")
print(f"  ml max: {ml_col.max():.6f}  at vox {ml_col.argmax()}")
if ml_col.max() > 0:
    print(f"  ratio py/ml max: {py_col.max()/ml_col.max():.4f}")

# Find voxels with nonzero dose in either
union_nz = np.where((py_col > 0) | (ml_col > 0))[0]
print(f"\n  Nonzero voxels: py={np.sum(py_col>0)}  ml={np.sum(ml_col>0)}")
if len(union_nz) > 0:
    print(f"\n  {'vox_idx':>8} {'py_dose':>12} {'ml_dose':>12} {'ratio':>8}")
    # Print top 20 highest-dose voxels
    top_vox = union_nz[np.argsort(ml_col[union_nz])[::-1][:20]]
    for vox in sorted(top_vox):
        ratio = py_col[vox] / ml_col[vox] if ml_col[vox] > 0 else float('nan')
        print(f"  {vox:>8} {py_col[vox]:>12.6f} {ml_col[vox]:>12.6f} {ratio:>8.4f}")

print("\nDone.")
