"""
Compare pyMatRad example2 output against MATLAB matRad reference.

MATLAB reference: matRad_example2_ref.mat (8 beams, 50° spacing)
- physicalDose stored on CT grid (167×167×129 = Ny×Nx×Nz_ct)
- CST voxel indices are 1-based linear into CT grid

Strategy:
  1. Load MATLAB physicalDose → compute per-structure stats
  2. Run pyMatRad with same gantry angles (8 beams, 50°)
  3. Compute per-structure stats for pyMatRad dose
  4. Compare
"""

import os
import sys
import numpy as np
import scipy.io as sio
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REF_FILE  = r'C:\Users\jkim20\Desktop\projects\tps\pyMatRad\examples\_matRad_ref_outputs\example2\matRad_example2_ref.mat'
TG119_MAT = r'C:\Users\jkim20\Desktop\projects\tps\matRad\matRad\phantoms\TG119.mat'

# ============================================================
# Helper functions for HDF5 MATLAB v7.3 reading
# ============================================================
def h5scalar(f, path):
    return float(np.array(f[path]).ravel()[0])

def h5vec(f, path):
    return np.array(f[path]).ravel()


# ============================================================
# 1. Load MATLAB reference
# ============================================================
print("=" * 60)
print("Loading MATLAB reference...")
print("=" * 60)

f = h5py.File(REF_FILE, 'r')

# Dose: stored as (Nz_ct, Nx, Ny) in h5py → .T gives (Ny, Nx, Nz_ct)
ref_dose_h5 = np.array(f['resultGUI/physicalDose'])  # h5py: (129, 167, 167)
ref_dose_mat = ref_dose_h5.T                          # → (Ny=167, Nx=167, Nz=129) on CT grid

ga_raw     = h5vec(f, 'pln/propStf/gantryAngles')
gantry_angles = sorted([float(a) for a in ga_raw])
num_fractions = int(round(h5scalar(f, 'pln/numOfFractions')))
dg_res_x   = h5scalar(f, 'dij/doseGrid/resolution/x')
dg_res_y   = h5scalar(f, 'dij/doseGrid/resolution/y')
dg_res_z   = h5scalar(f, 'dij/doseGrid/resolution/z')
n_bixels_ml = int(h5scalar(f, 'dij/totalNumOfBixels'))

f.close()

print(f"Gantry angles ({len(gantry_angles)} beams): {gantry_angles}")
print(f"Num fractions: {num_fractions}")
print(f"Dose grid res: ({dg_res_x},{dg_res_y},{dg_res_z}) mm")
print(f"MATLAB total bixels: {n_bixels_ml}")
print(f"MATLAB dose max: {ref_dose_mat.max():.4f} Gy/fx  "
      f"({ref_dose_mat.max()*num_fractions:.2f} Gy total)")


# ============================================================
# 2. Load CT + CST from TG119.mat (scipy handles it cleanly)
# ============================================================
print("\nLoading TG119.mat...")
raw = sio.loadmat(TG119_MAT, squeeze_me=True, struct_as_record=False)
ct_raw  = raw['ct']
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
Ny, Nx, Nz = ct['cubeDim']
print(f"CT cubeDim: [{Ny},{Nx},{Nz}]  res: {ct['resolution']}")

cst = []
for i in range(cst_raw.shape[0]):
    row = cst_raw[i]
    vox = np.asarray(row[3], dtype=np.int64).ravel()
    cst.append([int(row[0]), str(row[1]), str(row[2]), vox, {}, row[5]])
    print(f"  [{i}] {row[1]} ({row[2]}): {len(vox)} voxels")


# ============================================================
# 3. Compute MATLAB per-structure statistics
#    physicalDose is on CT grid → use CST 1-based linear indices directly
# ============================================================
print("\n" + "=" * 60)
print("MATLAB per-structure statistics (on CT grid, per fraction):")
print("=" * 60)

ref_flat = ref_dose_mat.ravel(order='F')   # MATLAB column-major flat
n_ct_vox = Ny * Nx * Nz

def struct_stats(dose_flat, vox_1based, n_total):
    vox_0 = np.asarray(vox_1based, dtype=np.int64).ravel() - 1
    valid = vox_0[(vox_0 >= 0) & (vox_0 < n_total)]
    if len(valid) == 0:
        return dict(D_mean=0, D_95=0, D_5=0, D_max=0, n=0)
    d = dose_flat[valid]
    return dict(
        D_mean = float(np.mean(d)),
        D_95   = float(np.percentile(d, 5)),    # D_95 = 5th percentile (covers 95% volume)
        D_5    = float(np.percentile(d, 95)),
        D_max  = float(np.max(d)),
        n      = len(valid),
    )

matlab_stats = {}
print(f"\n{'Structure':<20} {'D_mean':>10} {'D_95':>10} {'D_5':>10} {'D_max':>10}  [Gy total]")
print("-" * 68)
for row in cst:
    name, vtype, vox = row[1], row[2], row[3]
    s = struct_stats(ref_flat, vox, n_ct_vox)
    matlab_stats[name] = s
    fx = num_fractions
    print(f"  {name:<18} {s['D_mean']*fx:>10.2f} {s['D_95']*fx:>10.2f} "
          f"{s['D_5']*fx:>10.2f} {s['D_max']*fx:>10.2f}  ({vtype})")


# ============================================================
# 4. Run pyMatRad with same gantry angles
# ============================================================
from matRad.steering.stf_generator import generate_stf
from matRad.doseCalc.calc_dose_influence import calc_dose_influence
from matRad.optimization.fluence_optimization import fluence_optimization
from matRad.planAnalysis.plan_analysis import plan_analysis

pln = {
    'radiationMode': 'photons',
    'machine':       'Generic',
    'numOfFractions': num_fractions,
    'bioModel': 'none',
    'multScen': 'nomScen',
    'propStf': {
        'gantryAngles': gantry_angles,
        'couchAngles':  [0] * len(gantry_angles),
        'bixelWidth':   5,
        'isoCenter':    None,
        'visMode':      0,
        'addMargin':    True,
    },
    'propOpt': {'runDAO': False, 'runSequencing': False},
    'propDoseCalc': {
        'doseGrid': {'resolution': {'x': dg_res_x, 'y': dg_res_y, 'z': dg_res_z}}
    },
}

print(f"\n{'=' * 60}")
print(f"Running pyMatRad ({len(gantry_angles)} beams, "
      f"{gantry_angles[1]-gantry_angles[0]:.0f}° spacing)...")
print("=" * 60)

print("  Generating STF...")
stf = generate_stf(ct, cst, pln)
total_bixels = sum(b['totalNumOfBixels'] for b in stf)
print(f"  {len(stf)} beams, {total_bixels} bixels  "
      f"(MATLAB had {n_bixels_ml})")

print("  Computing dose influence matrix (this takes ~10 min)...")
dij = calc_dose_influence(ct, cst, stf, pln)
py_dij = dij['physicalDose'][0]
print(f"  DIJ: {py_dij.shape}, {py_dij.nnz} non-zeros")

print("  Optimizing fluence...")
result = fluence_optimization(dij, cst, pln)
result = plan_analysis(result, ct, cst, stf, pln)

py_dose = result['physicalDose']   # (Ny, Nx, Nz_dose) Gy/fraction
print(f"  pyMatRad dose shape: {py_dose.shape}")
print(f"  pyMatRad dose max: {py_dose.max():.4f} Gy/fx  "
      f"({py_dose.max()*num_fractions:.2f} Gy total)")


# ============================================================
# 5. Compute pyMatRad per-structure statistics using plan_analysis QI
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON: pyMatRad vs MATLAB")
print("=" * 60)
print(f"\n{'Structure':<20} {'Metric':<8} "
      f"{'pyMatRad (Gy)':>16} {'MATLAB (Gy)':>14} {'Diff%':>8}")
print("-" * 72)

for qi in result['qi']:
    name  = qi['name']
    if name not in matlab_stats:
        continue
    ms = matlab_stats[name]
    vtype = qi.get('type', '')

    rows = [
        ('D_mean', qi.get('D_mean', 0), ms['D_mean']),
        ('D_95',   qi.get('D_95',   0), ms['D_95']),
        ('D_5',    qi.get('D_5',    0), ms['D_5']),
    ]
    print(f"  {name} ({vtype})")
    for metric, py_val, ref_val in rows:
        py_gy  = py_val  * num_fractions
        ref_gy = ref_val * num_fractions
        if abs(ref_gy) > 0.1:
            diff = (py_gy - ref_gy) / ref_gy * 100
            flag = "  <<" if abs(diff) > 10 else ""
        else:
            diff, flag = 0.0, ""
        print(f"    {metric:<8} {py_gy:>16.2f} {ref_gy:>14.2f} {diff:>7.1f}%{flag}")
    print()

py_max  = py_dose.max() * num_fractions
ref_max = ref_dose_mat.max() * num_fractions
print(f"  Max dose:  pyMatRad={py_max:.2f} Gy  MATLAB={ref_max:.2f} Gy  "
      f"diff={(py_max-ref_max)/ref_max*100:.1f}%")

print("\nDone.")
