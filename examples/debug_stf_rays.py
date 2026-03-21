"""
Quick check: STF ray counts vs MATLAB reference for TG119.
Tests the margin fix in _initialize without running dose calc.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scipy.io as sio
import h5py

REF = r'C:\Users\jkim20\Desktop\projects\tps\pyMatRad\examples\_matRad_ref_outputs\example2\matRad_example2_ref.mat'
TG119_MAT = r'C:\Users\jkim20\Desktop\projects\tps\matRad\matRad\phantoms\TG119.mat'

# Load MATLAB STF ray counts
f = h5py.File(REF, 'r')
def h5val(f, ref):
    return np.array(f[ref]).ravel()

stf_h5 = f['stf']
n_beams = stf_h5['gantryAngle'].shape[0]
ml_rays = []
ml_bixels = []
ml_ga = []
for b in range(n_beams):
    ga = float(h5val(f, stf_h5['gantryAngle'][b, 0])[0])
    nr = int(h5val(f, stf_h5['numOfRays'][b, 0])[0])
    nb = int(h5val(f, stf_h5['totalNumOfBixels'][b, 0])[0])
    ml_rays.append(nr)
    ml_bixels.append(nb)
    ml_ga.append(ga)
f.close()

print(f"MATLAB: {n_beams} beams, {sum(ml_bixels)} total bixels")

# Load TG119
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

pln = {
    "radiationMode": "photons", "machine": "Generic",
    "bioModel": "none", "multScen": "nomScen", "numOfFractions": 30,
    "propStf": {"gantryAngles": ml_ga,
                "couchAngles": [0]*n_beams, "bixelWidth": 5,
                "isoCenter": None, "visMode": 0,
                "addMargin": True, "fillEmptyBixels": False},
    "propOpt": {"runDAO": False, "runSequencing": False},
    "propDoseCalc": {"doseGrid": {"resolution": {"x": 3, "y": 3, "z": 3}}},
}

from matRad.steering.stf_generator import generate_stf
stf = generate_stf(ct, cst, pln)

py_rays = [b['numOfRays'] for b in stf]
py_bixels = [b['totalNumOfBixels'] for b in stf]

print(f"\n{'GA':>6} | {'Py_rays':>8} {'ML_rays':>8} | {'Py_bix':>8} {'ML_bix':>8}")
print("-" * 50)
for b in range(n_beams):
    ga = ml_ga[b]
    print(f"  {ga:>4.0f}° | {py_rays[b]:>8} {ml_rays[b]:>8} | {py_bixels[b]:>8} {ml_bixels[b]:>8}")
print(f"  {'TOT':>4}  | {sum(py_rays):>8} {sum(ml_rays):>8} | {sum(py_bixels):>8} {sum(ml_bixels):>8}")

iso = stf[0]['isoCenter']
print(f"\nisoCenter: [{iso[0]:.4f}, {iso[1]:.4f}, {iso[2]:.4f}]")
print("MATLAB iso: [-1.6911, -16.5853, 0.1421]")
