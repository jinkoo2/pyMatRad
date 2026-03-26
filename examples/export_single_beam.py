"""
Export CT + single-beam dose for example2 (gantry=50 deg) to MHD.

Writes to _data/example2_beam50/:
  ct.mhd               -- TG119 CT
  dose_pymatrad.mhd    -- pyMatRad dose, 50 deg beam only, MATLAB weights
  dose_matrad_full.mhd -- full 8-beam MATLAB dose (for context)
"""

import os, sys
import numpy as np
import scipy.io as sio
import h5py

PYMATRAD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENTPS_CORE  = r'C:\Users\jkim20\Desktop\projects\tps\opentps2\opentps_core'
REF_EX2 = os.path.join(PYMATRAD_ROOT,
    r'examples\_matRad_ref_outputs\example2\matRad_example2_ref.mat')
TG119   = r'C:\Users\jkim20\Desktop\projects\tps\matRad\matRad\phantoms\TG119.mat'
DATA_DIR = os.path.join(PYMATRAD_ROOT, '_data', 'example2_beam50')

sys.path.insert(0, PYMATRAD_ROOT)
sys.path.insert(0, OPENTPS_CORE)

from opentps.core.data.images._ctImage import CTImage
from opentps.core.data.images._doseImage import DoseImage
from opentps.core.io.mhdIO import exportImageMHD


def h5val(f, ref):
    return np.array(f[ref]).ravel()

def to_opentps_array(cube_ny_nx_nz):
    return cube_ny_nx_nz.transpose(1, 0, 2).copy()

def flat_dose_to_opentps(flat, ny, nx, nz):
    return flat.reshape((ny, nx, nz), order='F').transpose(1, 0, 2).copy()

def save_dose(path, arr, origin, spacing, name, fractions=1):
    img = DoseImage(
        imageArray=(arr * fractions).astype(np.float32),
        name=name,
        origin=tuple(float(v) for v in origin),
        spacing=tuple(float(v) for v in spacing),
    )
    exportImageMHD(path, img)
    print(f"  Saved: {path}  max={img._imageArray.max():.3f} Gy")


TARGET_GANTRY = 50.0   # deg
os.makedirs(DATA_DIR, exist_ok=True)

# --- Load TG119 CT ---
print("Loading TG119.mat...")
raw  = sio.loadmat(TG119, squeeze_me=True, struct_as_record=False)
ct_raw = raw['ct']
cst_raw = raw['cst']

ct2 = {
    'cubeDim':    [int(ct_raw.cubeDim[0]), int(ct_raw.cubeDim[1]), int(ct_raw.cubeDim[2])],
    'resolution': {'x': float(ct_raw.resolution.x),
                   'y': float(ct_raw.resolution.y),
                   'z': float(ct_raw.resolution.z)},
    'x':    np.asarray(ct_raw.x).ravel(),
    'y':    np.asarray(ct_raw.y).ravel(),
    'z':    np.asarray(ct_raw.z).ravel(),
    'cubeHU': [np.asarray(ct_raw.cubeHU)],
    'cube':   [np.asarray(ct_raw.cube)],
    'numOfCtScen': 1,
    'hlut':   ct_raw.hlut,
}
Ny2, Nx2, Nz2 = ct2['cubeDim']
ct2_origin  = (ct2['x'][0], ct2['y'][0], ct2['z'][0])
ct2_spacing = (ct2['resolution']['x'], ct2['resolution']['y'], ct2['resolution']['z'])
print(f"  CT: {Ny2}x{Nx2}x{Nz2}  res={ct2_spacing}")

cst2 = []
for i in range(cst_raw.shape[0]):
    row = cst_raw[i]
    vox = np.asarray(row[3], dtype=np.int64).ravel()
    cst2.append([int(row[0]), str(row[1]), str(row[2]), [vox], {}, row[5]])

# Save CT
ct_img = CTImage(
    imageArray=to_opentps_array(ct2['cubeHU'][0]).astype(np.float32),
    name="ex2_CT",
    origin=ct2_origin, spacing=ct2_spacing,
)
exportImageMHD(os.path.join(DATA_DIR, 'ct.mhd'), ct_img)
print(f"  Saved: {DATA_DIR}/ct.mhd")

# --- Load MATLAB reference ---
print("\nLoading MATLAB example2 reference...")
f = h5py.File(REF_EX2, 'r')
num_fx2 = int(round(float(np.array(f['pln/numOfFractions']).ravel()[0])))

# Full MATLAB dose (all beams) for reference
ml_dose2_mat = np.array(f['resultGUI/physicalDose']).T   # (Ny,Nx,Nz)

stf_h5  = f['stf']
n_beams = stf_h5['gantryAngle'].shape[0]
stf_all = []
beam_idx = None

for ib in range(n_beams):
    ga  = float(h5val(f, stf_h5['gantryAngle'][ib, 0])[0])
    ca  = float(h5val(f, stf_h5['couchAngle'][ib, 0])[0])
    sad = float(h5val(f, stf_h5['SAD'][ib, 0])[0])
    bw  = float(h5val(f, stf_h5['bixelWidth'][ib, 0])[0])
    nr  = int(h5val(f, stf_h5['numOfRays'][ib, 0])[0])
    nb  = int(h5val(f, stf_h5['totalNumOfBixels'][ib, 0])[0])
    iso = h5val(f, stf_h5['isoCenter'][ib, 0])
    sp_b= h5val(f, stf_h5['sourcePoint_bev'][ib, 0])
    sp  = h5val(f, stf_h5['sourcePoint'][ib, 0])
    ray_grp = f[stf_h5['ray'][ib, 0]]
    rays = []
    for ir in range(nr):
        rpb = h5val(f, ray_grp['rayPos_bev'][ir, 0])
        tpb = h5val(f, ray_grp['targetPoint_bev'][ir, 0])
        rp  = h5val(f, ray_grp['rayPos'][ir, 0])
        tp  = h5val(f, ray_grp['targetPoint'][ir, 0])
        rays.append({
            'rayPos_bev':      rpb.astype(float),
            'targetPoint_bev': tpb.astype(float),
            'rayPos':          rp.astype(float),
            'targetPoint':     tp.astype(float),
            'energy':          np.array([6.0]),
        })
    beam = {'gantryAngle': ga, 'couchAngle': ca, 'SAD': sad, 'bixelWidth': bw,
            'isoCenter': iso, 'sourcePoint': sp, 'sourcePoint_bev': sp_b,
            'numOfRays': nr, 'totalNumOfBixels': nb, 'ray': rays,
            'radiationMode': 'photons', 'machine': 'Generic'}
    stf_all.append(beam)
    print(f"    beam[{ib}]: gantry={ga:.0f}°  rays={nr}  bixels={nb}")
    if abs(ga - TARGET_GANTRY) < 0.5:
        beam_idx = ib

if beam_idx is None:
    print(f"ERROR: no beam found near gantry={TARGET_GANTRY}°")
    f.close()
    sys.exit(1)

print(f"\n  Using beam[{beam_idx}] at gantry={stf_all[beam_idx]['gantryAngle']:.0f} deg")

# Extract weight slice for target beam
w_all = np.array(f['resultGUI/w_coarse']).ravel()
offset = sum(stf_all[i]['totalNumOfBixels'] for i in range(beam_idx))
count  = stf_all[beam_idx]['totalNumOfBixels']
w_beam = w_all[offset:offset + count]
print(f"  weight slice: [{offset}:{offset+count}]  nnz={np.sum(w_beam>0)}/{count}  max={w_beam.max():.4f}")

f.close()

# Save full MATLAB dose for reference
save_dose(os.path.join(DATA_DIR, 'dose_matrad_full.mhd'),
          to_opentps_array(ml_dose2_mat),
          ct2_origin, ct2_spacing, name="matRad full dose", fractions=num_fx2)

# --- Run pyMatRad dose calc (single beam) ---
print("\nRunning pyMatRad dose calc (single beam, gantry=50°)...")
stf_single = [stf_all[beam_idx]]

pln2 = {
    "radiationMode": "photons", "machine": "Generic",
    "bioModel": "none", "multScen": "nomScen", "numOfFractions": num_fx2,
    "propStf": {"gantryAngles": [TARGET_GANTRY], "couchAngles": [0], "bixelWidth": 5},
    "propOpt": {"runDAO": False},
    "propDoseCalc": {"doseGrid": {"resolution": ct2['resolution']}},
}

from matRad.doseCalc.calc_dose_influence import calc_dose_influence
dij = calc_dose_influence(ct2, cst2, stf_single, pln2)
D   = dij['physicalDose'][0].tocsc()
dose_flat = np.asarray(D @ w_beam).ravel()
print(f"  Single-beam dose max: {dose_flat.max():.4f} Gy/fx  (x{num_fx2} fx = {dose_flat.max()*num_fx2:.2f} Gy)")

dg = dij['doseGrid']
Ny_dg, Nx_dg, Nz_dg = [int(d) for d in dg['dimensions']]
dg_origin  = (dg['x'][0], dg['y'][0], dg['z'][0])
dg_spacing = (dg['resolution']['x'], dg['resolution']['y'], dg['resolution']['z'])

save_dose(os.path.join(DATA_DIR, 'dose_pymatrad.mhd'),
          flat_dose_to_opentps(dose_flat, Ny_dg, Nx_dg, Nz_dg),
          dg_origin, dg_spacing, name="pyMatRad 50deg beam", fractions=num_fx2)

print(f"\nDone. Data in: {DATA_DIR}")
print("  ct.mhd              -- TG119 CT")
print("  dose_pymatrad.mhd   -- pyMatRad, 50° beam only")
print("  dose_matrad_full.mhd-- MATLAB full 8-beam dose")
