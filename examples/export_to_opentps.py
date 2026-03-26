"""
Export CT + dose volumes to MHD format for viewing in OpenTPS GUI.

Exports:
  _data/example1/
    ct.mhd              -- water phantom CT (from PhantomBuilder)
    dose_matrad.mhd     -- MATLAB optimized dose (from resultGUI, on CT grid)
    dose_pymatrad.mhd   -- pyMatRad dose w/ MATLAB optimized weights (resultGUI/w)

  _data/example2/
    ct.mhd              -- TG119 CT
    dose_matrad.mhd     -- MATLAB optimized dose (from resultGUI, on CT grid)
    dose_pymatrad.mhd   -- pyMatRad dose w/ MATLAB optimized weights (resultGUI/w_coarse)

Coordinate conventions:
  matRad cubeHU shape: (Ny, Nx, Nz)
  OpenTPS Image3D._imageArray shape: (Nx, Ny, Nz)  [Fortran-order MHD]
  → transpose axis (1,0,2) to convert

  MATLAB HDF5 physicalDose loaded by h5py: (Nz, Nx, Ny)  [HDF5 reverses MATLAB dims]
  → .T → (Ny, Nx, Nz), then transpose (1,0,2) → (Nx, Ny, Nz)

  Flat dose from D @ w indexed by matRad linear index (Fortran order [Ny,Nx,Nz]):
  → reshape([Ny,Nx,Nz], order='F') → (Ny, Nx, Nz), then transpose → (Nx, Ny, Nz)
"""

import os, sys
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import h5py

# ---------- paths ----------
PYMATRAD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENTPS_CORE  = r'C:\Users\jkim20\Desktop\projects\tps\opentps2\opentps_core'
REF_EX1 = os.path.join(PYMATRAD_ROOT,
    r'examples\_matRad_ref_outputs\example1\matRad_example1_ref.mat')
REF_EX2 = os.path.join(PYMATRAD_ROOT,
    r'examples\_matRad_ref_outputs\example2\matRad_example2_ref.mat')
TG119   = r'C:\Users\jkim20\Desktop\projects\tps\matRad\matRad\phantoms\TG119.mat'
DATA_DIR = os.path.join(PYMATRAD_ROOT, '_data')

sys.path.insert(0, PYMATRAD_ROOT)
sys.path.insert(0, OPENTPS_CORE)

from opentps.core.data.images._image3D import Image3D
from opentps.core.data.images._ctImage import CTImage
from opentps.core.data.images._doseImage import DoseImage
from opentps.core.io.mhdIO import exportImageMHD


# ============================================================
# Helpers
# ============================================================

def h5val(f, ref):
    return np.array(f[ref]).ravel()

def h5scalar(f, path):
    return float(np.array(f[path]).ravel()[0])

def to_opentps_array(cube_ny_nx_nz):
    """Convert matRad (Ny,Nx,Nz) array to OpenTPS (Nx,Ny,Nz)."""
    return cube_ny_nx_nz.transpose(1, 0, 2).copy()

def flat_dose_to_opentps(flat, ny, nx, nz):
    """Reshape flat matRad dose to OpenTPS (Nx,Ny,Nz)."""
    return flat.reshape((ny, nx, nz), order='F').transpose(1, 0, 2).copy()

def save_ct(path, arr_nx_ny_nz, origin, spacing, name="CT"):
    img = CTImage(
        imageArray=arr_nx_ny_nz.astype(np.float32),
        name=name,
        origin=tuple(float(v) for v in origin),
        spacing=tuple(float(v) for v in spacing),
    )
    exportImageMHD(path, img)
    print(f"  Saved CT  : {path}")

def save_dose(path, arr_nx_ny_nz, origin, spacing, name="Dose", fractions=1):
    """Save dose in Gy (total = per_fraction * fractions)."""
    img = DoseImage(
        imageArray=(arr_nx_ny_nz * fractions).astype(np.float32),
        name=name,
        origin=tuple(float(v) for v in origin),
        spacing=tuple(float(v) for v in spacing),
    )
    exportImageMHD(path, img)
    print(f"  Saved dose: {path}  (×{fractions} fx, max={img._imageArray.max():.3f} Gy)")


# ============================================================
# EXAMPLE 1 — Water Phantom
# ============================================================
print("=" * 60)
print("EXAMPLE 1 — Water Phantom")
print("=" * 60)

ex1_dir = os.path.join(DATA_DIR, 'example1')
os.makedirs(ex1_dir, exist_ok=True)

# --- Build phantom CT (same as example1 script) ---
from matRad.phantoms.builder import PhantomBuilder
from matRad.optimization.DoseObjectives import SquaredDeviation, SquaredOverdosing

ct_dim = [200, 200, 100]
ct_res  = [2, 2, 3]
builder = PhantomBuilder(ct_dim, ct_res, num_of_ct_scen=1)
builder.add_spherical_target("Volume1", radius=20,
    objectives=[SquaredDeviation(penalty=800, d_ref=45).to_dict()], HU=0)
builder.add_box_oar("Volume2", [60, 30, 60], offset=[0, -15, 0],
    objectives=[SquaredOverdosing(penalty=400, d_ref=0).to_dict()], HU=0)
builder.add_box_oar("Volume3", [60, 30, 60], offset=[0, 15, 0],
    objectives=[SquaredOverdosing(penalty=10, d_ref=0).to_dict()], HU=0)
ct1, cst1 = builder.get_ct_cst()

# PhantomBuilder doesn't add x/y/z — compute them from cubeDim + resolution
from matRad.geometry.geometry import get_world_axes
ct1 = get_world_axes(ct1)

Ny1, Nx1, Nz1 = ct1['cubeDim']
ct1_hu = ct1['cubeHU'][0]              # (Ny, Nx, Nz)
ct1_origin  = (ct1['x'][0], ct1['y'][0], ct1['z'][0])
ct1_spacing = (ct_res[0], ct_res[1], ct_res[2])
print(f"  CT: {Ny1}×{Nx1}×{Nz1}  res={ct_res}  origin={ct1_origin}")

save_ct(os.path.join(ex1_dir, 'ct.mhd'),
        to_opentps_array(ct1_hu), ct1_origin, ct1_spacing, name="ex1_CT")

# --- Load MATLAB reference ---
print("\nLoading MATLAB example1 reference...")
f = h5py.File(REF_EX1, 'r')
num_fx1 = 30  # example1 uses 30 fractions (no pln key in example1 ref)

# MATLAB resultGUI dose is on CT grid: h5py gives (Nz,Nx,Ny) for MATLAB [Ny,Nx,Nz]
ml_dose_h5 = np.array(f['resultGUI/physicalDose'])   # (Nz,Nx,Ny)
ml_dose_mat = ml_dose_h5.T                            # (Ny,Nx,Nz) matRad convention

# Load MATLAB STF for pyMatRad dose calc
stf_h5 = f['stf']
n_beams1 = stf_h5['gantryAngle'].shape[0]
stf1 = []
for ib in range(n_beams1):
    ga   = float(h5val(f, stf_h5['gantryAngle'][ib, 0])[0])
    ca   = float(h5val(f, stf_h5['couchAngle'][ib, 0])[0])
    sad  = float(h5val(f, stf_h5['SAD'][ib, 0])[0])
    bw   = float(h5val(f, stf_h5['bixelWidth'][ib, 0])[0])
    nr   = int(h5val(f, stf_h5['numOfRays'][ib, 0])[0])
    nb   = int(h5val(f, stf_h5['totalNumOfBixels'][ib, 0])[0])
    iso  = h5val(f, stf_h5['isoCenter'][ib, 0])
    sp_b = h5val(f, stf_h5['sourcePoint_bev'][ib, 0])
    sp   = h5val(f, stf_h5['sourcePoint'][ib, 0])
    ray_grp = f[stf_h5['ray'][ib, 0]]
    rays = []
    for ir in range(nr):
        rpb = h5val(f, ray_grp['rayPos_bev'][ir, 0])
        tp  = h5val(f, ray_grp['targetPoint'][ir, 0])
        rays.append({
            'rayPos_bev':      rpb.astype(float),
            'targetPoint_bev': np.array([2*rpb[0], sad, 2*rpb[2]], float),
            'rayPos':          h5val(f, ray_grp['rayPos'][ir, 0]).astype(float) if 'rayPos' in ray_grp else rpb,
            'targetPoint':     tp.astype(float),
            'energy':          np.array([6.0]),
        })
    stf1.append({'gantryAngle': ga, 'couchAngle': ca, 'SAD': sad, 'bixelWidth': bw,
                 'isoCenter': iso, 'sourcePoint': sp, 'sourcePoint_bev': sp_b,
                 'numOfRays': nr, 'totalNumOfBixels': nb, 'ray': rays,
                 'radiationMode': 'photons', 'machine': 'Generic'})
    print(f"    beam[{ib}]: gantry={ga}°  rays={nr}  bixels={nb}")

# Load MATLAB optimized weights for example1 (w matches 2703 bixels)
w1 = np.array(f['resultGUI/w']).ravel()
print(f"  MATLAB w: {len(w1)} weights  max={w1.max():.4f}  nnz={np.sum(w1>0)}")

f.close()

# Save MATLAB dose (on CT grid)
save_dose(os.path.join(ex1_dir, 'dose_matrad.mhd'),
          to_opentps_array(ml_dose_mat),
          ct1_origin, ct1_spacing, name="ex1_dose_matrad", fractions=num_fx1)

# --- Run pyMatRad dose calc ---
print("\nRunning pyMatRad dose calc (example1)...")
pln1 = {
    "radiationMode": "photons", "machine": "Generic",
    "bioModel": "none", "multScen": "nomScen", "numOfFractions": num_fx1,
    "propStf": {"gantryAngles": [b['gantryAngle'] for b in stf1],
                "couchAngles": [0]*n_beams1, "bixelWidth": 5},
    "propOpt": {"runDAO": False},
    # Use CT grid resolution so pyMatRad dose grid matches matRad's resultGUI grid
    "propDoseCalc": {"doseGrid": {"resolution": {"x": ct_res[0], "y": ct_res[1], "z": ct_res[2]}}},
}
from matRad.doseCalc.calc_dose_influence import calc_dose_influence
dij1 = calc_dose_influence(ct1, cst1, stf1, pln1)
D1 = dij1['physicalDose'][0].tocsc()
# Apply MATLAB's optimized weights (same weights, different physics engine)
dose_py1 = np.asarray(D1 @ w1).ravel()
print(f"  pyMatRad dose w/ MATLAB weights: max={dose_py1.max():.4f} Gy/fx")

dg1 = dij1['doseGrid']
Ny_dg1, Nx_dg1, Nz_dg1 = [int(d) for d in dg1['dimensions']]
dg1_origin  = (dg1['x'][0], dg1['y'][0], dg1['z'][0])
dg1_spacing = (dg1['resolution']['x'], dg1['resolution']['y'], dg1['resolution']['z'])
print(f"  pyMatRad dose grid: {Ny_dg1}×{Nx_dg1}×{Nz_dg1}  res={dg1_spacing}")

save_dose(os.path.join(ex1_dir, 'dose_pymatrad.mhd'),
          flat_dose_to_opentps(dose_py1, Ny_dg1, Nx_dg1, Nz_dg1),
          dg1_origin, dg1_spacing, name="ex1_dose_pymatrad", fractions=num_fx1)


# ============================================================
# EXAMPLE 2 — TG119
# ============================================================
print("\n" + "=" * 60)
print("EXAMPLE 2 — TG119")
print("=" * 60)

ex2_dir = os.path.join(DATA_DIR, 'example2')
os.makedirs(ex2_dir, exist_ok=True)

# --- Load TG119 CT ---
print("Loading TG119.mat...")
raw = sio.loadmat(TG119, squeeze_me=True, struct_as_record=False)
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
print(f"  CT: {Ny2}×{Nx2}×{Nz2}  res={ct2_spacing}  origin={ct2_origin}")

cst2 = []
for i in range(cst_raw.shape[0]):
    row = cst_raw[i]
    vox = np.asarray(row[3], dtype=np.int64).ravel()
    cst2.append([int(row[0]), str(row[1]), str(row[2]), [vox], {}, row[5]])

save_ct(os.path.join(ex2_dir, 'ct.mhd'),
        to_opentps_array(ct2['cubeHU'][0]), ct2_origin, ct2_spacing, name="ex2_CT")

# --- Load MATLAB example2 reference ---
print("\nLoading MATLAB example2 reference...")
f = h5py.File(REF_EX2, 'r')
num_fx2 = int(round(h5scalar(f, 'pln/numOfFractions')))

ml_dose2_h5  = np.array(f['resultGUI/physicalDose'])   # (Nz,Nx,Ny) per h5py
ml_dose2_mat = ml_dose2_h5.T                            # (Ny,Nx,Nz) matRad convention

# Load MATLAB STF (example2)
stf_h5 = f['stf']
n_beams2 = stf_h5['gantryAngle'].shape[0]
stf2 = []
for ib in range(n_beams2):
    ga   = float(h5val(f, stf_h5['gantryAngle'][ib, 0])[0])
    ca   = float(h5val(f, stf_h5['couchAngle'][ib, 0])[0])
    sad  = float(h5val(f, stf_h5['SAD'][ib, 0])[0])
    bw   = float(h5val(f, stf_h5['bixelWidth'][ib, 0])[0])
    nr   = int(h5val(f, stf_h5['numOfRays'][ib, 0])[0])
    nb   = int(h5val(f, stf_h5['totalNumOfBixels'][ib, 0])[0])
    iso  = h5val(f, stf_h5['isoCenter'][ib, 0])
    sp_b = h5val(f, stf_h5['sourcePoint_bev'][ib, 0])
    sp   = h5val(f, stf_h5['sourcePoint'][ib, 0])
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
    stf2.append({'gantryAngle': ga, 'couchAngle': ca, 'SAD': sad, 'bixelWidth': bw,
                 'isoCenter': iso, 'sourcePoint': sp, 'sourcePoint_bev': sp_b,
                 'numOfRays': nr, 'totalNumOfBixels': nb, 'ray': rays,
                 'radiationMode': 'photons', 'machine': 'Generic'})
    print(f"    beam[{ib}]: gantry={ga}°  rays={nr}  bixels={nb}")

# Load MATLAB fine-optimized weights for example2 (resultGUI/w = 2851 bixels, bixelWidth=3)
w2 = np.array(f['resultGUI/w']).ravel()
print(f"  MATLAB w: {len(w2)} weights  max={w2.max():.4f}  nnz={np.sum(w2>0)}")

f.close()

# Save MATLAB dose (on CT grid)
save_dose(os.path.join(ex2_dir, 'dose_matrad.mhd'),
          to_opentps_array(ml_dose2_mat),
          ct2_origin, ct2_spacing, name="ex2_dose_matrad", fractions=num_fx2)

# --- Run pyMatRad dose calc (example2) ---
# Use bixelWidth=3 to match resultGUI/w (2851 bixels from fine optimization).
# Generate STF with pyMatRad so the ray layout matches the bixel count in w.
print("\nRunning pyMatRad dose calc (example2, bixelWidth=3, using resultGUI/w)...")
gantry_angles2 = [b['gantryAngle'] for b in stf2]
pln2 = {
    "radiationMode": "photons", "machine": "Generic",
    "bioModel": "none", "multScen": "nomScen", "numOfFractions": num_fx2,
    "propStf": {"gantryAngles": gantry_angles2, "couchAngles": [0]*n_beams2, "bixelWidth": 3},
    "propOpt": {"runDAO": False},
    # Use CT grid resolution (3x3x2.5mm) so pyMatRad dose grid matches matRad's resultGUI grid
    "propDoseCalc": {"doseGrid": {"resolution": ct2['resolution']}},
}
from matRad.steering.stf_generator import generate_stf
stf2_py = generate_stf(ct2, cst2, pln2)
total_bixels_py = sum(b['totalNumOfBixels'] for b in stf2_py)
print(f"  pyMatRad STF: {len(stf2_py)} beams, {total_bixels_py} total bixels")
print(f"  MATLAB w:     {len(w2)} weights")
if total_bixels_py != len(w2):
    print(f"  WARNING: bixel count mismatch ({total_bixels_py} vs {len(w2)}) — weights will not align!")
else:
    print(f"  Bixel count matches exactly.")
dij2 = calc_dose_influence(ct2, cst2, stf2_py, pln2)
D2 = dij2['physicalDose'][0].tocsc()
# Apply MATLAB's fine optimized weights (resultGUI/w, 2851 bixels)
dose_py2 = np.asarray(D2 @ w2).ravel()
print(f"  pyMatRad dose w/ MATLAB w: max={dose_py2.max():.4f} Gy/fx")

dg2 = dij2['doseGrid']
Ny_dg2, Nx_dg2, Nz_dg2 = [int(d) for d in dg2['dimensions']]
dg2_origin  = (dg2['x'][0], dg2['y'][0], dg2['z'][0])
dg2_spacing = (dg2['resolution']['x'], dg2['resolution']['y'], dg2['resolution']['z'])
print(f"  pyMatRad dose grid: {Ny_dg2}×{Nx_dg2}×{Nz_dg2}  res={dg2_spacing}")

save_dose(os.path.join(ex2_dir, 'dose_pymatrad.mhd'),
          flat_dose_to_opentps(dose_py2, Ny_dg2, Nx_dg2, Nz_dg2),
          dg2_origin, dg2_spacing, name="ex2_dose_pymatrad", fractions=num_fx2)

print("\n" + "=" * 60)
print("Export complete.")
print(f"Data written to: {DATA_DIR}")
print("  example1/  ct.mhd  dose_matrad.mhd  dose_pymatrad.mhd")
print("  example2/  ct.mhd  dose_matrad.mhd  dose_pymatrad.mhd")
print("=" * 60)
