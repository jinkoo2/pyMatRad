"""
Example 1 No-Opti: pyMatRad dose calculation with uniform weights,
comparison against MATLAB matRad, and OpenTPS GUI visualization.

Pipeline:
  1. Load MATLAB no-opti results  (STF + physicalDose as ground-truth reference)
  2. Rebuild water phantom CT with PhantomBuilder (same geometry as MATLAB)
  3. Run pyMatRad dose calc using MATLAB's exact STF + uniform weights (w = 1)
  4. Compare dose statistics (max, mean, DVH metrics)
  5. Export CT + both doses to MHD  →  _data/example1_no_opti/
  6. Launch OpenTPS GUI

Reference file:
  examples/_matRad_ref_outputs/example1_no_opti/example1_results.mat
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

REF_MAT = os.path.join(PYMATRAD_ROOT,
    r'examples\_matRad_ref_outputs\example1_no_opti\example1_results.mat')
DATA_DIR = os.path.join(PYMATRAD_ROOT, '_data', 'example1_no_opti')

sys.path.insert(0, PYMATRAD_ROOT)
sys.path.insert(0, OPENTPS_CORE)
sys.path.insert(0, OPENTPS_GUI)

os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------- helpers ---
def h5val(f, ref):
    return np.array(f[ref]).ravel()

def h5scalar(f, path):
    return float(np.array(f[path]).ravel()[0])

def to_opentps(cube_ny_nx_nz):
    """Convert matRad (Ny,Nx,Nz) → OpenTPS (Nx,Ny,Nz)."""
    return cube_ny_nx_nz.transpose(1, 0, 2).copy()

def flat_dose_to_opentps(flat, ny, nx, nz):
    """Reshape flat matRad-indexed dose → OpenTPS (Nx,Ny,Nz)."""
    return flat.reshape((ny, nx, nz), order='F').transpose(1, 0, 2).copy()


# ---------------------------------------------------------------- helpers ---
def _s(x):
    """Squeeze a potentially nested array/scalar to a plain Python scalar."""
    return float(np.array(x).ravel()[0])

def _v(x):
    """Squeeze a potentially nested array to a 1-D numpy float64 vector."""
    return np.array(x).ravel().astype(np.float64)


def _load_scipy(path):
    """Load example1_results.mat saved in MATLAB v5 format via scipy.io."""
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)

    dij_m    = mat['dij']
    result_m = mat['resultGUI']
    stf_m    = mat['stf']

    # --- dose grid ---
    dg = dij_m.doseGrid
    dg_dims = np.array(dg.dimensions).ravel().astype(int)
    Ny_dg, Nx_dg, Nz_dg = int(dg_dims[0]), int(dg_dims[1]), int(dg_dims[2])
    dg_res_x = _s(dg.resolution.x)
    dg_res_y = _s(dg.resolution.y)
    dg_res_z = _s(dg.resolution.z)
    dg_x = _v(dg.x);  dg_y = _v(dg.y);  dg_z = _v(dg.z)

    # --- resultGUI physicalDose ---
    # scipy.io returns MATLAB [Ny,Nx,Nz] as numpy (Ny,Nx,Nz) — no transpose needed
    ml_dose_mat = np.array(result_m.physicalDose, dtype=np.float64)
    if ml_dose_mat.ndim == 1:
        # Stored flat — reshape using dose grid dims
        ml_dose_mat = ml_dose_mat.reshape((Ny_dg, Nx_dg, Nz_dg), order='F')

    # --- resultGUI.w ---
    w_saved = _v(result_m.w)

    # --- DIJ sparse matrix ---
    # dij.physicalDose is a cell{1,1} of a sparse matrix.
    # squeeze_me=True flattens cell{1,1} to the sparse matrix itself.
    pd_raw = dij_m.physicalDose
    if sp.issparse(pd_raw):
        D_ml = pd_raw.tocsc()
    else:
        # object array wrapping the sparse matrix
        D_ml = pd_raw.flat[0].tocsc()

    # --- STF ---
    # stf_m is a numpy object array of shape (n_beams,) with squeeze_me
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

    # h5py reads MATLAB [Ny,Nx,Nz] as (Nz,Nx,Ny); .T → (Ny,Nx,Nz)
    ml_dose_mat = np.array(f['resultGUI/physicalDose']).T
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
print("STEP 1 — Load MATLAB no-opti results")
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

print("\n  STF beams:")
for ib, bm in enumerate(stf_ml):
    print(f"    beam[{ib}] gantry={bm['gantryAngle']:6.1f}°  "
          f"rays={bm['numOfRays']:4d}  bixels={bm['totalNumOfBixels']:4d}")
total_bixels_ml = sum(b['totalNumOfBixels'] for b in stf_ml)
print(f"  Total bixels (MATLAB) : {total_bixels_ml}")

# --- Verify weights ---
if w_saved is not None:
    is_uniform = np.allclose(w_saved, 1.0, atol=1e-6)
    print(f"\n  resultGUI/w : n={len(w_saved)}  min={w_saved.min():.4f}  "
          f"max={w_saved.max():.4f}  mean={w_saved.mean():.4f}")
    print(f"  weights are {'uniform (w=1) ✓' if is_uniform else 'NOT uniform — WARNING: optimized result!'}")


# ==========================================================================
# 2.  Rebuild water phantom CT with PhantomBuilder
# ==========================================================================
print("\n" + "=" * 65)
print("STEP 2 — Build water phantom CT")
print("=" * 65)

from matRad.phantoms.builder import PhantomBuilder
from matRad.optimization.DoseObjectives import SquaredDeviation, SquaredOverdosing
from matRad.geometry.geometry import get_world_axes

CT_DIM = [200, 200, 100]
CT_RES = [2, 2, 3]

builder = PhantomBuilder(CT_DIM, CT_RES, num_of_ct_scen=1)
builder.add_spherical_target("Volume1", radius=20,
    objectives=[SquaredDeviation(penalty=800, d_ref=45).to_dict()], HU=0)
builder.add_box_oar("Volume2", [60, 30, 60], offset=[0, -15, 0],
    objectives=[SquaredOverdosing(penalty=400, d_ref=0).to_dict()], HU=0)
builder.add_box_oar("Volume3", [60, 30, 60], offset=[0, 15, 0],
    objectives=[SquaredOverdosing(penalty=10, d_ref=0).to_dict()], HU=0)
ct, cst = builder.get_ct_cst()
ct = get_world_axes(ct)

Ny_ct, Nx_ct, Nz_ct = ct['cubeDim']
ct_origin  = (float(ct['x'][0]), float(ct['y'][0]), float(ct['z'][0]))
ct_spacing = (CT_RES[0], CT_RES[1], CT_RES[2])
print(f"  CT cubeDim : {Ny_ct}×{Nx_ct}×{Nz_ct}  res={ct_spacing}  origin={ct_origin}")


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
        "gantryAngles":  [b['gantryAngle'] for b in stf_ml],
        "couchAngles":   [0] * n_beams,
        "bixelWidth":    5,
    },
    "propOpt": {"runDAO": False, "runSequencing": False},
    # Match MATLAB dose grid resolution (3×3×3 mm)
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
    print(f"  WARNING: bixel count mismatch — DIJ shapes differ!")

# Apply uniform weights (w = 1 for all bixels)
w_uniform = np.ones(total_bixels_py)
dose_py_flat = np.asarray(D_py @ w_uniform).ravel()

dg_py   = dij['doseGrid']
Ny_py   = int(dg_py['dimensions'][0])
Nx_py   = int(dg_py['dimensions'][1])
Nz_py   = int(dg_py['dimensions'][2])
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

# MATLAB uniform dose (flat vector on dose grid)
w_ml_uniform  = np.ones(D_ml.shape[1])
dose_ml_flat  = np.asarray(D_ml @ w_ml_uniform).ravel()

def compare(label, v_py, v_ml):
    err = (v_py - v_ml) / v_ml * 100 if v_ml != 0 else float('nan')
    flag = "  <<<" if abs(err) > 5 else ""
    print(f"  {label:<35} {v_py:>12.4f}  {v_ml:>12.4f}  {err:>8.2f}%{flag}")

print(f"\n  {'Metric':<35} {'pyMatRad':>12}  {'MATLAB':>12}  {'Err%':>8}")
print("  " + "-" * 72)
compare("Max dose (Gy/fx)",       dose_py_flat.max(),                  dose_ml_flat.max())
compare("Mean dose > 0 (Gy/fx)",  dose_py_flat[dose_py_flat > 0].mean(), dose_ml_flat[dose_ml_flat > 0].mean())
compare("DIJ total sum",          D_py.data.sum(),                    D_ml.data.sum())
compare("DIJ nnz",                float(D_py.nnz),                    float(D_ml.nnz))
compare("DIJ max entry",          D_py.data.max(),                    D_ml.data.max())

print(f"\n  Per-beam DIJ column-sum comparison:")
print(f"  {'Beam':>5} {'Gantry':>8}  {'py_sum':>14}  {'ml_sum':>14}  {'Err%':>8}")
print("  " + "-" * 58)
bixel_offset = 0
for ib, beam in enumerate(stf_ml):
    nb   = beam['totalNumOfBixels']
    cols = slice(bixel_offset, bixel_offset + nb)
    py_s = float(D_py[:, cols].data.sum()) if D_py[:, cols].nnz > 0 else 0.0
    ml_s = float(D_ml[:, cols].data.sum()) if D_ml[:, cols].nnz > 0 else 0.0
    err  = (py_s - ml_s) / ml_s * 100 if ml_s != 0 else float('nan')
    flag = "  <<<" if abs(err) > 5 else ""
    print(f"  {ib+1:>5} {beam['gantryAngle']:>8.1f}°  {py_s:>14.4f}  {ml_s:>14.4f}  {err:>8.2f}%{flag}")
    bixel_offset += nb

# Per-structure dose metrics
print(f"\n  Per-structure dose metrics:")
print(f"  {'VOI':<12} {'py_mean':>10}  {'ml_mean':>10}  {'py_max':>10}  {'ml_max':>10}")
print("  " + "-" * 58)
for row in cst:
    name  = row[1]
    vtype = row[2]
    voxels = np.asarray(row[3][0], dtype=np.int64) - 1  # 1-based → 0-based
    # These voxels are on the CT grid — skip if we can't map to dose grid
    # (just report from dose_py on pyMatRad dose grid using dij mapping)
    print(f"  {name:<12}  (type={vtype}, {len(voxels)} CT voxels)")


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

# --- CT ---
ct_arr_opentps = to_opentps(ct['cubeHU'][0]).astype(np.float32)
ct_img = CTImage(imageArray=ct_arr_opentps, name="ex1_no_opti_CT",
                 origin=ct_origin, spacing=ct_spacing)
ct_path = os.path.join(DATA_DIR, 'ct.mhd')
exportImageMHD(ct_path, ct_img)
print(f"  Saved CT          : {ct_path}")

# --- MATLAB dose: always use D_ml @ ones on the dose grid ---
# This guarantees uniform weights regardless of what resultGUI/w was saved as.
# dose_ml_flat was already computed in Step 4 as D_ml @ ones.
ml_dose_opentps = flat_dose_to_opentps(dose_ml_flat, Ny_dg, Nx_dg, Nz_dg).astype(np.float32) * NUM_FX
ml_origin  = (float(dg_x[0]), float(dg_y[0]), float(dg_z[0]))
ml_spacing = (dg_res_x, dg_res_y, dg_res_z)
ml_dose_img = DoseImage(imageArray=ml_dose_opentps, name="matRad_uniform_dose",
                        origin=ml_origin, spacing=ml_spacing)
ml_path = os.path.join(DATA_DIR, 'dose_matrad.mhd')
exportImageMHD(ml_path, ml_dose_img)
print(f"  Saved matRad dose : {ml_path}  (max={ml_dose_opentps.max():.2f} Gy, ×{NUM_FX} fx)")

# --- pyMatRad dose (on pyMatRad dose grid) ---
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

# Re-load from MHD so we use exactly what was saved
ct_loaded      = importImageMHD(ct_path)
d_ml_loaded    = importImageMHD(ml_path)
d_py_loaded    = importImageMHD(py_path)

ct_image = CTImage(imageArray=ct_loaded._imageArray,   name="CT (water phantom)",
                   origin=ct_loaded._origin,            spacing=ct_loaded._spacing)
dose_matrad   = DoseImage(imageArray=d_ml_loaded._imageArray, name="matRad  (uniform, Gy)",
                          origin=d_ml_loaded._origin,         spacing=d_ml_loaded._spacing)
dose_pymatrad = DoseImage(imageArray=d_py_loaded._imageArray, name="pyMatRad (uniform, Gy)",
                          origin=d_py_loaded._origin,         spacing=d_py_loaded._spacing)

print(f"  CT           : {ct_image._imageArray.shape}  max HU={ct_image._imageArray.max():.0f}")
print(f"  matRad dose  : {dose_matrad._imageArray.shape}  max={dose_matrad._imageArray.max():.2f} Gy")
print(f"  pyMatRad dose: {dose_pymatrad._imageArray.shape}  max={dose_pymatrad._imageArray.max():.2f} Gy")

patient = Patient(name="example1_no_opti")
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
