"""
Compute uniform-fluence dose for example1 (water phantom) using pyMatRad,
then (once Octave result.mat is ready) export both to MHD and launch OpenTPS
showing matRad (Octave) vs pyMatRad side-by-side.

Saves to _data/example1_octave/:
  ct.mhd                -- water phantom CT
  dose_pymatrad.mhd     -- pyMatRad uniform fluence (w=1)
  dose_matrad.mhd       -- matRad/Octave uniform fluence (w=1)  [from result.mat]
"""

import os, sys, time
import numpy as np

PYMATRAD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENTPS_CORE  = r'C:\Users\jkim20\Desktop\projects\tps\opentps2\opentps_core'
OPENTPS_GUI   = r'C:\Users\jkim20\Desktop\projects\tps\opentps2\opentps_gui'
DATA_DIR      = os.path.join(PYMATRAD_ROOT, '_data', 'example1_octave')
MAT_FILE      = os.path.join(DATA_DIR, 'result.mat')

sys.path.insert(0, PYMATRAD_ROOT)
sys.path.insert(0, OPENTPS_CORE)
sys.path.insert(0, OPENTPS_GUI)

os.makedirs(DATA_DIR, exist_ok=True)

import logging
logging.basicConfig(level=logging.WARNING)

from opentps.core.data.images._ctImage import CTImage
from opentps.core.data.images._doseImage import DoseImage
from opentps.core.io.mhdIO import exportImageMHD, importImageMHD

def to_opentps(arr_ny_nx_nz):
    return arr_ny_nx_nz.transpose(1, 0, 2).copy()

def flat_to_opentps(flat, ny, nx, nz):
    return flat.reshape((ny, nx, nz), order='F').transpose(1, 0, 2).copy()

# ============================================================
# 1. pyMatRad: build phantom + compute dij + uniform dose
# ============================================================
print("=== pyMatRad: water phantom, uniform fluence ===")

from matRad.phantoms.builder import PhantomBuilder
from matRad.optimization.DoseObjectives import SquaredDeviation, SquaredOverdosing
from matRad.geometry.geometry import get_world_axes
from matRad.doseCalc.calc_dose_influence import calc_dose_influence

ct_dim = [200, 200, 100]
ct_res  = [2, 2, 3]
builder = PhantomBuilder(ct_dim, ct_res, num_of_ct_scen=1)
builder.add_spherical_target("Volume1", radius=20,
    objectives=[SquaredDeviation(penalty=800, d_ref=45).to_dict()], HU=0)
builder.add_box_oar("Volume2", [60, 30, 60], offset=[0, -15, 0],
    objectives=[SquaredOverdosing(penalty=400, d_ref=0).to_dict()], HU=0)
builder.add_box_oar("Volume3", [60, 30, 60], offset=[0, 15, 0],
    objectives=[SquaredOverdosing(penalty=10, d_ref=0).to_dict()], HU=0)
ct, cst = builder.get_ct_cst()
ct = get_world_axes(ct)

Ny, Nx, Nz = ct['cubeDim']
ct_origin  = (ct['x'][0], ct['y'][0], ct['z'][0])
ct_spacing = tuple(float(v) for v in ct_res)
print(f"  CT: {Ny}x{Nx}x{Nz}  res={ct_spacing}")

num_fx = 30
pln = {
    "radiationMode": "photons", "machine": "Generic",
    "bioModel": "none", "multScen": "nomScen", "numOfFractions": num_fx,
    "propStf": {"gantryAngles": list(range(0, 356, 70)),
                "couchAngles": [0]*6, "bixelWidth": 5},
    "propOpt": {"runDAO": False},
    "propDoseCalc": {"doseGrid": {"resolution": {"x": ct_res[0], "y": ct_res[1], "z": ct_res[2]}}},
}

print(f"  Computing dij (6 beams, bixelWidth=5)...")
dij = calc_dose_influence(ct, cst, None, pln)  # None = generate STF from pln
D   = dij['physicalDose'][0].tocsc()
nB  = D.shape[1]
dg  = dij['doseGrid']
Ny_dg, Nx_dg, Nz_dg = [int(d) for d in dg['dimensions']]
dg_origin  = (dg['x'][0], dg['y'][0], dg['z'][0])
dg_spacing = (dg['resolution']['x'], dg['resolution']['y'], dg['resolution']['z'])
print(f"  dij: {D.shape[0]} voxels x {nB} bixels")

# Uniform fluence: w = 1.0 for all bixels
w_uniform = np.ones(nB, dtype=np.float32)
dose_flat = np.asarray(D @ w_uniform).ravel() * num_fx   # total Gy
print(f"  Uniform dose: max={dose_flat.max():.2f} Gy")

# Save CT
ct_arr = to_opentps(ct['cubeHU'][0]).astype(np.float32)
ct_img = CTImage(imageArray=ct_arr, name="ex1 CT",
                 origin=ct_origin, spacing=ct_spacing)
exportImageMHD(os.path.join(DATA_DIR, 'ct.mhd'), ct_img)
print(f"  Saved ct.mhd")

# Save pyMatRad dose
dose_py_arr = flat_to_opentps(dose_flat, Ny_dg, Nx_dg, Nz_dg)
dose_py_img = DoseImage(imageArray=dose_py_arr, name="pyMatRad uniform",
                        origin=dg_origin, spacing=dg_spacing)
exportImageMHD(os.path.join(DATA_DIR, 'dose_pymatrad.mhd'), dose_py_img)
print(f"  Saved dose_pymatrad.mhd")

# ============================================================
# 2. Wait for Octave result.mat
# ============================================================
print(f"\nWaiting for Octave result.mat at:\n  {MAT_FILE}")
timeout = 600  # seconds
t0 = time.time()
while not os.path.isfile(MAT_FILE):
    elapsed = time.time() - t0
    if elapsed > timeout:
        print("ERROR: Timed out waiting for result.mat")
        sys.exit(1)
    print(f"  ... waiting ({elapsed:.0f}s)", end='\r')
    time.sleep(5)

# Give Octave a moment to finish writing
time.sleep(3)
print(f"\n  result.mat found. Loading...")

import scipy.io as sio
mat        = sio.loadmat(MAT_FILE, squeeze_me=True)
dose_3d_ml = mat['dose_3d'].astype(np.float32)   # (Ny,Nx,Nz) matRad convention
dg_origin_ml  = mat['dg_origin'].ravel()
dg_spacing_ml = mat['dg_spacing'].ravel()
print(f"  matRad dose: {dose_3d_ml.shape}  max={dose_3d_ml.max():.2f} Gy")

dose_ml_arr = to_opentps(dose_3d_ml)
dose_ml_img = DoseImage(imageArray=dose_ml_arr, name="matRad/Octave uniform",
                        origin=tuple(float(v) for v in dg_origin_ml),
                        spacing=tuple(float(v) for v in dg_spacing_ml))
exportImageMHD(os.path.join(DATA_DIR, 'dose_matrad.mhd'), dose_ml_img)
print(f"  Saved dose_matrad.mhd")

# ============================================================
# 3. Compute difference stats
# ============================================================
# Reload both on same grid for comparison
d_py = dose_py_arr
d_ml = dose_ml_arr
if d_py.shape == d_ml.shape:
    diff = d_py - d_ml
    print(f"\n=== Comparison (pyMatRad - matRad, uniform w=1) ===")
    print(f"  pyMatRad max:  {d_py.max():.2f} Gy")
    print(f"  matRad   max:  {d_ml.max():.2f} Gy")
    print(f"  diff min/max:  {diff.min():.2f} / {diff.max():.2f} Gy")
    print(f"  mean|diff|:    {np.mean(np.abs(diff)):.3f} Gy")
    print(f"  RMS:           {np.sqrt(np.mean(diff**2)):.3f} Gy")
else:
    print(f"  WARNING: grid mismatch {d_py.shape} vs {d_ml.shape} — skipping diff")

# ============================================================
# 4. Launch OpenTPS GUI
# ============================================================
print("\nLaunching OpenTPS GUI...")
from PyQt5.QtWidgets import QApplication
from opentps.core.data import PatientList
from opentps.core.data._patient import Patient
from opentps.core.utils.programSettings import ProgramSettings
from opentps.gui.viewController import ViewController

patient = Patient(name="example1 uniform fluence")
patient.appendPatientData(ct_img)
patient.appendPatientData(dose_py_img)
patient.appendPatientData(dose_ml_img)

patientList = PatientList()
patientList.append(patient)

app = QApplication.instance() or QApplication(sys.argv)
vc  = ViewController(patientList)
vc.mainConfig = ProgramSettings()
vc.dose1 = dose_py_img   # pyMatRad
vc.dose2 = dose_ml_img   # matRad/Octave

print("  dose1 = pyMatRad uniform fluence")
print("  dose2 = matRad/Octave uniform fluence")
print("  Toggle overlays in Patient Data panel.\n")

vc.mainWindow.show()
app.exec_()
