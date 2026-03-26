"""
Load result.mat from Octave matRad run and export CT + dose to MHD,
then launch OpenTPS GUI to view.

Usage:
    python examples/export_octave_result.py
"""

import os, sys
import numpy as np
import scipy.io as sio

PYMATRAD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENTPS_CORE  = r'C:\Users\jkim20\Desktop\projects\tps\opentps2\opentps_core'
OPENTPS_GUI   = r'C:\Users\jkim20\Desktop\projects\tps\opentps2\opentps_gui'
DATA_DIR      = os.path.join(PYMATRAD_ROOT, '_data', 'example1_octave')
MAT_FILE      = os.path.join(DATA_DIR, 'result.mat')

sys.path.insert(0, PYMATRAD_ROOT)
sys.path.insert(0, OPENTPS_CORE)
sys.path.insert(0, OPENTPS_GUI)

if not os.path.isfile(MAT_FILE):
    print(f"ERROR: result.mat not found at {MAT_FILE}")
    print("  Run: octave-cli matlab_scripts/run_matrad_example1.m first")
    sys.exit(1)

# ---- Load .mat ----
print(f"Loading {MAT_FILE} ...")
mat = sio.loadmat(MAT_FILE, squeeze_me=True)

dose_3d    = mat['dose_3d'].astype(np.float32)    # (Ny, Nx, Nz) total Gy
ct_hu      = mat['ct_hu'].astype(np.float32)       # (Ny, Nx, Nz)
ct_origin  = mat['ct_origin'].ravel()              # [x0, y0, z0] mm
ct_spacing = mat['ct_spacing'].ravel()             # [dx, dy, dz] mm
dg_origin  = mat['dg_origin'].ravel()
dg_spacing = mat['dg_spacing'].ravel()

print(f"  CT:   {ct_hu.shape}   origin={ct_origin}  spacing={ct_spacing}")
print(f"  Dose: {dose_3d.shape}  max={dose_3d.max():.2f} Gy  origin={dg_origin}  spacing={dg_spacing}")

# ---- Convert matRad (Ny,Nx,Nz) -> OpenTPS (Nx,Ny,Nz) ----
def to_opentps(arr_ny_nx_nz):
    return arr_ny_nx_nz.transpose(1, 0, 2).copy()

ct_opentps   = to_opentps(ct_hu)
dose_opentps = to_opentps(dose_3d)

# ---- Export MHD ----
import logging
logging.basicConfig(level=logging.WARNING)

from opentps.core.data.images._ctImage import CTImage
from opentps.core.data.images._doseImage import DoseImage
from opentps.core.io.mhdIO import exportImageMHD

os.makedirs(DATA_DIR, exist_ok=True)

ct_img = CTImage(
    imageArray=ct_opentps,
    name="example1_octave CT",
    origin=tuple(float(v) for v in ct_origin),
    spacing=tuple(float(v) for v in ct_spacing),
)
exportImageMHD(os.path.join(DATA_DIR, 'ct.mhd'), ct_img)
print(f"  Saved ct.mhd")

dose_img = DoseImage(
    imageArray=dose_opentps,
    name="matRad Octave dose",
    origin=tuple(float(v) for v in dg_origin),
    spacing=tuple(float(v) for v in dg_spacing),
)
exportImageMHD(os.path.join(DATA_DIR, 'dose_pymatrad.mhd'), dose_img)
print(f"  Saved dose_pymatrad.mhd  max={dose_opentps.max():.2f} Gy")

# ---- Launch OpenTPS GUI ----
print("\nLaunching OpenTPS GUI ...")
from PyQt5.QtWidgets import QApplication
from opentps.core.data import PatientList
from opentps.core.data._patient import Patient
from opentps.core.utils.programSettings import ProgramSettings
from opentps.gui.viewController import ViewController

patient = Patient(name="example1 (Octave matRad)")
patient.appendPatientData(ct_img)
patient.appendPatientData(dose_img)

patientList = PatientList()
patientList.append(patient)

app = QApplication.instance() or QApplication(sys.argv)
vc = ViewController(patientList)
vc.mainConfig = ProgramSettings()
vc.dose1 = dose_img

print(f"  dose = matRad Octave PGD result  (max {dose_opentps.max():.2f} Gy)")
print("  Use Patient Data panel to toggle dose overlay.\n")

vc.mainWindow.show()
app.exec_()
