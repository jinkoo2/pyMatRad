"""
Start the OpenTPS GUI pre-loaded with pyMatRad vs matRad dose comparison data.

Usage:
    python start_opentps.py [example1|example2] [diff]

    example1|example2  -- which dataset to load (default: example1)
    diff               -- show dose difference (pyMatRad - matRad) instead of both doses
                          requires identical grids (run export_to_opentps.py first)

Examples:
    python start_opentps.py example2        # show both doses
    python start_opentps.py example2 diff   # show difference map on CT

Requires:
    - opentps2 installed or on PYTHONPATH
    - _data/ folder populated by examples/export_to_opentps.py
"""

import os, sys, logging
import numpy as np

PYMATRAD_ROOT = os.path.dirname(os.path.abspath(__file__))
OPENTPS_CORE  = r'C:\Users\jkim20\Desktop\projects\tps\opentps2\opentps_core'
OPENTPS_GUI   = r'C:\Users\jkim20\Desktop\projects\tps\opentps2\opentps_gui'

sys.path.insert(0, OPENTPS_CORE)
sys.path.insert(0, OPENTPS_GUI)

logging.basicConfig(level=logging.WARNING)

# ---- parse args ----
args = sys.argv[1:]
example  = 'example1'
show_diff = False
for a in args:
    if a == 'diff':
        show_diff = True
    elif not a.startswith('-'):
        example = a

data_dir = os.path.join(PYMATRAD_ROOT, '_data', example)

# ---- verify exports exist ----
ct_mhd = os.path.join(data_dir, 'ct.mhd')
d2_mhd = os.path.join(data_dir, 'dose_pymatrad.mhd')
# prefer dose_matrad.mhd, fall back to dose_matrad_full.mhd
d1_mhd = os.path.join(data_dir, 'dose_matrad.mhd')
if not os.path.isfile(d1_mhd):
    d1_mhd = os.path.join(data_dir, 'dose_matrad_full.mhd')

missing = [p for p in [ct_mhd, d1_mhd, d2_mhd] if not os.path.isfile(p)]
if missing:
    print("ERROR: MHD files not found. Run the export first.")
    for m in missing:
        print(f"  Missing: {m}")
    sys.exit(1)

print(f"Loading {example} data from: {data_dir}  (mode: {'diff' if show_diff else 'both doses'})")

# ---- OpenTPS imports ----
from PyQt5.QtWidgets import QApplication
from opentps.core.data import PatientList
from opentps.core.data._patient import Patient
from opentps.core.data.images._ctImage import CTImage
from opentps.core.data.images._doseImage import DoseImage
from opentps.core.io.mhdIO import importImageMHD
from opentps.core.utils.programSettings import ProgramSettings
from opentps.gui.viewController import ViewController

# ---- load MHD files ----
print("  Reading ct.mhd ...")
ct_img   = importImageMHD(ct_mhd)
ct_image = CTImage(
    imageArray=ct_img._imageArray,
    name=f"{example} CT",
    origin=ct_img._origin,
    spacing=ct_img._spacing,
)

print("  Reading dose_matrad.mhd ...")
d1_img      = importImageMHD(d1_mhd)

print("  Reading dose_pymatrad.mhd ...")
d2_img      = importImageMHD(d2_mhd)

dose_matrad = DoseImage(
    imageArray=d1_img._imageArray,
    name="matRad dose",
    origin=d1_img._origin,
    spacing=d1_img._spacing,
)
dose_pymatrad = DoseImage(
    imageArray=d2_img._imageArray,
    name="pyMatRad dose",
    origin=d2_img._origin,
    spacing=d2_img._spacing,
)

print(f"  CT:           {ct_image._imageArray.shape}  max HU={ct_image._imageArray.max():.0f}")
print(f"  dose_matrad:  {dose_matrad._imageArray.shape}  max={dose_matrad._imageArray.max():.2f} Gy")
print(f"  dose_pymatrad:{dose_pymatrad._imageArray.shape}  max={dose_pymatrad._imageArray.max():.2f} Gy")

# ---- compute difference if requested ----
if show_diff:
    if d1_img._imageArray.shape != d2_img._imageArray.shape:
        print(f"ERROR: grids differ — {d1_img._imageArray.shape} vs {d2_img._imageArray.shape}")
        print("  Re-run export_to_opentps.py to regenerate on matching grids.")
        sys.exit(1)

    diff_arr = d2_img._imageArray.astype(np.float32) - d1_img._imageArray.astype(np.float32)
    dose_diff = DoseImage(
        imageArray=diff_arr,
        name="pyMatRad - matRad (Gy)",
        origin=d1_img._origin,
        spacing=d1_img._spacing,
    )
    print(f"\n  Difference (pyMatRad - matRad):")
    print(f"    min={diff_arr.min():.2f} Gy   max={diff_arr.max():.2f} Gy")
    print(f"    mean(abs)={np.mean(np.abs(diff_arr)):.2f} Gy")
    print(f"    RMS={np.sqrt(np.mean(diff_arr**2)):.2f} Gy")

# ---- build patient ----
patient = Patient(name=f"pyMatRad {example}")
patient.appendPatientData(ct_image)
patient.appendPatientData(dose_matrad)
patient.appendPatientData(dose_pymatrad)
if show_diff:
    patient.appendPatientData(dose_diff)

patientList = PatientList()
patientList.append(patient)

# ---- start GUI ----
app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

mainConfig = ProgramSettings()

vc = ViewController(patientList)
vc.mainConfig = mainConfig

if show_diff:
    vc.dose1 = dose_diff   # difference map as primary overlay
    vc.dose2 = dose_matrad # matRad as reference in background
    print(f"\nStarting OpenTPS GUI with {example} (DIFF mode)...")
    print("  dose1 = pyMatRad - matRad  [difference map, Gy]")
    print("  dose2 = matRad dose        [reference]")
else:
    vc.dose1 = dose_matrad
    vc.dose2 = dose_pymatrad
    print(f"\nStarting OpenTPS GUI with {example}...")
    print("  dose1 = matRad MATLAB optimized dose")
    print("  dose2 = pyMatRad dose (MATLAB weights)")

print("  Use the Patient Data panel to toggle overlays.\n")

vc.mainWindow.show()
app.exec_()
