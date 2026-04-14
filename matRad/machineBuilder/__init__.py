"""
Machine building utilities for pyMatRad.

Port of matRad/photonPencilBeamKernelCalc/ and my_scripts/truebeam_xxx/.

Quick start
-----------
from matRad.machineBuilder import build_all_truebeam

build_all_truebeam(
    gbd_root   = "/path/to/TrueBeamGBD",
    output_dir = "/path/to/pyMatRad/userdata/machines",
)
"""

from .build_truebeam import build_truebeam_machine, build_all_truebeam
from .kernel_calc import generate_machine, save_machine, load_machine_npy
from .read_gbd_data import read_output_factors, read_depth_dose_tpr, read_primary_fluence

__all__ = [
    "build_truebeam_machine",
    "build_all_truebeam",
    "generate_machine",
    "save_machine",
    "load_machine_npy",
    "read_output_factors",
    "read_depth_dose_tpr",
    "read_primary_fluence",
]
