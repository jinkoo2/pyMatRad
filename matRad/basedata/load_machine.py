"""
Machine base data loading.

Python port of matRad_loadMachine.m.
Loads MATLAB .mat files using scipy.io.
"""

import os
import numpy as np
from typing import Optional


def load_machine(pln: dict, matrad_src_root: Optional[str] = None) -> dict:
    """
    Load machine base data from .mat file.

    Port of matRad_loadMachine.m

    Looks for the machine file in:
    1. matRad/matRad/basedata/ (original MATLAB data)
    2. pyMatRad/userdata/machines/

    Parameters
    ----------
    pln : dict
        Plan struct with 'radiationMode' and 'machine' fields
    matrad_src_root : str, optional
        Path to matRad source root (defaults to sibling matRad dir)

    Returns
    -------
    dict
        Machine struct with 'meta' and 'data' fields
    """
    from ..config import MatRad_Config
    cfg = MatRad_Config.instance()

    radiation_mode = pln.get("radiationMode", None)
    machine_name = pln.get("machine", "Generic")

    if radiation_mode is None:
        cfg.disp_error("No radiation mode given in pln")

    base_name = f"{radiation_mode}_{machine_name}"

    # Build search paths
    pymatrad_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))

    # The original matRad basedata is a sibling directory
    matrad_root = os.path.join(os.path.dirname(pymatrad_root), "matRad")
    matrad_basedata = os.path.join(matrad_root, "matRad", "basedata")
    pymatrad_userdata = os.path.join(pymatrad_root, "userdata", "machines")

    # Search order: userdata first (custom machines override Generic),
    # then the original MATLAB basedata.
    # Within each folder prefer .npy (native Python) over .mat.
    search_paths = [pymatrad_userdata, matrad_basedata]

    filepath  = None
    file_name = None
    for folder in search_paths:
        for ext in (".npy", ".mat"):
            candidate = os.path.join(folder, base_name + ext)
            if os.path.isfile(candidate):
                filepath  = candidate
                file_name = base_name + ext
                break
        if filepath is not None:
            break

    if filepath is None:
        cfg.disp_error(
            f"Could not find machine file: {base_name}.npy or {base_name}.mat"
        )

    # Load
    try:
        if filepath.endswith(".npy"):
            machine = np.load(filepath, allow_pickle=True).item()
        else:
            machine = _load_mat_machine(filepath)
    except Exception as e:
        cfg.disp_error(f"Could not load machine file {file_name}: {e}")

    return machine


def _load_mat_machine(filepath: str) -> dict:
    """
    Load a MATLAB machine .mat file and convert to Python dict.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    dict
        Machine struct with nested Python dicts/arrays
    """
    import scipy.io as sio

    try:
        # Try loading as MATLAB v5 format
        raw = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    except Exception:
        # Try h5py for MATLAB v7.3 format
        try:
            raw = _load_mat_v73(filepath)
            return raw
        except Exception as e:
            raise RuntimeError(f"Failed to load {filepath}: {e}")

    if "machine" not in raw:
        raise ValueError(f"Machine file does not contain 'machine' variable")

    machine_raw = raw["machine"]
    return _matlab_struct_to_dict(machine_raw)


def _matlab_struct_to_dict(obj) -> dict:
    """Recursively convert MATLAB struct to Python dict."""
    import scipy.io as sio

    if isinstance(obj, sio.matlab.mat_struct):
        result = {}
        for key in obj._fieldnames:
            val = getattr(obj, key)
            result[key] = _matlab_struct_to_dict(val)
        return result
    elif isinstance(obj, np.ndarray):
        if obj.dtype.names:
            # Structured array
            result = {}
            for name in obj.dtype.names:
                result[name] = _matlab_struct_to_dict(obj[name])
            return result
        elif obj.dtype == object:
            # Cell array or array of structs
            if obj.ndim == 0:
                return _matlab_struct_to_dict(obj.item())
            return [_matlab_struct_to_dict(v) for v in obj.flat]
        else:
            # Numeric array - squeeze scalar to Python scalar
            if obj.ndim == 0:
                return obj.item()
            if obj.size == 1:
                return obj.flat[0]
            return obj
    else:
        return obj


def _load_mat_v73(filepath: str) -> dict:
    """Load MATLAB v7.3 (HDF5) .mat file."""
    import h5py

    def h5_to_dict(group):
        result = {}
        for key, val in group.items():
            if isinstance(val, h5py.Group):
                result[key] = h5_to_dict(val)
            elif isinstance(val, h5py.Dataset):
                data = val[()]
                if data.dtype.kind in ("S", "O"):
                    # String or object
                    try:
                        data = "".join(chr(c) for c in data.flat)
                    except Exception:
                        pass
                result[key] = data
        return result

    with h5py.File(filepath, "r") as f:
        raw = h5_to_dict(f)

    if "machine" in raw:
        return raw["machine"]
    return raw
