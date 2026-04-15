"""
pyMatRad DICOM importer.

Port of matRad's @matRad_DicomImporter MATLAB class.

Supported modalities
--------------------
- CT          → ct dict
- RTStruct    → cst list
- RTPlan      → pln dict
- RTDose      → dose ndarray

Usage
-----
from matRad.dicom import import_dicom

result = import_dicom("/path/to/dicom/folder")
ct          = result["ct"]
cst         = result["cst"]
pln         = result["pln"]
dose_cube   = result["dose"]   # ndarray (Ny, Nx, Nz) [Gy]
"""

import os
import glob
import warnings
import numpy as np

# ---------------------------------------------------------------------------
# Default HLUT — HU → relative electron density (photons)
# Matches matRad_default.hlut shipped with matRad
# ---------------------------------------------------------------------------
_DEFAULT_HLUT = np.array([
    [-1024.0, 0.001],
    [-999.0,  0.001],
    [   0.0,  1.000],
    [ 200.0,  1.200],
    [ 449.0,  1.200],
    [2000.0,  2.491],
    [2048.0,  2.531],
    [3071.0,  2.531],
])


def _hu_to_red(hu_cube: np.ndarray,
               hlut: np.ndarray = _DEFAULT_HLUT) -> np.ndarray:
    """Convert HU array to relative electron density using a piecewise-linear
    HLUT (Hounsfield Lookup Table)."""
    red = np.interp(hu_cube.astype(float),
                    hlut[:, 0], hlut[:, 1],
                    left=hlut[0, 1], right=hlut[-1, 1])
    return red


def _scan_folder(dicom_dir: str) -> dict:
    """Scan *dicom_dir* and classify files by DICOM modality.

    Returns
    -------
    dict with keys 'ct', 'rtplan', 'rtstruct', 'rtdose', each a list of paths.
    """
    try:
        import pydicom
    except ImportError as e:
        raise ImportError(
            "pydicom is required for DICOM import.  "
            "Install it with:  pip install pydicom"
        ) from e

    files = {"ct": [], "rtplan": [], "rtstruct": [], "rtdose": []}
    for path in glob.glob(os.path.join(dicom_dir, "**", "*.dcm"), recursive=True):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            modality = getattr(ds, "Modality", "").upper()
        except Exception:
            continue
        if modality == "CT":
            files["ct"].append(path)
        elif modality == "RTPLAN":
            files["rtplan"].append(path)
        elif modality == "RTSTRUCT":
            files["rtstruct"].append(path)
        elif modality == "RTDOSE":
            files["rtdose"].append(path)
    return files


# ---------------------------------------------------------------------------
# CT importer
# ---------------------------------------------------------------------------

def import_ct(ct_files: list, hlut: np.ndarray = _DEFAULT_HLUT) -> dict:
    """Import a set of DICOM CT slice files.

    Parameters
    ----------
    ct_files : list of str
        Paths to the CT DICOM files (any order — sorted by z internally).
    hlut : ndarray, optional
        HU → RED lookup table.  Defaults to the matRad standard HLUT.

    Returns
    -------
    ct : dict  matching pyMatRad's ct struct format.
    """
    import pydicom

    if not ct_files:
        raise ValueError("No CT DICOM files provided.")

    # Read all slice headers to build the z-ordering
    slices = []
    for path in ct_files:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        z = float(ds.ImagePositionPatient[2])
        slices.append((z, path))

    slices.sort(key=lambda x: x[0])   # sort ascending z

    # Validate consistency using first slice
    ds0 = pydicom.dcmread(slices[0][1], stop_before_pixels=True)
    n_rows = int(ds0.Rows)
    n_cols = int(ds0.Columns)
    dx = float(ds0.PixelSpacing[1])          # column spacing → x [mm]
    dy = float(ds0.PixelSpacing[0])          # row spacing    → y [mm]
    dz = float(ds0.SliceThickness) if hasattr(ds0, "SliceThickness") and ds0.SliceThickness else (
        float(slices[1][0] - slices[0][0]) if len(slices) > 1 else 2.0
    )
    rescale_slope     = float(getattr(ds0, "RescaleSlope", 1))
    rescale_intercept = float(getattr(ds0, "RescaleIntercept", -1000))
    origin_x = float(ds0.ImagePositionPatient[0])  # x of first pixel centre [mm]
    origin_y = float(ds0.ImagePositionPatient[1])  # y of first pixel centre [mm]

    n_slices = len(slices)

    # Build pixel cube (Ny, Nx, Nz) in Fortran/column-major memory layout
    # matRad convention: cubeDim = [Ny, Nx, Nz]
    Ny = n_rows
    Nx = n_cols
    Nz = n_slices

    pixel_cube = np.empty((Ny, Nx, Nz), dtype=np.float32, order="F")
    z_coords = np.empty(Nz, dtype=float)

    for k, (z_val, path) in enumerate(slices):
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)  # (Rows, Cols) = (Ny, Nx)
        pixel_cube[:, :, k] = arr
        z_coords[k] = z_val

    # Convert pixel values → HU
    hu_cube = pixel_cube * rescale_slope + rescale_intercept

    # Convert HU → relative electron density
    red_cube = _hu_to_red(hu_cube, hlut).astype(np.float32)

    # Build world coordinate arrays (voxel centres, [mm])
    x_coords = origin_x + np.arange(Nx) * dx   # DICOM LPS x
    y_coords = origin_y + np.arange(Ny) * dy   # DICOM LPS y

    ct = {
        "cube":          [red_cube],
        "cubeHU":        [hu_cube.astype(np.float32)],
        "cubeDim":       [Ny, Nx, Nz],
        "numOfCtScen":   1,
        "resolution":    {"x": dx, "y": dy, "z": dz},
        "x":             x_coords,
        "y":             y_coords,
        "z":             z_coords,
        "hlut":          hlut,
        "dicomInfo": {
            "PixelSpacing":             np.array([dy, dx]),
            "SlicePositions":           z_coords,
            "SliceThickness":           dz,
            "ImagePositionPatient":     np.array([origin_x, origin_y, z_coords[0]]),
            "ImageOrientationPatient":  np.array([1, 0, 0, 0, 1, 0], dtype=float),
            "RescaleSlope":             rescale_slope,
            "RescaleIntercept":         rescale_intercept,
            "Rows":                     n_rows,
            "Columns":                  n_cols,
        },
    }

    return ct


# ---------------------------------------------------------------------------
# RTStruct importer
# ---------------------------------------------------------------------------

def import_rtstruct(rtstruct_file: str, ct: dict) -> list:
    """Import a DICOM RTStruct file and return a pyMatRad-format cst list.

    Parameters
    ----------
    rtstruct_file : str
    ct : dict
        The ct dict returned by import_ct — needed for voxel indexing.

    Returns
    -------
    cst : list of [idx, name, type, [voxel_indices], props, objectives]
    """
    import pydicom

    ds = pydicom.dcmread(rtstruct_file)

    # Build ROI number → name lookup
    roi_names = {}
    for item in ds.StructureSetROISequence:
        roi_names[int(item.ROINumber)] = str(item.ROIName)

    # Build ROI number → colour lookup (optional)
    roi_colors = {}
    if hasattr(ds, "ROIContourSequence"):
        for item in ds.ROIContourSequence:
            n = int(item.ReferencedROINumber)
            if hasattr(item, "ROIDisplayColor"):
                roi_colors[n] = [int(v) for v in item.ROIDisplayColor]
            else:
                roi_colors[n] = [128, 128, 128]

    # CT grid
    x = ct["x"]
    y = ct["y"]
    z = ct["z"]
    Ny, Nx, Nz = ct["cubeDim"]
    dx = ct["resolution"]["x"]
    dy = ct["resolution"]["y"]
    dz = ct["resolution"]["z"]

    cst = []
    for roi_idx, roi_item in enumerate(ds.ROIContourSequence):
        roi_number = int(roi_item.ReferencedROINumber)
        name = roi_names.get(roi_number, f"ROI_{roi_number}")
        # Sanitise name (replace non-alphanumeric with space, like matRad)
        name = "".join(c if c.isalnum() else " " for c in name).strip()

        color = roi_colors.get(roi_number, [128, 128, 128])
        color_norm = [c / 255.0 for c in color]

        # Determine structure type
        name_lower = name.lower()
        if any(t in name_lower for t in ("ptv", "ctv", "gtv", "target", "boost")):
            struct_type = "TARGET"
            priority = 1
        else:
            struct_type = "OAR"
            priority = 2

        # Collect all voxel indices by rasterising contours slice-by-slice
        voxel_set = set()
        if hasattr(roi_item, "ContourSequence"):
            for contour in roi_item.ContourSequence:
                pts = np.array(contour.ContourData, dtype=float).reshape(-1, 3)
                # pts columns: x_lps, y_lps, z_lps [mm]
                z_c = pts[0, 2]
                # Find nearest z slice
                iz = int(np.argmin(np.abs(z - z_c)))
                if np.abs(z[iz] - z_c) > dz:
                    continue   # contour z doesn't match any slice

                # Rasterise polygon in x-y plane
                _add_contour_voxels(pts[:, 0], pts[:, 1], x, y, Ny, Nx, iz, Nz,
                                    voxel_set)

        voxel_indices = np.array(sorted(voxel_set), dtype=np.int64)

        props = {
            "Priority":     priority,
            "alphaX":       0.1,
            "betaX":        0.05,
            "Visible":      1,
            "visibleColor": color_norm,
        }

        objectives = []
        if struct_type == "TARGET":
            objectives = [{
                "type":       "SquaredDeviation",
                "penalty":    800,
                "parameters": [30.0],  # reference dose [Gy]
            }]

        cst.append([roi_idx, name, struct_type, [voxel_indices], props, objectives])

    return cst


def _add_contour_voxels(cx: np.ndarray, cy: np.ndarray,
                        x: np.ndarray, y: np.ndarray,
                        Ny: int, Nx: int, iz: int, Nz: int,
                        voxel_set: set):
    """Rasterise one closed 2-D contour into CT voxel linear indices.

    Uses a scanline fill (ray-casting) approach without external dependencies.
    Linear index formula (Fortran/column-major, 0-based):
        idx = iy + ix * Ny + iz * Ny * Nx   (then +1 for 1-based)
    """
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    dy = y[1] - y[0] if len(y) > 1 else 1.0

    # Bounding box in voxel space
    x0 = x[0];  y0 = y[0]
    ix_min = max(0, int(np.floor((cx.min() - x0) / dx)))
    ix_max = min(Nx - 1, int(np.ceil((cx.max() - x0) / dx)))
    iy_min = max(0, int(np.floor((cy.min() - y0) / dy)))
    iy_max = min(Ny - 1, int(np.ceil((cy.max() - y0) / dy)))

    # Close the polygon
    px = np.append(cx, cx[0])
    py = np.append(cy, cy[0])

    for iy in range(iy_min, iy_max + 1):
        scan_y = y0 + iy * dy
        # Ray-casting: count crossings at scan_y for each ix
        crossings = []
        for k in range(len(px) - 1):
            y1, y2 = py[k], py[k + 1]
            x1, x2 = px[k], px[k + 1]
            if (y1 <= scan_y < y2) or (y2 <= scan_y < y1):
                t = (scan_y - y1) / (y2 - y1)
                crossings.append(x1 + t * (x2 - x1))
        crossings.sort()
        # Fill between pairs of crossings
        for i in range(0, len(crossings) - 1, 2):
            fill_x0 = crossings[i]
            fill_x1 = crossings[i + 1]
            for ix in range(max(ix_min, int(np.ceil((fill_x0 - x0) / dx))),
                            min(ix_max + 1, int(np.floor((fill_x1 - x0) / dx)) + 1)):
                # 1-based Fortran-order linear index
                voxel_set.add(iy + ix * Ny + iz * Ny * Nx + 1)


# ---------------------------------------------------------------------------
# RTPlan importer
# ---------------------------------------------------------------------------

def import_rtplan(rtplan_file: str, ct: dict) -> dict:
    """Import a DICOM RTPlan file and return a pyMatRad pln dict.

    Parameters
    ----------
    rtplan_file : str
    ct : dict
        The ct dict (used for isocenter bounds check).

    Returns
    -------
    pln : dict
    """
    import pydicom

    ds = pydicom.dcmread(rtplan_file, stop_before_pixels=True)

    if hasattr(ds, "BeamSequence"):
        beam_seq = list(ds.BeamSequence)
        beam_type = "photons"
    elif hasattr(ds, "IonBeamSequence"):
        beam_seq = list(ds.IonBeamSequence)
        beam_type = "protons"
    else:
        raise ValueError("RTPlan contains neither BeamSequence nor IonBeamSequence.")

    # Filter to TREATMENT beams only (exclude SETUP, FIELD)
    treatment_beams = [
        b for b in beam_seq
        if getattr(b, "TreatmentDeliveryType", "TREATMENT").upper() == "TREATMENT"
    ]
    if not treatment_beams:
        treatment_beams = beam_seq  # fall back if attribute absent

    # Per-beam MU from FractionGroupSequence
    beam_mu = {}
    if hasattr(ds, "FractionGroupSequence"):
        fg = ds.FractionGroupSequence[0]
        for rb in fg.ReferencedBeamSequence:
            bnum = int(rb.ReferencedBeamNumber)
            mu = float(getattr(rb, "BeamMeterset", 0.0))
            beam_mu[bnum] = mu

    gantry_angles = []
    couch_angles  = []
    iso_centers   = []
    sad_mm        = []
    beam_names    = []
    beam_mus      = []
    energies      = []

    for b in treatment_beams:
        cp0 = (b.ControlPointSequence if hasattr(b, "ControlPointSequence")
               else b.IonControlPointSequence)[0]

        gantry = float(getattr(cp0, "GantryAngle", 0.0))
        couch  = float(getattr(cp0, "PatientSupportAngle", 0.0))

        # Isocenter: use plan value; fall back to CT centre if out of bounds
        iso_raw = [float(v) for v in cp0.IsocenterPosition] if hasattr(cp0, "IsocenterPosition") else None
        if iso_raw is not None:
            iso = np.array(iso_raw)
            # Bounds check
            x, y, z = ct["x"], ct["y"], ct["z"]
            if not (x[0] <= iso[0] <= x[-1] and y[0] <= iso[1] <= y[-1]
                    and z[0] <= iso[2] <= z[-1]):
                warnings.warn(
                    f"Beam '{getattr(b, 'BeamName', '')}' isocenter {iso} is "
                    "outside CT bounds — using CT centre.", stacklevel=2
                )
                iso = np.array([np.mean(x), np.mean(y), np.mean(z)])
        else:
            iso = np.array([np.mean(ct["x"]), np.mean(ct["y"]), np.mean(ct["z"])])

        energy = float(getattr(cp0, "NominalBeamEnergy", 6.0))
        sad    = float(getattr(b, "SourceAxisDistance", 1000.0))
        bnum   = int(getattr(b, "BeamNumber", 0))

        gantry_angles.append(gantry)
        couch_angles.append(couch)
        iso_centers.append(iso)
        sad_mm.append(sad)
        beam_names.append(str(getattr(b, "BeamName", f"Beam{bnum}")))
        beam_mus.append(beam_mu.get(bnum, 0.0))
        energies.append(energy)

    n_fractions = 1
    if hasattr(ds, "FractionGroupSequence"):
        n_fractions = int(
            getattr(ds.FractionGroupSequence[0], "NumberOfFractionsPlanned", 1)
        )

    # Map energy → machine name (user can override after import)
    energy_to_machine = {6.0: "TrueBeam_6X", 10.0: "TrueBeam_10XFFF",
                         15.0: "TrueBeam_15X"}
    machine = energy_to_machine.get(float(energies[0]), "Generic") if energies else "Generic"

    pln = {
        "radiationMode":  beam_type,
        "machine":        machine,
        "numOfFractions": n_fractions,
        "bioModel":       "none",
        "multScen":       "nomScen",
        "propStf": {
            "gantryAngles": np.array(gantry_angles),
            "couchAngles":  np.array(couch_angles),
            "isoCenter":    np.array(iso_centers[0]) if len(iso_centers) == 1
                            else np.array(iso_centers),
            "bixelWidth":   5.0,    # mm — default; adjust as needed
            "SAD":          float(sad_mm[0]) if sad_mm else 1000.0,
            "beamNames":    beam_names,
            "beamMU":       np.array(beam_mus),
            "energies_MV":  np.array(energies),
        },
        "propDoseCalc": {
            "doseGrid": {"resolution": {"x": 3.0, "y": 3.0, "z": 3.0}},
            "ignoreOutsideDensities": False,
        },
        "propOpt": {
            "runSequencing": False,
            "runDAO":        False,
        },
        "dicomInfo": {
            "PlanLabel":   str(getattr(ds, "RTPlanLabel", "")),
            "PlanName":    str(getattr(ds, "RTPlanName", "")),
            "NumBeams":    len(treatment_beams),
        },
    }

    return pln


# ---------------------------------------------------------------------------
# RTDose importer
# ---------------------------------------------------------------------------

def import_rtdose(rtdose_file: str, ct: dict) -> tuple:
    """Import a DICOM RTDose file and interpolate it onto the CT grid.

    Parameters
    ----------
    rtdose_file : str
    ct : dict

    Returns
    -------
    (dose_cube, dose_grid) where
        dose_cube : ndarray (Ny, Nx, Nz) [Gy] on the CT grid
        dose_grid : dict with keys 'x', 'y', 'z', 'resolution'
    """
    import pydicom
    from scipy.interpolate import RegularGridInterpolator

    ds = pydicom.dcmread(rtdose_file)

    scaling  = float(getattr(ds, "DoseGridScaling", 1.0))
    pixel    = ds.pixel_array.astype(np.float64)   # (Nz_dose, Ny_dose, Nx_dose)
    dose_raw = pixel * scaling                      # convert to Gy

    # Dose grid origin and spacing
    ipp     = [float(v) for v in ds.ImagePositionPatient]  # [x0, y0, z0]
    ps      = [float(v) for v in ds.PixelSpacing]           # [row_sp, col_sp]
    dz_dose_raw = getattr(ds, "SliceThickness", None)
    dz_dose = float(dz_dose_raw) if dz_dose_raw is not None else 2.0
    if hasattr(ds, "GridFrameOffsetVector"):
        offsets  = [float(v) for v in ds.GridFrameOffsetVector]
        z_dose   = ipp[2] + np.array(offsets)
    else:
        n_frames = dose_raw.shape[0]
        z_dose   = ipp[2] + np.arange(n_frames) * dz_dose

    dx_dose = float(ps[1])   # column → x
    dy_dose = float(ps[0])   # row → y
    Nz_dose, Ny_dose, Nx_dose = dose_raw.shape

    x_dose = ipp[0] + np.arange(Nx_dose) * dx_dose
    y_dose = ipp[1] + np.arange(Ny_dose) * dy_dose
    # dose_raw[frame, row, col] = dose_raw[iz, iy, ix]
    # shape for RegularGridInterpolator: axes (z, y, x) → data[iz, iy, ix]
    dose_zyx = dose_raw   # already (Nz, Ny, Nx)

    dose_grid = {
        "x": x_dose, "y": y_dose, "z": z_dose,
        "resolution": {"x": dx_dose, "y": dy_dose, "z": dz_dose},
    }

    # Interpolate onto CT grid
    interp_fn = RegularGridInterpolator(
        (z_dose, y_dose, x_dose), dose_zyx,
        method="linear", bounds_error=False, fill_value=0.0,
    )

    x_ct, y_ct, z_ct = ct["x"], ct["y"], ct["z"]
    Ny, Nx, Nz = ct["cubeDim"]
    zg, yg, xg = np.meshgrid(z_ct, y_ct, x_ct, indexing="ij")
    pts = np.column_stack([zg.ravel(), yg.ravel(), xg.ravel()])

    # interpolated shape: (Nz*Ny*Nx,) → reshape to (Nz, Ny, Nx) → transpose to (Ny, Nx, Nz)
    dose_ct = interp_fn(pts).reshape(Nz, Ny, Nx).transpose(1, 2, 0)
    dose_ct = np.asfortranarray(dose_ct)

    return dose_ct, dose_grid


# ---------------------------------------------------------------------------
# One-shot importer
# ---------------------------------------------------------------------------

def import_dicom(dicom_dir: str,
                 ct_dir: str = None,
                 hlut: np.ndarray = _DEFAULT_HLUT,
                 verbose: bool = True) -> dict:
    """Import all DICOM modalities from a folder.

    Parameters
    ----------
    dicom_dir : str
        Path to folder containing RTSTRUCT, RTPLAN, and/or RTDOSE files.
        May also contain CT files; if not, supply *ct_dir*.
    ct_dir : str, optional
        Separate folder containing CT DICOM slices.  Used when the plan
        folder contains no CT (e.g. ap_IMRT, ap_VMAT which share a CT
        with another plan).
    hlut : ndarray, optional
        Custom HU→RED lookup table.
    verbose : bool

    Returns
    -------
    dict with keys:
        'ct'        — ct dict
        'cst'       — cst list (empty list if no RTSTRUCT found)
        'pln'       — pln dict (None if no RTPLAN found)
        'dose'      — ndarray (Ny,Nx,Nz) [Gy] on CT grid (None if no RTDOSE)
        'dose_grid' — dose grid dict (None if no RTDOSE)
    """
    if verbose:
        print(f"Scanning {dicom_dir} ...")

    files = _scan_folder(dicom_dir)

    # If no CT in plan dir, try the supplied ct_dir
    if not files["ct"] and ct_dir is not None:
        if verbose:
            print(f"  No CT in plan dir, scanning ct_dir={ct_dir} ...")
        ct_files = _scan_folder(ct_dir)
        files["ct"] = ct_files["ct"]

    if not files["ct"]:
        raise FileNotFoundError(
            f"No CT DICOM files found in {dicom_dir}"
            + (f" or {ct_dir}" if ct_dir else "")
            + ".  Supply ct_dir= for plans that share a CT."
        )

    if verbose:
        print(f"  Found: {len(files['ct'])} CT slices, "
              f"{len(files['rtstruct'])} RTSTRUCT, "
              f"{len(files['rtplan'])} RTPLAN, "
              f"{len(files['rtdose'])} RTDOSE")

    # CT
    if verbose:
        print("Importing CT ...")
    ct = import_ct(files["ct"], hlut=hlut)
    if verbose:
        Ny, Nx, Nz = ct["cubeDim"]
        print(f"  CT grid: {Ny}×{Nx}×{Nz}  "
              f"resolution {ct['resolution']['x']:.2f}×"
              f"{ct['resolution']['y']:.2f}×"
              f"{ct['resolution']['z']:.2f} mm")

    # RTStruct
    cst = []
    if files["rtstruct"]:
        if verbose:
            print("Importing RTStruct ...")
        cst = import_rtstruct(files["rtstruct"][0], ct)
        if verbose:
            print(f"  {len(cst)} structures: "
                  + ", ".join(row[1] for row in cst[:6])
                  + ("..." if len(cst) > 6 else ""))

    # RTPlan
    pln = None
    if files["rtplan"]:
        if verbose:
            print("Importing RTPlan ...")
        pln = import_rtplan(files["rtplan"][0], ct)
        if verbose:
            n = len(pln["propStf"]["gantryAngles"])
            angles = pln["propStf"]["gantryAngles"]
            print(f"  {n} beams, gantry angles: {angles}")
            print(f"  Energy: {pln['propStf']['energies_MV'][0]} MV  "
                  f"→ machine: {pln['machine']}")

    # RTDose
    dose = None
    dose_grid = None
    if files["rtdose"]:
        if verbose:
            print("Importing RTDose ...")
        dose, dose_grid = import_rtdose(files["rtdose"][0], ct)
        if verbose:
            print(f"  Dose cube: {dose.shape}  "
                  f"max={dose.max():.3f} Gy  mean={dose[dose > 0].mean():.3f} Gy")

    return {
        "ct":        ct,
        "cst":       cst,
        "pln":       pln,
        "dose":      dose,
        "dose_grid": dose_grid,
    }


# ---------------------------------------------------------------------------
# MLC fluence importer
# ---------------------------------------------------------------------------

def _parse_beam_mlc(beam_ds) -> dict:
    """Extract jaw, leaf boundaries, and per-CP MLC positions from one DICOM beam.

    Returns a dict with:
        jaw_x      : [x_min, x_max] mm at iso (from ASYMX at CP0)
        jaw_y      : [y_min, y_max] mm at iso (from ASYMY at CP0)
        leaf_bounds: ndarray (n_leaves+1,) leaf y-boundaries at iso [mm]
        A          : ndarray (n_cp, n_leaves) bank-A (−x side) positions [mm]
        B          : ndarray (n_cp, n_leaves) bank-B (+x side) positions [mm]
        cum_w      : ndarray (n_cp,) cumulative meterset weights [0→1]
        beam_number: int DICOM BeamNumber
    """
    # Leaf boundaries from BeamLimitingDeviceSequence (static; independent of CP)
    leaf_bounds = None
    for dev in beam_ds.BeamLimitingDeviceSequence:
        if "MLC" in dev.RTBeamLimitingDeviceType.upper():
            leaf_bounds = np.array([float(v) for v in dev.LeafPositionBoundaries])
            break
    if leaf_bounds is None:
        raise ValueError(f"Beam {beam_ds.BeamName} has no MLC device sequence.")

    n_leaves = len(leaf_bounds) - 1

    # Extract MLC positions and jaw for each CP
    A_list, B_list, cum_w_list = [], [], []
    jaw_x = jaw_y = None

    # Track last seen MLC/jaw to fill gaps (DICOM only requires CPs that change)
    last_A = last_B = None
    last_jaw_x = last_jaw_y = None

    for cp in beam_ds.ControlPointSequence:
        cum_w_list.append(float(cp.CumulativeMetersetWeight))

        if hasattr(cp, "BeamLimitingDevicePositionSequence"):
            for dev in cp.BeamLimitingDevicePositionSequence:
                t = dev.RTBeamLimitingDeviceType.upper()
                pos = np.array([float(v) for v in dev.LeafJawPositions])
                if "MLCX" in t or (t == "MLC"):
                    last_A = pos[:n_leaves]
                    last_B = pos[n_leaves:]
                elif "ASYMX" in t or t == "X":
                    last_jaw_x = [float(pos[0]), float(pos[1])]
                elif "ASYMY" in t or t == "Y":
                    last_jaw_y = [float(pos[0]), float(pos[1])]

        A_list.append(last_A.copy() if last_A is not None else np.full(n_leaves, 1e4))
        B_list.append(last_B.copy() if last_B is not None else np.full(n_leaves, -1e4))
        if jaw_x is None and last_jaw_x is not None:
            jaw_x = last_jaw_x
        if jaw_y is None and last_jaw_y is not None:
            jaw_y = last_jaw_y

    # Fallbacks
    if jaw_x is None:
        jaw_x = [-200.0, 200.0]
    if jaw_y is None:
        jaw_y = [float(leaf_bounds[0]), float(leaf_bounds[-1])]

    return {
        "jaw_x":       np.array(jaw_x, dtype=float),
        "jaw_y":       np.array(jaw_y, dtype=float),
        "leaf_bounds": leaf_bounds,
        "A":           np.array(A_list),   # (n_cp, n_leaves)
        "B":           np.array(B_list),
        "cum_w":       np.array(cum_w_list),
        "beam_number": int(getattr(beam_ds, "BeamNumber", 0)),
    }


def _fluence_at_bixels(mlc: dict, x_bev: np.ndarray, z_bev: np.ndarray,
                        beam_mu: float, n_t: int = 30) -> np.ndarray:
    """Compute fluence (MU) at each bixel centre (x_bev, z_bev) in BEV.

    The BEV axes map directly to MLC coordinates at the isocenter plane:
        x_bev  → MLC x  (leaf-opening direction, A/B banks)
        z_bev  → MLC y  (leaf-selection direction, along leaf width)

    For each CP transition i→i+1 the leaves move linearly from A_i→A_{i+1}
    and B_i→B_{i+1}.  The open fraction at bixel x is estimated by sampling
    t uniformly in [0,1] (n_t samples).

    Parameters
    ----------
    mlc      : dict from _parse_beam_mlc
    x_bev    : (N,) lateral BEV positions of bixels [mm]
    z_bev    : (N,) cranio-caudal BEV positions of bixels [mm]
    beam_mu  : total MU for this beam
    n_t      : number of sub-samples per CP transition (default 30)

    Returns
    -------
    w : (N,) fluence in MU at each bixel centre
    """
    A         = mlc["A"]           # (n_cp, n_leaves)
    B         = mlc["B"]
    cum_w     = mlc["cum_w"]       # (n_cp,)
    bounds    = mlc["leaf_bounds"] # (n_leaves+1,)
    jaw_x     = mlc["jaw_x"]
    jaw_y     = mlc["jaw_y"]
    n_cp      = A.shape[0]
    delta_w   = np.diff(cum_w)     # (n_cp-1,)

    N = len(x_bev)
    fluence = np.zeros(N)

    # For each bixel, determine its leaf-pair index (from z_bev)
    # leaf_bounds are the y-edges of each leaf row
    lp_idx = np.searchsorted(bounds, z_bev, side="right") - 1
    lp_idx = np.clip(lp_idx, 0, A.shape[1] - 1)

    # Apply jaw mask: bixels outside jaws get zero fluence
    jaw_mask = (
        (x_bev >= jaw_x[0]) & (x_bev <= jaw_x[1]) &
        (z_bev >= jaw_y[0]) & (z_bev <= jaw_y[1])
    )

    # Also mask bixels outside the leaf boundary range
    jaw_mask &= (z_bev >= bounds[0]) & (z_bev <= bounds[-1])

    active = np.where(jaw_mask)[0]
    if len(active) == 0:
        return fluence

    x_a  = x_bev[active]          # (n_active,)
    lp_a = lp_idx[active]         # (n_active,) leaf-pair indices

    # t samples for linear interpolation within each CP gap
    t = np.linspace(0.0, 1.0, n_t)  # (n_t,)

    for i in range(n_cp - 1):
        dw = delta_w[i]
        if dw < 1e-10:
            continue

        # Bank positions for this transition, indexed by leaf pair for each bixel
        A_start = A[i,   lp_a]   # (n_active,)
        A_end   = A[i+1, lp_a]
        B_start = B[i,   lp_a]
        B_end   = B[i+1, lp_a]

        # Linear interpolation: shape (n_active, n_t)
        A_t = A_start[:, None] + t[None, :] * (A_end - A_start)[:, None]
        B_t = B_start[:, None] + t[None, :] * (B_end - B_start)[:, None]

        # Is bixel x open at each t?  shape (n_active, n_t)
        open_flag = (A_t < x_a[:, None]) & (x_a[:, None] < B_t)

        # Mean open fraction over the transition
        open_frac = open_flag.mean(axis=1)   # (n_active,)

        fluence[active] += dw * open_frac

    # Scale by total beam MU
    fluence *= beam_mu
    return fluence


def import_rtplan_fluence(rtplan_file: str, stf: list,
                          n_t: int = 30,
                          verbose: bool = True) -> np.ndarray:
    """Parse Eclipse DICOM RTPlan leaf sequences and return per-bixel weights.

    Supports sliding-window DMLC (continuous MLC motion) and step-and-shoot
    IMRT.  Both are represented identically in DICOM as a sequence of control
    points with cumulative meterset weights.

    The MLC leaf boundaries and jaw positions are read from the DICOM file —
    no external MLC model is required.

    The returned weight vector ``w`` has shape ``(totalNumOfBixels,)`` with
    units of **MU per bixel**.  Pass it directly to ``calc_dose_direct``:

    .. code-block:: python

        w = import_rtplan_fluence(plan_file, stf)
        result = calc_dose_direct(dij, w)

    Parameters
    ----------
    rtplan_file : str
        Path to the DICOM RTPLAN file.
    stf : list
        Beam geometry list returned by ``generate_stf()``.
        Used to read bixel BEV positions for each beam.
    n_t : int
        Number of sub-samples per CP transition for the open-fraction
        integral.  30 is accurate to < 1% for typical sliding-window fields.
    verbose : bool

    Returns
    -------
    w : ndarray, shape (totalNumOfBixels,)
        Per-bixel fluence weights in MU.  Zero for bixels outside the
        MLC/jaw aperture.

    Notes
    -----
    Coordinate mapping
        BEV ``rayPos_bev[0]`` (x) → MLC x (leaf-opening direction, A/B banks)
        BEV ``rayPos_bev[2]`` (z) → MLC y (leaf-selection, along leaf width)
        Both are defined at the isocenter plane, so no SAD magnification is
        needed.

    Beam matching
        DICOM beams are matched to STF beams by gantry angle (nearest, within
        ±0.5°).  Couch angle is checked as a secondary constraint when multiple
        beams share the same gantry angle.

    Leaf transmission / tongue-and-groove
        Not modelled — closed leaves contribute zero fluence.  The error is
        typically < 1% for IMRT fields.
    """
    import pydicom

    ds = pydicom.dcmread(rtplan_file, stop_before_pixels=True)

    # Beam sequence (photons only)
    if hasattr(ds, "BeamSequence"):
        beam_seq = list(ds.BeamSequence)
    elif hasattr(ds, "IonBeamSequence"):
        raise NotImplementedError("Ion beam (proton/carbon) MLC import not yet supported.")
    else:
        raise ValueError("RTPlan has neither BeamSequence nor IonBeamSequence.")

    # Filter to TREATMENT beams
    treatment_beams = [
        b for b in beam_seq
        if getattr(b, "TreatmentDeliveryType", "TREATMENT").upper() == "TREATMENT"
    ]
    if not treatment_beams:
        treatment_beams = beam_seq

    # Per-beam MU from FractionGroupSequence
    beam_mu_map = {}
    if hasattr(ds, "FractionGroupSequence"):
        for rb in ds.FractionGroupSequence[0].ReferencedBeamSequence:
            beam_mu_map[int(rb.ReferencedBeamNumber)] = float(rb.BeamMeterset)

    # Parse MLC data for each DICOM beam, keyed by (gantry, couch) angle
    dicom_beams = {}
    for b in treatment_beams:
        cp0 = b.ControlPointSequence[0]
        g = round(float(getattr(cp0, "GantryAngle", 0.0)), 2)
        c = round(float(getattr(cp0, "PatientSupportAngle", 0.0)), 2)
        mu = beam_mu_map.get(int(getattr(b, "BeamNumber", 0)), 0.0)
        try:
            mlc = _parse_beam_mlc(b)
        except ValueError as e:
            warnings.warn(str(e))
            continue
        mlc["beam_mu"] = mu
        dicom_beams[(g, c)] = mlc

    if verbose:
        print(f"  Parsed {len(dicom_beams)} DICOM treatment beams with MLC data")

    # Build per-bixel weight vector ordered by STF bixel sequence
    total_bixels = sum(b["totalNumOfBixels"] for b in stf)
    w = np.zeros(total_bixels)

    bixel_offset = 0
    for beam_idx, stf_beam in enumerate(stf):
        g_stf = round(float(stf_beam["gantryAngle"]), 2)
        c_stf = round(float(stf_beam["couchAngle"]), 2)

        # Match to DICOM beam by (gantry, couch) — try exact first, then nearest
        mlc = dicom_beams.get((g_stf, c_stf))
        if mlc is None:
            candidates = sorted(
                ((abs(g - g_stf), abs(c - c_stf), (g, c))
                 for (g, c) in dicom_beams),
            )
            if candidates and candidates[0][0] <= 0.5:
                mlc = dicom_beams[candidates[0][2]]
                if verbose:
                    print(f"  Beam {beam_idx}: STF gantry {g_stf}° matched to "
                          f"DICOM gantry {candidates[0][2][0]}° (Δ={candidates[0][0]:.2f}°)")
            else:
                warnings.warn(
                    f"STF beam {beam_idx} (gantry={g_stf}°) has no matching DICOM beam. "
                    "Weights will be zero."
                )
                bixel_offset += stf_beam["totalNumOfBixels"]
                continue

        # Collect BEV bixel positions for this beam
        # rayPos_bev[0] = x_bev (MLC x), rayPos_bev[2] = z_bev (MLC y/leaf)
        n_bixels = stf_beam["totalNumOfBixels"]
        x_bev = np.array([r["rayPos_bev"][0] for r in stf_beam["ray"]])
        z_bev = np.array([r["rayPos_bev"][2] for r in stf_beam["ray"]])

        if verbose:
            print(f"  Beam {beam_idx}: gantry={g_stf}°  "
                  f"{n_bixels} bixels  MU={mlc['beam_mu']:.1f}  "
                  f"x=[{x_bev.min():.0f},{x_bev.max():.0f}] mm  "
                  f"z=[{z_bev.min():.0f},{z_bev.max():.0f}] mm")

        beam_w = _fluence_at_bixels(mlc, x_bev, z_bev, mlc["beam_mu"], n_t=n_t)
        w[bixel_offset: bixel_offset + n_bixels] = beam_w
        bixel_offset += n_bixels

        if verbose:
            n_open = np.sum(beam_w > 0)
            print(f"    → {n_open}/{n_bixels} bixels open  "
                  f"total fluence: {beam_w.sum():.1f} MU·bixels")

    return w
