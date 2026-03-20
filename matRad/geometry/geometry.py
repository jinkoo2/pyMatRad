"""
Geometry utilities - Python port of matRad geometry functions.

Key coordinate systems:
  - Cube indices:  [i, j, k] = [row, col, slice] (Python 0-based, MATLAB 1-based)
                   MATLAB ordering: cubeDim = [Ny, Nx, Nz]
                   => i is y-direction, j is x-direction, k is z-direction
  - Cube coords:  [cx, cy, cz] in mm from (0,0,0)
  - World coords: [x, y, z] in mm (patient LPS coordinate system)
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple


def get_world_axes(grid_struct: dict) -> dict:
    """
    Compute world coordinate axes x, y, z for a CT grid struct.

    Port of matRad_getWorldAxes.m

    Parameters
    ----------
    grid_struct : dict
        CT or dose grid struct with fields:
        - cubeDim or dimensions: [Ny, Nx, Nz]
        - resolution: dict with x, y, z [mm/voxel]
        - optionally x, y, z arrays (already computed)

    Returns
    -------
    dict
        Updated grid_struct with x, y, z arrays added.
    """
    grid_struct = dict(grid_struct)  # shallow copy

    # Normalize dimension field
    if "cubeDim" in grid_struct and "dimensions" not in grid_struct:
        grid_struct["dimensions"] = grid_struct["cubeDim"]

    dims = grid_struct["dimensions"]  # [Ny, Nx, Nz]
    res = grid_struct["resolution"]

    # Check if already computed and non-empty
    if "x" in grid_struct and "y" in grid_struct and "z" in grid_struct:
        if (grid_struct["x"] is not None and len(grid_struct["x"]) > 0 and
                grid_struct["y"] is not None and len(grid_struct["y"]) > 0 and
                grid_struct["z"] is not None and len(grid_struct["z"]) > 0):
            return grid_struct

    # Check for DICOM origin
    if "dicomInfo" in grid_struct and "ImagePositionPatient" in grid_struct.get("dicomInfo", {}):
        first_vox = np.array(grid_struct["dicomInfo"]["ImagePositionPatient"])
    else:
        # Default: center origin
        # dimensions = [Ny, Nx, Nz], so x has Nx voxels, y has Ny, z has Nz
        Ny, Nx, Nz = dims[0], dims[1], dims[2]
        first_vox = np.array([
            -Nx / 2 * res["x"],
            -Ny / 2 * res["y"],
            -Nz / 2 * res["z"],
        ])

    Ny, Nx, Nz = dims[0], dims[1], dims[2]
    grid_struct["x"] = first_vox[0] + res["x"] * np.arange(Nx)
    grid_struct["y"] = first_vox[1] + res["y"] * np.arange(Ny)
    grid_struct["z"] = first_vox[2] + res["z"] * np.arange(Nz)

    return grid_struct


def cube_coords_to_world_coords(cube_coords: np.ndarray, grid_struct: dict, allow_outside: bool = True) -> np.ndarray:
    """
    Convert cube coordinates [mm from (res.x, res.y, res.z)] to world coordinates.

    Port of matRad_cubeCoords2worldCoords.m

    Parameters
    ----------
    cube_coords : np.ndarray, shape (N, 3)
        Cube coordinates [cx, cy, cz] in mm
    grid_struct : dict
        CT or dose grid struct
    allow_outside : bool

    Returns
    -------
    np.ndarray, shape (N, 3)
        World coordinates [x, y, z] in mm
    """
    grid_struct = get_world_axes(grid_struct)
    res = grid_struct["resolution"]

    first_vox_world = np.array([
        np.min(grid_struct["x"]),
        np.min(grid_struct["y"]),
        np.min(grid_struct["z"]),
    ])
    first_vox_cube = np.array([res["x"], res["y"], res["z"]])
    translation = first_vox_world - first_vox_cube

    return cube_coords + translation


def cube_index_to_world_coords(cube_ix: np.ndarray, grid_struct: dict) -> np.ndarray:
    """
    Convert cube voxel indices to world coordinates [mm].

    Port of matRad_cubeIndex2worldCoords.m

    In MATLAB the cube has dimensions [Ny, Nx, Nz], indexed as (i, j, k).
    The cube coordinate of voxel (i, j, k) is:
        cx = j * res.x   (j is x-direction)
        cy = i * res.y   (i is y-direction)
        cz = k * res.z   (k is z-direction)

    Parameters
    ----------
    cube_ix : np.ndarray
        Either (N, 3) subscript indices [i, j, k] (0-based, Python style)
        or (N,) linear indices (0-based, Python/C-order)
    grid_struct : dict

    Returns
    -------
    np.ndarray, shape (N, 3)
        World coordinates [x, y, z] in mm
    """
    if "cubeDim" in grid_struct and "dimensions" not in grid_struct:
        grid_struct = dict(grid_struct)
        grid_struct["dimensions"] = grid_struct["cubeDim"]

    dims = grid_struct["dimensions"]  # [Ny, Nx, Nz]
    res = grid_struct["resolution"]

    orig = np.asarray(cube_ix, dtype=np.int64)
    # Detect whether input is linear indices (1D array or column vector) vs [i,j,k] subscripts
    is_linear = orig.ndim == 1 or (orig.ndim == 2 and orig.shape[1] == 1)
    cube_ix = np.atleast_2d(orig)

    if is_linear:
        # Linear indices -> subscript
        cube_ix = cube_ix.ravel()
        Ny, Nx, Nz = dims[0], dims[1], dims[2]
        # MATLAB-style linear index: col-major (Fortran-order), 1-based
        # ix = i + (j-1)*Ny + (k-1)*Ny*Nx  (MATLAB 1-based i, j, k)
        # Convert to 0-based Python first
        ix_0 = cube_ix - 1  # Convert 1-based MATLAB to 0-based Python
        k = ix_0 // (Ny * Nx)
        rem = ix_0 % (Ny * Nx)
        j = rem // Ny
        i = rem % Ny
        cube_ix = np.column_stack([i, j, k])  # 0-based Python subscripts

    # cube_ix is (N, 3) = [i, j, k] (0-based)
    # Cube coords: x = (j+1)*res.x, y = (i+1)*res.y, z = (k+1)*res.z (1-based MATLAB -> Python +1)
    # Python: x = (j)*res.x + res.x ... or just use j+1 equivalent
    # MATLAB: cubeCoord = cubeIx(:,[2 1 3]) .* [res.x res.y res.z]
    # With 1-based MATLAB indexing: cubeCoord(j) = j * res.x where j is the column (1-based)
    # Python 0-based equivalent: cubeCoord = (j+1) * res.x

    # Convert from 0-based Python to 1-based MATLAB
    cube_ix_1based = cube_ix + 1  # [i, j, k] 1-based
    # Permute: MATLAB does cubeIx(:,[2 1 3]) to swap i,j -> [j, i, k] for [x, y, z]
    cube_coords = np.column_stack([
        cube_ix_1based[:, 1] * res["x"],   # j -> x
        cube_ix_1based[:, 0] * res["y"],   # i -> y
        cube_ix_1based[:, 2] * res["z"],   # k -> z
    ])

    return cube_coords_to_world_coords(cube_coords, grid_struct)


def world_to_cube_coords(world_coords: np.ndarray, grid_struct: dict, allow_outside: bool = True) -> np.ndarray:
    """
    Convert world coordinates [mm] to cube coordinates [mm].

    Port of matRad_world2cubeCoords.m

    Parameters
    ----------
    world_coords : np.ndarray, shape (N, 3)
        World coordinates [x, y, z] in mm
    grid_struct : dict
    allow_outside : bool

    Returns
    -------
    np.ndarray, shape (N, 3)
        Cube coordinates [cx, cy, cz] in mm
    """
    grid_struct = get_world_axes(grid_struct)
    res = grid_struct["resolution"]

    first_vox_world = np.array([
        np.min(grid_struct["x"]),
        np.min(grid_struct["y"]),
        np.min(grid_struct["z"]),
    ])
    first_vox_cube = np.array([res["x"], res["y"], res["z"]])
    translation = first_vox_cube - first_vox_world

    return world_coords + translation


def world_to_cube_index(world_coords: np.ndarray, grid_struct: dict) -> np.ndarray:
    """
    Convert world coordinates to 0-based linear cube indices.

    Port of matRad_world2cubeIndex.m

    Parameters
    ----------
    world_coords : np.ndarray, shape (N, 3) or (3,)
        World coordinates [x, y, z] in mm

    Returns
    -------
    np.ndarray, shape (N, 3)
        Subscript indices [i, j, k] (0-based Python)
    """
    if "cubeDim" in grid_struct and "dimensions" not in grid_struct:
        grid_struct = dict(grid_struct)
        grid_struct["dimensions"] = grid_struct["cubeDim"]

    dims = grid_struct["dimensions"]  # [Ny, Nx, Nz]
    res = grid_struct["resolution"]

    world_coords = np.atleast_2d(np.asarray(world_coords, dtype=float))

    # Get cube coords
    cube_coords = world_to_cube_coords(world_coords, grid_struct)

    # Convert to indices (1-based MATLAB)
    # cube coord = index * resolution
    j = np.round(cube_coords[:, 0] / res["x"]).astype(int) - 1  # x -> j (col), 0-based
    i = np.round(cube_coords[:, 1] / res["y"]).astype(int) - 1  # y -> i (row), 0-based
    k = np.round(cube_coords[:, 2] / res["z"]).astype(int) - 1  # z -> k (slice), 0-based

    Ny, Nx, Nz = dims[0], dims[1], dims[2]
    i = np.clip(i, 0, Ny - 1)
    j = np.clip(j, 0, Nx - 1)
    k = np.clip(k, 0, Nz - 1)

    return np.column_stack([i, j, k])


def linear_index_to_subscript(lin_ix: np.ndarray, dims: tuple) -> np.ndarray:
    """
    Convert MATLAB-style (Fortran/column-major) linear indices to [i, j, k] subscripts.

    MATLAB uses column-major linear indexing: ix = i + j*Ny + k*Ny*Nx (1-based)
    Python: 0-based equivalent

    Parameters
    ----------
    lin_ix : np.ndarray
        1-based MATLAB linear indices
    dims : tuple (Ny, Nx, Nz)

    Returns
    -------
    np.ndarray (N, 3), 0-based [i, j, k]
    """
    lin_ix = np.asarray(lin_ix, dtype=np.int64) - 1  # convert to 0-based
    Ny, Nx, Nz = dims
    k = lin_ix // (Ny * Nx)
    rem = lin_ix % (Ny * Nx)
    j = rem // Ny
    i = rem % Ny
    return np.column_stack([i, j, k])


def subscript_to_linear_index(ijk: np.ndarray, dims: tuple) -> np.ndarray:
    """
    Convert [i, j, k] subscripts to MATLAB-style 1-based linear indices.

    Parameters
    ----------
    ijk : np.ndarray (N, 3), 0-based
    dims : tuple (Ny, Nx, Nz)

    Returns
    -------
    np.ndarray (N,), 1-based MATLAB linear indices
    """
    Ny, Nx, Nz = dims
    ijk = np.asarray(ijk)
    i = ijk[:, 0]
    j = ijk[:, 1]
    k = ijk[:, 2]
    return i + j * Ny + k * Ny * Nx + 1  # 1-based


def get_rotation_matrix(gantry_angle: float, couch_angle: float) -> np.ndarray:
    """
    Returns rotation matrix for beam geometry in LPS coordinate system.

    Port of matRad_getRotationMatrix.m

    Active, counter-clockwise rotation:
    - Gantry rotates around z-axis
    - Couch rotates around y-axis

    R = R_Couch @ R_Gantry

    Parameters
    ----------
    gantry_angle : float
        Gantry angle in degrees
    couch_angle : float
        Couch angle in degrees

    Returns
    -------
    np.ndarray (3, 3)
        Rotation matrix (for pre-multiplication: R @ x for column vectors)
    """
    g = np.radians(gantry_angle)
    c = np.radians(couch_angle)

    R_gantry = np.array([
        [np.cos(g), -np.sin(g), 0],
        [np.sin(g),  np.cos(g), 0],
        [0,          0,         1],
    ])

    R_couch = np.array([
        [ np.cos(c), 0, np.sin(c)],
        [ 0,         1, 0        ],
        [-np.sin(c), 0, np.cos(c)],
    ])

    return R_couch @ R_gantry


def get_iso_center(cst: list, ct: dict, vis_bool: bool = False) -> np.ndarray:
    """
    Compute isocenter as center of gravity of all TARGET VOIs.

    Port of matRad_getIsoCenter.m

    Parameters
    ----------
    cst : list
        List of structure rows: [idx, name, type, voxels, properties, objectives]
    ct : dict
        CT struct
    vis_bool : bool
        Enable visualization (not supported in Python; ignored)

    Returns
    -------
    np.ndarray (3,)
        Isocenter [x, y, z] in mm
    """
    from ..config import MatRad_Config
    cfg = MatRad_Config.instance()

    # Check if any objectives are defined
    no_obj_or_const = True
    if len(cst[0]) >= 6:
        no_obj_or_const = all(
            (len(row) < 6 or row[5] is None or len(row[5]) == 0)
            for row in cst
        )

    # Collect target voxel indices
    all_target_voxels = []
    for row in cst:
        voi_type = row[2]
        voi_indices = row[3][0] if isinstance(row[3], list) else row[3]
        has_obj = len(row) >= 6 and row[5] is not None and len(row[5]) > 0

        if voi_type == "TARGET" and (no_obj_or_const or has_obj):
            all_target_voxels.append(np.asarray(voi_indices))

    if not all_target_voxels:
        cfg.disp_error("Could not find target!")

    V = np.unique(np.concatenate(all_target_voxels))

    # Convert linear indices to world coordinates
    coords = cube_index_to_world_coords(V, ct)

    return np.mean(coords, axis=0)


def set_overlap_priorities(cst: list) -> list:
    """
    Set overlap priorities for VOIs.

    VOIs with lower priority overlap index will overwrite higher ones.
    Port of matRad_setOverlapPriorities.m

    Parameters
    ----------
    cst : list of lists

    Returns
    -------
    list
        Updated cst with overlap priorities set
    """
    num_vois = len(cst)
    for i, row in enumerate(cst):
        if len(row) < 5 or row[4] is None:
            cst[i] = list(cst[i]) + [{}] * (5 - len(cst[i]))
            cst[i][4] = {}
        if not isinstance(cst[i][4], dict):
            cst[i][4] = {}
        if "Priority" not in cst[i][4]:
            cst[i][4]["Priority"] = i + 1  # Default priority based on order

    # Sort by priority: lower number = higher priority
    # Find TARGET structures - they get highest priority (lowest number)
    for i, row in enumerate(cst):
        if row[2] == "TARGET":
            cst[i][4]["Priority"] = min(cst[i][4].get("Priority", i + 1), i + 1)

    return cst


def resize_cst_to_grid(cst: list, ct: dict, dose_grid: dict) -> list:
    """
    Resize structure set VOI voxel indices to a different grid resolution.

    Port of matRad_resizeCstToGrid.m

    The input cst has voxel indices in ct grid space.
    Output cst will have voxel indices in dose_grid space.

    Parameters
    ----------
    cst : list
        Structure set in CT grid
    ct : dict
        Original CT grid
    dose_grid : dict
        Target dose grid (potentially coarser)

    Returns
    -------
    list
        Updated cst with voxel indices in dose_grid space
    """
    ct = get_world_axes(ct)
    dose_grid = get_world_axes(dose_grid)

    # Map each VOI's voxel indices from ct grid to dose grid
    cst_dose = [list(row) for row in cst]

    for i, row in enumerate(cst_dose):
        if len(row) < 4 or row[3] is None:
            continue

        voxel_list = row[3]
        if isinstance(voxel_list, list) and len(voxel_list) > 0:
            ct_voxels = np.asarray(voxel_list[0], dtype=np.int64)
        else:
            ct_voxels = np.asarray(voxel_list, dtype=np.int64)

        if len(ct_voxels) == 0:
            cst_dose[i][3] = [np.array([], dtype=np.int64)]
            continue

        # Convert CT linear indices to world coords
        ct_dims = ct.get("cubeDim", ct.get("dimensions"))
        world_coords = cube_index_to_world_coords(ct_voxels, ct)

        # Convert world coords to dose grid indices
        dose_dims = dose_grid.get("cubeDim", dose_grid.get("dimensions"))
        dose_res = dose_grid["resolution"]

        # Find which dose grid voxels overlap with these world coords
        # Snap world coords to nearest dose grid voxel
        dose_ijk = world_to_cube_index(world_coords, dose_grid)
        Ny_d, Nx_d, Nz_d = dose_dims[0], dose_dims[1], dose_dims[2]
        mask = (
            (dose_ijk[:, 0] >= 0) & (dose_ijk[:, 0] < Ny_d) &
            (dose_ijk[:, 1] >= 0) & (dose_ijk[:, 1] < Nx_d) &
            (dose_ijk[:, 2] >= 0) & (dose_ijk[:, 2] < Nz_d)
        )
        valid_ijk = dose_ijk[mask]
        if len(valid_ijk) == 0:
            cst_dose[i][3] = [np.array([], dtype=np.int64)]
            continue

        # Convert to MATLAB-style linear indices (1-based)
        lin_ix = subscript_to_linear_index(valid_ijk, tuple(dose_dims))
        lin_ix = np.unique(lin_ix)
        cst_dose[i][3] = [lin_ix]

    return cst_dose


def add_margin(cst: list, ct: dict, margin: float = 5.0) -> list:
    """
    Add margin to target structures.

    Parameters
    ----------
    cst : list
    ct : dict
    margin : float
        Margin in mm

    Returns
    -------
    list
        Updated cst with margins added to targets
    """
    # Simple implementation: expand TARGET voxels by margin
    ct = get_world_axes(ct)
    dims = ct.get("cubeDim", ct.get("dimensions"))
    Ny, Nx, Nz = dims[0], dims[1], dims[2]
    res = ct["resolution"]

    margin_vox_x = int(np.ceil(margin / res["x"]))
    margin_vox_y = int(np.ceil(margin / res["y"]))
    margin_vox_z = int(np.ceil(margin / res["z"]))

    for i, row in enumerate(cst):
        if row[2] != "TARGET":
            continue

        voxel_list = row[3]
        if isinstance(voxel_list, list):
            ct_voxels = np.asarray(voxel_list[0], dtype=np.int64)
        else:
            ct_voxels = np.asarray(voxel_list, dtype=np.int64)

        if len(ct_voxels) == 0:
            continue

        # Convert to subscripts
        ijk = linear_index_to_subscript(ct_voxels, (Ny, Nx, Nz))

        # Create boolean mask
        mask = np.zeros((Ny, Nx, Nz), dtype=bool)
        mask[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

        # Dilate by margin using binary dilation
        from scipy.ndimage import binary_dilation
        struct = np.ones((2 * margin_vox_y + 1, 2 * margin_vox_x + 1, 2 * margin_vox_z + 1), dtype=bool)
        mask_dilated = binary_dilation(mask, structure=struct)

        new_voxels = np.where(mask_dilated.ravel(order="F"))[0] + 1  # 1-based linear
        cst[i] = list(cst[i])
        cst[i][3] = [new_voxels]

    return cst
