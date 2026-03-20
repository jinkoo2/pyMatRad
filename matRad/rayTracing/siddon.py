"""
Siddon ray tracer for 3D CT cubes.

Python port of matRad_siddonRayTracer.m and matRad_rayTracing.m

Reference: Siddon 1985 Medical Physics (PMID: 4000088)
"""

import numpy as np
from typing import List, Optional, Tuple


def siddon_ray_tracer(
    isocenter_cube: np.ndarray,
    resolution: dict,
    source_point: np.ndarray,
    target_point: np.ndarray,
    cubes: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], float, np.ndarray]:
    """
    Siddon ray tracing through 3D cube(s).

    Port of matRad_siddonRayTracer.m

    The cube uses MATLAB [Ny, Nx, Nz] ordering.
    Coordinates correspond to cube voxel centers:
    - X axis: column index j (index in x-direction)
    - Y axis: row index i (index in y-direction)
    - Z axis: slice index k (index in z-direction)

    Parameters
    ----------
    isocenter_cube : np.ndarray (3,)
        Isocenter in cube coordinates [x, y, z] in mm
    resolution : dict
        {'x': rx, 'y': ry, 'z': rz} in mm/voxel
    source_point : np.ndarray (3,)
        Source point [x, y, z] in mm (relative to isocenter)
    target_point : np.ndarray (3,)
        Target point [x, y, z] in mm (relative to isocenter)
    cubes : list of np.ndarray
        3D cubes with shape [Ny, Nx, Nz] to trace through

    Returns
    -------
    alphas : np.ndarray
        Parametric intersection values
    l : np.ndarray
        Path lengths through each voxel segment
    rho : list of np.ndarray
        Densities from each cube at hit voxels
    d12 : float
        Total distance between source and target
    ix : np.ndarray
        1-based MATLAB linear indices of hit voxels
    """
    # Shift source/target by isocenter (make isocenter the origin offset)
    sp = source_point + isocenter_cube
    tp = target_point + isocenter_cube

    rx = resolution["x"]
    ry = resolution["y"]
    rz = resolution["z"]

    cube_shape = cubes[0].shape  # [Ny, Nx, Nz]
    Ny_vox = cube_shape[0]
    Nx_vox = cube_shape[1]
    Nz_vox = cube_shape[2]

    # Number of planes = number of voxels + 1
    y_num_planes = Ny_vox + 1
    x_num_planes = Nx_vox + 1
    z_num_planes = Nz_vox + 1

    # eq 11: distance
    d12 = float(np.linalg.norm(sp - tp))
    if d12 == 0:
        return np.array([]), np.array([]), [np.array([]) for _ in cubes], 0.0, np.array([])

    # eq 3: position of first and last planes
    x_plane_1 = 0.5 * rx
    y_plane_1 = 0.5 * ry
    z_plane_1 = 0.5 * rz

    x_plane_end = (x_num_planes - 0.5) * rx
    y_plane_end = (y_num_planes - 0.5) * ry
    z_plane_end = (z_num_planes - 0.5) * rz

    # Check if ray intersects the cube
    t_values = []
    for t in [(np.array([x_plane_1, y_plane_1, z_plane_1]) - sp) / (tp - sp + 1e-15),
              (np.array([x_plane_end, y_plane_end, z_plane_end]) - sp) / (tp - sp + 1e-15)]:
        for tv in t:
            if not np.isfinite(tv):
                continue
            p = sp + tv * (tp - sp)
            lower = np.array([x_plane_1, y_plane_1, z_plane_1]) - np.sqrt(np.finfo(float).eps)
            upper = np.array([x_plane_end, y_plane_end, z_plane_end]) + np.sqrt(np.finfo(float).eps)
            if np.all(p > lower) and np.all(p < upper):
                t_values.append(tv)

    does_hit = len(t_values) > 0
    if not does_hit:
        return np.array([]), np.array([]), [np.array([]) for _ in cubes], d12, np.array([])

    # eq 4: parametric alpha values at cube boundary planes
    def safe_alpha(plane_1, plane_end, s, t_):
        if abs(t_ - s) < 1e-12:
            return None, None
        a1 = (plane_1 - s) / (t_ - s)
        ae = (plane_end - s) / (t_ - s)
        return a1, ae

    aX_1, aX_end = safe_alpha(x_plane_1, x_plane_end, sp[0], tp[0])
    aY_1, aY_end = safe_alpha(y_plane_1, y_plane_end, sp[1], tp[1])
    aZ_1, aZ_end = safe_alpha(z_plane_1, z_plane_end, sp[2], tp[2])

    # eq 5: alpha_min, alpha_max
    candidates_min = [0.0]
    candidates_max = [1.0]
    for a1, ae in [(aX_1, aX_end), (aY_1, aY_end), (aZ_1, aZ_end)]:
        if a1 is not None and ae is not None:
            candidates_min.append(min(a1, ae))
            candidates_max.append(max(a1, ae))
    alpha_min = max(candidates_min)
    alpha_max = min(candidates_max)

    if alpha_min >= alpha_max:
        return np.array([]), np.array([]), [np.array([]) for _ in cubes], d12, np.array([])

    # eq 6: index ranges
    def calc_index_range(num_planes, plane_1, plane_end, s, t_, alpha_min_, alpha_max_):
        if abs(t_ - s) < 1e-12:
            return None, None
        dt = t_ - s
        if dt > 0:
            i_min = num_planes - (plane_end - alpha_min_ * dt - s) / (plane_end - plane_1) * (num_planes - 1) - 1
            i_max = 1 + (s + alpha_max_ * dt - plane_1) / (plane_end - plane_1) * (num_planes - 1)
        else:
            i_min = num_planes - (plane_end - alpha_max_ * dt - s) / (plane_end - plane_1) * (num_planes - 1) - 1
            i_max = 1 + (s + alpha_min_ * dt - plane_1) / (plane_end - plane_1) * (num_planes - 1)
        # Rounding trick from MATLAB
        i_min = int(np.ceil(np.round(i_min * 1000) / 1000))
        i_max = int(np.floor(np.round(i_max * 1000) / 1000))
        return i_min, i_max

    i_min_x, i_max_x = calc_index_range(x_num_planes, x_plane_1, x_plane_end, sp[0], tp[0], alpha_min, alpha_max)
    j_min_y, j_max_y = calc_index_range(y_num_planes, y_plane_1, y_plane_end, sp[1], tp[1], alpha_min, alpha_max)
    k_min_z, k_max_z = calc_index_range(z_num_planes, z_plane_1, z_plane_end, sp[2], tp[2], alpha_min, alpha_max)

    # eq 7: parametric alpha values for all intersected planes
    def calc_alphas(i_min_, i_max_, plane_1, resolution, s, t_):
        if i_min_ is None or i_max_ is None or i_min_ == i_max_:
            return np.array([])
        dt = t_ - s
        if abs(dt) < 1e-12:
            return np.array([])
        if dt > 0:
            indices = np.arange(i_min_, i_max_ + 1)
        else:
            indices = np.arange(i_max_, i_min_ - 1, -1)
        return (resolution * indices - s - 0.5 * resolution) / dt

    alpha_x = calc_alphas(i_min_x, i_max_x, x_plane_1, rx, sp[0], tp[0])
    alpha_y = calc_alphas(j_min_y, j_max_y, y_plane_1, ry, sp[1], tp[1])
    alpha_z = calc_alphas(k_min_z, k_max_z, z_plane_1, rz, sp[2], tp[2])

    # eq 8: merge all alpha sets
    all_alphas = np.concatenate([[alpha_min], alpha_x, alpha_y, alpha_z, [alpha_max]])
    alphas = np.unique(all_alphas)
    alphas = alphas[(alphas >= alpha_min - 1e-10) & (alphas <= alpha_max + 1e-10)]

    if len(alphas) < 2:
        return np.array([]), np.array([]), [np.array([]) for _ in cubes], d12, np.array([])

    # eq 10: voxel intersection lengths
    l = d12 * np.diff(alphas)

    # eq 13: midpoints
    alphas_mid = 0.5 * (alphas[:-1] + alphas[1:])

    # eq 12: voxel coordinates at midpoints
    i_mm = sp[0] + alphas_mid * (tp[0] - sp[0])
    j_mm = sp[1] + alphas_mid * (tp[1] - sp[1])
    k_mm = sp[2] + alphas_mid * (tp[2] - sp[2])

    # Convert to voxel indices (1-based MATLAB style)
    i_idx = np.round(i_mm / rx).astype(int)  # x-direction -> column j
    j_idx = np.round(j_mm / ry).astype(int)  # y-direction -> row i
    k_idx = np.round(k_mm / rz).astype(int)  # z-direction -> slice k

    # Handle numerical instabilities at borders
    i_idx = np.clip(i_idx, 1, Nx_vox)  # x -> column (MATLAB j)
    j_idx = np.clip(j_idx, 1, Ny_vox)  # y -> row (MATLAB i)
    k_idx = np.clip(k_idx, 1, Nz_vox)  # z -> slice

    # Convert to MATLAB-style 1-based column-major linear indices
    # MATLAB: ix = j + (i-1)*Ny + (k-1)*Ny*Nx
    # where i=row (y), j=col (x), k=slice (z)
    # cube_shape = [Ny, Nx, Nz]
    ix = j_idx + (i_idx - 1) * Ny_vox + (k_idx - 1) * Ny_vox * Nx_vox

    # Extract densities from cubes
    ix_0based = ix - 1  # Convert to 0-based
    rho_list = []
    flat_cubes = [c.ravel(order="F") for c in cubes]
    for flat in flat_cubes:
        rho_list.append(flat[ix_0based])

    return alphas, l, rho_list, d12, ix


def ray_tracing(
    stf_beam: dict,
    ct: dict,
    V_ct_grid: np.ndarray,
    rot_coords_v: np.ndarray,
    effective_lateral_cutoff: float,
) -> List[np.ndarray]:
    """
    Calculate radiological depth for all voxels in the CT grid.

    Port of matRad_rayTracing.m

    For each voxel in V_ct_grid, finds its corresponding ray and computes
    the radiological depth (sum of density * path length along the ray).

    Parameters
    ----------
    stf_beam : dict
        Single beam from stf, with:
        - sourcePoint_bev: source in BEV coords
        - ray: list of rays with targetPoint_bev
        - SAD
    ct : dict
        CT with cube field (relative electron density)
    V_ct_grid : np.ndarray
        Linear indices (1-based) of valid CT voxels
    rot_coords_v : np.ndarray (N, 3)
        Rotated world coords of valid voxels (BEV coords - source)
    effective_lateral_cutoff : float
        Lateral cutoff in mm

    Returns
    -------
    list
        One radiological depth vector per CT scenario.
        Each vector has length = len(V_ct_grid) with NaN for out-of-range voxels.
    """
    dims = ct.get("cubeDim", ct.get("dimensions"))  # [Ny, Nx, Nz]
    res = ct["resolution"]
    num_ct_scen = ct.get("numOfCtScen", 1)

    # Get world axes
    from ..geometry import get_world_axes
    ct = get_world_axes(ct)

    # Create isocenter in cube coordinates
    isocenter_world = stf_beam["isoCenter"]
    from ..geometry.geometry import world_to_cube_coords
    isocenter_cube = world_to_cube_coords(
        np.atleast_2d(isocenter_world), ct
    )[0]

    source_point_bev = np.asarray(stf_beam["sourcePoint_bev"])

    # Get unique rays
    rays = stf_beam["ray"]

    # Initialize rad depth output: one per CT scenario
    rad_depth_v = [np.full(len(V_ct_grid), np.nan) for _ in range(num_ct_scen)]

    # For each voxel, find the closest ray and do ray tracing
    # Get all ray positions in BEV
    ray_pos_bev = np.array([r["rayPos_bev"] for r in rays])  # (nRays, 3)

    # Get rotated coords as BEV positions from source (shifted by source offset)
    # rot_coords_v already has source subtracted in initBeam
    # These are the voxel positions in BEV relative to source

    # For each voxel, find the closest ray by lateral distance
    # Lateral distance is in x-z plane of BEV (perpendicular to beam axis y)
    n_voxels = len(V_ct_grid)

    if n_voxels == 0:
        return rad_depth_v

    # Voxel BEV coords (including source point offset)
    # rot_coords_v = voxel_world - isoCenter, rotated to BEV, then subtract sourcePoint_bev
    # So: voxel in BEV relative to source = rot_coords_v
    # The actual BEV position is rot_coords_v (relative to source)
    # Ray pos_bev is relative to isocenter, so relative to source = rayPos_bev - sourcePoint_bev = rayPos_bev - [0,-SAD,0]

    vox_bev = rot_coords_v  # (N, 3) relative to source

    # Project each voxel onto ray positions at isocenter plane
    # Rays are at y=0 in BEV (isocenter), source at y=-SAD
    # For a voxel at BEV position (vbx, vby, vbz) relative to source,
    # its projection at isocenter plane (y=0) is:
    # Using similar triangles: proj = vbev * SAD / (SAD + vbev_y) where vbev_y from isocenter
    # But rot_coords_v = voxel_world_coords - isoCenter rotated to BEV, then - sourcePoint_bev
    # sourcePoint_bev = [0, -SAD, 0]
    # So vox_bev[:,1] = (voxel_y_in_bev_from_iso) - (-SAD) = voxel_y_in_bev_from_iso + SAD
    # Projection at y_bev=0 (isocenter):
    # For a ray from source (0,-SAD,0) through bev position (rx,0,rz):
    # Parametric: p = source + t*(target - source)
    # At y=0: t = SAD / (SAD + vox_bev_from_iso_y)
    # vox_bev_from_iso = rot_coords_v + sourcePoint_bev (re-add source)
    # = rot_coords_v + [0, -SAD, 0]

    vox_bev_from_iso = vox_bev + source_point_bev  # (N, 3) relative to iso

    SAD = stf_beam["SAD"]

    # Project to isocenter plane
    proj_y = vox_bev_from_iso[:, 1]  # depth in beam direction from iso
    # Avoid division by zero
    denom = SAD + proj_y
    safe_denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    proj_x = vox_bev_from_iso[:, 0] * SAD / safe_denom
    proj_z = vox_bev_from_iso[:, 2] * SAD / safe_denom

    # Find nearest ray for each voxel
    ray_x = ray_pos_bev[:, 0]
    ray_z = ray_pos_bev[:, 2]

    # Use KD-tree to find nearest ray (avoids large N×M distance matrix)
    from scipy.spatial import cKDTree
    ray_pos_2d = np.column_stack([ray_x, ray_z])  # (M, 2)
    tree = cKDTree(ray_pos_2d)
    proj_2d = np.column_stack([proj_x, proj_z])    # (N, 2)
    nearest_dist, nearest_ray = tree.query(proj_2d, k=1, workers=-1)

    # Only trace voxels within lateral cutoff
    valid_mask = nearest_dist <= effective_lateral_cutoff

    # Group voxels by ray to minimize ray tracing calls
    for ray_idx in range(len(rays)):
        ray = rays[ray_idx]
        ray_voxels = np.where(valid_mask & (nearest_ray == ray_idx))[0]

        if len(ray_voxels) == 0:
            continue

        target_point_bev = np.asarray(ray["targetPoint_bev"])

        for scen_idx in range(num_ct_scen):
            cube = ct["cube"][scen_idx]  # [Ny, Nx, Nz] relative electron density

            # For each voxel in this ray group, compute rad depth
            # We use the actual voxel positions
            vox_lin_ix = V_ct_grid[ray_voxels] - 1  # 0-based

            # Get voxel BEV positions
            vox_bev_local = vox_bev[ray_voxels]  # relative to source

            # Compute rad depth for each voxel using siddon
            for vi, vox_idx in enumerate(ray_voxels):
                target_bev = vox_bev[vox_idx] + source_point_bev  # in BEV from iso
                # Actually target is the voxel itself; source is beam source
                # source_bev = [0, -SAD, 0]
                # target_bev = vox_bev[vox_idx] + [0, -SAD, 0]

                _, l_seg, rho_seg, d12, _ = siddon_ray_tracer(
                    isocenter_cube,
                    res,
                    source_point_bev,  # source relative to iso
                    vox_bev[vox_idx] + source_point_bev,  # voxel relative to iso
                    [cube],
                )

                if len(l_seg) > 0 and len(rho_seg[0]) > 0:
                    rad_depth_v[scen_idx][vox_idx] = float(np.sum(rho_seg[0] * l_seg))

    return rad_depth_v


def ray_tracing_fast(
    stf_beam: dict,
    ct: dict,
    V_ct_grid: np.ndarray,
    rot_coords_v: np.ndarray,
    effective_lateral_cutoff: float,
) -> List[np.ndarray]:
    """
    Fast radiological depth calculation using vectorized ray tracing.

    Uses a single ray per beam position to compute the full depth profile,
    then assigns depth values to voxels based on their proximity to each ray.
    This is significantly faster than ray_tracing() for dense grids.

    Parameters
    ----------
    (same as ray_tracing)
    """
    dims = ct.get("cubeDim", ct.get("dimensions"))
    res = ct["resolution"]
    num_ct_scen = ct.get("numOfCtScen", 1)

    from ..geometry import get_world_axes
    ct = get_world_axes(ct)

    isocenter_world = stf_beam["isoCenter"]
    from ..geometry.geometry import world_to_cube_coords
    isocenter_cube = world_to_cube_coords(np.atleast_2d(isocenter_world), ct)[0]

    source_point_bev = np.asarray(stf_beam["sourcePoint_bev"])
    rays = stf_beam["ray"]
    n_voxels = len(V_ct_grid)

    rad_depth_v = [np.full(n_voxels, np.nan) for _ in range(num_ct_scen)]

    if n_voxels == 0:
        return rad_depth_v

    SAD = stf_beam["SAD"]
    vox_bev_from_iso = rot_coords_v + source_point_bev

    # Project voxels to isocenter plane
    proj_y = vox_bev_from_iso[:, 1]
    denom = SAD + proj_y
    safe_denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    proj_x = vox_bev_from_iso[:, 0] * SAD / safe_denom
    proj_z = vox_bev_from_iso[:, 2] * SAD / safe_denom

    ray_pos_bev = np.array([r["rayPos_bev"] for r in rays])
    ray_x = ray_pos_bev[:, 0]
    ray_z = ray_pos_bev[:, 2]

    # Use KD-tree to find nearest ray (avoids large N×M distance matrix)
    from scipy.spatial import cKDTree
    ray_pos_2d = np.column_stack([ray_x, ray_z])  # (M, 2)
    tree = cKDTree(ray_pos_2d)
    proj_2d = np.column_stack([proj_x, proj_z])    # (N, 2)
    nearest_dist, nearest_ray = tree.query(proj_2d, k=1, workers=-1)
    valid_mask = nearest_dist <= effective_lateral_cutoff

    # For each ray, do one full ray trace and assign depths to voxels
    for ray_idx, ray in enumerate(rays):
        in_ray = valid_mask & (nearest_ray == ray_idx)
        ray_voxel_indices = np.where(in_ray)[0]
        if len(ray_voxel_indices) == 0:
            continue

        # Target point for this ray (at far side of patient)
        target_bev = np.asarray(ray["targetPoint_bev"])

        for scen_idx in range(num_ct_scen):
            cube = ct["cube"][scen_idx]

            alphas, l_seg, rho_seg, d12, ix = siddon_ray_tracer(
                isocenter_cube, res, source_point_bev, target_bev, [cube]
            )

            if len(alphas) < 2:
                continue

            # For each voxel in this ray group, its rad depth is the
            # cumulative sum along the ray up to its depth
            # The voxel's BEV depth is vox_bev_from_iso[:,1] (from iso center, + = distal)
            # The ray goes from source (alpha=0) to target (alpha=1)
            # Alpha for a voxel at depth d_y from iso:
            # d_y = alpha * (target_y - source_y) + source_y
            # alpha = (d_y - source_y) / (target_y - source_y)
            src_y = source_point_bev[1]
            tgt_y = target_bev[1]
            dt_y = tgt_y - src_y

            if abs(dt_y) < 1e-6:
                continue

            # Vectorized: compute rad depth for all voxels in this ray group at once
            alphas_start = alphas[:-1]  # start alpha of each segment (n_seg,)
            vox_depths_y = vox_bev_from_iso[ray_voxel_indices, 1]  # (n_vox,)
            alpha_vox_arr = (vox_depths_y - src_y) / dt_y           # (n_vox,)

            # Cumulative radiological depth along ray
            cum_rho_l_ray = np.cumsum(rho_seg[0] * l_seg)  # (n_seg,)
            n_seg = len(cum_rho_l_ray)

            # For each voxel, find the last segment where alpha_start <= alpha_vox
            idx_arr = np.searchsorted(alphas_start, alpha_vox_arr, side='right') - 1
            idx_arr_clipped = np.clip(idx_arr, 0, n_seg - 1)
            depths = np.where(idx_arr < 0, 0.0, cum_rho_l_ray[idx_arr_clipped])
            rad_depth_v[scen_idx][ray_voxel_indices] = depths

    return rad_depth_v
