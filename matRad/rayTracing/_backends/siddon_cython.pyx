# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython backend for Siddon ray tracer.

Mirrors the algorithm in siddon.py but uses typed variables to eliminate
Python object overhead in the inner loops.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, fabs, ceil as cceil, floor as cfloor, round as cround

cnp.import_array()


cdef inline double _round1k(double v) nogil:
    """Round to 3 decimal places (matches MATLAB rounding trick)."""
    return cround(v * 1000.0) / 1000.0


def siddon_ray_tracer(
    cnp.ndarray[double, ndim=1] isocenter_cube,
    dict resolution,
    cnp.ndarray[double, ndim=1] source_point,
    cnp.ndarray[double, ndim=1] target_point,
    list cubes,
):
    """
    Cython-accelerated Siddon ray tracer.  Drop-in replacement for the
    pure-Python version in siddon.py.
    """
    cdef double iso_x = isocenter_cube[0]
    cdef double iso_y = isocenter_cube[1]
    cdef double iso_z = isocenter_cube[2]

    cdef double rx = resolution["x"]
    cdef double ry = resolution["y"]
    cdef double rz = resolution["z"]

    cdef double spx = source_point[0] + iso_x
    cdef double spy = source_point[1] + iso_y
    cdef double spz = source_point[2] + iso_z

    cdef double tpx = target_point[0] + iso_x
    cdef double tpy = target_point[1] + iso_y
    cdef double tpz = target_point[2] + iso_z

    cdef cnp.ndarray cube_arr = np.asfortranarray(cubes[0], dtype=np.float64)
    cdef int Ny = cube_arr.shape[0]
    cdef int Nx = cube_arr.shape[1]
    cdef int Nz = cube_arr.shape[2]

    cdef double dx = tpx - spx
    cdef double dy = tpy - spy
    cdef double dz = tpz - spz
    cdef double d12 = sqrt(dx*dx + dy*dy + dz*dz)

    empty_f = np.array([], dtype=np.float64)
    empty_i = np.array([], dtype=np.int64)
    empty_rho = [np.array([], dtype=np.float64) for _ in cubes]

    if d12 < 1e-12:
        return empty_f, empty_f, empty_rho, 0.0, empty_i

    # Plane boundaries
    cdef double x_p1 = 0.5 * rx, y_p1 = 0.5 * ry, z_p1 = 0.5 * rz
    cdef double x_pe = (Nx + 0.5) * rx
    cdef double y_pe = (Ny + 0.5) * ry
    cdef double z_pe = (Nz + 0.5) * rz

    # alpha_min / alpha_max  (eq 4 + 5)
    cdef double alpha_min = 0.0, alpha_max = 1.0
    cdef double a1, ae, tmp

    if fabs(dx) > 1e-12:
        a1 = (x_p1 - spx) / dx
        ae = (x_pe - spx) / dx
        if a1 > ae:
            tmp = a1; a1 = ae; ae = tmp
        if a1 > alpha_min: alpha_min = a1
        if ae < alpha_max: alpha_max = ae

    if fabs(dy) > 1e-12:
        a1 = (y_p1 - spy) / dy
        ae = (y_pe - spy) / dy
        if a1 > ae:
            tmp = a1; a1 = ae; ae = tmp
        if a1 > alpha_min: alpha_min = a1
        if ae < alpha_max: alpha_max = ae

    if fabs(dz) > 1e-12:
        a1 = (z_p1 - spz) / dz
        ae = (z_pe - spz) / dz
        if a1 > ae:
            tmp = a1; a1 = ae; ae = tmp
        if a1 > alpha_min: alpha_min = a1
        if ae < alpha_max: alpha_max = ae

    if alpha_min >= alpha_max:
        return empty_f, empty_f, empty_rho, d12, empty_i

    # Index ranges (eq 6)
    cdef int i_min_x = 0, i_max_x = -1
    cdef int i_min_y = 0, i_max_y = -1
    cdef int i_min_z = 0, i_max_z = -1

    cdef double span

    if fabs(dx) > 1e-12:
        span = x_pe - x_p1
        if dx > 0:
            i_min_x = <int>cceil(_round1k(  (Nx + 1) - (x_pe - alpha_min * dx - spx) / span * Nx  ))
            i_max_x = <int>cfloor(_round1k( 1 + (spx + alpha_max * dx - x_p1) / span * Nx  ))
        else:
            i_min_x = <int>cceil(_round1k(  (Nx + 1) - (x_pe - alpha_max * dx - spx) / span * Nx  ))
            i_max_x = <int>cfloor(_round1k( 1 + (spx + alpha_min * dx - x_p1) / span * Nx  ))

    if fabs(dy) > 1e-12:
        span = y_pe - y_p1
        if dy > 0:
            i_min_y = <int>cceil(_round1k(  (Ny + 1) - (y_pe - alpha_min * dy - spy) / span * Ny  ))
            i_max_y = <int>cfloor(_round1k( 1 + (spy + alpha_max * dy - y_p1) / span * Ny  ))
        else:
            i_min_y = <int>cceil(_round1k(  (Ny + 1) - (y_pe - alpha_max * dy - spy) / span * Ny  ))
            i_max_y = <int>cfloor(_round1k( 1 + (spy + alpha_min * dy - y_p1) / span * Ny  ))

    if fabs(dz) > 1e-12:
        span = z_pe - z_p1
        if dz > 0:
            i_min_z = <int>cceil(_round1k(  (Nz + 1) - (z_pe - alpha_min * dz - spz) / span * Nz  ))
            i_max_z = <int>cfloor(_round1k( 1 + (spz + alpha_max * dz - z_p1) / span * Nz  ))
        else:
            i_min_z = <int>cceil(_round1k(  (Nz + 1) - (z_pe - alpha_max * dz - spz) / span * Nz  ))
            i_max_z = <int>cfloor(_round1k( 1 + (spz + alpha_min * dz - z_p1) / span * Nz  ))

    # Build alpha list (eq 7)
    cdef list ax_list = [alpha_min, alpha_max]
    cdef int ii

    if fabs(dx) > 1e-12 and i_min_x <= i_max_x:
        for ii in range(i_min_x, i_max_x + 1):
            ax_list.append((rx * ii - spx - 0.5 * rx) / dx)

    if fabs(dy) > 1e-12 and i_min_y <= i_max_y:
        for ii in range(i_min_y, i_max_y + 1):
            ax_list.append((ry * ii - spy - 0.5 * ry) / dy)

    if fabs(dz) > 1e-12 and i_min_z <= i_max_z:
        for ii in range(i_min_z, i_max_z + 1):
            ax_list.append((rz * ii - spz - 0.5 * rz) / dz)

    alphas = np.unique(np.array(ax_list, dtype=np.float64))
    alphas = alphas[(alphas >= alpha_min - 1e-10) & (alphas <= alpha_max + 1e-10)]

    if len(alphas) < 2:
        return empty_f, empty_f, empty_rho, d12, empty_i

    # Segment lengths (eq 10)
    l = d12 * np.diff(alphas)

    # Midpoints (eq 13)
    cdef cnp.ndarray[double, ndim=1] amids = 0.5 * (alphas[:-1] + alphas[1:])
    cdef int n_seg = len(amids)

    # Voxel coordinates at midpoints
    cdef cnp.ndarray[double, ndim=1] i_mm = spx + amids * dx
    cdef cnp.ndarray[double, ndim=1] j_mm = spy + amids * dy
    cdef cnp.ndarray[double, ndim=1] k_mm = spz + amids * dz

    # Convert to 1-based indices, clipped to valid range
    i_idx = np.clip(np.round(i_mm / rx).astype(np.int64), 1, Nx)
    j_idx = np.clip(np.round(j_mm / ry).astype(np.int64), 1, Ny)
    k_idx = np.clip(np.round(k_mm / rz).astype(np.int64), 1, Nz)

    # MATLAB-style 1-based Fortran-order linear index
    ix = j_idx + (i_idx - 1) * Ny + (k_idx - 1) * Ny * Nx

    # Extract densities
    ix0 = ix - 1
    rho_list = []
    for cube in cubes:
        flat = np.asfortranarray(cube, dtype=np.float64).ravel(order='F')
        rho_list.append(flat[ix0])

    return alphas, l, rho_list, d12, ix
