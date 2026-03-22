/**
 * siddon_c.h — Plain C99 Siddon ray tracer
 *
 * Port of matRad_siddonRayTracer.m for use via ctypes.
 *
 * Cube ordering: [Ny, Nx, Nz] Fortran-order (column-major), 1-based MATLAB
 * linear indices.
 */

#ifndef SIDDON_C_H
#define SIDDON_C_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * siddon_ray_tracer_c — trace one ray through a 3-D CT cube.
 *
 * Parameters
 * ----------
 * iso_x, iso_y, iso_z     : isocenter in cube coordinates (mm)
 * rx, ry, rz              : voxel resolution (mm)
 * sp_x,  sp_y,  sp_z      : source point relative to isocenter (mm)
 * tp_x,  tp_y,  tp_z      : target point relative to isocenter (mm)
 * cube                    : flat Fortran-order array of length Ny*Nx*Nz
 * Ny, Nx, Nz              : cube dimensions
 * alphas_out              : output buffer for alpha values (size max_buf+1)
 * l_out                   : output buffer for segment lengths (size max_buf)
 * rho_out                 : output buffer for densities   (size max_buf)
 * ix_out                  : output buffer for 1-based MATLAB lin. indices
 * d12_out                 : output: total source-target distance
 * max_buf                 : capacity of each output buffer
 *
 * Returns
 * -------
 * Number of segments written (0 if ray misses the cube, -1 if buffer too small).
 */
int siddon_ray_tracer_c(
    double iso_x, double iso_y, double iso_z,
    double rx,    double ry,    double rz,
    double sp_x,  double sp_y,  double sp_z,
    double tp_x,  double tp_y,  double tp_z,
    const double *cube, int Ny, int Nx, int Nz,
    double       *alphas_out,
    double       *l_out,
    double       *rho_out,
    long long    *ix_out,
    double       *d12_out,
    int           max_buf
);

#ifdef __cplusplus
}
#endif

#endif /* SIDDON_C_H */
