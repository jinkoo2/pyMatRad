/**
 * siddon.hpp — C++ Siddon ray tracer interface.
 */

#pragma once
#include <vector>

namespace siddon {

struct SiddonResult {
    std::vector<double>    alphas;  ///< parametric values (n_seg+1)
    std::vector<double>    l;       ///< segment lengths   (n_seg)
    std::vector<double>    rho;     ///< densities         (n_seg)
    std::vector<long long> ix;      ///< 1-based MATLAB linear indices (n_seg)
    double                 d12 = 0.0;
};

/**
 * Trace one ray through a Fortran-order CT cube.
 *
 * @param iso_x/y/z   isocenter in cube coordinates (mm)
 * @param rx/ry/rz    voxel resolution (mm)
 * @param sp_*        source relative to isocenter (mm)
 * @param tp_*        target relative to isocenter (mm)
 * @param cube        flat Fortran-order array of length Ny*Nx*Nz
 * @param Ny,Nx,Nz    cube dimensions
 */
SiddonResult trace(
    double iso_x, double iso_y, double iso_z,
    double rx,    double ry,    double rz,
    double sp_x,  double sp_y,  double sp_z,
    double tp_x,  double tp_y,  double tp_z,
    const double *cube, int Ny, int Nx, int Nz
);

} // namespace siddon
