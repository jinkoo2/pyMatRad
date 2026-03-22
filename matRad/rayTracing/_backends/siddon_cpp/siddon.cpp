/**
 * siddon.cpp — C++ implementation of the Siddon ray tracer.
 *
 * Algorithm mirrors siddon.py exactly so results are numerically identical.
 * Exposed to Python via pybind11 bindings in bindings.cpp.
 */

#include "siddon.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace siddon {

/* ------------------------------------------------------------------ helpers */

static inline double round1k(double v) {
    return std::round(v * 1000.0) / 1000.0;
}

/* ------------------------------------------------------------------ public  */

SiddonResult trace(
    double iso_x, double iso_y, double iso_z,
    double rx,    double ry,    double rz,
    double sp_x,  double sp_y,  double sp_z,
    double tp_x,  double tp_y,  double tp_z,
    const double *cube, int Ny, int Nx, int Nz
) {
    SiddonResult res;

    /* Shift by isocenter */
    double spx = sp_x + iso_x,  spy = sp_y + iso_y,  spz = sp_z + iso_z;
    double tpx = tp_x + iso_x,  tpy = tp_y + iso_y,  tpz = tp_z + iso_z;

    double dx = tpx - spx,  dy = tpy - spy,  dz = tpz - spz;
    res.d12 = std::sqrt(dx*dx + dy*dy + dz*dz);

    if (res.d12 < 1e-12) return res;

    /* Plane boundaries */
    double x_p1 = 0.5*rx,       y_p1 = 0.5*ry,       z_p1 = 0.5*rz;
    double x_pe = (Nx + 0.5)*rx, y_pe = (Ny + 0.5)*ry, z_pe = (Nz + 0.5)*rz;

    /* alpha_min / alpha_max */
    double alpha_min = 0.0, alpha_max = 1.0;

    auto clamp_range = [&](double s, double t_, double p1, double pe) {
        if (std::abs(t_ - s) < 1e-12) return;
        double a1 = (p1 - s) / (t_ - s);
        double ae = (pe - s) / (t_ - s);
        if (a1 > ae) std::swap(a1, ae);
        if (a1 > alpha_min) alpha_min = a1;
        if (ae < alpha_max) alpha_max = ae;
    };
    clamp_range(spx, tpx, x_p1, x_pe);
    clamp_range(spy, tpy, y_p1, y_pe);
    clamp_range(spz, tpz, z_p1, z_pe);

    if (alpha_min >= alpha_max) return res;

    /* Index ranges (eq 6) */
    auto idx_range = [&](int n_vox, double s, double t_, double p1, double pe,
                         int &imin, int &imax) {
        imin = 0; imax = -1;
        if (std::abs(t_ - s) < 1e-12) return;
        double dt = t_ - s;
        double span = pe - p1;
        if (dt > 0) {
            imin = (int)std::ceil( round1k( (n_vox+1) - (pe - alpha_min*dt - s)/span*n_vox ));
            imax = (int)std::floor(round1k( 1         + (s + alpha_max*dt - p1)/span*n_vox ));
        } else {
            imin = (int)std::ceil( round1k( (n_vox+1) - (pe - alpha_max*dt - s)/span*n_vox ));
            imax = (int)std::floor(round1k( 1         + (s + alpha_min*dt - p1)/span*n_vox ));
        }
    };

    int i_min_x, i_max_x, i_min_y, i_max_y, i_min_z, i_max_z;
    idx_range(Nx, spx, tpx, x_p1, x_pe, i_min_x, i_max_x);
    idx_range(Ny, spy, tpy, y_p1, y_pe, i_min_y, i_max_y);
    idx_range(Nz, spz, tpz, z_p1, z_pe, i_min_z, i_max_z);

    /* Collect alphas */
    std::vector<double> avec;
    avec.reserve(4 + (i_max_x - i_min_x + 1) + (i_max_y - i_min_y + 1) + (i_max_z - i_min_z + 1));
    avec.push_back(alpha_min);
    avec.push_back(alpha_max);

    if (std::abs(dx) > 1e-12)
        for (int i = i_min_x; i <= i_max_x; i++)
            avec.push_back((rx * i - spx - 0.5*rx) / dx);
    if (std::abs(dy) > 1e-12)
        for (int i = i_min_y; i <= i_max_y; i++)
            avec.push_back((ry * i - spy - 0.5*ry) / dy);
    if (std::abs(dz) > 1e-12)
        for (int i = i_min_z; i <= i_max_z; i++)
            avec.push_back((rz * i - spz - 0.5*rz) / dz);

    std::sort(avec.begin(), avec.end());
    avec.erase(std::unique(avec.begin(), avec.end(),
                           [](double a, double b){ return b - a < 1e-14; }),
               avec.end());

    /* Filter */
    {
        std::vector<double> tmp;
        tmp.reserve(avec.size());
        for (double a : avec)
            if (a >= alpha_min - 1e-10 && a <= alpha_max + 1e-10)
                tmp.push_back(a);
        avec = std::move(tmp);
    }

    int na = (int)avec.size();
    if (na < 2) return res;

    int n_seg = na - 1;
    res.alphas.resize(na);
    res.l.resize(n_seg);
    res.rho.resize(n_seg);
    res.ix.resize(n_seg);

    for (int s = 0; s < n_seg; s++) {
        res.alphas[s] = avec[s];
        double a_lo = avec[s], a_hi = avec[s + 1];
        res.l[s] = res.d12 * (a_hi - a_lo);

        double amid = 0.5*(a_lo + a_hi);
        double xm = spx + amid * dx;
        double ym = spy + amid * dy;
        double zm = spz + amid * dz;

        long long xi = (long long)std::round(xm / rx);
        long long yi = (long long)std::round(ym / ry);
        long long zi = (long long)std::round(zm / rz);

        if (xi < 1) xi = 1; if (xi > Nx) xi = Nx;
        if (yi < 1) yi = 1; if (yi > Ny) yi = Ny;
        if (zi < 1) zi = 1; if (zi > Nz) zi = Nz;

        long long lin_ix = yi + (xi - 1)*Ny + (zi - 1)*(long long)Ny*Nx;
        res.ix[s]  = lin_ix;
        res.rho[s] = cube[lin_ix - 1];
    }
    res.alphas[n_seg] = avec[n_seg];

    return res;
}

} // namespace siddon
