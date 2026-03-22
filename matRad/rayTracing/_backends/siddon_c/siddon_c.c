/**
 * siddon_c.c — Plain C99 Siddon ray tracer
 *
 * Algorithm reference: Siddon 1985 Medical Physics (PMID: 4000088)
 * Closely mirrors siddon.py so results are numerically identical.
 */

#include "siddon_c.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ helpers */

static inline double _round1k(double v) {
    return round(v * 1000.0) / 1000.0;
}

static inline double _clamp(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* qsort comparator for doubles */
static int _cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return  1;
    return 0;
}

/* Remove duplicate doubles (assumes sorted array). Returns new length. */
static int _unique_sorted(double *arr, int n) {
    if (n <= 1) return n;
    int w = 1;
    for (int i = 1; i < n; i++) {
        if (arr[i] - arr[w - 1] > 1e-14) {
            arr[w++] = arr[i];
        }
    }
    return w;
}

/* ------------------------------------------------------------------ main API */

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
) {
    /* Shift by isocenter */
    double spx = sp_x + iso_x;
    double spy = sp_y + iso_y;
    double spz = sp_z + iso_z;
    double tpx = tp_x + iso_x;
    double tpy = tp_y + iso_y;
    double tpz = tp_z + iso_z;

    double dx = tpx - spx;
    double dy = tpy - spy;
    double dz = tpz - spz;
    double d12 = sqrt(dx*dx + dy*dy + dz*dz);
    *d12_out = d12;

    if (d12 < 1e-12) return 0;

    /* Plane boundaries (eq 3) */
    double x_p1 = 0.5 * rx,  y_p1 = 0.5 * ry,  z_p1 = 0.5 * rz;
    double x_pe = (Nx + 0.5) * rx;
    double y_pe = (Ny + 0.5) * ry;
    double z_pe = (Nz + 0.5) * rz;

    /* alpha_min / alpha_max  (eq 4 + 5) */
    double alpha_min = 0.0, alpha_max = 1.0;
    double a1, ae, tmp;

    if (fabs(dx) > 1e-12) {
        a1 = (x_p1 - spx) / dx;
        ae = (x_pe - spx) / dx;
        if (a1 > ae) { tmp = a1; a1 = ae; ae = tmp; }
        if (a1 > alpha_min) alpha_min = a1;
        if (ae < alpha_max) alpha_max = ae;
    }
    if (fabs(dy) > 1e-12) {
        a1 = (y_p1 - spy) / dy;
        ae = (y_pe - spy) / dy;
        if (a1 > ae) { tmp = a1; a1 = ae; ae = tmp; }
        if (a1 > alpha_min) alpha_min = a1;
        if (ae < alpha_max) alpha_max = ae;
    }
    if (fabs(dz) > 1e-12) {
        a1 = (z_p1 - spz) / dz;
        ae = (z_pe - spz) / dz;
        if (a1 > ae) { tmp = a1; a1 = ae; ae = tmp; }
        if (a1 > alpha_min) alpha_min = a1;
        if (ae < alpha_max) alpha_max = ae;
    }

    if (alpha_min >= alpha_max) return 0;

    /* Index ranges (eq 6) */
    int i_min_x = 0, i_max_x = -1;
    int i_min_y = 0, i_max_y = -1;
    int i_min_z = 0, i_max_z = -1;

    if (fabs(dx) > 1e-12) {
        if (dx > 0) {
            i_min_x = (int)ceil(_round1k(  (Nx+1) - (x_pe - alpha_min*dx - spx)/(x_pe-x_p1)*Nx  ));
            i_max_x = (int)floor(_round1k( 1       + (spx + alpha_max*dx - x_p1)/(x_pe-x_p1)*Nx  ));
        } else {
            i_min_x = (int)ceil(_round1k(  (Nx+1) - (x_pe - alpha_max*dx - spx)/(x_pe-x_p1)*Nx  ));
            i_max_x = (int)floor(_round1k( 1       + (spx + alpha_min*dx - x_p1)/(x_pe-x_p1)*Nx  ));
        }
    }
    if (fabs(dy) > 1e-12) {
        if (dy > 0) {
            i_min_y = (int)ceil(_round1k(  (Ny+1) - (y_pe - alpha_min*dy - spy)/(y_pe-y_p1)*Ny  ));
            i_max_y = (int)floor(_round1k( 1       + (spy + alpha_max*dy - y_p1)/(y_pe-y_p1)*Ny  ));
        } else {
            i_min_y = (int)ceil(_round1k(  (Ny+1) - (y_pe - alpha_max*dy - spy)/(y_pe-y_p1)*Ny  ));
            i_max_y = (int)floor(_round1k( 1       + (spy + alpha_min*dy - y_p1)/(y_pe-y_p1)*Ny  ));
        }
    }
    if (fabs(dz) > 1e-12) {
        if (dz > 0) {
            i_min_z = (int)ceil(_round1k(  (Nz+1) - (z_pe - alpha_min*dz - spz)/(z_pe-z_p1)*Nz  ));
            i_max_z = (int)floor(_round1k( 1       + (spz + alpha_max*dz - z_p1)/(z_pe-z_p1)*Nz  ));
        } else {
            i_min_z = (int)ceil(_round1k(  (Nz+1) - (z_pe - alpha_max*dz - spz)/(z_pe-z_p1)*Nz  ));
            i_max_z = (int)floor(_round1k( 1       + (spz + alpha_min*dz - z_p1)/(z_pe-z_p1)*Nz  ));
        }
    }

    /* Collect all alpha values into a temporary buffer (eq 7).
       Upper bound on count: 2 + (i_max_x-i_min_x+1) + ... */
    int n_x = (i_max_x >= i_min_x && fabs(dx) > 1e-12) ? (i_max_x - i_min_x + 1) : 0;
    int n_y = (i_max_y >= i_min_y && fabs(dy) > 1e-12) ? (i_max_y - i_min_y + 1) : 0;
    int n_z = (i_max_z >= i_min_z && fabs(dz) > 1e-12) ? (i_max_z - i_min_z + 1) : 0;
    int cap  = 2 + n_x + n_y + n_z + 4;  /* +4 margin */

    double *abuf = (double *)malloc((size_t)cap * sizeof(double));
    if (!abuf) return -1;

    int na = 0;
    abuf[na++] = alpha_min;
    abuf[na++] = alpha_max;

    int ii;
    if (fabs(dx) > 1e-12) {
        for (ii = i_min_x; ii <= i_max_x; ii++)
            abuf[na++] = (rx * ii - spx - 0.5 * rx) / dx;
    }
    if (fabs(dy) > 1e-12) {
        for (ii = i_min_y; ii <= i_max_y; ii++)
            abuf[na++] = (ry * ii - spy - 0.5 * ry) / dy;
    }
    if (fabs(dz) > 1e-12) {
        for (ii = i_min_z; ii <= i_max_z; ii++)
            abuf[na++] = (rz * ii - spz - 0.5 * rz) / dz;
    }

    /* Sort & unique (eq 8) */
    qsort(abuf, na, sizeof(double), _cmp_double);
    na = _unique_sorted(abuf, na);

    /* Filter to [alpha_min-eps, alpha_max+eps] */
    int nfilt = 0;
    double *afilt = abuf;  /* reuse buffer in-place (ascending, so prefix is ok) */
    for (int i = 0; i < na; i++) {
        if (abuf[i] >= alpha_min - 1e-10 && abuf[i] <= alpha_max + 1e-10) {
            afilt[nfilt++] = abuf[i];
        }
    }
    na = nfilt;

    if (na < 2) {
        free(abuf);
        return 0;
    }

    int n_seg = na - 1;
    if (n_seg > max_buf) {
        free(abuf);
        return -1;  /* buffer too small */
    }

    /* Fill output arrays */
    for (int s = 0; s < n_seg; s++) {
        double a_lo = afilt[s], a_hi = afilt[s + 1];
        double seg_l = d12 * (a_hi - a_lo);

        double amid = 0.5 * (a_lo + a_hi);
        double xm = spx + amid * dx;
        double ym = spy + amid * dy;
        double zm = spz + amid * dz;

        /* 1-based voxel indices */
        long long xi = (long long)round(xm / rx);
        long long yi = (long long)round(ym / ry);
        long long zi = (long long)round(zm / rz);

        if (xi < 1) xi = 1; if (xi > Nx) xi = Nx;
        if (yi < 1) yi = 1; if (yi > Ny) yi = Ny;
        if (zi < 1) zi = 1; if (zi > Nz) zi = Nz;

        /* Fortran column-major: ix = yi + (xi-1)*Ny + (zi-1)*Ny*Nx */
        long long lin_ix = yi + (xi - 1) * Ny + (zi - 1) * (long long)Ny * Nx;

        alphas_out[s]   = a_lo;
        l_out[s]        = seg_l;
        ix_out[s]       = lin_ix;
        rho_out[s]      = cube[lin_ix - 1];  /* 0-based Fortran flat index */
    }
    alphas_out[n_seg] = afilt[na - 1];  /* final alpha */

    free(abuf);
    return n_seg;
}
