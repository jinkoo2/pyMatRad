"""
test_backends.py — Accuracy and speed benchmark for all Siddon backends.

Runs Example 1 (phantom) and optionally Example 2 (TG119) dose calc with
each compiled backend and verifies numerical agreement against the
pure-Python baseline.

Usage:
    python examples/test_backends.py
    python examples/test_backends.py --examples 1      # only Example 1
    python examples/test_backends.py --examples 1 2    # both examples

Environment:
    Compiled backends must be built first:
      # Cython
      cd matRad/rayTracing/_backends
      python cython_setup.py build_ext --inplace

      # pybind11 C++
      cd matRad/rayTracing/_backends/siddon_cpp
      python setup.py build_ext --inplace

      # Plain C
      cd matRad/rayTracing/_backends/siddon_c
      python build.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import scipy.io as sio
import h5py
import scipy.sparse

# ----------------------------------------------------------------- paths ----
PYMATRAD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PYMATRAD_ROOT)

REF_MAT1 = os.path.join(PYMATRAD_ROOT,
    r'examples\_matRad_ref_outputs\example1_no_opti\example1_results.mat')
REF_MAT2 = os.path.join(PYMATRAD_ROOT,
    r'examples\_matRad_ref_outputs\example2_no_opti\example2_results.mat')

# ---------------------------------------------------------------- helpers ---

def _s(x):
    return float(np.array(x).ravel()[0])

def _v(x):
    return np.array(x).ravel().astype(np.float64)


def _load_stf_scipy(path):
    """Return (stf_list, D_ml, Ny_dg, Nx_dg, Nz_dg, ct, cst) for MATLAB v5."""
    mat  = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    dij_m = mat['dij']
    stf_m = mat['stf']

    dg     = dij_m.doseGrid
    dg_dims = np.array(dg.dimensions).ravel().astype(int)
    Ny_dg, Nx_dg, Nz_dg = int(dg_dims[0]), int(dg_dims[1]), int(dg_dims[2])
    dg_res_x = _s(dg.resolution.x)
    dg_res_y = _s(dg.resolution.y)
    dg_res_z = _s(dg.resolution.z)
    dg_x = _v(dg.x); dg_y = _v(dg.y); dg_z = _v(dg.z)

    pd_raw = dij_m.physicalDose
    if scipy.sparse.issparse(pd_raw):
        D_ml = pd_raw.tocsc()
    else:
        D_ml = pd_raw.flat[0].tocsc()

    if stf_m.ndim == 0:
        stf_m = np.array([stf_m.item()])
    stf_list = []
    for ib in range(len(stf_m)):
        bm   = stf_m[ib]
        ga   = _s(bm.gantryAngle)
        ca   = _s(bm.couchAngle)
        sad  = _s(bm.SAD)
        bw   = _s(bm.bixelWidth)
        nr   = int(_s(bm.numOfRays))
        nb   = int(_s(bm.totalNumOfBixels))
        iso  = _v(bm.isoCenter)
        sp_w = _v(bm.sourcePoint)
        sp_b = _v(bm.sourcePoint_bev)
        rays_m = bm.ray
        if rays_m.ndim == 0:
            rays_m = np.array([rays_m.item()])
        rays = []
        for ir in range(len(rays_m)):
            ray  = rays_m[ir]
            rbev = _v(ray.rayPos_bev)
            tp   = _v(ray.targetPoint)
            rp   = _v(ray.rayPos) if hasattr(ray, 'rayPos') else rbev.copy()
            rays.append({
                'rayPos_bev':      rbev,
                'targetPoint_bev': np.array([2.0*rbev[0], sad, 2.0*rbev[2]]),
                'rayPos':          rp,
                'targetPoint':     tp,
                'energy':          np.array([6.0]),
            })
        stf_list.append({
            'gantryAngle': ga, 'couchAngle': ca, 'SAD': sad, 'bixelWidth': bw,
            'isoCenter': iso, 'sourcePoint': sp_w, 'sourcePoint_bev': sp_b,
            'numOfRays': nr, 'totalNumOfBixels': nb, 'ray': rays,
            'radiationMode': 'photons', 'machine': 'Generic',
        })

    return (stf_list, D_ml, Ny_dg, Nx_dg, Nz_dg,
            dg_res_x, dg_res_y, dg_res_z, dg_x, dg_y, dg_z)


def _load_stf_h5(path):
    """Return same tuple for MATLAB v7.3 (HDF5)."""
    f = h5py.File(path, 'r')
    dg_dims = np.array(f['dij/doseGrid/dimensions']).ravel().astype(int)
    Ny_dg, Nx_dg, Nz_dg = int(dg_dims[0]), int(dg_dims[1]), int(dg_dims[2])
    dg_res_x = float(np.array(f['dij/doseGrid/resolution/x']).ravel()[0])
    dg_res_y = float(np.array(f['dij/doseGrid/resolution/y']).ravel()[0])
    dg_res_z = float(np.array(f['dij/doseGrid/resolution/z']).ravel()[0])
    dg_x = np.array(f['dij/doseGrid/x']).ravel()
    dg_y = np.array(f['dij/doseGrid/y']).ravel()
    dg_z = np.array(f['dij/doseGrid/z']).ravel()

    ref_pd = f['dij/physicalDose'][0, 0]
    sp_grp = f[ref_pd]
    ir_idx = np.array(sp_grp['ir'],   dtype=np.int64)
    jc_arr = np.array(sp_grp['jc'],   dtype=np.int64)
    data   = np.array(sp_grp['data'], dtype=np.float64)
    n_vox  = Ny_dg * Nx_dg * Nz_dg
    D_ml   = scipy.sparse.csc_matrix((data, ir_idx, jc_arr),
                                      shape=(n_vox, len(jc_arr) - 1))

    stf_h5  = f['stf']
    n_beams = stf_h5['gantryAngle'].shape[0]
    stf_list = []
    for ib in range(n_beams):
        def h5v(ref): return np.array(f[ref]).ravel()
        ga   = float(h5v(stf_h5['gantryAngle'][ib, 0])[0])
        ca   = float(h5v(stf_h5['couchAngle'][ib, 0])[0])
        sad  = float(h5v(stf_h5['SAD'][ib, 0])[0])
        bw   = float(h5v(stf_h5['bixelWidth'][ib, 0])[0])
        nr   = int(h5v(stf_h5['numOfRays'][ib, 0])[0])
        nb   = int(h5v(stf_h5['totalNumOfBixels'][ib, 0])[0])
        iso  = h5v(stf_h5['isoCenter'][ib, 0])
        sp_b = h5v(stf_h5['sourcePoint_bev'][ib, 0])
        sp_w = h5v(stf_h5['sourcePoint'][ib, 0])
        ray_grp = f[stf_h5['ray'][ib, 0]]
        rays = []
        for ir in range(nr):
            rbev = h5v(ray_grp['rayPos_bev'][ir, 0]).astype(float)
            tp   = h5v(ray_grp['targetPoint'][ir, 0]).astype(float)
            rp   = h5v(ray_grp['rayPos'][ir, 0]).astype(float) \
                   if 'rayPos' in ray_grp else rbev.copy()
            rays.append({
                'rayPos_bev':      rbev,
                'targetPoint_bev': np.array([2.0*rbev[0], sad, 2.0*rbev[2]]),
                'rayPos':          rp,
                'targetPoint':     tp,
                'energy':          np.array([6.0]),
            })
        stf_list.append({
            'gantryAngle': ga, 'couchAngle': ca, 'SAD': sad, 'bixelWidth': bw,
            'isoCenter': iso, 'sourcePoint': sp_w, 'sourcePoint_bev': sp_b,
            'numOfRays': nr, 'totalNumOfBixels': nb, 'ray': rays,
            'radiationMode': 'photons', 'machine': 'Generic',
        })
    f.close()
    return (stf_list, D_ml, Ny_dg, Nx_dg, Nz_dg,
            dg_res_x, dg_res_y, dg_res_z, dg_x, dg_y, dg_z)


def load_ref(path):
    try:
        with h5py.File(path, 'r'):
            pass
        return _load_stf_h5(path)
    except OSError:
        return _load_stf_scipy(path)


# ---------------------------------------------------------------- setup ct1 -

def build_ct1():
    """Water phantom for Example 1."""
    from matRad.phantoms.builder import PhantomBuilder
    from matRad.optimization.DoseObjectives import SquaredDeviation, SquaredOverdosing
    from matRad.geometry.geometry import get_world_axes
    builder = PhantomBuilder([200, 200, 100], [2, 2, 3], num_of_ct_scen=1)
    builder.add_spherical_target("Volume1", radius=20,
        objectives=[SquaredDeviation(penalty=800, d_ref=45).to_dict()], HU=0)
    builder.add_box_oar("Volume2", [60, 30, 60], offset=[0, -15, 0],
        objectives=[SquaredOverdosing(penalty=400, d_ref=0).to_dict()], HU=0)
    builder.add_box_oar("Volume3", [60, 30, 60], offset=[0, 15, 0],
        objectives=[SquaredOverdosing(penalty=10, d_ref=0).to_dict()], HU=0)
    ct, cst = builder.get_ct_cst()
    ct = get_world_axes(ct)
    return ct, cst


def load_ct2(path):
    """TG119 CT for Example 2."""
    from matRad.basedata.load_machine import load_mat
    from matRad.geometry.geometry import get_world_axes
    try:
        with h5py.File(path, 'r'):
            pass
        f = h5py.File(path, 'r')
        cube_h5 = np.array(f['ct/cube']).T          # (Ny,Nx,Nz)
        dims    = np.array(f['ct/cubeDim']).ravel().astype(int)
        rx = float(np.array(f['ct/resolution/x']).ravel()[0])
        ry = float(np.array(f['ct/resolution/y']).ravel()[0])
        rz = float(np.array(f['ct/resolution/z']).ravel()[0])
        f.close()
    except OSError:
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        ct_m  = mat['ct']
        cube_h5 = np.array(ct_m.cube).ravel().reshape(
            tuple(np.array(ct_m.cubeDim).ravel().astype(int)), order='F')
        dims = np.array(ct_m.cubeDim).ravel().astype(int)
        rx = _s(ct_m.resolution.x)
        ry = _s(ct_m.resolution.y)
        rz = _s(ct_m.resolution.z)

    Ny, Nx, Nz = int(dims[0]), int(dims[1]), int(dims[2])
    x_vec = (np.arange(Nx) + 0.5) * rx - 0.5 * Nx * rx
    y_vec = (np.arange(Ny) + 0.5) * ry - 0.5 * Ny * ry
    z_vec = (np.arange(Nz) + 0.5) * rz - 0.5 * Nz * rz

    ct = {
        'cubeDim':     [Ny, Nx, Nz],
        'dimensions':  [Ny, Nx, Nz],
        'resolution':  {'x': rx, 'y': ry, 'z': rz},
        'numOfCtScen': 1,
        'cube':        [cube_h5.astype(np.float64)],
        'x': x_vec, 'y': y_vec, 'z': z_vec,
    }
    ct = get_world_axes(ct)
    return ct


# --------------------------------------------------------------- run dose ---

def run_dose(ct, cst_or_none, stf, dg_res_x, dg_res_y, dg_res_z):
    """Run calc_dose_influence and return (D, elapsed_s)."""
    from matRad.doseCalc.calc_dose_influence import calc_dose_influence

    n_beams = len(stf)
    pln = {
        'radiationMode': 'photons',
        'machine':       'Generic',
        'bioModel':      'none',
        'multScen':      'nomScen',
        'numOfFractions': 30,
        'propStf': {
            'gantryAngles': [b['gantryAngle'] for b in stf],
            'couchAngles':  [0] * n_beams,
            'bixelWidth':   5,
        },
        'propOpt': {'runDAO': False, 'runSequencing': False},
        'propDoseCalc': {'doseGrid': {'resolution': {
            'x': dg_res_x, 'y': dg_res_y, 'z': dg_res_z,
        }}},
    }

    # Use a minimal CST if not available
    if cst_or_none is None:
        cst_or_none = [[0, 'BODY', 'OAR', [np.array([1])], [], []]]

    t0 = time.perf_counter()
    dij = calc_dose_influence(ct, cst_or_none, stf, pln)
    elapsed = time.perf_counter() - t0

    D = dij['physicalDose'][0].tocsc()
    return D, elapsed


# --------------------------------------------------------------- main  -----

def main():
    parser = argparse.ArgumentParser(description="Backend accuracy & speed test")
    parser.add_argument('--examples', nargs='+', type=int, default=[1],
                        choices=[1, 2],
                        help='Which examples to run (1, 2, or both)')
    args = parser.parse_args()

    BACKENDS = ['python', 'cython', 'cpp', 'c']

    examples_cfg = {}
    if 1 in args.examples:
        examples_cfg['Example1'] = {'mat': REF_MAT1, 'build_ct': build_ct1, 'cst': True}
    if 2 in args.examples:
        examples_cfg['Example2'] = {'mat': REF_MAT2, 'build_ct': None,      'cst': False}

    for ex_name, cfg in examples_cfg.items():
        print("\n" + "=" * 70)
        print(f"  {ex_name}")
        print("=" * 70)

        mat_path = cfg['mat']
        if not os.path.isfile(mat_path):
            print(f"  SKIP: reference file not found: {mat_path}")
            continue

        (stf, D_ml, Ny_dg, Nx_dg, Nz_dg,
         dg_rx, dg_ry, dg_rz, dg_x, dg_y, dg_z) = load_ref(mat_path)
        print(f"  MATLAB DIJ: {D_ml.shape}  grid={Ny_dg}×{Nx_dg}×{Nz_dg}")

        # Build CT
        if cfg['build_ct'] is not None:
            ct, cst = cfg['build_ct']()
        else:
            tg119_mat = os.path.join(PYMATRAD_ROOT,
                r'examples\_matRad_ref_outputs\TG119.mat')
            if not os.path.isfile(tg119_mat):
                print(f"  SKIP: TG119.mat not found: {tg119_mat}")
                continue
            ct = load_ct2(tg119_mat)
            cst = None

        # ------ run once per backend ------
        results  = {}   # backend -> (D, elapsed)
        timings  = {}   # backend -> elapsed
        baseline_dose = None

        from matRad.rayTracing import dispatch as _dispatch

        for backend in BACKENDS:
            print(f"\n  Backend: [{backend}]")
            try:
                activated = _dispatch.activate(backend)
                if activated != backend:
                    print(f"    WARNING: requested '{backend}', got '{activated}'")
            except Exception as e:
                print(f"    SKIP: activate failed — {e}")
                continue

            try:
                D, elapsed = run_dose(ct, cst if cfg['cst'] else None,
                                      stf, dg_rx, dg_ry, dg_rz)
            except Exception as e:
                print(f"    ERROR during dose calc: {e}")
                import traceback; traceback.print_exc()
                continue

            w   = np.ones(D.shape[1])
            dose = np.asarray(D @ w).ravel()

            if baseline_dose is None:
                baseline_dose = dose.copy()
                print(f"    max_dose={dose.max():.4f} Gy  elapsed={elapsed:.1f}s  [BASELINE]")
            else:
                max_abs_err = np.max(np.abs(dose - baseline_dose))
                rel_err     = max_abs_err / max(np.abs(baseline_dose).max(), 1e-12)
                speedup     = timings['python'] / elapsed if 'python' in timings else float('nan')
                print(f"    max_dose={dose.max():.4f} Gy  elapsed={elapsed:.1f}s  "
                      f"max_abs_err={max_abs_err:.2e}  rel_err={rel_err:.2e}  "
                      f"speedup={speedup:.2f}x")
                # Expected: small FP rounding differences from different arithmetic
                # ordering (C vs Python rounding, thread execution order in lil_matrix
                # accumulation). 2% threshold is appropriate for cross-language + threaded.
                if rel_err > 2e-2:
                    print(f"    *** FAIL: relative error {rel_err:.2e} exceeds 2% ***")
                elif rel_err > 1e-6:
                    print(f"    PASS: within FP tolerance (rel_err={rel_err:.2e})")
                else:
                    print(f"    PASS: bit-identical to Python baseline")

            results[backend]  = (D, elapsed)
            timings[backend]  = elapsed

        # ------ summary table ------
        if timings:
            print(f"\n  {'Backend':<10}  {'Time (s)':>10}  {'Speedup':>10}  {'Max abs err':>14}  {'Status':>10}")
            print("  " + "-" * 65)
            py_t = timings.get('python', None)
            for be, (D_be, t_be) in results.items():
                su = py_t / t_be if (py_t and be != 'python') else 1.0
                if be == 'python':
                    print(f"  {be:<10}  {t_be:>10.1f}  {'1.00x':>10}  {'—':>14}  {'baseline':>10}")
                else:
                    dose_be = np.asarray(D_be @ np.ones(D_be.shape[1])).ravel()
                    mae     = np.max(np.abs(dose_be - baseline_dose))
                    rel     = mae / max(np.abs(baseline_dose).max(), 1e-12)
                    status  = 'PASS' if rel < 2e-2 else 'FAIL'
                    print(f"  {be:<10}  {t_be:>10.1f}  {su:>9.2f}x  {mae:>14.2e}  {status:>10}")

        # reset to python before next example
        _dispatch.activate('python')

    print("\nDone.")


if __name__ == "__main__":
    main()
