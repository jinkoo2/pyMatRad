"""
Compare pyMatRad dose output against MATLAB photons_testData.mat reference.

Reconstructs CT/CST/STF directly from the MATLAB test data and runs pyMatRad's
dose engine, then compares DIJ structure and dose distributions.
"""

import os
import sys
import numpy as np
import scipy.io as sio
import scipy.sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matRad.doseCalc.calc_dose_influence import calc_dose_influence

# ---------------------------------------------------------------------------
# 1. Load MATLAB reference
# ---------------------------------------------------------------------------
MATRAD_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "..", "matRad")
TEST_DATA = os.path.join(MATRAD_ROOT, "test", "testData", "photons_testData.mat")

print("Loading MATLAB reference:", os.path.normpath(TEST_DATA))
raw = sio.loadmat(TEST_DATA, squeeze_me=True, struct_as_record=False)

ref_result = raw["resultGUI"]
ref_dose   = ref_result.physicalDose          # (20, 10, 10)
ref_w      = ref_result.w                     # (10,) all ones
ref_dij_obj = raw["dij"]
ref_dij    = ref_dij_obj.physicalDose         # sparse (2000, 10)

print(f"Reference dose:  max={ref_dose.max():.4f}  mean(>0)={ref_dose[ref_dose>0].mean():.4f} Gy/fx")
print(f"Reference DIJ:   shape={ref_dij.shape}  nnz={ref_dij.nnz}")

# ---------------------------------------------------------------------------
# 2. Reconstruct CT
# ---------------------------------------------------------------------------
ct_mat = raw["ct"]
res_mat = ct_mat.resolution
cube_hu  = ct_mat.cubeHU   # (Ny, Nx, Nz) = (20, 10, 10)
cube_rsp = ct_mat.cube     # relative electron density

ct = {
    "cubeDim":    list(cube_hu.shape),   # [Ny, Nx, Nz]
    "resolution": {"x": float(res_mat.x),
                   "y": float(res_mat.y),
                   "z": float(res_mat.z)},
    "cubeHU":    [cube_hu],
    "cube":      [cube_rsp],
    "numOfCtScen": 1,
    "hlut":      ct_mat.hlut,
}

# ---------------------------------------------------------------------------
# 3. Reconstruct CST
# ---------------------------------------------------------------------------
cst_mat = raw["cst"]   # (2, 6) object array
cst = []
for i in range(cst_mat.shape[0]):
    row = cst_mat[i]
    vox_1based = np.asarray(row[3], dtype=np.int64).ravel()
    cst.append([
        int(row[0]),        # index (1-based MATLAB)
        str(row[1]),        # name
        str(row[2]),        # type
        vox_1based,         # voxel indices (1-based, as in pyMatRad convention)
        {},                 # parameters
        [],                 # objectives
    ])

print(f"\nCST: {len(cst)} structures")
for s in cst:
    print(f"  [{s[0]}] {s[1]} ({s[2]}): {len(s[3])} voxels")

# ---------------------------------------------------------------------------
# 4. Reconstruct STF from MATLAB stf struct
# ---------------------------------------------------------------------------
stf_mat = raw["stf"]   # numpy array of mat_struct, shape (2,)

def reconstruct_stf_beam(b):
    """Convert a MATLAB stf beam mat_struct to pyMatRad dict."""
    rays = []
    for r in b.ray:
        ray_dict = {
            "rayPos_bev":      np.asarray(getattr(r, "rayPos_bev"),      dtype=float),
            "targetPoint_bev": np.asarray(getattr(r, "targetPoint_bev"), dtype=float),
            "rayPos":          np.asarray(getattr(r, "rayPos"),          dtype=float),
            "targetPoint":     np.asarray(getattr(r, "targetPoint"),     dtype=float),
            "energy":          np.array([6.0]),   # 6 MV photons (Generic machine)
        }
        rays.append(ray_dict)

    n_bixels_per_ray = [1] * len(rays)   # 1 bixel per ray for photons

    beam = {
        "gantryAngle":       float(b.gantryAngle),
        "couchAngle":        float(b.couchAngle),
        "bixelWidth":        float(b.bixelWidth),
        "radiationMode":     str(b.radiationMode),
        "machine":           str(b.machine),
        "SAD":               float(b.SAD),
        "isoCenter":         np.asarray(b.isoCenter, dtype=float),
        "numOfRays":         int(b.numOfRays),
        "ray":               rays,
        "sourcePoint_bev":   np.asarray(b.sourcePoint_bev, dtype=float),
        "sourcePoint":       np.asarray(b.sourcePoint,     dtype=float),
        "numOfBixelsPerRay": n_bixels_per_ray,
        "totalNumOfBixels":  int(b.totalNumOfBixels),
    }
    return beam

stf = [reconstruct_stf_beam(stf_mat[i]) for i in range(len(stf_mat))]
print(f"\nSTF: {len(stf)} beams")
for i, beam in enumerate(stf):
    print(f"  Beam {i+1}: gantry={beam['gantryAngle']:.0f}°  "
          f"rays={beam['numOfRays']}  bixels={beam['totalNumOfBixels']}")

# ---------------------------------------------------------------------------
# 5. Build PLN
# ---------------------------------------------------------------------------
pln = {
    "radiationMode": "photons",
    "machine":       "Generic",
    "numOfFractions": 30,
    "propStf": {
        "gantryAngles": [0, 180],
        "couchAngles":  [0, 0],
        "bixelWidth":   10,
        "numOfBeams":   2,
        "addMargin":    False,
    },
    "propDoseCalc": {
        "doseGrid": {"resolution": {"x": 10, "y": 10, "z": 10}},
    },
    "propOpt": {},
}

# ---------------------------------------------------------------------------
# 6. Run pyMatRad dose calculation
# ---------------------------------------------------------------------------
print("\nRunning pyMatRad dose calculation...")
try:
    dij = calc_dose_influence(ct, cst, stf, pln)
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit(1)

py_dij = dij.get("physicalDose")
if py_dij is None:
    print("ERROR: dij['physicalDose'] not found. Keys:", list(dij.keys()))
    sys.exit(1)

# physicalDose may be a list (per scenario) or a single sparse matrix
if isinstance(py_dij, list):
    py_dij = py_dij[0]

print(f"\npyMatRad DIJ: shape={py_dij.shape}  nnz={py_dij.nnz}")
print(f"Reference DIJ: shape={ref_dij.shape}  nnz={ref_dij.nnz}")

# ---------------------------------------------------------------------------
# 7. Compute dose with uniform fluence (w=1) and compare
# ---------------------------------------------------------------------------
w_uniform = np.ones(py_dij.shape[1])
py_dose_flat  = np.asarray(py_dij.dot(w_uniform)).ravel()
ref_dose_flat = np.asarray(ref_dij.dot(ref_w)).ravel()

Ny, Nx, Nz = ct["cubeDim"]
py_dose  = py_dose_flat.reshape(Ny, Nx, Nz)

print(f"\n{'=== Dose comparison (uniform fluence, per fraction) ':=<60}")
print(f"{'Metric':<30} {'pyMatRad':>12} {'MATLAB':>12} {'rel err':>10}")
print("-" * 66)

def print_row(name, py_val, ref_val):
    rel = abs(py_val - ref_val) / (abs(ref_val) + 1e-12)
    print(f"{name:<30} {py_val:>12.4f} {ref_val:>12.4f} {rel*100:>9.2f}%")

print_row("max dose (Gy/fx)",       py_dose.max(),                   ref_dose.max())
print_row("mean dose >0 (Gy/fx)",   py_dose[py_dose>1e-6].mean(),   ref_dose[ref_dose>1e-6].mean())
print_row("DIJ nnz",                float(py_dij.nnz),               float(ref_dij.nnz))
print_row("DIJ total weight sum",   float(py_dij.sum()),             float(ref_dij.sum()))

# Target voxel comparison (1-based → 0-based)
target_vox_0 = cst[0][3] - 1
py_target  = py_dose_flat[target_vox_0]
ref_target = ref_dose_flat[target_vox_0]

print(f"\n{'=== Target voxel doses (Gy/fx) ':=<60}")
print(f"{'Voxel':>6} {'pyMatRad':>12} {'MATLAB':>12} {'rel err':>10}")
print("-" * 44)
for vi, (pv, rv) in enumerate(zip(py_target, ref_target)):
    rel = abs(pv - rv) / (abs(rv) + 1e-12)
    flag = " <--" if rel > 0.10 else ""
    print(f"{vi:>6} {pv:>12.4f} {rv:>12.4f} {rel*100:>9.2f}%{flag}")

print(f"\nTarget mean: pyMatRad={py_target.mean():.4f}  MATLAB={ref_target.mean():.4f}")
print("\nDone.")
