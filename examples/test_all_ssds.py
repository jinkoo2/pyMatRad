"""
Compute SSDs for all beams in the 8-beam example2 plan with the masked cube.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scipy.io as sio
import h5py

from matRad.geometry.geometry import get_world_axes, world_to_cube_coords
from matRad.rayTracing.siddon import siddon_ray_tracer
from matRad.doseCalc.DoseEngines.photon_svd_engine import PhotonPencilBeamSVDEngine

TG119_MAT = r'C:\Users\jkim20\Desktop\projects\tps\matRad\matRad\phantoms\TG119.mat'
REF_FILE  = r'C:\Users\jkim20\Desktop\projects\tps\pyMatRad\examples\_matRad_ref_outputs\example2\matRad_example2_ref.mat'

raw = sio.loadmat(TG119_MAT, squeeze_me=True, struct_as_record=False)
ct_raw = raw['ct']
cst_raw = raw['cst']

cube = np.asarray(ct_raw.cube)
Ny, Nx, Nz = int(ct_raw.cubeDim[0]), int(ct_raw.cubeDim[1]), int(ct_raw.cubeDim[2])
rx, ry, rz = float(ct_raw.resolution.x), float(ct_raw.resolution.y), float(ct_raw.resolution.z)

ct = {
    'cubeDim': [Ny, Nx, Nz],
    'resolution': {'x': rx, 'y': ry, 'z': rz},
    'cube': [cube],
    'numOfCtScen': 1,
}
ct = get_world_axes(ct)

# Build CST + V_ct_grid
cst = []
for i in range(cst_raw.shape[0]):
    row = cst_raw[i]
    vox = np.asarray(row[3], dtype=np.int64).ravel()
    cst.append([row[0], row[1], row[2], [vox], None])

engine = PhotonPencilBeamSVDEngine()
all_voxels = [np.asarray(row[3][0] if isinstance(row[3], list) else row[3]) for row in cst]
engine._V_ct_grid = np.unique(np.concatenate(all_voxels))
engine._cube_wed = [cube.copy()]
engine._apply_outside_density_mask()
masked_cube = engine._cube_wed[0]

# Load ref stf for beam geometry
def h5val(f, ref):
    return np.array(f[ref]).ravel()

f = h5py.File(REF_FILE, 'r')
stf_h5 = f['stf']
n_beams = stf_h5['gantryAngle'].shape[0]

print(f"{'Beam':>5} {'Gantry':>8} {'SSD_masked':>12} {'SSD_unmasked':>14}")
print("-" * 45)

for b in range(n_beams):
    ga = float(h5val(f, stf_h5['gantryAngle'][b, 0])[0])
    iso = h5val(f, stf_h5['isoCenter'][b, 0])
    sp  = h5val(f, stf_h5['sourcePoint'][b, 0])
    nr  = int(h5val(f, stf_h5['numOfRays'][b, 0])[0])

    ray_grp = f[stf_h5['ray'][b, 0]]
    rays_bev = np.array([h5val(f, ray_grp['rayPos_bev'][i, 0]) for i in range(nr)])
    center_idx = int(np.argmin(np.sum(rays_bev**2, axis=1)))
    tp = h5val(f, ray_grp['targetPoint'][center_idx, 0])

    iso_cube = world_to_cube_coords(np.atleast_2d(iso), ct)[0]

    def compute_ssd(cube_):
        alphas, _, rho, d12, _ = siddon_ray_tracer(
            iso_cube, ct['resolution'], sp, tp, [cube_]
        )
        above = np.where(rho[0] > 0.05)[0]
        return d12 * alphas[above[0]] if len(above) else float('nan')

    ssd_m = compute_ssd(masked_cube)
    ssd_u = compute_ssd(cube)
    print(f"{b+1:>5} {ga:>8.1f} {ssd_m:>12.1f} {ssd_u:>14.1f}")

f.close()
