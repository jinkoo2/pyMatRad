"""
Test that ignoreOutsideDensities masking zeroes out artifact voxels outside BODY.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scipy.io as sio
from matRad.geometry.geometry import get_world_axes
from matRad.doseCalc.DoseEngines.photon_svd_engine import PhotonPencilBeamSVDEngine

TG119_MAT = r'C:\Users\jkim20\Desktop\projects\tps\matRad\matRad\phantoms\TG119.mat'
raw = sio.loadmat(TG119_MAT, squeeze_me=True, struct_as_record=False)
ct_raw = raw['ct']
cst_raw = raw['cst']

cube = np.asarray(ct_raw.cube)

# Build CST
cst = []
for i in range(cst_raw.shape[0]):
    row = cst_raw[i]
    vox = np.asarray(row[3], dtype=np.int64).ravel()
    cst.append([row[0], row[1], row[2], [vox], None])

engine = PhotonPencilBeamSVDEngine()
print(f"ignore_outside_densities = {engine.ignore_outside_densities}")

# Build V_ct_grid
all_voxels = []
for row in cst:
    vox = row[3][0] if isinstance(row[3], list) else row[3]
    all_voxels.append(np.asarray(vox))
engine._V_ct_grid = np.unique(np.concatenate(all_voxels))
print(f"V_ct_grid: {len(engine._V_ct_grid)} voxels (of {cube.size} total)")

engine._cube_wed = [cube.copy()]

print(f"\nBefore masking:")
print(f"  Problem voxel (yvox=145, xvox=122, zvox=66): {engine._cube_wed[0][144, 121, 65]:.4f}")
print(f"  BODY top voxel (yvox=109, xvox=101, zvox=66): {engine._cube_wed[0][108, 100, 65]:.4f}")

engine._apply_outside_density_mask()

print(f"\nAfter masking:")
print(f"  Problem voxel (yvox=145, xvox=122, zvox=66): {engine._cube_wed[0][144, 121, 65]:.4f}  (should be 0.0)")
print(f"  BODY top voxel (yvox=109, xvox=101, zvox=66): {engine._cube_wed[0][108, 100, 65]:.4f}  (should be ~0.97)")

# Now run Siddon on the masked cube and check SSD
from matRad.geometry.geometry import world_to_cube_coords
from matRad.rayTracing.siddon import siddon_ray_tracer

Ny, Nx, Nz = int(ct_raw.cubeDim[0]), int(ct_raw.cubeDim[1]), int(ct_raw.cubeDim[2])
rx, ry, rz = float(ct_raw.resolution.x), float(ct_raw.resolution.y), float(ct_raw.resolution.z)
ct = {
    'cubeDim': [Ny, Nx, Nz],
    'resolution': {'x': rx, 'y': ry, 'z': rz},
    'cube': [cube],
    'numOfCtScen': 1,
}
ct = get_world_axes(ct)

# Beam 4 (gantry=150deg)
iso = np.array([-1.69106999, -16.58527755, 0.14212926])
iso_cube = world_to_cube_coords(np.atleast_2d(iso), ct)[0]
source_world = np.array([500.0, 866.02540378, 0.0])
target_world = np.array([-500.0, -866.02540378, 0.0])

alphas, l_seg, rho, d12, ix = siddon_ray_tracer(
    iso_cube, ct['resolution'], source_world, target_world, [engine._cube_wed[0]]
)

above_thresh = np.where(rho[0] > 0.05)[0]
if len(above_thresh):
    ssd = d12 * alphas[above_thresh[0]]
    print(f"\nWith masked cube, beam 4 SSD = {ssd:.1f} mm  (MATLAB: ~895 mm)")
    print(f"  First above-threshold at segment {above_thresh[0]}, alpha={alphas[above_thresh[0]]:.6f}")
    sp_cube = source_world + iso_cube
    tp_cube = target_world + iso_cube
    am = 0.5 * (alphas[above_thresh[0]] + alphas[above_thresh[0]+1])
    xpos = sp_cube[0] + am * (tp_cube[0] - sp_cube[0])
    ypos = sp_cube[1] + am * (tp_cube[1] - sp_cube[1])
    print(f"  At xvox={round(xpos/rx)}, yvox={round(ypos/ry)}, rho={rho[0][above_thresh[0]]:.4f}")
else:
    print("\nNo above-threshold voxels found!")
