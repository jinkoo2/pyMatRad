"""
Debug SSD discrepancy for beam 4 (gantry=150°).
Print Siddon inputs and first-above-threshold voxel details.
"""
import os, sys
import numpy as np
import scipy.io as sio
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REF_FILE  = r'C:\Users\jkim20\Desktop\projects\tps\pyMatRad\examples\_matRad_ref_outputs\example2\matRad_example2_ref.mat'
TG119_MAT = r'C:\Users\jkim20\Desktop\projects\tps\matRad\matRad\phantoms\TG119.mat'

DENSITY_THRESHOLD = 0.05

def h5val(f, ref):
    return np.array(f[ref]).ravel()

# Load just beam 4 (index 3, gantry=150°)
f = h5py.File(REF_FILE, 'r')
stf_h5 = f['stf']

TARGET_BEAM = 3  # 0-based, gantry=150°

ga   = float(h5val(f, stf_h5['gantryAngle'][TARGET_BEAM, 0])[0])
sad  = float(h5val(f, stf_h5['SAD'][TARGET_BEAM, 0])[0])
nr   = int(h5val(f, stf_h5['numOfRays'][TARGET_BEAM, 0])[0])
iso  = h5val(f, stf_h5['isoCenter'][TARGET_BEAM, 0])
sp   = h5val(f, stf_h5['sourcePoint'][TARGET_BEAM, 0])
sp_b = h5val(f, stf_h5['sourcePoint_bev'][TARGET_BEAM, 0])

print(f"Beam {TARGET_BEAM+1}: gantry={ga}°, SAD={sad}")
print(f"  isoCenter (world): {iso}")
print(f"  sourcePoint (world, from iso): {sp}")
print(f"  sourcePoint_bev: {sp_b}")

ray_grp = f[stf_h5['ray'][TARGET_BEAM, 0]]
rays_bev = []
rays_tp = []
for ir_idx in range(nr):
    rpb = h5val(f, ray_grp['rayPos_bev'][ir_idx, 0])
    tp  = h5val(f, ray_grp['targetPoint'][ir_idx, 0])
    rays_bev.append(rpb)
    rays_tp.append(tp)

f.close()

rays_bev = np.array(rays_bev)
# center ray = minimum |rayPos_bev| norm
center_idx = int(np.argmin(np.sum(rays_bev**2, axis=1)))
print(f"  Center ray index: {center_idx} (rayPos_bev={rays_bev[center_idx]})")
target_point = rays_tp[center_idx]
print(f"  targetPoint (world, from iso): {target_point}")

print("\nLoading CT...")
raw = sio.loadmat(TG119_MAT, squeeze_me=True, struct_as_record=False)
ct_raw = raw['ct']
ct = {
    'cubeDim':    [int(ct_raw.cubeDim[0]), int(ct_raw.cubeDim[1]), int(ct_raw.cubeDim[2])],
    'resolution': {'x': float(ct_raw.resolution.x),
                   'y': float(ct_raw.resolution.y),
                   'z': float(ct_raw.resolution.z)},
    'cube':       [np.asarray(ct_raw.cube)],
    'numOfCtScen': 1,
}

from matRad.geometry.geometry import world_to_cube_coords, get_world_axes
from matRad.rayTracing.siddon import siddon_ray_tracer

ct = get_world_axes(ct)
res = ct['resolution']
print(f"\nCT cubeDim: {ct['cubeDim']}, res: {res}")
print(f"CT x: [{ct['x'][0]:.1f}, {ct['x'][-1]:.1f}]")
print(f"CT y: [{ct['y'][0]:.1f}, {ct['y'][-1]:.1f}]")
print(f"CT z: [{ct['z'][0]:.1f}, {ct['z'][-1]:.1f}]")

iso_world = np.asarray(iso)
iso_cube = world_to_cube_coords(np.atleast_2d(iso_world), ct)[0]
print(f"\niso_world: {iso_world}")
print(f"iso_cube:  {iso_cube}")

source_point = np.asarray(sp)
target_point_np = np.asarray(target_point)

print(f"\nSource (world, from iso): {source_point}")
print(f"Target (world, from iso): {target_point_np}")

# What Siddon sees after adding iso_cube:
sp_in_cube = source_point + iso_cube
tp_in_cube = target_point_np + iso_cube
print(f"\nSource in cube coords: {sp_in_cube}")
print(f"Target in cube coords: {tp_in_cube}")

# Cube bounds
Ny, Nx, Nz = ct['cubeDim']
rx, ry, rz = res['x'], res['y'], res['z']
print(f"\nCube plane bounds:")
print(f"  x: [{0.5*rx:.1f}, {(Nx+0.5)*rx:.1f}]  (Nx={Nx})")
print(f"  y: [{0.5*ry:.1f}, {(Ny+0.5)*ry:.1f}]  (Ny={Ny})")
print(f"  z: [{0.5*rz:.1f}, {(Nz+0.5)*rz:.1f}]  (Nz={Nz})")

cube = ct['cube'][0]
alphas, l_seg, rho, d12, ix = siddon_ray_tracer(
    iso_cube, res, source_point, target_point_np, [cube]
)

print(f"\nd12 = {d12:.2f} mm")
print(f"alpha_min = {alphas[0]:.6f}, alpha_max = {alphas[-1]:.6f}")
print(f"n_segments = {len(l_seg)}")

# Show first 20 segments
above_thresh = np.where(rho[0] > DENSITY_THRESHOLD)[0]
print(f"\nDensity threshold = {DENSITY_THRESHOLD}")
print(f"First above-threshold segment index: {above_thresh[0] if len(above_thresh) > 0 else 'NONE'}")
if len(above_thresh) > 0:
    k = above_thresh[0]
    alpha_entry = alphas[k]
    ssd_py = d12 * alpha_entry
    print(f"alpha at entry of first above-thresh seg: {alpha_entry:.6f}")
    print(f"pyMatRad SSD = {ssd_py:.2f} mm (MATLAB = 894 mm)")

    print(f"\nFirst 5 segment densities:")
    for i in range(min(5, len(rho[0]))):
        print(f"  seg[{i}]: rho={rho[0][i]:.4f}, alpha_start={alphas[i]:.6f}, SSD={d12*alphas[i]:.1f}")

# Check: is source inside or outside the cube?
print(f"\nIs source inside cube bounds?")
print(f"  x: {0.5*rx:.1f} <= {sp_in_cube[0]:.1f} <= {(Nx+0.5)*rx:.1f}: {0.5*rx <= sp_in_cube[0] <= (Nx+0.5)*rx}")
print(f"  y: {0.5*ry:.1f} <= {sp_in_cube[1]:.1f} <= {(Ny+0.5)*ry:.1f}: {0.5*ry <= sp_in_cube[1] <= (Ny+0.5)*ry}")
print(f"  z: {0.5*rz:.1f} <= {sp_in_cube[2]:.1f} <= {(Nz+0.5)*rz:.1f}: {0.5*rz <= sp_in_cube[2] <= (Nz+0.5)*rz}")

# What if we use the ct.cube (relative electron density) vs ct.cube (electron density)?
print(f"\nCube stats: min={cube.min():.4f}, max={cube.max():.4f}, mean={cube.mean():.4f}")
print(f"Fraction > 0.05: {(cube > 0.05).mean():.3f}")
print(f"Fraction > 0.5: {(cube > 0.5).mean():.3f}")
