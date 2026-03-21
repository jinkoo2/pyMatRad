"""
Debug SSD for beam 4 (gantry=150°) — check cube values and voxel coords.
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

# Load beam 4 (index 3) from MATLAB ref
f = h5py.File(REF_FILE, 'r')
stf_h5 = f['stf']
TARGET_BEAM = 3  # gantry=150°

ga   = float(h5val(f, stf_h5['gantryAngle'][TARGET_BEAM, 0])[0])
nr   = int(h5val(f, stf_h5['numOfRays'][TARGET_BEAM, 0])[0])
iso  = h5val(f, stf_h5['isoCenter'][TARGET_BEAM, 0])
sp   = h5val(f, stf_h5['sourcePoint'][TARGET_BEAM, 0])

ray_grp = f[stf_h5['ray'][TARGET_BEAM, 0]]
rays_bev = []; rays_tp = []
for ir_idx in range(nr):
    rpb = h5val(f, ray_grp['rayPos_bev'][ir_idx, 0])
    tp  = h5val(f, ray_grp['targetPoint'][ir_idx, 0])
    rays_bev.append(rpb); rays_tp.append(tp)
f.close()

rays_bev = np.array(rays_bev)
center_idx = int(np.argmin(np.sum(rays_bev**2, axis=1)))
target_point = np.asarray(rays_tp[center_idx])
source_point = np.asarray(sp)

print(f"Beam {TARGET_BEAM+1}: gantry={ga}°")
print(f"  source_point (world from iso): {source_point}")
print(f"  target_point (world from iso): {target_point}")

# Load CT
raw = sio.loadmat(TG119_MAT, squeeze_me=True, struct_as_record=False)
ct_raw = raw['ct']
cube = np.asarray(ct_raw.cube)
Ny, Nx, Nz = int(ct_raw.cubeDim[0]), int(ct_raw.cubeDim[1]), int(ct_raw.cubeDim[2])
rx, ry, rz = float(ct_raw.resolution.x), float(ct_raw.resolution.y), float(ct_raw.resolution.z)

print(f"\nCT: [{Ny},{Nx},{Nz}], res=({rx},{ry},{rz})")
print(f"Cube shape: {cube.shape}")  # Should be [Ny, Nx, Nz]
print(f"Cube min/max: {cube.min():.4f} / {cube.max():.4f}")

# Compute isocenter in cube coordinates (same as MATLAB's matRad_world2cubeCoords)
from matRad.geometry.geometry import world_to_cube_coords, get_world_axes
ct_dict = {
    'cubeDim': [Ny, Nx, Nz],
    'resolution': {'x': rx, 'y': ry, 'z': rz},
    'cube': [cube], 'numOfCtScen': 1,
}
ct_dict = get_world_axes(ct_dict)
iso_cube = world_to_cube_coords(np.atleast_2d(iso), ct_dict)[0]
print(f"\niso_world: {iso}")
print(f"iso_cube:  {iso_cube}")
print(f"CT x: [{ct_dict['x'][0]:.2f}, {ct_dict['x'][-1]:.2f}]")
print(f"CT y: [{ct_dict['y'][0]:.2f}, {ct_dict['y'][-1]:.2f}]")

# Siddon inputs
sp_cube = source_point + iso_cube
tp_cube = target_point + iso_cube
print(f"\nSiddon source in cube: {sp_cube}")
print(f"Siddon target in cube: {tp_cube}")

from matRad.rayTracing.siddon import siddon_ray_tracer
res = {'x': rx, 'y': ry, 'z': rz}
alphas, l_seg, rho, d12, ix = siddon_ray_tracer(iso_cube, res, source_point, target_point, [cube])

print(f"\nd12 = {d12:.2f} mm, n_segments = {len(l_seg)}")
print(f"alpha_min = {alphas[0]:.6f}, alpha_max = {alphas[-1]:.6f}")

above_thresh = np.where(rho[0] > DENSITY_THRESHOLD)[0]
print(f"\nFirst above-threshold (> {DENSITY_THRESHOLD}) at segment: {above_thresh[0] if len(above_thresh) else 'NONE'}")

# Print segments 30-45 with position info
print(f"\nSegments around first above-threshold:")
print(f"  {'seg':>4} {'alpha':>10} {'SSD':>8} {'rho':>8} {'x_cube':>8} {'y_cube':>8} {'xvox':>6} {'yvox':>6}")
start_seg = max(0, above_thresh[0] - 3) if len(above_thresh) else 0
end_seg = min(len(l_seg), above_thresh[0] + 5) if len(above_thresh) else 10
for i in range(start_seg, end_seg):
    alpha_s = alphas[i]
    alpha_m = 0.5 * (alphas[i] + alphas[i+1])
    x_pos = sp_cube[0] + alpha_m * (tp_cube[0] - sp_cube[0])
    y_pos = sp_cube[1] + alpha_m * (tp_cube[1] - sp_cube[1])
    xvox = int(round(x_pos / rx))
    yvox = int(round(y_pos / ry))
    ssd = d12 * alpha_s
    flag = "  << SURFACE" if i in above_thresh and above_thresh[0] == i else ""
    print(f"  {i:>4} {alpha_s:>10.6f} {ssd:>8.1f} {rho[0][i]:>8.4f} {x_pos:>8.1f} {y_pos:>8.1f} {xvox:>6} {yvox:>6}{flag}")

# Now compute what MATLAB should see: SSD=894mm → alpha=894/2000=0.447
ml_alpha = 894 / d12
ml_x = sp_cube[0] + ml_alpha * (tp_cube[0] - sp_cube[0])
ml_y = sp_cube[1] + ml_alpha * (tp_cube[1] - sp_cube[1])
ml_xvox = int(round(ml_x / rx))
ml_yvox = int(round(ml_y / ry))
print(f"\nAt MATLAB's expected SSD=894 (alpha={ml_alpha:.4f}):")
print(f"  x_cube={ml_x:.1f}, y_cube={ml_y:.1f}, xvox={ml_xvox}, yvox={ml_yvox}")
# Check density at that voxel (0-based)
ml_zvox = int(round((sp_cube[2] + ml_alpha * (tp_cube[2] - sp_cube[2])) / rz))
ml_zvox = max(0, min(Nz-1, ml_zvox-1))
if 1 <= ml_xvox <= Nx and 1 <= ml_yvox <= Ny:
    rho_at_ml = cube[ml_yvox-1, ml_xvox-1, ml_zvox]
    print(f"  density at that voxel: {rho_at_ml:.4f}")

# Also show the cube density profile in x-direction at the beam entry y position
entry_alpha = alphas[0]
entry_y = sp_cube[1] + entry_alpha * (tp_cube[1] - sp_cube[1])
entry_z = sp_cube[2] + entry_alpha * (tp_cube[2] - sp_cube[2])
entry_x = sp_cube[0] + entry_alpha * (tp_cube[0] - sp_cube[0])
print(f"\nRay entry point (cube coords): ({entry_x:.1f}, {entry_y:.1f}, {entry_z:.1f})")
entry_yvox = int(round(entry_y / ry))
entry_zvox = int(round(entry_z / rz))
entry_xvox = int(round(entry_x / rx))
print(f"Entry voxel: ({entry_xvox}, {entry_yvox}, {entry_zvox})")

print(f"\nDensity slice at z={entry_zvox}, x={entry_xvox} (y-direction):")
for yv in range(max(1, entry_yvox-3), min(Ny, entry_yvox+10)):
    d = cube[yv-1, entry_xvox-1, entry_zvox-1]
    marker = " <<" if d > DENSITY_THRESHOLD else ""
    print(f"  y_vox={yv:3d}: {d:.4f}{marker}")
