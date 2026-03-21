"""
Verify SSD correctness using CST BODY voxels.
For beam at gantry=150°: check if Python's detected surface voxel
is at the actual BODY boundary, by inspecting CST BODY voxels.
"""
import os, sys
import numpy as np
import scipy.io as sio
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TG119_MAT = r'C:\Users\jkim20\Desktop\projects\tps\matRad\matRad\phantoms\TG119.mat'
REF_FILE  = r'C:\Users\jkim20\Desktop\projects\tps\pyMatRad\examples\_matRad_ref_outputs\example2\matRad_example2_ref.mat'

# Load TG119 CT and CST
raw = sio.loadmat(TG119_MAT, squeeze_me=True, struct_as_record=False)
ct_raw = raw['ct']
cst_raw = raw['cst']
cube = np.asarray(ct_raw.cube)
Ny, Nx, Nz = int(ct_raw.cubeDim[0]), int(ct_raw.cubeDim[1]), int(ct_raw.cubeDim[2])
rx, ry, rz = float(ct_raw.resolution.x), float(ct_raw.resolution.y), float(ct_raw.resolution.z)

print(f"CT: [{Ny},{Nx},{Nz}], res=({rx},{ry},{rz})")
print(f"Cube shape: {cube.shape}")

# Print all structures
print("\nCST structures:")
for i in range(cst_raw.shape[0]):
    row = cst_raw[i]
    vox = np.asarray(row[3], dtype=np.int64).ravel()
    print(f"  [{i}] {row[1]} ({row[2]}): {len(vox)} voxels")

# Get BODY voxels (should be index 2, name='BODY')
body_row = cst_raw[2]
body_vox = np.asarray(body_row[3], dtype=np.int64).ravel()  # 1-based MATLAB linear
print(f"\nBODY has {len(body_vox)} voxels")

# Convert MATLAB 1-based linear indices to (y, x, z) subscripts
# MATLAB linear index ix = y + (x-1)*Ny + (z-1)*Ny*Nx (1-based, Fortran/column-major)
ix_0 = body_vox - 1  # 0-based
k_z = ix_0 // (Ny * Nx)
rem = ix_0 % (Ny * Nx)
i_x = rem // Ny
j_y = rem % Ny
# Python 0-based: j_y=y-dim, i_x=x-dim, k_z=z-dim

print(f"BODY voxel ranges:")
print(f"  x (col): [{i_x.min()+1}, {i_x.max()+1}]  →  [{(i_x.min()+1)*rx:.0f}, {(i_x.max()+1)*rx:.0f}] mm cube")
print(f"  y (row): [{j_y.min()+1}, {j_y.max()+1}]  →  [{(j_y.min()+1)*ry:.0f}, {(j_y.max()+1)*ry:.0f}] mm cube")
print(f"  z (slice): [{k_z.min()+1}, {k_z.max()+1}]")

# Phantom center in cube coords from BODY voxels
x_center_vox = (i_x.min() + i_x.max()) / 2 + 1  # 1-based
y_center_vox = (j_y.min() + j_y.max()) / 2 + 1
print(f"\nBODY center (voxel, 1-based): x={x_center_vox:.1f}, y={y_center_vox:.1f}")
print(f"BODY center (cube mm): x={x_center_vox*rx:.1f}, y={y_center_vox*ry:.1f}")

# Load isocenter from ref file
f = h5py.File(REF_FILE, 'r')
stf_h5 = f['stf']
TARGET_BEAM = 3  # gantry=150°

iso  = np.array(f['dij/doseGrid/resolution/x'])  # just to access file
# Actually get iso from stf
def h5val(f, ref): return np.array(f[ref]).ravel()
iso = h5val(f, stf_h5['isoCenter'][TARGET_BEAM, 0])
sp  = h5val(f, stf_h5['sourcePoint'][TARGET_BEAM, 0])
ga  = float(h5val(f, stf_h5['gantryAngle'][TARGET_BEAM, 0])[0])
nr  = int(h5val(f, stf_h5['numOfRays'][TARGET_BEAM, 0])[0])

ray_grp = f[stf_h5['ray'][TARGET_BEAM, 0]]
rays_bev = [h5val(f, ray_grp['rayPos_bev'][i, 0]) for i in range(nr)]
rays_tp = [h5val(f, ray_grp['targetPoint'][i, 0]) for i in range(nr)]
f.close()

rays_bev = np.array(rays_bev)
center_idx = int(np.argmin(np.sum(rays_bev**2, axis=1)))
target_world = np.asarray(rays_tp[center_idx])
source_world = np.asarray(sp)

print(f"\nBeam {TARGET_BEAM+1} (gantry={ga}°):")
print(f"  source (world from iso): {source_world}")
print(f"  target (world from iso): {target_world}")
print(f"  iso_world: {iso}")

# Compute iso_cube using MATLAB's actual x,y,z axes
x_ml = np.asarray(ct_raw.x).ravel()
y_ml = np.asarray(ct_raw.y).ravel()
z_ml = np.asarray(ct_raw.z).ravel()
first_vox_world = np.array([x_ml[0], y_ml[0], z_ml[0]])
first_vox_cube = np.array([rx, ry, rz])
translation = first_vox_cube - first_vox_world
iso_cube = np.asarray(iso) + translation
print(f"  iso_cube (MATLAB axes): {iso_cube}")

# iso_xvox, iso_yvox
iso_xvox = round(iso_cube[0] / rx)
iso_yvox = round(iso_cube[1] / ry)
print(f"  iso_xvox={iso_xvox}, iso_yvox={iso_yvox}")

# Check if Python's surface voxel (xvox=122, yvox=145) is in BODY
py_surf_xvox, py_surf_yvox, py_surf_zvox = 122, 145, 66  # From previous debug
py_surf_ix = py_surf_yvox - 1 + (py_surf_xvox - 1) * Ny + (py_surf_zvox - 1) * Ny * Nx
py_surf_ix_1based = py_surf_ix + 1
in_body = py_surf_ix_1based in set(body_vox.tolist())
print(f"\nPython's surface voxel (xvox={py_surf_xvox}, yvox={py_surf_yvox}, zvox={py_surf_zvox}):")
print(f"  1-based linear index: {py_surf_ix_1based}")
print(f"  In BODY structure: {in_body}")
print(f"  Density: {cube[py_surf_yvox-1, py_surf_xvox-1, py_surf_zvox-1]:.4f}")

# Find BODY voxels along the beam direction for beam 4 (gantry=150°)
print(f"\nBODY voxels along beam direction (gantry=150°) at z={py_surf_zvox}:")
# Body voxels at z=py_surf_zvox
z_mask = k_z == (py_surf_zvox - 1)  # 0-based
body_z_x = i_x[z_mask]  # 0-based x
body_z_y = j_y[z_mask]  # 0-based y

# The beam travels from (xvox≈135, yvox≈168) down-left to lower yvox
# Find BODY voxels with xvox near the beam path
# Beam path at z=66: x decreases from 135 to ~84 as y decreases from 168 to ~79
# Relationship: along the ray, for each yvov, xvov ≈ 135 - (168-yvov)*0.577
print(f"  Body y-range at z={py_surf_zvox}: [{body_z_y.min()+1}, {body_z_y.max()+1}]")
print(f"  First body yvox (top, max y in cube): {body_z_y.max()+1}")
print(f"  (in mm cube): {(body_z_y.max()+1)*ry:.0f} mm")

# What SSD would this give?
# At the top body voxel, find where the beam is
# Beam: sp_cube = source_world + iso_cube = (500+iso_cx, 866+iso_cy, iso_cz)
sp_cube = source_world + iso_cube
tp_cube = target_world + iso_cube
body_top_yvox = body_z_y.max() + 1  # 1-based
body_top_y_cube = body_top_yvox * ry

# Find x at this y along the beam
dt_y = tp_cube[1] - sp_cube[1]
if abs(dt_y) > 1e-6:
    alpha_at_body_top_y = (body_top_y_cube - sp_cube[1]) / dt_y
    x_at_alpha = sp_cube[0] + alpha_at_body_top_y * (tp_cube[0] - sp_cube[0])
    body_top_xvox_along_ray = round(x_at_alpha / rx)
    d12 = np.linalg.norm(sp_cube - tp_cube)
    ssd_at_body_top = alpha_at_body_top_y * d12
    print(f"  At top body y ({body_top_yvox}, {body_top_y_cube:.0f}mm), beam is at xvox={body_top_xvox_along_ray}")
    print(f"  SSD at body top-y: {ssd_at_body_top:.1f} mm")

    # Check if that (xvox, yvox) is in BODY
    test_ix = (body_top_yvox - 1) + (body_top_xvox_along_ray - 1) * Ny + (py_surf_zvox - 1) * Ny * Nx
    test_in_body = (test_ix + 1) in set(body_vox.tolist())
    print(f"  That voxel in BODY: {test_in_body}")
    print(f"  Density there: {cube[body_top_yvox-1, body_top_xvox_along_ray-1, py_surf_zvox-1]:.4f}")
