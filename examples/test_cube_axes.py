"""
Check if the TG119 cube axes are correctly oriented.
Key test: center voxel should be ~1.0 (water), boundary should be ~0.027 (air).
Also check if swapping x/y changes the density at Python's "surface" voxel.
"""
import os, sys
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TG119_MAT = r'C:\Users\jkim20\Desktop\projects\tps\matRad\matRad\phantoms\TG119.mat'

raw = sio.loadmat(TG119_MAT, squeeze_me=True, struct_as_record=False)
ct_raw = raw['ct']
cube = np.asarray(ct_raw.cube)
Ny, Nx, Nz = int(ct_raw.cubeDim[0]), int(ct_raw.cubeDim[1]), int(ct_raw.cubeDim[2])
rx, ry, rz = float(ct_raw.resolution.x), float(ct_raw.resolution.y), float(ct_raw.resolution.z)

print(f"CT cubeDim: [{Ny},{Nx},{Nz}]  (Ny,Nx,Nz)")
print(f"Cube shape from scipy.io: {cube.shape}")
print(f"Cube min/max: {cube.min():.4f} / {cube.max():.4f}")

# ISO center (world): (-1.69, -16.59, 0.14)
# ISO cube: (251.81, 236.91, 163.89)
# ISO voxel (1-based): x=84, y=79, z=66
iso_xvox, iso_yvox, iso_zvox = 84, 79, 66

print(f"\nIsocenter voxel (1-based): xvox={iso_xvox}, yvox={iso_yvox}, zvox={iso_zvox}")
print(f"Cube at center [yvox-1, xvox-1, zvox-1] = cube[{iso_yvox-1},{iso_xvox-1},{iso_zvox-1}]: {cube[iso_yvox-1, iso_xvox-1, iso_zvox-1]:.4f}")
print(f"Cube at center swapped [xvox-1, yvox-1, zvox-1] = cube[{iso_xvox-1},{iso_yvox-1},{iso_zvox-1}]: {cube[iso_xvox-1, iso_yvox-1, iso_zvox-1]:.4f}")

# Python's "wrong" surface voxel: xvox=122, yvox=145, zvox=66
pyvox_x, pyvox_y, pyvox_z = 122, 145, 66
print(f"\nPython's surface voxel (1-based): xvox={pyvox_x}, yvox={pyvox_y}, zvox={pyvox_z}")
print(f"Cube [yvox-1, xvox-1, zvox-1] = cube[{pyvox_y-1},{pyvox_x-1},{pyvox_z-1}]: {cube[pyvox_y-1, pyvox_x-1, pyvox_z-1]:.4f}")
print(f"Cube swapped [xvox-1, yvox-1, zvox-1] = cube[{pyvox_x-1},{pyvox_y-1},{pyvox_z-1}]: {cube[pyvox_x-1, pyvox_y-1, pyvox_z-1]:.4f}")

# MATLAB's expected surface voxel: xvox=102, yvox=110, zvox=66
mlvox_x, mlvox_y, mlvox_z = 102, 110, 66
print(f"\nMATLAB's expected surface voxel (1-based): xvox={mlvox_x}, yvox={mlvox_y}, zvox={mlvox_z}")
print(f"Cube [yvox-1, xvox-1, zvox-1] = cube[{mlvox_y-1},{mlvox_x-1},{mlvox_z-1}]: {cube[mlvox_y-1, mlvox_x-1, mlvox_z-1]:.4f}")

# Print x-profile through iso y,z to see cylinder boundary
print(f"\nX-profile through iso (yvox={iso_yvox}, zvox={iso_zvox}):")
x_profile = cube[iso_yvox-1, :, iso_zvox-1]
above = np.where(x_profile > 0.05)[0]
if len(above) > 0:
    print(f"  First above 0.05 at xvox={above[0]+1}, Last at xvox={above[-1]+1}")
    print(f"  Range: [{above[0]+1}, {above[-1]+1}] → width={(above[-1]-above[0]+1)*rx:.0f} mm, radius={(above[-1]-above[0])*rx/2:.0f} mm")

# Print y-profile through iso x,z
print(f"\nY-profile through iso (xvox={iso_xvox}, zvox={iso_zvox}):")
y_profile = cube[:, iso_xvox-1, iso_zvox-1]
above = np.where(y_profile > 0.05)[0]
if len(above) > 0:
    print(f"  First above 0.05 at yvox={above[0]+1}, Last at yvox={above[-1]+1}")
    print(f"  Range: [{above[0]+1}, {above[-1]+1}] → width={(above[-1]-above[0]+1)*ry:.0f} mm, radius={(above[-1]-above[0])*ry/2:.0f} mm")

# Diagonal profile for gantry=150° beam direction
# Ray enters cube at (x_cube=405, y_cube=502.5) with direction (-1000, -1732) per 2000mm
# In voxels: x decreases by 1000/2000=0.5 voxels/mm*rx = 0.5*3=1.5 vox/alpha step...
# Just trace along the ray manually
print(f"\nDiagonal density profile along gantry=150° ray (first 50 voxels from cube entry):")
# Siddon cube source: (751.8, 1103, 163.9), target: (-248.2, -629.1, 163.9)
sp_x, sp_y = 751.8, 1102.9
tp_x, tp_y = -248.2, -629.1
alpha_min = 0.3467  # entry alpha
d12 = 2000.0
print(f"  {'alpha':>8} {'SSD':>8} {'x_cube':>8} {'y_cube':>8} {'xvox':>6} {'yvox':>6} {'rho_correct':>12} {'rho_swapped':>12}")
for step in range(55):
    alpha = alpha_min + step * (0.635917 - 0.346664) / 100
    x_c = sp_x + alpha * (tp_x - sp_x)
    y_c = sp_y + alpha * (tp_y - sp_y)
    xvox = int(round(x_c / rx))
    yvox = int(round(y_c / ry))
    zvox = 66  # z is constant for this beam
    # Clip
    xvox_c = max(1, min(Nx, xvox))
    yvox_c = max(1, min(Ny, yvox))
    zvox_c = max(1, min(Nz, zvox))
    # Correct access: cube[yvox-1, xvox-1, zvox-1]
    rho_correct = cube[yvox_c-1, xvox_c-1, zvox_c-1]
    # Swapped access: cube[xvox-1, yvox-1, zvox-1]
    rho_swapped = cube[xvox_c-1, yvox_c-1, zvox_c-1]
    ssd = d12 * alpha
    flag = ""
    if rho_correct > 0.05 and (step == 0 or cube[max(0,yvox_c-2), xvox_c-1, zvox_c-1] <= 0.05):
        flag = " <<SURF_correct"
    if rho_swapped > 0.05 and (step == 0 or cube[xvox_c-1, max(0,yvox_c-2), zvox_c-1] <= 0.05):
        flag += " <<SURF_swapped"
    print(f"  {alpha:>8.5f} {ssd:>8.1f} {x_c:>8.1f} {y_c:>8.1f} {xvox:>6} {yvox:>6} {rho_correct:>12.4f} {rho_swapped:>12.4f}{flag}")
