"""
Check if TG119.mat has pre-defined x, y, z axes that differ from Python's computed ones.
"""
import os, sys
import numpy as np
import scipy.io as sio

TG119_MAT = r'C:\Users\jkim20\Desktop\projects\tps\matRad\matRad\phantoms\TG119.mat'

raw = sio.loadmat(TG119_MAT, squeeze_me=True, struct_as_record=False)
ct_raw = raw['ct']

print("Fields in ct struct:", [f for f in dir(ct_raw) if not f.startswith('_')])
print()

# Check for pre-defined world axes
for field in ['x', 'y', 'z']:
    if hasattr(ct_raw, field):
        arr = np.asarray(getattr(ct_raw, field)).ravel()
        print(f"ct.{field}: shape={arr.shape}, range=[{arr[0]:.3f}, {arr[-1]:.3f}], step={arr[1]-arr[0]:.3f}")
    else:
        print(f"ct.{field}: NOT PRESENT")

print()
Ny, Nx, Nz = int(ct_raw.cubeDim[0]), int(ct_raw.cubeDim[1]), int(ct_raw.cubeDim[2])
rx, ry, rz = float(ct_raw.resolution.x), float(ct_raw.resolution.y), float(ct_raw.resolution.z)
print(f"cubeDim=[{Ny},{Nx},{Nz}], res=({rx},{ry},{rz})")

# Python's computed axes
x_py = -Nx/2 * rx + rx * np.arange(Nx)
y_py = -Ny/2 * ry + ry * np.arange(Ny)
z_py = -Nz/2 * rz + rz * np.arange(Nz)
print(f"\nPython computed axes:")
print(f"  x: [{x_py[0]:.3f}, {x_py[-1]:.3f}]")
print(f"  y: [{y_py[0]:.3f}, {y_py[-1]:.3f}]")
print(f"  z: [{z_py[0]:.3f}, {z_py[-1]:.3f}]")

# Compare
if hasattr(ct_raw, 'x'):
    x_ml = np.asarray(ct_raw.x).ravel()
    y_ml = np.asarray(ct_raw.y).ravel()
    z_ml = np.asarray(ct_raw.z).ravel()
    print(f"\nMatlab vs Python axes comparison:")
    print(f"  x diff: max={np.max(np.abs(x_ml - x_py[:len(x_ml)])):.6f}")
    print(f"  y diff: max={np.max(np.abs(y_ml - y_py[:len(y_ml)])):.6f}")
    print(f"  z diff: max={np.max(np.abs(z_ml - z_py[:len(z_ml)])):.6f}")
    print(f"  x MATLAB[0]={x_ml[0]:.3f}  Python[0]={x_py[0]:.3f}  diff={x_ml[0]-x_py[0]:.3f}")
    print(f"  y MATLAB[0]={y_ml[0]:.3f}  Python[0]={y_py[0]:.3f}  diff={y_ml[0]-y_py[0]:.3f}")
    print(f"  z MATLAB[0]={z_ml[0]:.3f}  Python[0]={z_py[0]:.3f}  diff={z_ml[0]-z_py[0]:.3f}")

    # Compute iso_cube with MATLAB axes
    iso_world = np.array([-1.69106999, -16.58527755, 0.14212926])
    first_vox_world_ml = np.array([x_ml[0], y_ml[0], z_ml[0]])
    first_vox_cube = np.array([rx, ry, rz])
    translation_ml = first_vox_cube - first_vox_world_ml
    iso_cube_ml = iso_world + translation_ml
    print(f"\niso_cube with MATLAB axes: {iso_cube_ml}")

    first_vox_world_py = np.array([x_py[0], y_py[0], z_py[0]])
    translation_py = first_vox_cube - first_vox_world_py
    iso_cube_py = iso_world + translation_py
    print(f"iso_cube with Python axes: {iso_cube_py}")
    print(f"Difference: {iso_cube_ml - iso_cube_py}")
