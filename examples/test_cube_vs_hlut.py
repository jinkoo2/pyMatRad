"""
Check if applying HLUT to cubeHU gives different density values
at Python's "wrong" surface voxel (xvox=122, yvox=145, zvox=66).
"""
import os, sys
import numpy as np
import scipy.io as sio

TG119_MAT = r'C:\Users\jkim20\Desktop\projects\tps\matRad\matRad\phantoms\TG119.mat'

raw = sio.loadmat(TG119_MAT, squeeze_me=True, struct_as_record=False)
ct_raw = raw['ct']
cube = np.asarray(ct_raw.cube)
cubeHU = np.asarray(ct_raw.cubeHU)
hlut = np.asarray(ct_raw.hlut)  # shape (N, 2): column 0 = HU, column 1 = density

print(f"HLUT shape: {hlut.shape}")
print(f"HLUT range: HU=[{hlut[:,0].min():.0f}, {hlut[:,0].max():.0f}], density=[{hlut[:,1].min():.4f}, {hlut[:,1].max():.4f}]")
print(f"HLUT data:\n{hlut}")

print(f"\ncubeHU shape: {cubeHU.shape}, min={cubeHU.min():.0f}, max={cubeHU.max():.0f}")

# Check the problematic voxel (xvox=122, yvox=145, zvox=66, 1-based)
# Python access: cube[yvox-1, xvox-1, zvox-1]
pv = (144, 121, 65)  # 0-based (yvox-1, xvox-1, zvox-1)
print(f"\nProblem voxel (yvox=145, xvox=122, zvox=66):")
print(f"  cube[{pv}] = {cube[pv]:.4f}  (current density used by Python)")
print(f"  cubeHU[{pv}] = {cubeHU[pv]:.1f} HU")

# Apply HLUT to convert HU to density
def apply_hlut(hu_values, hlut):
    """Apply HU-to-density HLUT lookup."""
    return np.interp(hu_values, hlut[:, 0], hlut[:, 1])

hu_at_pv = cubeHU[pv]
density_from_hlut = apply_hlut(hu_at_pv, hlut)
print(f"  density from HLUT: {density_from_hlut:.4f}")
print(f"  Above 0.05 threshold? cube={cube[pv]>0.05}, HLUT={density_from_hlut>0.05}")

# Check the BODY top voxel (xvox=101, yvox=109, zvox=66) - MATLAB's surface
mv = (108, 100, 65)  # 0-based
print(f"\nMATLAB surface voxel (yvox=109, xvox=101, zvox=66):")
print(f"  cube[{mv}] = {cube[mv]:.4f}")
print(f"  cubeHU[{mv}] = {cubeHU[mv]:.1f} HU")
density_from_hlut_mv = apply_hlut(cubeHU[mv], hlut)
print(f"  density from HLUT: {density_from_hlut_mv:.4f}")

# Apply HLUT to the full cube and compare with stored cube
print(f"\nApplying HLUT to full cubeHU...")
cube_recomputed = apply_hlut(cubeHU, hlut)
print(f"Recomputed cube: min={cube_recomputed.min():.4f}, max={cube_recomputed.max():.4f}")
print(f"Original cube:   min={cube.min():.4f}, max={cube.max():.4f}")

# Compare
diff = np.abs(cube - cube_recomputed)
print(f"\nDifference original vs recomputed:")
print(f"  max diff: {diff.max():.4f}")
print(f"  mean diff: {diff.mean():.6f}")
print(f"  Fraction with diff > 0.01: {(diff > 0.01).mean():.4f}")

# Check: with recomputed cube, what density do problem voxels have?
print(f"\nWith HLUT-recomputed cube:")
print(f"  Problem voxel (yvox=145, xvox=122): {cube_recomputed[pv]:.4f}  (threshold=0.05?  {cube_recomputed[pv]>0.05})")
print(f"  BODY top voxel (yvox=109, xvox=101): {cube_recomputed[mv]:.4f}  (threshold=0.05? {cube_recomputed[mv]>0.05})")

# Show a y-profile along the beam at xvox=84, zvov=66 for both cubes
print(f"\nY-profile at xvox=84, zvox=66:")
print(f"  {'yvox':>6} {'original':>10} {'recomputed':>12} {'HU':>8}")
for yv in range(55, 160):
    od = cube[yv-1, 83, 65]
    rd = cube_recomputed[yv-1, 83, 65]
    hu = cubeHU[yv-1, 83, 65]
    if od > 0.04 or rd > 0.04:
        flag = " << orig" if od > 0.05 else ""
        flag += " << recomp" if rd > 0.05 else ""
        print(f"  {yv:>6} {od:>10.4f} {rd:>12.4f} {hu:>8.1f}{flag}")
