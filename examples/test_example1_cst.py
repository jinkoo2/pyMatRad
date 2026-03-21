"""
Check CST voxel coverage in example1 phantom and identify the angle-dependent error source.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matRad.phantoms.builder import PhantomBuilder
from matRad.optimization.DoseObjectives import SquaredDeviation, SquaredOverdosing

ct_dim = [200, 200, 100]
ct_resolution = [2, 2, 3]
builder = PhantomBuilder(ct_dim, ct_resolution, num_of_ct_scen=1)

objective1 = SquaredDeviation(penalty=800, d_ref=45)
objective2 = SquaredOverdosing(penalty=400, d_ref=0)
objective3 = SquaredOverdosing(penalty=10, d_ref=0)

builder.add_spherical_target("Volume1", radius=20, objectives=[objective1.to_dict()], HU=0)
builder.add_box_oar("Volume2", [60, 30, 60], offset=[0, -15, 0],
                     objectives=[objective2.to_dict()], HU=0)
builder.add_box_oar("Volume3", [60, 30, 60], offset=[0, 15, 0],
                     objectives=[objective3.to_dict()], HU=0)
ct, cst = builder.get_ct_cst()

total_voxels = int(np.prod(ct_dim))
print(f"Total CT voxels: {total_voxels}")
print(f"\nCST structures:")
all_voxels = []
for i, row in enumerate(cst):
    vox = row[3][0] if isinstance(row[3], list) else row[3]
    vox = np.asarray(vox)
    all_voxels.append(vox)
    print(f"  [{i}] {row[1]} ({row[2]}): {len(vox)} voxels ({len(vox)/total_voxels*100:.1f}%)")

union = np.unique(np.concatenate(all_voxels))
print(f"\nUnion of all CST voxels: {len(union)} of {total_voxels} ({len(union)/total_voxels*100:.1f}%)")
print(f"Voxels OUTSIDE CST: {total_voxels - len(union)} ({(total_voxels-len(union))/total_voxels*100:.1f}%)")

# Check: are the missing voxels near the beam paths for 70-210 degree beams?
Ny, Nx, Nz = ct_dim[1], ct_dim[0], ct_dim[2]  # cubeDim order
print(f"\nCT cubeDim interpretation: [Ny={Ny}, Nx={Nx}, Nz={Nz}]")

# Check voxel index 0 (corners)
print(f"\nSample cube values at corners (HU):")
cube_hu = ct['cubeHU'][0]
print(f"  [0,0,0] = {cube_hu[0,0,0]}")
print(f"  [Ny/2, Nx/2, Nz/2] = {cube_hu[Ny//2, Nx//2, Nz//2]}")
print(f"  Center voxel HU: {cube_hu[Ny//2, Nx//2, Nz//2]}")

# With ignoreOutsideDensities=True:
# V_ct_grid = union of all CST voxels = only 3 small structures
# This means ~96% of the water box voxels get zeroed out!
# For beams at angles where the beam passes through the zeroed region,
# the radiological depth computation will be wrong.
print(f"\n*** KEY FINDING ***")
print(f"With ignoreOutsideDensities=True, {total_voxels - len(union)} voxels")
print(f"({(total_voxels-len(union))/total_voxels*100:.1f}%) will be set to density=0!")
print(f"This means the water box OUTSIDE the 3 small structures is treated as air!")
print(f"The beam still enters water (SSD is computed first), but the radiological")
print(f"depth cube used for dose calc has most of the box zeroed out.")
