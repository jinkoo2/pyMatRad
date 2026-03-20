"""Inspect TG119.mat to debug voxel index issue."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scipy.io as sio

mat_file = r'C:\Users\jkim20\Desktop\projects\tps\matRad\matRad\phantoms\TG119.mat'
raw = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)

ct = raw['ct']
print('CT fields:', ct._fieldnames)
print('cubeDim:', ct.cubeDim)
res = ct.resolution
print('resolution: x=%g y=%g z=%g' % (res.x, res.y, res.z))

cst_raw = raw['cst']
print('\nCST shape:', cst_raw.shape, 'dtype:', cst_raw.dtype)

for i in range(cst_raw.shape[0]):
    row = cst_raw[i]
    name = row[1]
    vox = row[3]
    print(f'\n  [{i}] name={name}')
    print(f'       vox type={type(vox)} dtype={getattr(vox,"dtype","N/A")}')
    if hasattr(vox, '__len__'):
        vox_arr = np.asarray(vox).ravel()
        print(f'       vox shape={vox.shape} min={vox_arr.min()} max={vox_arr.max()} count={len(vox_arr)}')
    else:
        print(f'       vox={vox}')

# Check what cubeDim means
cubeDim = ct.cubeDim
Ny, Nx, Nz = int(cubeDim[0]), int(cubeDim[1]), int(cubeDim[2])
total_vox = Ny * Nx * Nz
print(f'\nTotal CT voxels: {Ny}x{Nx}x{Nz} = {total_vox}')

# Check max voxel index in first structure
vox0 = np.asarray(cst_raw[0, 3]).ravel()
print(f'Structure 0 max index: {vox0.max()} (should be <= {total_vox})')
print(f'Structure 0 index dtype: {vox0.dtype}')

# Manually decompose one voxel index
ix = int(vox0[0]) - 1  # 0-based
k = ix // (Ny * Nx)
rem = ix % (Ny * Nx)
j = rem // Ny
i_row = rem % Ny
print(f'\nTest index {vox0[0]} (0-based: {ix}):')
print(f'  i={i_row} j={j} k={k}  (Ny={Ny} Nx={Nx} Nz={Nz})')
print(f'  k < Nz? {k < Nz}')
