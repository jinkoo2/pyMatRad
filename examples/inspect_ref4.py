"""Inspect STF ray structure in detail."""
import h5py
import numpy as np

REF = r'C:\Users\jkim20\Desktop\projects\tps\pyMatRad\examples\_matRad_ref_outputs\example2\matRad_example2_ref.mat'
f = h5py.File(REF, 'r')
stf = f['stf']

# Check all 8 beams' gantry angles and bixel counts
print("=== All beams ===")
for i in range(8):
    ga = float(np.array(f[stf['gantryAngle'][i, 0]]).ravel()[0])
    nb = int(np.array(f[stf['totalNumOfBixels'][i, 0]]).ravel()[0])
    nr = int(np.array(f[stf['numOfRays'][i, 0]]).ravel()[0])
    print(f"  beam[{i}]: gantry={ga:.0f}°  rays={nr}  bixels={nb}")

# Check ray structure for beam 0
print("\n=== beam[0] ray structure ===")
ray_group = f[stf['ray'][0, 0]]
print(f"  ray keys: {list(ray_group.keys())}")
for k in ray_group.keys():
    ds = ray_group[k]
    print(f"  {k}: shape={ds.shape} dtype={ds.dtype}")

# Sample first ray's positions
print("\n  First ray positions:")
for k in ['rayPos_bev', 'targetPoint_bev', 'rayPos', 'targetPoint']:
    if k in ray_group:
        val = np.array(ray_group[k])
        print(f"    {k}: shape={val.shape}  first={val.T[0] if val.ndim>1 else val}")

# Check DIJ dimensions
print("\n=== DIJ info ===")
ref_pd = f['dij/physicalDose'][0, 0]
sp = f[ref_pd]
ir = np.array(sp['ir'])
jc = np.array(sp['jc'])
data = np.array(sp['data'])
print(f"  ir range: [{ir.min()}, {ir.max()}]")
print(f"  jc shape: {jc.shape}  (n_bixels = {len(jc)-1})")
print(f"  data nnz: {len(data)}")
n_vox = int(ir.max()) + 1
print(f"  estimated n_voxels_dose: {n_vox}")

# Check dose grid
dg_dims = np.array(f['dij/doseGrid/dimensions']).ravel().astype(int)
print(f"  dose grid dims: {dg_dims}  product={np.prod(dg_dims)}")

f.close()
