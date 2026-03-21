"""Inspect STF and DIJ structure in the MATLAB .mat file."""
import h5py
import numpy as np

REF = r'C:\Users\jkim20\Desktop\projects\tps\pyMatRad\examples\_matRad_ref_outputs\example2\matRad_example2_ref.mat'

def deref(f, ref):
    """Dereference an HDF5 object reference."""
    return f[ref]

f = h5py.File(REF, 'r')

# --- Check DIJ physicalDose (sparse matrix) ---
print("=== dij/physicalDose ===")
pd = f['dij/physicalDose']
print(f"  shape: {pd.shape}  dtype: {pd.dtype}")
# It's (1,1) object → dereference
ref = pd[0, 0]
sp_group = f[ref]
print(f"  Sparse matrix group keys: {list(sp_group.keys())}")
for k in sp_group.keys():
    ds = sp_group[k]
    print(f"    {k}: shape={ds.shape} dtype={ds.dtype}")

print()

# --- Check STF (first beam) ---
print("=== stf (first beam) ===")
stf = f['stf']
print(f"  stf keys: {list(stf.keys())}")

# Dereference first beam's scalar fields
for field in ['gantryAngle', 'couchAngle', 'SAD', 'bixelWidth', 'numOfRays', 'totalNumOfBixels']:
    ref = stf[field][0, 0]
    val = np.array(f[ref]).ravel()
    print(f"  beam[0].{field} = {val}")

# sourcePoint_bev
ref = stf['sourcePoint_bev'][0, 0]
val = np.array(f[ref]).ravel()
print(f"  beam[0].sourcePoint_bev = {val}")

# isoCenter
ref = stf['isoCenter'][0, 0]
val = np.array(f[ref]).ravel()
print(f"  beam[0].isoCenter = {val}")

# numOfBixelsPerRay
ref = stf['numOfBixelsPerRay'][0, 0]
val = np.array(f[ref]).ravel()
print(f"  beam[0].numOfBixelsPerRay shape={val.shape} sample={val[:5]}")

# ray (first beam, first ray)
print("\n=== stf ray[0][0] ===")
ray_ref = stf['ray'][0, 0]
ray_group = f[ray_ref]
print(f"  ray group type: {type(ray_group)}")
if isinstance(ray_group, h5py.Dataset):
    print(f"  ray dataset shape={ray_group.shape} dtype={ray_group.dtype}")
    # Try dereferencing first ray
    r0_ref = ray_group[0, 0]
    r0 = f[r0_ref]
    print(f"  ray[0] type={type(r0)}")
    if hasattr(r0, 'keys'):
        print(f"  ray[0] keys: {list(r0.keys())}")
        for rk in ['rayPos_bev', 'targetPoint_bev', 'rayPos', 'targetPoint']:
            if rk in r0:
                print(f"    {rk}: {np.array(r0[rk]).ravel()}")
elif isinstance(ray_group, h5py.Group):
    print(f"  ray group keys: {list(ray_group.keys())}")

f.close()
