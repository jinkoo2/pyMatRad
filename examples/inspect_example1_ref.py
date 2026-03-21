"""
Inspect structure of the MATLAB example1 reference file.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import h5py

REF = r'C:\Users\jkim20\Desktop\projects\tps\pyMatRad\examples\_matRad_ref_outputs\example1\matRad_example1_ref.mat'

def print_h5_tree(group, prefix='', max_depth=3, depth=0):
    if depth > max_depth:
        return
    for key in list(group.keys())[:20]:
        item = group[key]
        if isinstance(item, h5py.Dataset):
            print(f"{prefix}{key}: shape={item.shape} dtype={item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"{prefix}{key}/")
            print_h5_tree(item, prefix + '  ', max_depth, depth+1)

f = h5py.File(REF, 'r')
print("Top-level keys:", list(f.keys()))
print()
print_h5_tree(f, max_depth=3)

# Check CT dimensions and resolution
print("\n--- CT ---")
if 'ct' in f:
    ct_grp = f['ct']
    print("ct keys:", list(ct_grp.keys()))
    for k in ['cubeDim', 'resolution']:
        if k in ct_grp:
            print(f"  ct.{k}:", np.array(ct_grp[k]).ravel())

# Check STF
print("\n--- STF ---")
if 'stf' in f:
    stf = f['stf']
    print("stf keys:", list(stf.keys()))
    ga = np.array(stf['gantryAngle']).ravel()
    print(f"  gantryAngles: {ga}")
    n_beams = stf['gantryAngle'].shape[0]
    print(f"  n_beams: {n_beams}")

# Check DIJ
print("\n--- DIJ ---")
if 'dij' in f:
    dij = f['dij']
    print("dij keys:", list(dij.keys()))
    if 'physicalDose' in dij:
        pd = dij['physicalDose']
        print(f"  physicalDose shape: {pd.shape}")
        # Try to get sparse data
        ref = pd[0, 0]
        sp_grp = f[ref]
        print(f"  sparse keys: {list(sp_grp.keys())}")
        ir = np.array(sp_grp['ir'], dtype=np.int64)
        jc = np.array(sp_grp['jc'], dtype=np.int64)
        data = np.array(sp_grp['data'], dtype=np.float64)
        n_bixels = len(jc) - 1
        print(f"  n_bixels={n_bixels}, nnz={len(data)}")
        print(f"  data range: [{data.min():.6f}, {data.max():.6f}]")

    if 'doseGrid' in dij:
        dg = dij['doseGrid']
        if 'dimensions' in dg:
            print(f"  doseGrid dims: {np.array(dg['dimensions']).ravel()}")
        for k in ['resolution']:
            if k in dg:
                grp = dg[k]
                for ax in ['x', 'y', 'z']:
                    if ax in grp:
                        print(f"  doseGrid.resolution.{ax}: {float(np.array(grp[ax]).ravel()[0])}")

f.close()
