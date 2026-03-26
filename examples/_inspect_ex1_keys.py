import h5py, numpy as np

path = r'C:\Users\jkim20\Desktop\projects\tps\pyMatRad\examples\_matRad_ref_outputs\example2\matRad_example2_ref.mat'
f = h5py.File(path, 'r')

# Check all weight vectors
for key in ['w', 'wUnsequenced', 'w_coarse', 'wUnsequenced_coarse']:
    if f'resultGUI/{key}' in f:
        v = np.array(f[f'resultGUI/{key}']).ravel()
        print(f"resultGUI/{key}: len={len(v)}  nnz={np.sum(v>0)}  max={v.max():.4f}")

# Check the stf bixel counts per beam
stf_h5 = f['stf']
n_beams = stf_h5['gantryAngle'].shape[0]
total = 0
print(f"\nSTF beams: {n_beams}")
for b in range(n_beams):
    nb = int(np.array(f[stf_h5['totalNumOfBixels'][b,0]]).ravel()[0])
    ga = float(np.array(f[stf_h5['gantryAngle'][b,0]]).ravel()[0])
    print(f"  beam[{b}] gantry={ga}° bixels={nb}")
    total += nb
print(f"  TOTAL bixels in STF: {total}")

# Check dij
print(f"\ndij/totalNumOfBixels: {int(np.array(f['dij/totalNumOfBixels']).ravel()[0])}")

f.close()
