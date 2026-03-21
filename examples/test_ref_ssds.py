"""
Extract actual MATLAB reference SSDs from the ref stf.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import h5py

REF_FILE = r'C:\Users\jkim20\Desktop\projects\tps\pyMatRad\examples\_matRad_ref_outputs\example2\matRad_example2_ref.mat'

def h5val(f, ref):
    return np.array(f[ref]).ravel()

f = h5py.File(REF_FILE, 'r')
stf_h5 = f['stf']
n_beams = stf_h5['gantryAngle'].shape[0]

print(f"{'Beam':>5} {'Gantry':>8} {'MATLAB_SSD_center':>18}")
print("-" * 35)

for b in range(n_beams):
    ga = float(h5val(f, stf_h5['gantryAngle'][b, 0])[0])
    nr = int(h5val(f, stf_h5['numOfRays'][b, 0])[0])
    ray_grp = f[stf_h5['ray'][b, 0]]

    # Get SSD for each ray
    ssds = []
    rays_bev = []
    for i in range(nr):
        bev = h5val(f, ray_grp['rayPos_bev'][i, 0])
        rays_bev.append(bev)
        if 'SSD' in ray_grp:
            try:
                ssd = float(h5val(f, ray_grp['SSD'][i, 0])[0])
                ssds.append(ssd)
            except:
                pass

    rays_bev = np.array(rays_bev)
    center_idx = int(np.argmin(np.sum(rays_bev**2, axis=1)))

    if ssds and center_idx < len(ssds):
        center_ssd = ssds[center_idx]
        print(f"{b+1:>5} {ga:>8.1f} {center_ssd:>18.1f}")
    else:
        print(f"{b+1:>5} {ga:>8.1f} {'no SSD in ref':>18}")

f.close()
