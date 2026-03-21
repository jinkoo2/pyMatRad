"""
Quick test to verify Siddon i_min fix corrects SSD for posterior beams.
Uses MATLAB's exact STF and computes SSD manually via siddon_ray_tracer.
"""
import os, sys
import numpy as np
import scipy.io as sio
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REF_FILE  = r'C:\Users\jkim20\Desktop\projects\tps\pyMatRad\examples\_matRad_ref_outputs\example2\matRad_example2_ref.mat'
TG119_MAT = r'C:\Users\jkim20\Desktop\projects\tps\matRad\matRad\phantoms\TG119.mat'

# MATLAB SSDs from console.log (coarse plan, 8 beams, 50° spacing)
matlab_ssd = {0: 939, 50: 905, 100: 849, 150: 894,
              200: 902, 250: 840, 300: 878, 350: 938}

DENSITY_THRESHOLD = 0.05

def h5val(f, ref):
    return np.array(f[ref]).ravel()

print("Loading STF from MATLAB ref...")
f = h5py.File(REF_FILE, 'r')
stf_h5 = f['stf']
n_beams = stf_h5['gantryAngle'].shape[0]

stf = []
for ib in range(n_beams):
    ga   = float(h5val(f, stf_h5['gantryAngle'][ib, 0])[0])
    sad  = float(h5val(f, stf_h5['SAD'][ib, 0])[0])
    nr   = int(h5val(f, stf_h5['numOfRays'][ib, 0])[0])
    iso  = h5val(f, stf_h5['isoCenter'][ib, 0])
    sp   = h5val(f, stf_h5['sourcePoint'][ib, 0])

    ray_grp = f[stf_h5['ray'][ib, 0]]
    rays = []
    for ir_idx in range(nr):
        rpb = h5val(f, ray_grp['rayPos_bev'][ir_idx, 0])
        tp  = h5val(f, ray_grp['targetPoint'][ir_idx, 0])
        rays.append({'rayPos_bev': rpb.astype(float), 'targetPoint': tp.astype(float)})

    stf.append({'gantryAngle': ga, 'SAD': sad, 'numOfRays': nr,
                'isoCenter': iso.astype(float), 'sourcePoint': sp.astype(float),
                'ray': rays})

f.close()

print("Loading CT from TG119.mat...")
raw = sio.loadmat(TG119_MAT, squeeze_me=True, struct_as_record=False)
ct_raw = raw['ct']
ct = {
    'cubeDim':    [int(ct_raw.cubeDim[0]), int(ct_raw.cubeDim[1]), int(ct_raw.cubeDim[2])],
    'resolution': {'x': float(ct_raw.resolution.x),
                   'y': float(ct_raw.resolution.y),
                   'z': float(ct_raw.resolution.z)},
    'cube':       [np.asarray(ct_raw.cube)],
    'numOfCtScen': 1,
}

from matRad.geometry.geometry import world_to_cube_coords, get_world_axes
from matRad.rayTracing.siddon import siddon_ray_tracer

ct = get_world_axes(ct)
res = ct['resolution']

print(f"\n  {'Beam':>5} {'Gantry':>8} {'pyMatRad SSD':>14} {'MATLAB SSD':>12} {'Diff mm':>9} {'Diff%':>7}")
print("  " + "-" * 55)

for ib, beam in enumerate(stf):
    ga = int(round(beam['gantryAngle']))
    iso_world = np.asarray(beam['isoCenter'])
    iso_cube = world_to_cube_coords(np.atleast_2d(iso_world), ct)[0]
    cube = ct['cube'][0]

    # Find center ray (smallest ray_pos_bev norm)
    ray_pos = np.array([r['rayPos_bev'] for r in beam['ray']])
    center_idx = int(np.argmin(np.sum(ray_pos**2, axis=1)))
    ray = beam['ray'][center_idx]

    source_point = np.asarray(beam['sourcePoint'])
    target_point = np.asarray(ray['targetPoint'])

    alphas, l_seg, rho, d12, _ = siddon_ray_tracer(
        iso_cube, res, source_point, target_point, [cube]
    )

    ssd = None
    if len(rho[0]) > 0:
        above_thresh = np.where(rho[0] > DENSITY_THRESHOLD)[0]
        if len(above_thresh) > 0 and len(alphas) > above_thresh[0]:
            ssd = float(d12 * alphas[above_thresh[0]])

    ml_ssd = matlab_ssd.get(ga, None)
    if ssd is not None and ml_ssd is not None:
        diff_mm = ssd - ml_ssd
        diff_pct = diff_mm / ml_ssd * 100
        flag = "  <<" if abs(diff_pct) > 3 else ""
        print(f"  {ib+1:>5} {ga:>7}°  {ssd:>12.1f} {ml_ssd:>12.0f} {diff_mm:>+8.1f} {diff_pct:>+6.1f}%{flag}")
    else:
        print(f"  {ib+1:>5} {ga:>7}°  py={ssd}  ml={ml_ssd}")

print("\nDone.")
