"""Quick test of plan_analysis with MATLAB test data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scipy.io as sio

from matRad.doseCalc.calc_dose_influence import calc_dose_influence
from matRad.planAnalysis.plan_analysis import plan_analysis

path = r'C:\Users\jkim20\Desktop\projects\tps\matRad\test\testData\photons_testData.mat'
raw = sio.loadmat(path, squeeze_me=True, struct_as_record=False)

ct_mat = raw['ct']
res_mat = ct_mat.resolution
ct = {
    'cubeDim': list(ct_mat.cubeHU.shape),
    'resolution': {'x': float(res_mat.x), 'y': float(res_mat.y), 'z': float(res_mat.z)},
    'cubeHU': [ct_mat.cubeHU], 'cube': [ct_mat.cube], 'numOfCtScen': 1, 'hlut': ct_mat.hlut,
}

cst_mat = raw['cst']
cst = []
for i in range(cst_mat.shape[0]):
    row = cst_mat[i]
    cst.append([int(row[0]), str(row[1]), str(row[2]),
                np.asarray(row[3], dtype=np.int64).ravel(), {}, []])

stf_mat = raw['stf']
def make_beam(b):
    rays = [{'rayPos_bev': np.asarray(getattr(r,'rayPos_bev'), dtype=float),
             'targetPoint_bev': np.asarray(getattr(r,'targetPoint_bev'), dtype=float),
             'rayPos': np.asarray(getattr(r,'rayPos'), dtype=float),
             'targetPoint': np.asarray(getattr(r,'targetPoint'), dtype=float),
             'energy': np.array([6.0])} for r in b.ray]
    return {'gantryAngle': float(b.gantryAngle), 'couchAngle': float(b.couchAngle),
            'bixelWidth': float(b.bixelWidth), 'radiationMode': str(b.radiationMode),
            'machine': str(b.machine), 'SAD': float(b.SAD),
            'isoCenter': np.asarray(b.isoCenter, dtype=float), 'numOfRays': int(b.numOfRays),
            'ray': rays, 'sourcePoint_bev': np.asarray(b.sourcePoint_bev, dtype=float),
            'sourcePoint': np.asarray(b.sourcePoint, dtype=float),
            'numOfBixelsPerRay': [1]*len(rays), 'totalNumOfBixels': int(b.totalNumOfBixels)}
stf = [make_beam(stf_mat[i]) for i in range(len(stf_mat))]

pln = {'radiationMode': 'photons', 'machine': 'Generic', 'numOfFractions': 30,
       'propStf': {'gantryAngles': [0,180], 'couchAngles': [0,0], 'bixelWidth': 10, 'addMargin': False},
       'propDoseCalc': {'doseGrid': {'resolution': {'x':10,'y':10,'z':10}}}, 'propOpt': {}}

dij = calc_dose_influence(ct, cst, stf, pln)
print(f'DIJ OK: shape={dij["physicalDose"][0].shape} nnz={dij["physicalDose"][0].nnz}')

# Uniform fluence, no optimization
w = np.ones(dij['totalNumOfBixels'])
dose_vec = dij['physicalDose'][0].dot(w)
dims = dij['doseGrid']['dimensions']
dose_cube = dose_vec.reshape(dims, order='F')
result = {'physicalDose': dose_cube, 'w': w, 'doseGrid': dij['doseGrid']}

result = plan_analysis(result, ct, cst, None, pln)
print('\nPlan analysis QI:')
for qi in result['qi']:
    print(f"  {qi['name']} ({qi['type']}): D_mean={qi.get('D_mean',0):.4f}  D_95={qi.get('D_95',0):.4f}  D_5={qi.get('D_5',0):.4f} Gy/fx")

# Compare dose with MATLAB
ref_dose = raw['resultGUI'].physicalDose
print(f'\nDose max: pyMatRad={dose_cube.max():.4f}  MATLAB={ref_dose.max():.4f}  rel_err={abs(dose_cube.max()-ref_dose.max())/ref_dose.max()*100:.2f}%')

# Check target D_mean is reasonable (should be ~0.7 Gy/fx)
target_qi = next(q for q in result['qi'] if q.get('type','') == 'TARGET')
print(f'\nTarget D_mean={target_qi.get("D_mean",0):.4f} Gy/fx (expected ~0.70)')
assert target_qi.get('D_mean', 0) > 0.5, f'Target D_mean too low: {target_qi.get("D_mean",0)}'
print('PASS: Target D_mean is reasonable')
