"""Quick inspection of key values in matRad_example2_ref.mat."""
import h5py
import numpy as np

def s(f, path):
    """Read scalar from HDF5 dataset."""
    return float(np.array(f[path]).ravel()[0])

def v(f, path):
    """Read array from HDF5 dataset."""
    return np.array(f[path]).ravel()

f = h5py.File(r'U:\matRad_refdata\matRad_example2_ref.mat', 'r')

ga   = sorted(v(f, 'pln/propStf/gantryAngles').tolist())
nfx  = int(round(s(f, 'pln/numOfFractions')))
dg_dims  = v(f, 'dij/doseGrid/dimensions').astype(int)
dg_res_x = s(f, 'dij/doseGrid/resolution/x')
dg_res_y = s(f, 'dij/doseGrid/resolution/y')
dg_res_z = s(f, 'dij/doseGrid/resolution/z')
n_bixels = int(s(f, 'dij/totalNumOfBixels'))
ct_dims  = v(f, 'ct/cubeDim').astype(int)
ct_res_x = s(f, 'ct/resolution/x')
ct_res_y = s(f, 'ct/resolution/y')
ct_res_z = s(f, 'ct/resolution/z')

ref_dose = np.array(f['resultGUI/physicalDose'])   # h5py order: (Nz,Nx,Ny)

print(f"Gantry angles ({len(ga)} beams): {ga}")
print(f"Spacing: {ga[1]-ga[0]:.0f} deg")
print(f"Num fractions: {nfx}")
print(f"Total bixels in dij: {n_bixels}")
print(f"CT dims (h5py): {ct_dims}  res: ({ct_res_x},{ct_res_y},{ct_res_z}) mm")
print(f"Dose grid dims (h5py): {dg_dims}  res: ({dg_res_x},{dg_res_y},{dg_res_z}) mm")
print(f"Reference dose shape (h5py order): {ref_dose.shape}")
ref_dose_T = ref_dose.T  # → (Ny,Nx,Nz)
print(f"Reference dose .T shape: {ref_dose_T.shape}")
print(f"Reference dose max: {ref_dose_T.max():.4f} Gy/fx  ({ref_dose_T.max()*nfx:.2f} Gy total)")

f.close()
