"""
Inspect Generic photon machine kernel structure.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scipy.io as sio

MACHINE = r'C:\Users\jkim20\Desktop\projects\tps\matRad\matRad\basedata\photons_Generic.mat'
raw = sio.loadmat(MACHINE, squeeze_me=True, struct_as_record=False)
machine = raw['machine']

print("Machine fields:", [f for f in dir(machine) if not f.startswith('_')])
data = machine.data
print("\nData fields:", [f for f in dir(data) if not f.startswith('_')])

# Check kernel structure
if hasattr(data, 'kernel'):
    kern = data.kernel
    if hasattr(kern, '__len__'):
        print(f"\nNumber of kernels: {len(kern)}")
        for i, k in enumerate(kern):
            print(f"  kernel[{i}]: SSD={getattr(k, 'SSD', 'N/A')}")
    else:
        print(f"\nSingle kernel: SSD={getattr(kern, 'SSD', 'N/A')}")
        for f in dir(kern):
            if not f.startswith('_'):
                val = getattr(kern, f)
                if hasattr(val, '__len__'):
                    print(f"  {f}: len={len(np.asarray(val).ravel())}")
                else:
                    print(f"  {f}: {val}")

# Check SSD values
print(f"\npenumbraFWHMatIso: {getattr(data, 'penumbraFWHMatIso', 'N/A')}")
if hasattr(data, 'kernelPos'):
    kpos = np.asarray(data.kernelPos).ravel()
    print(f"kernelPos: [{kpos[0]:.1f}, {kpos[-1]:.1f}] mm (n={len(kpos)})")

# Check machine meta
meta = machine.meta
print("\nMeta fields:", [f for f in dir(meta) if not f.startswith('_')])
for f in ['radiationMode', 'machine', 'SAD']:
    if hasattr(meta, f):
        print(f"  meta.{f}: {getattr(meta, f)}")
