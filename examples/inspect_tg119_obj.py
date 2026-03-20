"""Inspect TG119 CST objectives."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scipy.io as sio

mat_file = r'C:\Users\jkim20\Desktop\projects\tps\matRad\matRad\phantoms\TG119.mat'
raw = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
cst_raw = raw['cst']

for i in range(cst_raw.shape[0]):
    row = cst_raw[i]
    name = row[1]
    vtype = row[2]
    obj = row[5]
    print(f'\n[{i}] {name} ({vtype}):')
    print(f'  obj type: {type(obj)}')
    if hasattr(obj, '_fieldnames'):
        print(f'  obj fields: {obj._fieldnames}')
        for f in obj._fieldnames:
            print(f'    {f}: {getattr(obj, f)}')
    elif isinstance(obj, np.ndarray):
        print(f'  obj shape={obj.shape} dtype={obj.dtype}')
        for j, o in enumerate(obj.flat):
            print(f'  obj[{j}] type={type(o)}')
            if hasattr(o, '_fieldnames'):
                for f in o._fieldnames:
                    print(f'    {f}: {getattr(o, f)}')
    else:
        print(f'  obj value: {obj}')
