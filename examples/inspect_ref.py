"""Inspect matRad_example2_ref.mat structure."""
import h5py
import numpy as np

mat_file = r'U:\matRad_refdata\matRad_example2_ref.mat'

def print_tree(name, obj, depth=0):
    indent = "  " * depth
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}{name}: shape={obj.shape} dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"{indent}{name}/")
        if depth < 3:
            for k in list(obj.keys())[:20]:
                print_tree(k, obj[k], depth + 1)

with h5py.File(mat_file, 'r') as f:
    print("Top-level keys:", list(f.keys()))
    print()
    for key in f.keys():
        print_tree(key, f[key], depth=0)
        print()
