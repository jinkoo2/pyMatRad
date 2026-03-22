"""
Build the plain-C Siddon shared library.

Usage:
    cd matRad/rayTracing/_backends/siddon_c
    python build.py

Produces siddon_c.dll (Windows) or siddon_c.so (Linux/macOS)
next to this file.
"""

import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "siddon_c.c")


def _find_gcc_windows():
    """Find gcc.exe on Windows — checks conda envs and common locations."""
    import sysconfig
    prefix = sysconfig.get_config_var("prefix") or sys.prefix

    candidates = [
        # conda-forge m2w64 gcc in the active conda env
        os.path.join(prefix, "Library", "bin", "x86_64-w64-mingw32-gcc.exe"),
        os.path.join(prefix, "Library", "bin", "gcc.exe"),
        # Git for Windows MinGW
        r"C:\Program Files\Git\mingw64\bin\gcc.exe",
    ]

    # Also search PATH
    for p in os.environ.get("PATH", "").split(os.pathsep):
        candidates.append(os.path.join(p, "gcc.exe"))
        candidates.append(os.path.join(p, "x86_64-w64-mingw32-gcc.exe"))

    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


if platform.system() == "Windows":
    OUT = os.path.join(HERE, "siddon_c.dll")
    gcc = _find_gcc_windows()
    if gcc:
        cmd = [gcc, "-O3", "-shared", "-fPIC", SRC, "-o", OUT, "-lm"]
    else:
        # Fall back to MSVC if gcc not found
        cmd = ["cl.exe", "/O2", "/LD", SRC, f"/Fe:{OUT}", "/link", "/DLL"]
elif platform.system() == "Darwin":
    OUT = os.path.join(HERE, "siddon_c.so")
    cmd = ["cc", "-O3", "-shared", "-fPIC", SRC, "-o", OUT, "-lm"]
else:
    OUT = os.path.join(HERE, "siddon_c.so")
    cmd = ["cc", "-O3", "-shared", "-fPIC", SRC, "-o", OUT, "-lm"]

print(f"Building: {' '.join(cmd)}")
result = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
if result.returncode != 0:
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    sys.exit(result.returncode)
print(f"Built: {OUT}")
