"""
Native ray-tracing backends for pyMatRad.

Each backend provides a drop-in replacement for
``matRad.rayTracing.siddon.siddon_ray_tracer``.

Available submodules (must be compiled before use):
  - siddon_cython  — Cython backend
  - siddon_cpp     — pybind11 C++ backend (inside siddon_cpp/)
  - siddon_ctypes  — plain-C backend via ctypes (inside siddon_c/)
"""
