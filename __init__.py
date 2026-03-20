"""
pyMatRad - Python port of matRad radiation treatment planning system.

matRad is a free and open-source software for radiation treatment planning.
This Python port provides the same functionality without requiring MATLAB.
"""

from .matRad.config import MatRad_Config
from .matRad import (
    generate_stf,
    calc_dose_influence,
    fluence_optimization,
    plan_analysis,
)

__version__ = "0.1.0"
__all__ = [
    "MatRad_Config",
    "generate_stf",
    "calc_dose_influence",
    "fluence_optimization",
    "plan_analysis",
]
