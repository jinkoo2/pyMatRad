# pyMatRad - Python Port of matRad

## Overview
Python port of the MATLAB-based matRad radiation treatment planning system.
- Source MATLAB code: `/gpfs/projects/KimGroup/projects/tps/matRad/`
- Python port: `/gpfs/projects/KimGroup/projects/tps/pyMatRad/`

## Running Examples
```bash
conda activate scikit-learn
cd /path/to/pyMatRad
python examples/example1_phantom.py
python examples/example2_photons.py
```

## Dependencies
- Python >= 3.9
- numpy >= 1.20
- scipy >= 1.7
- matplotlib >= 3.4
- h5py >= 3.0 (for MATLAB v7.3 .mat files)

## Project Structure
```
pyMatRad/
├── matRad/
│   ├── config.py              # Singleton config
│   ├── scenarios.py           # NominalScenario
│   ├── geometry/geometry.py   # Coordinate transforms, rotation matrices
│   ├── phantoms/builder/      # Synthetic phantom creation
│   ├── basedata/load_machine.py  # Load .mat machine files
│   ├── rayTracing/siddon.py   # Siddon ray tracer
│   ├── steering/stf_generator.py  # Beam/ray geometry (STF)
│   ├── doseCalc/
│   │   ├── calc_dose_influence.py  # Entry point
│   │   └── DoseEngines/
│   │       ├── dose_engine_base.py     # Base class
│   │       └── photon_svd_engine.py    # SVD photon engine
│   ├── optimization/
│   │   ├── fluence_optimization.py    # L-BFGS-B optimizer
│   │   └── DoseObjectives/objectives.py  # Objective functions
│   └── planAnalysis/plan_analysis.py  # DVH, quality indicators
├── gui/matrad_gui.py          # matplotlib GUI
└── examples/
    ├── example1_phantom.py    # Synthetic phantom plan
    └── example2_photons.py    # TG119 photon plan
```

## Key Coordinate System Notes
- `cubeDim = [Ny, Nx, Nz]` (MATLAB row=y, col=x, slice=z)
- MATLAB linear indices: 1-based, Fortran column-major: `ix = i + (j-1)*Ny + (k-1)*Ny*Nx`
- Python code converts 1-based MATLAB linear indices by subtracting 1 before decomposition
- BEV (Beam's-Eye-View): source at `[0, -SAD, 0]`, beam travels in +y direction
- Rotation: `R = R_Couch(y-axis) @ R_Gantry(z-axis)`

## Known Issues / Status
See Progress.md for current status.
