"""
Base class for dose calculation engines.

Python port of matRad_DoseEngineBase.m / matRad_PencilBeamEngineAbstract.m
"""

import numpy as np
import scipy.sparse as sp
from typing import Optional, List, Dict, Any
from ...config import MatRad_Config
from ...geometry import get_world_axes, set_overlap_priorities, resize_cst_to_grid
from ...geometry.geometry import cube_index_to_world_coords
from ...scenarios import NominalScenario


class DoseEngineBase:
    """
    Abstract base class for dose calculation engines.
    """

    name = "Base Dose Engine"
    possible_radiation_modes = []

    def __init__(self, pln: Optional[dict] = None):
        cfg = MatRad_Config.instance()
        self.calc_dose_direct = False
        self.mult_scen = "nomScen"
        self.bio_model = "none"
        self.machine = None
        self._pln = pln

        # Dose grid settings
        self.dose_grid = {
            "resolution": {
                "x": cfg.defaults.propDoseCalc.get("doseGridResolution", 3.0) if hasattr(cfg.defaults.propDoseCalc, 'get') else 3.0,
                "y": cfg.defaults.propDoseCalc.get("doseGridResolution", 3.0) if hasattr(cfg.defaults.propDoseCalc, 'get') else 3.0,
                "z": cfg.defaults.propDoseCalc.get("doseGridResolution", 3.0) if hasattr(cfg.defaults.propDoseCalc, 'get') else 3.0,
            }
        }

        # Pencil beam properties
        defaults = cfg.defaults.propDoseCalc
        self.geometric_lateral_cutoff = 50.0   # mm
        self.dosimetric_lateral_cutoff = 0.995
        self.use_given_eq_density_cube = False
        self.ignore_outside_densities = True
        self.ssd_density_threshold = 0.05
        self.keep_rad_depth_cubes = False
        self.num_of_dij_fill_steps = 10

        # Internal variables (set during calc)
        self._V_ct_grid = None
        self._V_dose_grid = None
        self._vox_world_coords = None
        self._vox_world_coords_dose_grid = None
        self._cube_wed = None
        self._cst_dose_grid = None
        self._effective_lateral_cutoff = 50.0

        if pln is not None:
            self._assign_from_pln(pln)

    def _assign_from_pln(self, pln: dict):
        """Assign properties from pln struct."""
        if "multScen" in pln:
            self.mult_scen = pln["multScen"]
        if "bioModel" in pln:
            self.bio_model = pln.get("bioModel", "none")

        prop = pln.get("propDoseCalc", {})
        if "doseGrid" in prop:
            dg = prop["doseGrid"]
            if "resolution" in dg:
                r = dg["resolution"]
                self.dose_grid["resolution"]["x"] = r.get("x", 3.0)
                self.dose_grid["resolution"]["y"] = r.get("y", 3.0)
                self.dose_grid["resolution"]["z"] = r.get("z", 3.0)
        if "ignoreOutsideDensities" in prop:
            self.ignore_outside_densities = bool(prop["ignoreOutsideDensities"])

    def calc_dose_influence(self, ct: dict, cst: list, stf: list) -> dict:
        """
        Calculate dose influence matrix (dij).

        Parameters
        ----------
        ct : dict
        cst : list
        stf : list

        Returns
        -------
        dict
            dij struct
        """
        dij = self._init_dose_calc(ct, cst, stf)
        dij = self._calc_dose(ct, cst, stf, dij)
        return dij

    def _init_dose_calc(self, ct: dict, cst: list, stf: list) -> dict:
        """Initialize dose calculation - set up grids and structures."""
        cfg = MatRad_Config.instance()

        ct = get_world_axes(ct)

        # Set up CT grid
        dij = {
            "ctGrid": {
                "x": ct["x"],
                "y": ct["y"],
                "z": ct["z"],
                "resolution": ct["resolution"],
                "dimensions": np.array([len(ct["y"]), len(ct["x"]), len(ct["z"])]),
            }
        }
        dij["ctGrid"]["numOfVoxels"] = int(np.prod(dij["ctGrid"]["dimensions"]))

        # Set up dose grid
        dg_res = self.dose_grid.get("resolution", ct["resolution"])
        # Use tiny epsilon (not half-step) to match MATLAB's colon semantics: a:step:b
        # MATLAB stops at the last value <= b; using step/2 would add an extra point.
        _eps = dg_res["x"] * 1e-6
        dij["doseGrid"] = {
            "resolution": dg_res,
            "x": np.arange(ct["x"][0], ct["x"][-1] + _eps, dg_res["x"]),
            "y": np.arange(ct["y"][0], ct["y"][-1] + _eps, dg_res["y"]),
            "z": np.arange(ct["z"][0], ct["z"][-1] + _eps, dg_res["z"]),
        }
        dij["doseGrid"]["dimensions"] = np.array([
            len(dij["doseGrid"]["y"]),
            len(dij["doseGrid"]["x"]),
            len(dij["doseGrid"]["z"]),
        ])
        dij["doseGrid"]["numOfVoxels"] = int(np.prod(dij["doseGrid"]["dimensions"]))
        dij["doseGrid"]["cubeDim"] = dij["doseGrid"]["dimensions"]

        cfg.disp_info(f"Dose grid has dimensions {dij['doseGrid']['dimensions'][0]}x{dij['doseGrid']['dimensions'][1]}x{dij['doseGrid']['dimensions'][2]}\n")

        # Meta information
        dij["numOfBeams"] = len(stf)
        dij["numOfRaysPerBeam"] = [b["numOfRays"] for b in stf]
        dij["totalNumOfRays"] = sum(dij["numOfRaysPerBeam"])
        dij["totalNumOfBixels"] = sum(b["totalNumOfBixels"] for b in stf)

        # Bookkeeping arrays
        n_cols = dij["totalNumOfBixels"]
        dij["bixelNum"] = np.full(n_cols, np.nan)
        dij["rayNum"] = np.full(n_cols, np.nan)
        dij["beamNum"] = np.full(n_cols, np.nan)
        dij["minMU"] = np.zeros(n_cols)
        dij["maxMU"] = np.full(n_cols, np.inf)
        dij["numOfParticlesPerMU"] = np.ones(n_cols) * 1e6

        # Set up scenario model
        self._mult_scen_obj = NominalScenario.from_pln(self.mult_scen, ct)

        # Find valid voxels (union of all structures)
        num_ct_scen = ct.get("numOfCtScen", 1)
        all_voxels = []
        for row in cst:
            vox = row[3][0] if isinstance(row[3], list) else row[3]
            all_voxels.append(np.asarray(vox))

        if all_voxels:
            self._V_ct_grid = np.unique(np.concatenate(all_voxels))
        else:
            self._V_ct_grid = np.arange(1, dij["ctGrid"]["numOfVoxels"] + 1)

        # Get dose grid voxels by interpolation from CT grid
        ct_dims = dij["ctGrid"]["dimensions"]
        dose_dims = dij["doseGrid"]["dimensions"]

        # Create boolean mask on CT grid
        ct_mask = np.zeros(ct_dims, dtype=float, order="F")
        ix_0 = self._V_ct_grid - 1
        ct_mask.ravel(order="F")[ix_0] = 1.0

        # Interpolate to dose grid
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (ct["y"], ct["x"], ct["z"]),
            ct_mask,
            method="nearest",
            bounds_error=False,
            fill_value=0.0,
        )

        # Create dose grid meshgrid
        dose_y, dose_x, dose_z = np.meshgrid(
            dij["doseGrid"]["y"],
            dij["doseGrid"]["x"],
            dij["doseGrid"]["z"],
            indexing="ij",
        )
        dose_pts = np.column_stack([
            dose_y.ravel(),
            dose_x.ravel(),
            dose_z.ravel(),
        ])
        dose_mask = interp(dose_pts).reshape(dose_dims, order="C")

        # V_dose_grid: 1-based Fortran-order linear indices of valid dose voxels
        valid_dose = dose_mask.ravel(order="F") > 0.5
        self._V_dose_grid = np.where(valid_dose)[0] + 1  # 1-based

        # Store world coordinates
        self._vox_world_coords = cube_index_to_world_coords(self._V_ct_grid, dij["ctGrid"])
        self._vox_world_coords_dose_grid = cube_index_to_world_coords(self._V_dose_grid, dij["doseGrid"])

        # Store dose grid structure
        self._dose_grid = dij["doseGrid"]
        self._ct_grid = dij["ctGrid"]

        # Set overlap priorities and resize CST to dose grid
        cst_dose = set_overlap_priorities(list(cst))
        self._cst_dose_grid = resize_cst_to_grid(cst_dose, dij["ctGrid"], dij["doseGrid"])

        return dij

    def _calc_dose(self, ct: dict, cst: list, stf: list, dij: dict) -> dict:
        """Override in subclass."""
        raise NotImplementedError

    @staticmethod
    def get_engine_from_pln(pln: dict) -> "DoseEngineBase":
        """
        Return the appropriate dose engine for the given plan.

        Engine is selected via pln['propDoseCalc']['engine']:
          'SVPB'   — SVD photon pencil beam (default)
          'ompMC'  — ompMC-style TERMA + scatter engine
          'TOPAS'  — TOPAS Geant4 MC (requires TOPAS binary)
        """
        radiation_mode = pln.get("radiationMode", "photons")
        engine_name    = (
            pln.get("propDoseCalc", {}).get("engine", "SVPB").upper()
        )

        if radiation_mode == "photons":
            if engine_name in ("SVPB", "SVDPB", "SVD"):
                from .photon_svd_engine import PhotonPencilBeamSVDEngine
                return PhotonPencilBeamSVDEngine(pln)
            elif engine_name in ("OMPC", "OMPMC"):
                from .photon_ompc_engine import PhotonOmpMCEngine
                return PhotonOmpMCEngine(pln)
            elif engine_name == "TOPAS":
                from .topas_mc_engine import TopasMCEngine
                return TopasMCEngine(pln)
            else:
                raise NotImplementedError(
                    f"Unknown engine '{engine_name}' for photons. "
                    "Choose 'SVPB', 'ompMC', or 'TOPAS'."
                )
        else:
            raise NotImplementedError(
                f"Dose engine for '{radiation_mode}' not implemented"
            )
