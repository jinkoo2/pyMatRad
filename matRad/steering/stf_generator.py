"""
Steering information (STF) generators.

Python port of:
- matRad_StfGeneratorBase.m
- matRad_StfGeneratorExternalRayBixelAbstract.m
- matRad_StfGeneratorPhotonIMRT.m
"""

import numpy as np
from typing import List, Optional, Dict, Any


class StfGeneratorBase:
    """
    Abstract base class for STF generators.
    Port of matRad_StfGeneratorBase.m
    """

    name = "Base STF Generator"
    short_name = "Base"
    possible_radiation_modes = []

    def __init__(self, pln: Optional[dict] = None):
        self.vis_mode = 0
        self.add_margin = True
        self.mult_scen = None
        self.bio_model = None
        self.radiation_mode = None
        self.machine_name = "Generic"
        self.machine = None

        self._pln = None
        self._ct = None
        self._cst = None
        self._vox_target_world_coords = None

        if pln is not None:
            self.assign_properties_from_pln(pln)

    def set_defaults(self):
        """Set default values."""
        pass

    def assign_properties_from_pln(self, pln: dict):
        """Assign properties from pln dict."""
        self._pln = pln
        self.radiation_mode = pln.get("radiationMode", self.radiation_mode)
        self.machine_name = pln.get("machine", "Generic")

        if "multScen" in pln:
            self.mult_scen = pln["multScen"]
        if "bioModel" in pln:
            self.bio_model = pln.get("bioModel", "none")

        # Assign from propStf
        prop_stf = pln.get("propStf", {})
        if "gantryAngles" in prop_stf:
            self.gantry_angles = np.asarray(prop_stf["gantryAngles"]).ravel()
        if "couchAngles" in prop_stf:
            self.couch_angles = np.asarray(prop_stf["couchAngles"]).ravel()
        if "bixelWidth" in prop_stf:
            self.bixel_width = float(prop_stf["bixelWidth"])
        if "isoCenter" in prop_stf and prop_stf["isoCenter"] is not None:
            self.iso_center = np.asarray(prop_stf["isoCenter"])
        if "visMode" in prop_stf:
            self.vis_mode = int(prop_stf["visMode"])
        if "addMargin" in prop_stf:
            self.add_margin = bool(prop_stf["addMargin"])
        if "fillEmptyBixels" in prop_stf:
            self.fill_empty_bixels = bool(prop_stf["fillEmptyBixels"])
        if "centered" in prop_stf:
            self.centered = bool(prop_stf["centered"])

    def generate(self, ct: dict, cst: list) -> list:
        """Generate STF from CT and CST."""
        self._ct = ct
        self._cst = cst
        self._initialize(ct, cst)
        return self._generate_source_geometry()

    def _initialize(self, ct: dict, cst: list):
        """Initialize generator with CT and CST data."""
        from ..geometry import get_world_axes, get_iso_center, add_margin
        from ..config import MatRad_Config

        self._ct = get_world_axes(ct)
        self._cst = cst

        # Get target voxel world coords
        target_voxels = []
        for row in cst:
            if row[2] == "TARGET":
                vox = row[3][0] if isinstance(row[3], list) else row[3]
                target_voxels.append(np.asarray(vox))

        if not target_voxels:
            MatRad_Config.instance().disp_error("No target structure found!")

        V = np.unique(np.concatenate(target_voxels))

        # Optionally add margin
        if self.add_margin:
            # MATLAB's PhotonIMRT uses pbMargin = bixelWidth (5mm).
            # For nominal scenario: margin = max(ct_res, bixelWidth).
            # This gives round(5/3)=2 voxel expansion to match MATLAB.
            pb_margin = getattr(self, 'bixel_width', 0.0)
            max_ct_res = max(ct["resolution"]["x"], ct["resolution"]["y"], ct["resolution"]["z"])
            margin_mm = max(max_ct_res, pb_margin)
            cst_margin = add_margin(cst, ct, margin=margin_mm)
            target_voxels_margin = []
            for row in cst_margin:
                if row[2] == "TARGET":
                    vox = row[3][0] if isinstance(row[3], list) else row[3]
                    target_voxels_margin.append(np.asarray(vox))
            if target_voxels_margin:
                V = np.unique(np.concatenate(target_voxels_margin))

        from ..geometry.geometry import cube_index_to_world_coords
        self._vox_target_world_coords = cube_index_to_world_coords(V, ct)

    def _generate_source_geometry(self) -> list:
        raise NotImplementedError("Subclass must implement _generate_source_geometry")


class StfGeneratorExternalRayBixelAbstract(StfGeneratorBase):
    """
    Abstract base for external beam ray-bixel STF generators.
    Port of matRad_StfGeneratorExternalRayBixelAbstract.m
    """

    def __init__(self, pln: Optional[dict] = None):
        self.gantry_angles = np.array([0.0])
        self.couch_angles = np.array([0.0])
        self.bixel_width = 5.0
        self.iso_center = None
        self.fill_empty_bixels = False
        self.centered = True

        super().__init__(pln)

    @property
    def num_of_beams(self) -> int:
        return len(self.gantry_angles)

    def _initialize(self, ct: dict, cst: list):
        """Initialize and set up isocenter."""
        super()._initialize(ct, cst)

        from ..geometry import get_iso_center
        from ..config import MatRad_Config

        if self.iso_center is None:
            self.iso_center = get_iso_center(cst, ct)
            self.iso_center = np.tile(self.iso_center, (self.num_of_beams, 1))
        else:
            iso = np.atleast_2d(self.iso_center)
            if iso.shape[0] == 1:
                self.iso_center = np.tile(iso, (self.num_of_beams, 1))
            else:
                self.iso_center = iso

    def _get_ray_position_matrix(self, beam: dict) -> np.ndarray:
        """
        Calculate ray positions in BEV for a given beam.

        Port of getRayPositionMatrix in StfGeneratorExternalRayBixelAbstract.m
        """
        from ..geometry.geometry import get_rotation_matrix

        iso_center = beam["isoCenter"]
        gantry_angle = beam["gantryAngle"]
        couch_angle = beam["couchAngle"]
        bixel_width = beam["bixelWidth"]
        SAD = beam["SAD"]

        # Shift target coords to isocenter reference
        iso_coords = self._vox_target_world_coords - iso_center

        # Get rotation matrix (passive/system rotation with row vectors)
        rot_mat_T = get_rotation_matrix(gantry_angle, couch_angle)

        # Rotate target coords to BEV
        coords_bev = iso_coords @ rot_mat_T  # (N, 3)

        # Project to isocenter plane using perspective projection
        # SAD + y_bev = depth from source to point, SAD = depth from source to iso
        proj_denom = SAD + coords_bev[:, 1]
        safe_denom = np.where(np.abs(proj_denom) < 1e-6, 1e-6, proj_denom)
        coords_iso_plane = coords_bev * SAD / safe_denom[:, np.newaxis]
        coords_iso_plane[:, 1] = 0.0  # Force y=0 at isocenter plane

        # Quantize to bixel grid
        ray_pos = np.unique(
            bixel_width * np.round(coords_iso_plane / bixel_width),
            axis=0
        )

        # Pad ray positions if bixel width < CT resolution
        ct_res = self._ct["resolution"]
        max_ct_res = max(ct_res["x"], ct_res["y"], ct_res["z"])
        if bixel_width < max_ct_res:
            pad_range = range(-int(np.floor(max_ct_res / bixel_width)),
                              int(np.floor(max_ct_res / bixel_width)) + 1)
            orig_ray_pos = ray_pos.copy()
            extra_rays = []
            for j in pad_range:
                for k in pad_range:
                    if abs(j) + abs(k) == 0:
                        continue
                    extra = orig_ray_pos.copy()
                    extra[:, 0] += j * bixel_width
                    extra[:, 2] += k * bixel_width
                    extra_rays.append(extra)
            if extra_rays:
                ray_pos = np.vstack([ray_pos] + extra_rays)

        # Fill empty bixels (DAo mode)
        if self.fill_empty_bixels:
            unique_z = np.unique(ray_pos[:, 2])
            extra = []
            for uz in unique_z:
                mask = ray_pos[:, 2] == uz
                x_vals = ray_pos[mask, 0]
                x_min, x_max = x_vals.min(), x_vals.max()
                new_x = np.arange(x_min, x_max + bixel_width * 0.5, bixel_width)
                new_rays = np.column_stack([new_x, np.zeros_like(new_x), np.full_like(new_x, uz)])
                extra.append(new_rays)
            if extra:
                ray_pos = np.vstack([ray_pos] + extra)

        return np.unique(ray_pos, axis=0)

    def _init_beam_data(self, beam_index: int) -> dict:
        """Initialize beam metadata dict."""
        machine = self.machine
        if isinstance(machine, dict):
            meta = machine.get("meta", {})
            # Try 'name' or 'machine' field
            machine_name = meta.get("name", meta.get("machine", self.machine_name))
            SAD = float(meta.get("SAD", 1000.0))
        else:
            machine_name = self.machine_name
            SAD = 1000.0

        beam = {
            "gantryAngle": float(self.gantry_angles[beam_index]),
            "couchAngle": float(self.couch_angles[beam_index]),
            "isoCenter": np.asarray(self.iso_center[beam_index]),
            "bixelWidth": self.bixel_width,
            "radiationMode": self.radiation_mode,
            "machine": machine_name,
            "SAD": SAD,
        }
        return beam

    def _init_rays(self, beam: dict) -> dict:
        """
        Initialize ray geometry for a beam.
        Port of initRays in StfGeneratorExternalRayBixelAbstract.m
        """
        from ..geometry.geometry import get_rotation_matrix

        beam["sourcePoint_bev"] = np.array([0.0, -beam["SAD"], 0.0])

        ray_pos_bev = self._get_ray_position_matrix(beam)
        beam["numOfRays"] = len(ray_pos_bev)

        # Save ray and target positions in BEV
        rays = []
        for j in range(beam["numOfRays"]):
            ray = {
                "rayPos_bev": ray_pos_bev[j],
                "targetPoint_bev": np.array([
                    2.0 * ray_pos_bev[j, 0],
                    beam["SAD"],
                    2.0 * ray_pos_bev[j, 2],
                ]),
            }
            rays.append(ray)

        # Get rotation matrix (transpose for row vector multiplication)
        rot_mat_T = get_rotation_matrix(beam["gantryAngle"], beam["couchAngle"]).T

        # Transform source point to world coords
        beam["sourcePoint"] = beam["sourcePoint_bev"] @ rot_mat_T

        # Transform ray positions to world coords
        for j in range(beam["numOfRays"]):
            rays[j]["rayPos"] = rays[j]["rayPos_bev"] @ rot_mat_T
            rays[j]["targetPoint"] = rays[j]["targetPoint_bev"] @ rot_mat_T

        beam["ray"] = rays
        return beam

    def _set_beamlet_energies(self, beam: dict) -> dict:
        """Set energies for each ray. To be overridden by subclasses."""
        raise NotImplementedError("Subclass must implement _set_beamlet_energies")

    def _finalize_beam(self, beam: dict) -> dict:
        """Finalize beam metadata."""
        n_rays = len(beam["ray"])
        beam["numOfRays"] = n_rays
        beam["numOfBixelsPerRay"] = [len(r.get("energy", [1.0])) for r in beam["ray"]]
        beam["totalNumOfBixels"] = sum(beam["numOfBixelsPerRay"])
        return beam

    def _generate_source_geometry(self) -> list:
        """
        Generate beam geometry for all gantry angles.
        Port of generateSourceGeometry.m
        """
        from ..config import MatRad_Config
        cfg = MatRad_Config.instance()

        # Load machine data
        if self._pln is not None:
            from ..basedata import load_machine
            self.machine = load_machine(self._pln)

        stf = []
        n_beams = self.num_of_beams

        for i in range(n_beams):
            cfg.disp_info(f"\rGenerating STF: beam {i+1}/{n_beams}...")

            beam = self._init_beam_data(i)
            beam = self._init_rays(beam)
            beam = self._set_beamlet_energies(beam)
            beam = self._finalize_beam(beam)
            stf.append(beam)

        cfg.disp_info("\n")
        return stf


class StfGeneratorPhotonRayBixelAbstract(StfGeneratorExternalRayBixelAbstract):
    """
    Abstract base for photon ray-bixel STF generators.
    Port of matRad_StfGeneratorPhotonRayBixelAbstract.m
    """

    possible_radiation_modes = ["photons"]

    def _get_pb_margin(self) -> float:
        """Return pencil beam margin."""
        return self.bixel_width

    def _set_beamlet_energies(self, beam: dict) -> dict:
        """
        Set photon energies for each ray.
        For photons: one energy per ray (monoenergetic).
        """
        machine = self.machine
        if isinstance(machine, dict):
            data = machine.get("data", {})
            # Get nominal energy from machine
            energy = float(data.get("energy", 6.0))
        else:
            energy = 6.0

        for j in range(beam["numOfRays"]):
            beam["ray"][j]["energy"] = np.array([energy])
            beam["ray"][j]["SSD"] = 0.0  # Will be computed during dose calc

        return beam


class StfGeneratorPhotonIMRT(StfGeneratorPhotonRayBixelAbstract):
    """
    Photon IMRT STF generator.
    Port of matRad_StfGeneratorPhotonIMRT.m
    """

    name = "Photon IMRT stf Generator"
    short_name = "PhotonIMRT"
    possible_radiation_modes = ["photons"]

    def _get_pb_margin(self) -> float:
        return self.bixel_width


def generate_stf(ct: dict, cst: list, pln: dict) -> list:
    """
    Generate steering information for treatment planning.

    Python port of matRad_generateStf.m

    Parameters
    ----------
    ct : dict
        CT struct
    cst : list
        Structure set
    pln : dict
        Plan struct with:
        - radiationMode: 'photons', 'protons', or 'carbon'
        - machine: machine name (e.g. 'Generic')
        - propStf.gantryAngles, couchAngles, bixelWidth, etc.

    Returns
    -------
    list
        List of beam dicts (stf)
    """
    radiation_mode = pln.get("radiationMode", "photons")

    # Select appropriate generator
    if radiation_mode == "photons":
        generator = StfGeneratorPhotonIMRT(pln)
    else:
        raise NotImplementedError(f"STF generator for '{radiation_mode}' not implemented yet")

    return generator.generate(ct, cst)
