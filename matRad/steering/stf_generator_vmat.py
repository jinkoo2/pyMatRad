"""
VMAT (Volumetric Modulated Arc Therapy) STF generator.

Python port of matRad_StfGeneratorPhotonVMAT.m

gantryAngles (from pln['propStf']['gantryAngles']) are interpreted as arc
anchor points: the first and last define the arc start/finish; intermediate
values are waypoints the arc must pass through. Two anchor points define a
simple full or partial arc.
"""

import numpy as np
from typing import Optional
from .stf_generator import StfGeneratorPhotonRayBixelAbstract


class StfGeneratorPhotonVMAT(StfGeneratorPhotonRayBixelAbstract):
    """
    Photon VMAT arc STF generator.
    Port of matRad_StfGeneratorPhotonVMAT.m

    Three angle hierarchies are computed from the user-specified arc anchors:
      - Fine (dose-calc) angles  : one beam per maxGantryAngleSpacing degrees
      - DAO angles                : subset at maxDAOGantryAngleSpacing
      - FMO angles                : subset at maxFMOGantryAngleSpacing (odd multiple of DAO)

    Each beam in the returned stf list has a 'propVMAT' dict with the beam's
    role (FMOBeam / DAOBeam), angle-sector borders, parent/child relationships,
    and time-fraction metadata needed for arc sequencing and DAO optimisation.
    """

    name = "Photon VMAT stf Generator"
    short_name = "PhotonVMAT"
    possible_radiation_modes = ["photons"]

    def __init__(self, pln: Optional[dict] = None):
        # VMAT angle-spacing properties (defaults match MATLAB)
        self.max_gantry_angle_spacing = 4.0
        self.max_dao_gantry_angle_spacing = 8.0
        self.max_fmo_gantry_angle_spacing = 32.0
        self.continuous_aperture = False
        self.arc_index = None  # scalar or per-anchor array; None → all-ones

        # Internal arc angle arrays (populated by _setup_arc_angles)
        self._arc_gantry_angles = None
        self._arc_couch_angles = None
        self._arc_dao_angles = None
        self._arc_fmo_angles = None
        self._arc_start_angle = None
        self._arc_finish_angle = None

        # Anchor state saved during _initialize and restored afterwards
        self._saved_anchor_gantry_angles = None
        self._saved_anchor_couch_angles = None
        self._saved_iso_center = None

        # Shared between _prepare_arcs and _finalize_arcs
        self._dao_dose_angle_borders = None

        super().__init__(pln)

    # ------------------------------------------------------------------
    # Property assignment from pln
    # ------------------------------------------------------------------

    def assign_properties_from_pln(self, pln: dict):
        super().assign_properties_from_pln(pln)
        prop_stf = pln.get("propStf", {})
        if "maxGantryAngleSpacing" in prop_stf:
            self.max_gantry_angle_spacing = float(prop_stf["maxGantryAngleSpacing"])
        if "maxDAOGantryAngleSpacing" in prop_stf:
            self.max_dao_gantry_angle_spacing = float(prop_stf["maxDAOGantryAngleSpacing"])
        if "maxFMOGantryAngleSpacing" in prop_stf:
            self.max_fmo_gantry_angle_spacing = float(prop_stf["maxFMOGantryAngleSpacing"])
        if "continuousAperture" in prop_stf:
            self.continuous_aperture = bool(prop_stf["continuousAperture"])
        if "arcIndex" in prop_stf:
            self.arc_index = prop_stf["arcIndex"]

    # ------------------------------------------------------------------
    # Arc angle setup
    # ------------------------------------------------------------------

    def _setup_arc_angles(self):
        """
        Compute fine / DAO / FMO angle arrays from user-specified anchor points.
        Port of setupArcAngles() (lines 143-237 of matRad_StfGeneratorPhotonVMAT.m).
        """
        anchor_gantry = np.asarray(self.gantry_angles, dtype=float).ravel()
        anchor_couch = np.asarray(self.couch_angles, dtype=float).ravel()
        n_anchors = len(anchor_gantry)

        # Broadcast scalar arc_index to per-anchor vector
        if self.arc_index is None or np.isscalar(self.arc_index):
            idx_val = 1 if self.arc_index is None else int(self.arc_index)
            arc_idx = np.full(n_anchors, idx_val, dtype=int)
        else:
            arc_idx = np.asarray(self.arc_index, dtype=int).ravel()

        # Unique arc IDs in stable order
        arc_ids = list(dict.fromkeys(arc_idx.tolist()))

        all_gantry, all_couch, all_dao, all_fmo = [], [], [], []

        for arc_id in arc_ids:
            mask = arc_idx == arc_id
            anchors = anchor_gantry[mask]
            couch = anchor_couch[mask]

            start_angle = float(anchors[0])
            finish_angle = float(anchors[-1])
            couch_val = float(couch[0])
            angular_range = abs(finish_angle - start_angle)

            if self.continuous_aperture:
                # Continuous mode: beams centred half-spacing inside arc boundaries
                num_fine = max(1, int(np.ceil(angular_range / self.max_gantry_angle_spacing)))
                fine_spacing = angular_range / num_fine

                num_dao = max(2, int(np.ceil((num_fine - 1) * fine_spacing / self.max_dao_gantry_angle_spacing)) + 1)
                # Realign num_fine so every DAO angle lands on a fine angle
                ratio = int(np.ceil((num_fine - 1) / (num_dao - 1)))
                num_fine = (num_dao - 1) * ratio + 1
                fine_spacing = angular_range / num_fine
                dao_spacing = (angular_range - fine_spacing) / (num_dao - 1)

                first_fine = start_angle + fine_spacing / 2.0
                last_fine = finish_angle - fine_spacing / 2.0
            else:
                # Step-and-shoot: first/last beams sit at arc boundaries
                num_dao = max(1, int(np.ceil(angular_range / self.max_dao_gantry_angle_spacing)))
                dao_spacing = angular_range / num_dao
                num_fine = max(1, int(np.ceil(num_dao * dao_spacing / self.max_gantry_angle_spacing)))
                fine_spacing = angular_range / num_fine

                first_fine = start_angle
                last_fine = finish_angle

            # FMO spacing = largest odd-integer multiple of DAO spacing ≤ maxFMO
            num_apertures = max(1, int(np.floor(self.max_fmo_gantry_angle_spacing / dao_spacing)))
            if num_apertures % 2 == 0:
                num_apertures -= 1
            if num_apertures < 1:
                num_apertures = 1
            fmo_spacing = num_apertures * dao_spacing
            first_fmo = first_fine + dao_spacing * (num_apertures // 2)
            last_fmo = last_fine - dao_spacing * (num_apertures // 2)

            # Generate arrays with linspace for numerical stability
            n_fine_pts = int(round((last_fine - first_fine) / fine_spacing)) + 1
            fine_angles = np.linspace(first_fine, last_fine, n_fine_pts)

            n_dao_pts = int(round((last_fine - first_fine) / dao_spacing)) + 1
            dao_angles = np.linspace(first_fine, last_fine, n_dao_pts)

            if last_fmo >= first_fmo - 1e-9:
                n_fmo_pts = max(1, int(round((last_fmo - first_fmo) / fmo_spacing)) + 1)
                fmo_angles = np.linspace(first_fmo, last_fmo, n_fmo_pts)
            else:
                fmo_angles = np.array([first_fine])

            all_gantry.append(fine_angles)
            all_couch.append(np.full(len(fine_angles), couch_val))
            all_dao.append(dao_angles)
            all_fmo.append(fmo_angles)

        self._arc_gantry_angles = np.concatenate(all_gantry)
        self._arc_couch_angles = np.concatenate(all_couch)
        self._arc_dao_angles = np.concatenate(all_dao)
        self._arc_fmo_angles = np.concatenate(all_fmo)

        # Arc extent boundaries (first anchor of first arc / last anchor of last arc)
        first_arc_anchors = anchor_gantry[arc_idx == arc_ids[0]]
        last_arc_anchors = anchor_gantry[arc_idx == arc_ids[-1]]
        self._arc_start_angle = float(first_arc_anchors[0])
        self._arc_finish_angle = float(last_arc_anchors[-1])

    # ------------------------------------------------------------------
    # Override _initialize: expand anchors → fine grid before parent
    # ------------------------------------------------------------------

    def _initialize(self, ct: dict, cst: list):
        """
        Expand anchor angles to fine grid, then run parent initialisation.
        Port of initialize() (lines 88-137 of matRad_StfGeneratorPhotonVMAT.m).
        """
        # Save user-supplied anchor state for restoration after geometry build
        self._saved_anchor_gantry_angles = np.asarray(self.gantry_angles, dtype=float).copy()
        self._saved_anchor_couch_angles = np.asarray(self.couch_angles, dtype=float).copy()
        self._saved_iso_center = (
            np.asarray(self.iso_center).copy() if self.iso_center is not None else None
        )
        n_anchors = len(self._saved_anchor_gantry_angles)

        # Compute fine / DAO / FMO angles from anchor points
        self._setup_arc_angles()
        n_fine = len(self._arc_gantry_angles)

        # Expand isoCenter to [n_fine × 3] so parent sees correct beam count
        if self.iso_center is not None:
            iso = np.atleast_2d(np.asarray(self.iso_center, dtype=float))
            if iso.shape[0] == 1:
                self.iso_center = np.tile(iso, (n_fine, 1))
            elif iso.shape[0] == n_anchors:
                iso_full = np.zeros((n_fine, 3))
                for k in range(n_fine):
                    ia = int(np.argmin(np.abs(self._saved_anchor_gantry_angles - self._arc_gantry_angles[k])))
                    iso_full[k] = iso[ia]
                self.iso_center = iso_full
            # Otherwise leave untouched; parent will validate

        # Swap angles to fine grid so parent initialises with the correct count
        self.gantry_angles = self._arc_gantry_angles
        self.couch_angles = self._arc_couch_angles

        # Parent: loads machine, validates/computes isoCenter, builds geometry
        super()._initialize(ct, cst)

    # ------------------------------------------------------------------
    # Override _generate_source_geometry
    # ------------------------------------------------------------------

    def _generate_source_geometry(self) -> list:
        """
        Build arc STF: per-beam geometry + master ray set + propVMAT metadata.
        Port of generateSourceGeometry() (lines 239-290 of matRad_StfGeneratorPhotonVMAT.m).
        """
        from ..config import MatRad_Config
        cfg = MatRad_Config.instance()

        # Build per-beam geometry using the parent (fine angles already active)
        stf = super()._generate_source_geometry()

        cfg.disp_info("Apply VMAT configuration to stf...\n")

        # ----------------------------------------------------------------
        # Build master ray set: union of all per-beam rayPos_bev, gap-filled
        # so that every leaf row has a contiguous set of bixel columns.
        # Port of lines 248-272 in generateSourceGeometry().
        # ----------------------------------------------------------------
        master_ray_pos_bev = np.zeros((0, 3))
        for beam in stf:
            if beam["numOfRays"] > 0:
                ray_pos = np.array([r["rayPos_bev"] for r in beam["ray"]])
                combined = np.vstack([master_ray_pos_bev, ray_pos])
                master_ray_pos_bev = np.unique(combined, axis=0)

        if len(master_ray_pos_bev) > 0:
            x_all = master_ray_pos_bev[:, 0]
            z_all = master_ray_pos_bev[:, 2]
            uni_z = np.unique(z_all)
            extra_rows = []
            for uz in uni_z:
                mask = z_all == uz
                x_loc = x_all[mask]
                x_min, x_max = x_loc.min(), x_loc.max()
                if x_max > x_min:
                    n_new = int(round((x_max - x_min) / self.bixel_width)) + 1
                    new_x = np.linspace(x_min, x_max, n_new)
                    extra_rows.append(
                        np.column_stack([new_x, np.zeros(n_new), np.full(n_new, uz)])
                    )
            if extra_rows:
                master_ray_pos_bev = np.unique(
                    np.vstack([master_ray_pos_bev] + extra_rows), axis=0
                )

        machine = self.machine
        SAD = float(machine.get("meta", {}).get("SAD", 1000.0)) if isinstance(machine, dict) else 1000.0

        master_target_point_bev = np.column_stack([
            2.0 * master_ray_pos_bev[:, 0],
            np.full(len(master_ray_pos_bev), SAD),
            2.0 * master_ray_pos_bev[:, 2],
        ])

        # ----------------------------------------------------------------
        # Pass 1: assign propVMAT fields to every beam
        # ----------------------------------------------------------------
        cfg.disp_info("VMAT stf beam type and geometry setup...\n")
        stf = self._prepare_arcs(stf, master_ray_pos_bev, master_target_point_bev)

        # ----------------------------------------------------------------
        # Pass 2: derived quantities that need the complete propVMAT data
        # ----------------------------------------------------------------
        cfg.disp_info("VMAT stf cleanup...\n")
        stf = self._finalize_arcs(stf)

        # Restore object state to user-specified anchor configuration
        self.gantry_angles = self._saved_anchor_gantry_angles
        self.couch_angles = self._saved_anchor_couch_angles
        self.iso_center = self._saved_iso_center

        return stf

    # ------------------------------------------------------------------
    # Pass 1: prepareArcs
    # ------------------------------------------------------------------

    def _prepare_arcs(
        self,
        stf: list,
        master_ray_pos_bev: np.ndarray,
        master_target_point_bev: np.ndarray,
    ) -> list:
        """
        Assign propVMAT metadata to every beam in stf.
        Port of prepareArcs() (lines 292-464 of matRad_StfGeneratorPhotonVMAT.m).
        """
        from ..geometry.geometry import get_rotation_matrix

        n_beams = len(stf)
        num_dao_counter = 1

        # Carry-forward state tracking the bracketing DAO beams for non-DAO beams
        last_dao_index = 0
        next_dao_index = 0

        dao_dose_angle_borders: list = []

        machine = self.machine
        SAD = float(machine.get("meta", {}).get("SAD", 1000.0)) if isinstance(machine, dict) else 1000.0
        energy = float(machine.get("data", {}).get("energy", 6.0)) if isinstance(machine, dict) else 6.0

        fmo_angles = self._arc_fmo_angles
        dao_angles = self._arc_dao_angles

        # Pre-build gantry angle array for fast index lookups
        all_ga = np.array([b["gantryAngle"] for b in stf])

        for i, beam in enumerate(stf):
            ga = float(beam["gantryAngle"])
            vmat: dict = {}

            # ---- FMO parent ----
            fmo_diffs = np.abs(fmo_angles - ga)
            fmo_parent_fmo_idx = int(np.argmin(fmo_diffs))
            vmat["beamParentFMOIndex"] = fmo_parent_fmo_idx
            vmat["beamParentGantryAngle"] = float(fmo_angles[fmo_parent_fmo_idx])
            fmo_parent_stf_idx = int(np.argmin(np.abs(all_ga - vmat["beamParentGantryAngle"])))
            vmat["beamParentIndex"] = fmo_parent_stf_idx

            vmat["FMOBeam"] = bool(np.any(fmo_diffs < 1e-6))
            vmat["DAOBeam"] = bool(np.any(np.abs(dao_angles - ga) < 1e-6))

            # ---- Dose angle borders ----
            if i == 0:
                vmat["doseAngleBorders"] = [
                    self._arc_start_angle,
                    (stf[1]["gantryAngle"] + ga) / 2.0,
                ]
            elif i == n_beams - 1:
                vmat["doseAngleBorders"] = [
                    (stf[i - 1]["gantryAngle"] + ga) / 2.0,
                    self._arc_finish_angle,
                ]
            else:
                vmat["doseAngleBorders"] = [
                    (stf[i - 1]["gantryAngle"] + ga) / 2.0,
                    (stf[i + 1]["gantryAngle"] + ga) / 2.0,
                ]
            vmat["doseAngleBorderCentreDiff"] = [
                ga - vmat["doseAngleBorders"][0],
                vmat["doseAngleBorders"][1] - ga,
            ]
            vmat["doseAngleBordersDiff"] = sum(vmat["doseAngleBorderCentreDiff"])

            # ---- DAO-beam branch ----
            if vmat["DAOBeam"]:
                dao_dose_angle_borders.extend(vmat["doseAngleBorders"])

                # Register as child of FMO parent (forward/backward reference safe)
                parent_vmat = stf[fmo_parent_stf_idx].setdefault("propVMAT", {})
                if "beamChildrenGantryAngles" not in parent_vmat:
                    parent_vmat["numOfBeamChildren"] = 0
                    parent_vmat["beamChildrenGantryAngles"] = []
                    parent_vmat["beamChildrenIndex"] = []
                parent_vmat["numOfBeamChildren"] += 1
                parent_vmat["beamChildrenGantryAngles"].append(ga)
                parent_vmat["beamChildrenIndex"].append(i)

                # Locate this beam in the DAO angle array
                dao_match = np.where(np.abs(dao_angles - ga) < 1e-8)[0]
                dao_idx = int(dao_match[0])
                n_dao_total = len(dao_angles)

                if dao_idx == 0:
                    vmat["DAOAngleBorders"] = [
                        self._arc_start_angle,
                        (dao_angles[dao_idx + 1] + dao_angles[dao_idx]) / 2.0,
                    ]
                    last_dao_index = i
                    nxt = np.where(np.abs(all_ga - dao_angles[dao_idx + 1]) < 1e-8)[0]
                    next_dao_index = int(nxt[0])
                elif dao_idx == n_dao_total - 1:
                    vmat["DAOAngleBorders"] = [
                        (dao_angles[dao_idx - 1] + dao_angles[dao_idx]) / 2.0,
                        self._arc_finish_angle,
                    ]
                    prv = np.where(np.abs(all_ga - dao_angles[dao_idx - 1]) < 1e-8)[0]
                    last_dao_index = int(prv[0])
                    next_dao_index = i
                else:
                    vmat["DAOAngleBorders"] = [
                        (dao_angles[dao_idx - 1] + dao_angles[dao_idx]) / 2.0,
                        (dao_angles[dao_idx + 1] + dao_angles[dao_idx]) / 2.0,
                    ]
                    last_dao_index = i
                    nxt = np.where(np.abs(all_ga - dao_angles[dao_idx + 1]) < 1e-8)[0]
                    next_dao_index = int(nxt[0])

                vmat["lastDAOIndex"] = last_dao_index
                vmat["nextDAOIndex"] = next_dao_index
                vmat["DAOIndex"] = num_dao_counter
                num_dao_counter += 1

                vmat["DAOAngleBorderCentreDiff"] = [
                    ga - vmat["DAOAngleBorders"][0],
                    vmat["DAOAngleBorders"][1] - ga,
                ]
                vmat["DAOAngleBordersDiff"] = sum(vmat["DAOAngleBorderCentreDiff"])

                # Time factor: fraction of DAO sector time covered by this dose sector
                vmat["timeFacCurr"] = vmat["doseAngleBordersDiff"] / vmat["DAOAngleBordersDiff"]

                if self.continuous_aperture:
                    vmat["timeFac"] = [
                        (vmat["DAOAngleBorderCentreDiff"][0] - vmat["doseAngleBorderCentreDiff"][0])
                        / vmat["DAOAngleBordersDiff"],
                        vmat["timeFacCurr"],
                        (vmat["DAOAngleBorderCentreDiff"][1] - vmat["doseAngleBorderCentreDiff"][1])
                        / vmat["DAOAngleBordersDiff"],
                    ]
                else:
                    vmat["timeFac"] = [
                        vmat["DAOAngleBorderCentreDiff"][0] / vmat["DAOAngleBordersDiff"],
                        vmat["DAOAngleBorderCentreDiff"][1] / vmat["DAOAngleBordersDiff"],
                    ]

            # ---- Non-DAO beam branch ----
            else:
                parent_vmat = stf[fmo_parent_stf_idx].setdefault("propVMAT", {})
                if "beamSubChildrenGantryAngles" not in parent_vmat:
                    parent_vmat["numOfBeamSubChildren"] = 0
                    parent_vmat["beamSubChildrenGantryAngles"] = []
                    parent_vmat["beamSubChildrenIndex"] = []
                parent_vmat["numOfBeamSubChildren"] += 1
                parent_vmat["beamSubChildrenGantryAngles"].append(ga)
                parent_vmat["beamSubChildrenIndex"].append(i)

                next_dao_ga = stf[next_dao_index]["gantryAngle"]
                last_dao_ga = stf[last_dao_index]["gantryAngle"]
                denom = next_dao_ga - last_dao_ga
                vmat["fracFromLastDAO"] = (next_dao_ga - ga) / denom if abs(denom) > 1e-12 else 0.5
                vmat["lastDAOIndex"] = last_dao_index
                vmat["nextDAOIndex"] = next_dao_index

            # ---- FMO beam: FMO angle borders ----
            if vmat["FMOBeam"]:
                fmo_idx = int(np.argmin(np.abs(fmo_angles - ga)))
                n_fmo = len(fmo_angles)
                if n_fmo == 1:
                    vmat["FMOAngleBorders"] = [self._arc_start_angle, self._arc_finish_angle]
                elif fmo_idx == 0:
                    vmat["FMOAngleBorders"] = [
                        self._arc_start_angle,
                        (fmo_angles[fmo_idx + 1] + fmo_angles[fmo_idx]) / 2.0,
                    ]
                elif fmo_idx == n_fmo - 1:
                    vmat["FMOAngleBorders"] = [
                        (fmo_angles[fmo_idx - 1] + fmo_angles[fmo_idx]) / 2.0,
                        self._arc_finish_angle,
                    ]
                else:
                    vmat["FMOAngleBorders"] = [
                        (fmo_angles[fmo_idx - 1] + fmo_angles[fmo_idx]) / 2.0,
                        (fmo_angles[fmo_idx + 1] + fmo_angles[fmo_idx]) / 2.0,
                    ]
                vmat["FMOAngleBorderCentreDiff"] = [
                    ga - vmat["FMOAngleBorders"][0],
                    vmat["FMOAngleBorders"][1] - ga,
                ]
                vmat["FMOAngleBordersDiff"] = sum(vmat["FMOAngleBorderCentreDiff"])

            # ---- Assign master ray set to this beam (replaces per-beam rays) ----
            beam["numOfRays"] = len(master_ray_pos_bev)
            beam["numOfBixelsPerRay"] = [1] * beam["numOfRays"]
            beam["totalNumOfBixels"] = beam["numOfRays"]

            beam["sourcePoint_bev"] = np.array([0.0, -SAD, 0.0])
            rot_mat_T = get_rotation_matrix(beam["gantryAngle"], beam["couchAngle"]).T
            beam["sourcePoint"] = beam["sourcePoint_bev"] @ rot_mat_T

            rays = []
            for j in range(beam["numOfRays"]):
                rp_bev = master_ray_pos_bev[j]
                tp_bev = master_target_point_bev[j]
                rays.append({
                    "rayPos_bev": rp_bev,
                    "targetPoint_bev": tp_bev,
                    "rayPos": rp_bev @ rot_mat_T,
                    "targetPoint": tp_bev @ rot_mat_T,
                    "energy": np.array([energy]),
                    "SSD": 0.0,
                })
            beam["ray"] = rays

            # Merge with any propVMAT already written by forward-reference
            existing = beam.get("propVMAT", {})
            existing.update(vmat)
            beam["propVMAT"] = existing

        self._dao_dose_angle_borders = dao_dose_angle_borders
        return stf

    # ------------------------------------------------------------------
    # Pass 2: finalizeArcs
    # ------------------------------------------------------------------

    def _finalize_arcs(self, stf: list) -> list:
        """
        Compute derived interpolation fractions and clean up child lists.
        Port of finalizeArcs() (lines 466-514 of matRad_StfGeneratorPhotonVMAT.m).
        """
        dao_borders = self._dao_dose_angle_borders or []

        for beam in stf:
            vmat = beam.get("propVMAT", {})

            # ---- FMO beam: ensure child / sub-child lists exist ----
            if vmat.get("FMOBeam", False):
                if "beamChildrenGantryAngles" not in vmat:
                    vmat["numOfBeamChildren"] = 0
                    vmat["beamChildrenGantryAngles"] = []
                    vmat["beamChildrenIndex"] = []
                if "beamSubChildrenGantryAngles" not in vmat:
                    vmat["numOfBeamSubChildren"] = 0
                    vmat["beamSubChildrenGantryAngles"] = []
                    vmat["beamSubChildrenIndex"] = []

            # ---- DAO beam + continuous aperture: shared-border flag ----
            if vmat.get("DAOBeam", False) and self.continuous_aperture:
                dose_border_2 = vmat["doseAngleBorders"][1]
                count = sum(1 for b in dao_borders if abs(b - dose_border_2) < 1e-8)
                vmat["doseAngleDAO"] = [1, 0 if count > 1 else 1]

            # ---- Interpolated beam: leaf position and time fractions ----
            if not vmat.get("FMOBeam", False) and not vmat.get("DAOBeam", False):
                last_idx = vmat["lastDAOIndex"]
                next_idx = vmat["nextDAOIndex"]

                last_border = stf[last_idx]["propVMAT"]["doseAngleBorders"][1]
                next_border = stf[next_idx]["propVMAT"]["doseAngleBorders"][0]
                span = next_border - last_border

                d1 = vmat["doseAngleBorders"][0]
                d2 = vmat["doseAngleBorders"][1]

                if abs(span) > 1e-12:
                    vmat["fracFromLastDAO_I"] = (next_border - d1) / span
                    vmat["fracFromLastDAO_F"] = (next_border - d2) / span
                    vmat["fracFromNextDAO_I"] = (d1 - last_border) / span
                    vmat["fracFromNextDAO_F"] = (d2 - last_border) / span
                else:
                    vmat["fracFromLastDAO_I"] = 0.5
                    vmat["fracFromLastDAO_F"] = 0.5
                    vmat["fracFromNextDAO_I"] = 0.5
                    vmat["fracFromNextDAO_F"] = 0.5

                # Time interpolation fractions (clamped to [0, 1])
                last_dao_border_2 = stf[last_idx]["propVMAT"]["DAOAngleBorders"][1]
                ddiff = vmat["doseAngleBordersDiff"]
                vmat["timeFracFromLastDAO"] = float(
                    np.clip((last_dao_border_2 - d1) / ddiff if abs(ddiff) > 1e-12 else 0.5, 0.0, 1.0)
                )
                vmat["timeFracFromNextDAO"] = float(
                    np.clip((d2 - last_dao_border_2) / ddiff if abs(ddiff) > 1e-12 else 0.5, 0.0, 1.0)
                )

        return stf
