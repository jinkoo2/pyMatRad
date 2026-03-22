"""
Photon pencil beam SVD dose calculation engine.

Python port of matRad_PhotonPencilBeamSVDEngine.m

References:
    [1] Scholz 1994 PMB (PMID: 8497215) - SVD-based photon kernel decomposition
"""

import numpy as np
import scipy.sparse as sp
from scipy.fft import fft2, ifft2
from scipy.interpolate import RegularGridInterpolator
from typing import List, Optional, Dict, Any

from .dose_engine_base import DoseEngineBase
from ...config import MatRad_Config
from ...geometry import get_world_axes
from ...geometry.geometry import get_rotation_matrix, world_to_cube_coords
from ...rayTracing.dispatch import siddon_ray_tracer


class PhotonPencilBeamSVDEngine(DoseEngineBase):
    """
    Photon pencil beam dose calculation with SVD kernel decomposition.

    Port of matRad_PhotonPencilBeamSVDEngine.m
    """

    name = "SVD Pencil Beam"
    short_name = "SVDPB"
    possible_radiation_modes = ["photons"]

    def __init__(self, pln: Optional[dict] = None):
        super().__init__(pln)

        cfg = MatRad_Config.instance()

        # SVD-specific properties
        self.use_custom_primary_photon_fluence = False
        self.kernel_cutoff = 20.0        # mm
        self.random_seed = 0
        self.int_conv_resolution = 0.5   # mm
        self.is_field_based_dose_calc = False
        self.enable_dij_sampling = True
        self.ignore_invalid_values = False

        # DIJ sampling defaults
        self.dij_sampling = {
            "relDoseThreshold": 0.01,
            "latCutOff": 20.0,           # mm
            "type": "radius",
            "deltaRadDepth": 5.0,
        }

        # Calculated during init
        self._penumbra_fwhm = 5.0
        self._field_width = 5.0
        self._kernel_conv_size = None
        self._kernel_x = None
        self._kernel_z = None
        self._kernel_mxs = None
        self._gauss_filter = None
        self._gauss_conv_size = None
        self._conv_mx_x = None
        self._conv_mx_z = None
        self._F_x = None
        self._F_z = None
        self._Fpre = None
        self._interp_kernel_cache = None

    def _init_dose_calc(self, ct: dict, cst: list, stf: list) -> dict:
        """Initialize photon SVD dose calculation."""
        cfg = MatRad_Config.instance()

        # First do parent initialization
        dij = super()._init_dose_calc(ct, cst, stf)

        # Load machine
        pln = self._pln or {}
        from ...basedata import load_machine
        self.machine = load_machine(pln)
        machine_data = self.machine.get("data", {})
        machine_meta = self.machine.get("meta", {})

        # Get penumbra FWHM
        if "penumbraFWHMatIso" in machine_data:
            self._penumbra_fwhm = float(machine_data["penumbraFWHMatIso"])
        else:
            self._penumbra_fwhm = 5.0
            cfg.disp_warning("Machine file does not contain penumbraFWHMatIso. Using 5mm.")

        # Correct kernel cutoff
        kernel_pos = machine_data.get("kernelPos", np.array([0, 200]))
        if isinstance(kernel_pos, (list, np.ndarray)):
            kernel_pos = np.asarray(kernel_pos).ravel()
            max_kernel = float(kernel_pos[-1])
        else:
            max_kernel = 200.0

        if self.kernel_cutoff > max_kernel:
            cfg.disp_warning(f"Kernel cutoff {self.kernel_cutoff}mm > machine range {max_kernel}mm. Using {max_kernel}mm.")
            self.kernel_cutoff = max_kernel

        if self.kernel_cutoff < self.geometric_lateral_cutoff:
            self.kernel_cutoff = self.geometric_lateral_cutoff

        # Check if field-based dose calc
        bixel_widths = [b.get("bixelWidth", 5) for b in stf]
        self.is_field_based_dose_calc = any(str(bw) == "field" for bw in bixel_widths)

        # Set field width
        if not self.is_field_based_dose_calc:
            unique_bw = list(set(bixel_widths))
            if len(unique_bw) > 1:
                cfg.disp_warning("Different bixelWidths detected. Using first one.")
            self._field_width = float(unique_bw[0])

        # Setup convolution grids
        self._setup_convolution_grids()

        # Pre-compute fluence if uniform
        if not self.is_field_based_dose_calc:
            field_limit = int(np.ceil(self._field_width / (2 * self.int_conv_resolution)))
            field_size = int(self._field_width / self.int_conv_resolution)
            self._Fpre = np.ones((field_size, field_size))

            if not self.use_custom_primary_photon_fluence:
                # Gaussian convolution for penumbra
                s = (self._gauss_conv_size, self._gauss_conv_size)
                self._Fpre = np.real(ifft2(
                    fft2(self._Fpre, s=s) *
                    fft2(self._gauss_filter, s=s)
                ))

        # Set effective lateral cutoff
        self._effective_lateral_cutoff = (
            self.geometric_lateral_cutoff + self._field_width / np.sqrt(2)
        )

        # Set random seed
        np.random.seed(self.random_seed)

        # Convert CT to water equivalent density
        ct = self._calc_water_eq_density(ct, stf)
        self._cube_wed = ct.get("cube", ct.get("cubeHU"))
        self._apply_outside_density_mask()

        # Preallocate dij sparse matrix
        n_voxels = dij["doseGrid"]["numOfVoxels"]
        n_bixels = dij["totalNumOfBixels"]
        dij["physicalDose"] = [sp.lil_matrix((n_voxels, n_bixels))]

        return dij

    def _setup_convolution_grids(self):
        """Set up kernel convolution grids."""
        sigma_gauss = self._penumbra_fwhm / np.sqrt(8 * np.log(2))  # mm
        field_limit = int(np.ceil(self._field_width / (2 * self.int_conv_resolution)))
        gauss_limit = int(np.ceil(5 * sigma_gauss / self.int_conv_resolution))
        kernel_limit = int(np.ceil(self.kernel_cutoff / self.int_conv_resolution))

        res = self.int_conv_resolution

        # Field grid
        x_f = np.arange(-field_limit * res, field_limit * res, res)
        self._F_x, self._F_z = np.meshgrid(x_f, x_f)

        # Gaussian filter
        gx = np.arange(-gauss_limit * res, gauss_limit * res, res)
        GX, GZ = np.meshgrid(gx, gx)
        self._gauss_filter = (
            1 / (2 * np.pi * sigma_gauss ** 2 / res ** 2) *
            np.exp(-(GX ** 2 + GZ ** 2) / (2 * sigma_gauss ** 2))
        )
        self._gauss_conv_size = 2 * (field_limit + gauss_limit)

        # Kernel grid
        kx = np.arange(-kernel_limit * res, kernel_limit * res, res)
        self._kernel_x, self._kernel_z = np.meshgrid(kx, kx)

        # Convolution output grid
        kernel_conv_limit = field_limit + gauss_limit + kernel_limit
        cx = np.arange(-kernel_conv_limit * res, kernel_conv_limit * res, res)
        self._conv_mx_x, self._conv_mx_z = np.meshgrid(cx, cx)
        self._kernel_conv_size = 2 * kernel_conv_limit

    def _apply_outside_density_mask(self):
        """
        Zero out density for voxels outside all CST structures.

        Port of ignoreOutsideDensities logic in matRad_PencilBeamEngineAbstract.m:
            eraseCtDensMask = ones(prod(ct.cubeDim),1);
            eraseCtDensMask(this.VctGrid) = 0;
            this.cubeWED{i}(eraseCtDensMask == 1) = 0;
        """
        if not self.ignore_outside_densities:
            return
        if self._V_ct_grid is None or self._cube_wed is None:
            return

        for i, cube in enumerate(self._cube_wed):
            n_total = cube.size
            erase_mask = np.ones(n_total, dtype=bool)
            erase_mask[self._V_ct_grid - 1] = False  # V_ct_grid is 1-based Fortran-order
            flat = cube.ravel(order='F').copy()
            flat[erase_mask] = 0.0
            self._cube_wed[i] = flat.reshape(cube.shape, order='F')

    def _calc_water_eq_density(self, ct: dict, stf: list) -> dict:
        """
        Convert HU to relative electron density (water equivalent density).

        Port of matRad_calcWaterEqD / matRad_calcHU
        """
        ct = get_world_axes(ct)

        # If cube already exists (pre-computed), use it — ensure it's a list
        if "cube" in ct and ct["cube"] is not None:
            if not isinstance(ct["cube"], list):
                ct["cube"] = [np.asarray(ct["cube"])]
            return ct

        # Convert HU to relative electron density
        # Simple linear conversion: RED = 1 + HU/1000 (approx for water-like materials)
        # More accurate: use Hounsfield lookup table (HLUT) if available
        num_scen = ct.get("numOfCtScen", 1)
        cube_hu_list = ct.get("cubeHU", [])

        ct["cube"] = []
        for s in range(num_scen):
            if s < len(cube_hu_list):
                hu = cube_hu_list[s].astype(float)
            else:
                hu = cube_hu_list[0].astype(float)

            # Basic HU to relative electron density conversion
            # Air: HU = -1000 -> RED = 0
            # Water: HU = 0 -> RED = 1
            # Bone: HU = 1000 -> RED = 1.8 (approx)
            red = np.where(hu <= -1000, 0.0, 1.0 + hu / 1000.0)
            red = np.clip(red, 0.0, 3.0)
            ct["cube"].append(red)

        return ct

    def _get_kernel_interpolators(self, Fx: np.ndarray):
        """
        Create kernel interpolators by convolving fluence with kernels.

        Port of getKernelInterpolators in PhotonPencilBeamSVDEngine.
        """
        machine_data = self.machine.get("data", {})
        kernel_mxs = self._kernel_mxs
        n_kernels = len(kernel_mxs)
        interp_kernels = []

        for ik in range(n_kernels):
            # 2D FFT convolution of fluence and kernel
            s = (self._kernel_conv_size, self._kernel_conv_size)
            conv_mx = np.real(ifft2(
                fft2(Fx, s=s) *
                fft2(kernel_mxs[ik], s=s)
            ))

            # Create 2D interpolator
            conv_x = self._conv_mx_x[0, :]  # x-values (columns)
            conv_z = self._conv_mx_z[:, 0]  # z-values (rows)

            # Trim to valid conv_mx size
            Nx_k = self._kernel_conv_size
            Nz_k = self._kernel_conv_size
            conv_mx_trimmed = conv_mx[:Nz_k, :Nx_k]
            conv_x_trimmed = conv_x[:Nx_k]
            conv_z_trimmed = conv_z[:Nz_k]

            interp = RegularGridInterpolator(
                (conv_z_trimmed, conv_x_trimmed),
                conv_mx_trimmed,
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            interp_kernels.append(interp)

        return interp_kernels

    def _init_beam(self, ct: dict, cst: list, stf: list, beam_idx: int, dij: dict) -> dict:
        """Initialize a single beam for dose calculation."""
        cfg = MatRad_Config.instance()
        beam = stf[beam_idx]

        cfg.disp_info(f"\n  Beam {beam_idx+1}/{len(stf)}: gantry={beam['gantryAngle']}°, couch={beam['couchAngle']}°\n")

        machine_data = self.machine.get("data", {})
        machine_meta = self.machine.get("meta", {})

        # Get center ray SSD for kernel selection
        rays = beam["ray"]
        ray_pos_bev = np.array([r["rayPos_bev"] for r in rays])
        center_idx = int(np.argmin(np.sum(ray_pos_bev ** 2, axis=1)))
        center_ssd = float(rays[center_idx].get("SSD", beam["SAD"]))

        # Get kernel data (find closest SSD)
        kernels = machine_data.get("kernel", [])
        if isinstance(kernels, dict):
            kernels = [kernels]

        if len(kernels) > 0:
            ssd_vals = [float(k.get("SSD", beam["SAD"])) for k in kernels]
            ssd_idx = int(np.argmin(np.abs(np.array(ssd_vals) - center_ssd)))
            cfg.disp_info(f"    SSD = {ssd_vals[ssd_idx]:.1f} mm\n")
            kernel_data = kernels[ssd_idx]
        else:
            kernel_data = {}

        kernel_pos = np.asarray(machine_data.get("kernelPos", np.linspace(0, 200, 100))).ravel()

        # Compute kernel matrices at the convolution grid
        use_kernels = ["kernel1", "kernel2", "kernel3"]
        self._kernel_mxs = []
        r_grid = np.sqrt(self._kernel_x ** 2 + self._kernel_z ** 2)

        for kname in use_kernels:
            if kname in kernel_data:
                kvals = np.asarray(kernel_data[kname]).ravel()
                # Interpolate kernel at grid positions
                k_mx = np.interp(r_grid.ravel(), kernel_pos, kvals, left=0, right=0)
                k_mx = k_mx.reshape(r_grid.shape)
            else:
                k_mx = np.zeros_like(r_grid)
            self._kernel_mxs.append(k_mx)

        # Pre-compute kernel convolution if uniform fluence
        if self._Fpre is not None and not self.use_custom_primary_photon_fluence:
            cfg.disp_info("    Uniform fluence -> pre-computing kernel convolution...\n")
            self._interp_kernel_cache = self._get_kernel_interpolators(self._Fpre)

        return beam

    def _compute_ssd(self, ct: dict, stf: list):
        """
        Compute Source-to-Surface Distance for all rays.
        Port of matRad_computeSSD.m
        """
        ct = get_world_axes(ct)
        density_threshold = self.ssd_density_threshold

        for beam_idx, beam in enumerate(stf):
            iso_world = np.asarray(beam["isoCenter"])
            iso_cube = world_to_cube_coords(np.atleast_2d(iso_world), ct)[0]

            # Use first CT scenario for SSD
            cube = self._cube_wed[0] if (isinstance(self._cube_wed, list) and len(self._cube_wed) > 0) else ct["cube"][0]

            ray_pos_bev = []
            ssd_values = []

            for j, ray in enumerate(beam["ray"]):
                target_point = np.asarray(ray["targetPoint"])
                source_point = np.asarray(beam["sourcePoint"])

                _, _, rho, d12, _ = siddon_ray_tracer(
                    iso_cube, ct["resolution"],
                    source_point, target_point,
                    [cube]
                )

                if len(rho[0]) > 0:
                    # Find first voxel where density > threshold
                    above_thresh = np.where(rho[0] > density_threshold)[0]
                    if len(above_thresh) > 0:
                        # SSD = distance along ray to first surface intersection
                        # Compute corresponding alpha
                        alphas_tmp, _, _, _, _ = siddon_ray_tracer(
                            iso_cube, ct["resolution"],
                            source_point, target_point,
                            [cube]
                        )
                        if len(alphas_tmp) > above_thresh[0]:
                            ssd = float(d12 * alphas_tmp[above_thresh[0]])
                        else:
                            ssd = float(d12 * 0.5)
                    else:
                        ssd = None
                else:
                    ssd = None

                ray_pos_bev.append(ray["rayPos_bev"])
                ssd_values.append(ssd)

            # Fix missing SSDs with nearest neighbor
            ray_pos_arr = np.array(ray_pos_bev)
            for j, ssd in enumerate(ssd_values):
                if ssd is None:
                    # Find nearest ray with valid SSD
                    dists = np.sum((ray_pos_arr - ray_pos_arr[j]) ** 2, axis=1)
                    sorted_idx = np.argsort(dists)
                    for k in sorted_idx:
                        if ssd_values[k] is not None:
                            ssd_values[j] = ssd_values[k]
                            break
                    if ssd_values[j] is None:
                        ssd_values[j] = float(beam["SAD"])

                stf[beam_idx]["ray"][j]["SSD"] = float(ssd_values[j])

    def _calc_dose(self, ct: dict, cst: list, stf: list, dij: dict) -> dict:
        """
        Main dose calculation loop.

        Port of calcDose method in matRad_PencilBeamEngineAbstract.m
        """
        cfg = MatRad_Config.instance()

        ct = get_world_axes(ct)
        ct = self._calc_water_eq_density(ct, stf)
        self._cube_wed = ct.get("cube", [])
        self._apply_outside_density_mask()

        # Compute SSD for all rays
        cfg.disp_info("Computing SSD for all rays...\n")
        self._compute_ssd(ct, stf)

        n_voxels_dose = dij["doseGrid"]["numOfVoxels"]
        n_bixels = dij["totalNumOfBixels"]
        dose_matrix = sp.lil_matrix((n_voxels_dose, n_bixels))

        bixel_counter = 0

        for beam_idx, beam in enumerate(stf):
            cfg.disp_info(f"\nBeam {beam_idx+1}/{len(stf)}: gantry={beam['gantryAngle']}°\n")

            # Initialize beam
            beam = self._init_beam(ct, cst, stf, beam_idx, dij)

            # Get rotation matrix
            rot_mat = get_rotation_matrix(beam["gantryAngle"], beam["couchAngle"])

            # Compute BEV coordinates for dose grid voxels
            iso_center = np.asarray(beam["isoCenter"])
            coords_dose_grid = self._vox_world_coords_dose_grid - iso_center
            rot_coords = coords_dose_grid @ rot_mat  # (N, 3) in BEV

            # Subtract source point
            source_bev = np.asarray(beam["sourcePoint_bev"])
            rot_coords_relative = rot_coords - source_bev  # relative to source

            # Geometric distances from source
            geo_dists = np.sqrt(np.sum(rot_coords_relative ** 2, axis=1))

            # Compute radiological depths using ray tracing
            cfg.disp_info("  Ray tracing for radiological depths...\n")
            iso_cube = world_to_cube_coords(np.atleast_2d(iso_center), ct)[0]

            from ...rayTracing.dispatch import ray_tracing_fast
            beam_stf_for_ray = {
                "isoCenter": iso_center,
                "sourcePoint_bev": source_bev,
                "sourcePoint": beam.get("sourcePoint", source_bev),
                "ray": beam["ray"],
                "SAD": beam["SAD"],
            }
            rad_depths_list = ray_tracing_fast(
                beam_stf_for_ray, ct, self._V_dose_grid,
                rot_coords_relative, self._effective_lateral_cutoff
            )

            # Use first scenario
            rad_depths = rad_depths_list[0]

            # Project voxel coords onto isocenter plane for lateral distance
            SAD = beam["SAD"]
            vox_bev_from_iso = rot_coords  # [y,x,z] in BEV, relative to iso
            proj_factor = SAD / np.where(np.abs(SAD + vox_bev_from_iso[:, 1]) < 1e-6,
                                          1e-6, SAD + vox_bev_from_iso[:, 1])
            iso_lat_x = vox_bev_from_iso[:, 0] * proj_factor  # lateral X at iso
            iso_lat_z = vox_bev_from_iso[:, 2] * proj_factor  # lateral Z at iso

            # Process each ray
            for ray_idx, ray in enumerate(beam["ray"]):
                ray_pos_bev = np.asarray(ray["rayPos_bev"])

                # Radial distance from this ray
                rad_dist_sq = (iso_lat_x - ray_pos_bev[0]) ** 2 + (iso_lat_z - ray_pos_bev[2]) ** 2
                cutoff_sq = self._effective_lateral_cutoff ** 2

                # Valid voxels: within lateral cutoff and with finite rad depth
                valid = (rad_dist_sq <= cutoff_sq) & np.isfinite(rad_depths)

                if not np.any(valid):
                    for _ in ray.get("energy", [1.0]):
                        bixel_counter += 1
                    continue

                vox_ix = np.where(valid)[0]

                # Get interp kernels for this ray
                if self.use_custom_primary_photon_fluence:
                    interp_kernels = self._get_kernel_interpolators(self._Fpre)
                else:
                    interp_kernels = self._interp_kernel_cache

                if interp_kernels is None:
                    for _ in ray.get("energy", [1.0]):
                        bixel_counter += 1
                    continue

                # Compute bixel dose
                bixel_dose = self._calc_single_bixel(
                    float(SAD),
                    self.machine["data"].get("m", 0.03),
                    np.asarray(self.machine["data"].get("betas", [0.04, 0.15, 0.60])).ravel(),
                    interp_kernels,
                    rad_depths[vox_ix],
                    geo_dists[vox_ix],  # geometric distance from source
                    iso_lat_x[vox_ix] - ray_pos_bev[0],
                    iso_lat_z[vox_ix] - ray_pos_bev[2],
                    self.ignore_invalid_values,
                )

                # Apply DIJ sampling
                if self.enable_dij_sampling and not self.is_field_based_dose_calc:
                    sampled_ix, sampled_dose = self._sample_dij(
                        vox_ix, bixel_dose, rad_depths[vox_ix],
                        rad_dist_sq[vox_ix], self._field_width
                    )
                else:
                    sampled_ix, sampled_dose = vox_ix, bixel_dose

                # Map local dose voxel indices to dose grid linear indices
                dose_lin_ix = self._V_dose_grid[sampled_ix] - 1  # 0-based

                # Store in dij bookkeeping
                dij["bixelNum"][bixel_counter] = ray_idx + 1
                dij["rayNum"][bixel_counter] = ray_idx + 1
                dij["beamNum"][bixel_counter] = beam_idx + 1

                # Fill dose matrix
                valid_dose = sampled_dose > 0
                if np.any(valid_dose):
                    dose_matrix[dose_lin_ix[valid_dose], bixel_counter] = sampled_dose[valid_dose]

                bixel_counter += 1

                # Progress
                if bixel_counter % 100 == 0:
                    cfg.disp_info(f"\r  Progress: {bixel_counter}/{dij['totalNumOfBixels']} bixels")

        cfg.disp_info(f"\nDone. Computed {bixel_counter} bixels.\n")

        # Convert to CSC sparse matrix for efficiency
        dij["physicalDose"] = [dose_matrix.tocsc()]

        return dij

    @staticmethod
    def _calc_single_bixel(
        SAD: float,
        m: float,
        betas: np.ndarray,
        interp_kernels: list,
        rad_depths: np.ndarray,
        geo_dists: np.ndarray,
        iso_lat_dists_x: np.ndarray,
        iso_lat_dists_z: np.ndarray,
        ignore_invalid: bool = False,
    ) -> np.ndarray:
        """
        Calculate photon dose for a single bixel.
        Port of calcSingleBixel in PhotonPencilBeamSVDEngine.m

        Based on Scholz 1994 PMB, Eq. 17-19.

        Parameters
        ----------
        SAD : float
            Source-axis distance [mm]
        m : float
            Primary photon attenuation coefficient
        betas : np.ndarray
            Scatter kernel decay parameters
        interp_kernels : list
            Pre-computed kernel interpolators
        rad_depths : np.ndarray
            Radiological depths [mm]
        geo_dists : np.ndarray
            Geometric distances from source [mm]
        iso_lat_dists_x, iso_lat_dists_z : np.ndarray
            Lateral distances at isocenter plane

        Returns
        -------
        np.ndarray
            Bixel dose values
        """
        betas = np.asarray(betas).ravel()
        n_kernels = len(betas)

        # Eq. 17: depth dose components
        # D_i(z) = beta_i / (beta_i - m) * (exp(-m*z) - exp(-beta_i*z))
        dose_component = np.zeros((len(rad_depths), n_kernels))
        for ik in range(n_kernels):
            beta = betas[ik]
            if abs(beta - m) < 1e-10:
                dose_component[:, ik] = m * rad_depths * np.exp(-m * rad_depths)
            else:
                dose_component[:, ik] = (
                    beta / (beta - m) * (np.exp(-m * rad_depths) - np.exp(-beta * rad_depths))
                )

        # Eq. 19: multiply with lateral kernel values
        pts = np.column_stack([iso_lat_dists_z, iso_lat_dists_x])  # (N, 2) for [z, x]
        for ik in range(n_kernels):
            if ik < len(interp_kernels):
                kernel_vals = interp_kernels[ik](pts)
                kernel_vals = np.where(np.isnan(kernel_vals), 0.0, kernel_vals)
                dose_component[:, ik] *= kernel_vals

        # Sum components
        bixel_dose = np.sum(dose_component, axis=1)

        # Inverse square correction
        bixel_dose *= (SAD / geo_dists) ** 2

        # Clean up numerical artifacts
        bixel_dose[bixel_dose < 0] = np.where(
            bixel_dose[bixel_dose < 0] > -1e-14, 0.0,
            bixel_dose[bixel_dose < 0]
        )

        return np.maximum(bixel_dose, 0.0)

    def _sample_dij(
        self,
        ix: np.ndarray,
        bixel_dose: np.ndarray,
        rad_depths: np.ndarray,
        rad_dist_sq: np.ndarray,
        bixel_width: float,
    ):
        """
        Sample DIJ matrix to reduce storage.
        Port of sampleDij in PhotonPencilBeamSVDEngine.m

        Reference: http://dx.doi.org/10.1118/1.1469633
        """
        rel_dose_threshold = self.dij_sampling["relDoseThreshold"]
        lat_cutoff = self.dij_sampling["latCutOff"] + bixel_width / np.sqrt(2)
        sample_type = self.dij_sampling["type"]
        delta_rad_depth = self.dij_sampling["deltaRadDepth"]

        # Identify core voxels
        if sample_type == "radius":
            ix_core = rad_dist_sq < lat_cutoff ** 2
        elif sample_type == "dose":
            ix_core = bixel_dose > rel_dose_threshold * np.max(bixel_dose)
        else:
            ix_core = np.ones(len(ix), dtype=bool)

        dose_core = bixel_dose[ix_core]
        ix_core_vals = ix[ix_core]

        if np.all(ix_core):
            return ix, bixel_dose

        # Sample tail
        ix_tail = ~ix_core
        tail_ix = ix[ix_tail]
        tail_dose = bixel_dose[ix_tail]
        tail_rad_depth = rad_depths[ix_tail]

        if len(tail_ix) == 0:
            return ix_core_vals, dose_core

        # Cluster by radiological depth
        B_r = np.ceil(tail_rad_depth).astype(int)
        max_rad_depth = int(np.max(B_r))
        if max_rad_depth == 0:
            return ix_core_vals, dose_core

        n_clusters = max(1, int(np.round(max_rad_depth / delta_rad_depth)))
        C = np.linspace(0, max_rad_depth, n_clusters + 1).astype(int)

        new_ix = []
        new_dose = []

        for c_idx in range(len(C) - 1):
            mask = (B_r >= C[c_idx]) & (B_r < C[c_idx + 1])
            if not np.any(mask):
                continue

            sub_dose = tail_dose[mask]
            sub_ix = tail_ix[mask]
            threshold_dose = np.max(sub_dose)
            if threshold_dose <= 0:
                continue

            # Probabilistic sampling
            r = np.random.rand(len(sub_dose))
            sampled = r <= (sub_dose / threshold_dose)
            new_ix.extend(sub_ix[sampled].tolist())
            new_dose.extend(np.full(np.sum(sampled), threshold_dose).tolist())

        if new_ix:
            all_ix = np.concatenate([ix_core_vals, np.array(new_ix)])
            all_dose = np.concatenate([dose_core, np.array(new_dose)])
        else:
            all_ix = ix_core_vals
            all_dose = dose_core

        return all_ix, all_dose
