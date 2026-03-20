"""
Phantom VOI (Volume of Interest) classes.

Python port of matRad_PhantomVOIBox.m, matRad_PhantomVOISphere.m,
and matRad_PhantomVOIVolume.m.
"""

import numpy as np
from typing import Optional, List, Union


class PhantomVOIVolume:
    """
    Base class for phantom VOIs.
    Port of matRad_PhantomVOIVolume.m
    """

    def __init__(
        self,
        name: str,
        voi_type: str,
        offset: Optional[np.ndarray] = None,
        objectives: Optional[list] = None,
        HU: float = 0.0,
    ):
        self.name = name
        self.voi_type = voi_type  # 'TARGET' or 'OAR'
        self.offset = np.array(offset if offset is not None else [0, 0, 0], dtype=float)
        self.objectives = objectives if objectives is not None else []
        self.HU = HU

        # Ensure objectives is a list
        if not isinstance(self.objectives, list):
            self.objectives = [self.objectives]

    def initialize_parameters(self, ct: dict, cst: list) -> list:
        """
        Add this VOI to the cst structure.
        Port of initializeParameters@matRad_PhantomVOIVolume.

        Parameters
        ----------
        ct : dict
        cst : list

        Returns
        -------
        list
            Updated cst
        """
        new_row = [
            len(cst) + 1,       # index (1-based)
            self.name,          # name
            self.voi_type,      # type: 'TARGET' or 'OAR'
            [np.array([], dtype=np.int64)],  # voxel indices (to be filled by subclass)
            {                   # properties
                "Priority": len(cst) + 1,
                "alphaX": 0.1,
                "betaX": 0.05,
                "TissueClass": 1,
                "Visible": True,
                "visibleColor": [0.5, 0.5, 0.5],
            },
            list(self.objectives),  # objectives
        ]
        cst.append(new_row)
        return cst


class PhantomVOIBox(PhantomVOIVolume):
    """
    Box-shaped VOI.
    Port of matRad_PhantomVOIBox.m
    """

    def __init__(
        self,
        name: str,
        voi_type: str,
        box_dimensions: np.ndarray,
        offset: Optional[np.ndarray] = None,
        objectives: Optional[list] = None,
        HU: float = 0.0,
    ):
        """
        Parameters
        ----------
        box_dimensions : array-like [x_size, y_size, z_size] in voxels
        """
        super().__init__(name, voi_type, offset, objectives, HU)
        self.box_dimensions = np.asarray(box_dimensions)

    def initialize_parameters(self, ct: dict, cst: list) -> list:
        """Build box VOI in the CT grid and update cst."""
        cst = super().initialize_parameters(ct, cst)

        dims = ct["cubeDim"]  # [Ny, Nx, Nz]
        Ny, Nx, Nz = dims[0], dims[1], dims[2]

        center = np.array([Ny // 2, Nx // 2, Nz // 2])

        # Convert offsets: [x_offset, y_offset, z_offset] in voxels
        offset = self.offset  # [ox, oy, oz]
        dims_box = self.box_dimensions  # [dx, dy, dz] in voxels

        # Box bounds in CT array indices
        # MATLAB: xMinMax = center(2)+offsets(1) + round(dims(1)/2)*[-1,1]
        # center(2) is Nx/2 (x-direction), offsets(1) is x-offset
        x_min = int(center[1] + offset[0] - np.round(dims_box[0] / 2))
        x_max = int(center[1] + offset[0] + np.round(dims_box[0] / 2))
        y_min = int(center[0] + offset[1] - np.round(dims_box[1] / 2))
        y_max = int(center[0] + offset[1] + np.round(dims_box[1] / 2))
        z_min = int(center[2] + offset[2] - np.round(dims_box[2] / 2))
        z_max = int(center[2] + offset[2] + np.round(dims_box[2] / 2))

        # Clamp to valid range
        x_min = max(0, x_min)
        x_max = min(Nx - 1, x_max)
        y_min = max(0, y_min)
        y_max = min(Ny - 1, y_max)
        z_min = max(0, z_min)
        z_max = min(Nz - 1, z_max)

        # Create boolean mask
        voi_helper = np.zeros((Ny, Nx, Nz), dtype=bool)
        voi_helper[y_min:y_max + 1, x_min:x_max + 1, z_min:z_max + 1] = True

        # Convert to MATLAB-style 1-based Fortran-order (column-major) linear indices
        # In MATLAB: ind = find(VOIHelper) uses column-major order
        lin_ix = np.where(voi_helper.ravel(order="F"))[0] + 1  # 1-based

        cst[-1][3] = [lin_ix]
        return cst


class PhantomVOISphere(PhantomVOIVolume):
    """
    Spherical VOI.
    Port of matRad_PhantomVOISphere.m
    """

    def __init__(
        self,
        name: str,
        voi_type: str,
        radius: float,
        offset: Optional[np.ndarray] = None,
        objectives: Optional[list] = None,
        HU: float = 0.0,
    ):
        """
        Parameters
        ----------
        radius : float
            Radius in voxels
        """
        super().__init__(name, voi_type, offset, objectives, HU)
        self.radius = radius

    def initialize_parameters(self, ct: dict, cst: list) -> list:
        """Build spherical VOI in the CT grid and update cst."""
        cst = super().initialize_parameters(ct, cst)

        dims = ct["cubeDim"]  # [Ny, Nx, Nz]
        Ny, Nx, Nz = dims[0], dims[1], dims[2]

        center = np.array([Ny // 2, Nx // 2, Nz // 2], dtype=float)

        # MATLAB: for x in 1:Nx, for y in 1:Ny, for z in 1:Nz
        #         currPost = [y x z] + offsets - center
        #         if norm(currPost) < radius: VOIHelper(y,x,z) = 1
        # offsets in MATLAB = [offset_x, offset_y, offset_z]
        # [y, x, z] are 1-based MATLAB indices
        # Python 0-based: pos = [i, j, k] + 1 (to match 1-based)

        # Vectorized implementation
        i_arr = np.arange(Ny)  # y dimension
        j_arr = np.arange(Nx)  # x dimension
        k_arr = np.arange(Nz)  # z dimension

        # meshgrid: [i, j, k]
        I, J, K = np.meshgrid(i_arr, j_arr, k_arr, indexing="ij")

        # MATLAB 1-based positions: [y, x, z] = [i+1, j+1, k+1]
        # offsets = [ox, oy, oz] in voxels
        # center 1-based = [Ny/2, Nx/2, Nz/2] (approximately)
        center_1based = np.round(np.array([Ny, Nx, Nz]) / 2)
        offset = self.offset  # [ox, oy, oz]

        # Distance from center
        # MATLAB: currPost = [y x z] + offsets - center
        # [y, x, z] corresponds to [I+1, J+1, K+1] in 0-based
        dy = (I + 1) + offset[1] - center_1based[0]  # y-direction (offset[1] = oy)
        dx = (J + 1) + offset[0] - center_1based[1]  # x-direction (offset[0] = ox)
        dz = (K + 1) + offset[2] - center_1based[2]  # z-direction (offset[2] = oz)

        dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        voi_helper = dist < self.radius

        # MATLAB find() uses Fortran/column-major order
        lin_ix = np.where(voi_helper.ravel(order="F"))[0] + 1  # 1-based
        cst[-1][3] = [lin_ix]
        return cst
