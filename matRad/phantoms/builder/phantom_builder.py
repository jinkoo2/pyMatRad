"""
PhantomBuilder - Create synthetic radiotherapy phantoms.

Python port of matRad_PhantomBuilder.m
"""

import numpy as np
from typing import Optional, List, Union, Tuple
from .phantom_voi import PhantomVOIBox, PhantomVOISphere


class PhantomBuilder:
    """
    Helps create synthetic radiotherapy phantoms with VOIs.
    Port of matRad_PhantomBuilder.m

    Usage:
        builder = PhantomBuilder([200, 200, 100], [2, 2, 3], 1)
        builder.add_spherical_target('Target', 20, objectives=[obj])
        builder.add_box_oar('OAR1', [60, 30, 60], offset=[0, -15, 0])
        ct, cst = builder.get_ct_cst()
    """

    def __init__(
        self,
        ct_dim: List[int],
        ct_resolution: List[float],
        num_of_ct_scen: int = 1,
    ):
        """
        Parameters
        ----------
        ct_dim : list [x, y, z]
            Dimensions in voxels
        ct_resolution : list [x, y, z]
            Resolution in mm/voxel
        num_of_ct_scen : int
            Number of CT scenarios
        """
        self.volumes = []
        self._cst = []

        # Build CT struct
        # MATLAB: cubeDim = [ctDim(2), ctDim(1), ctDim(3)] = [Ny, Nx, Nz]
        self._ct = {
            "cubeDim": [ct_dim[1], ct_dim[0], ct_dim[2]],  # [Ny, Nx, Nz]
            "resolution": {
                "x": ct_resolution[0],
                "y": ct_resolution[1],
                "z": ct_resolution[2],
            },
            "numOfCtScen": num_of_ct_scen,
            "cubeHU": [np.ones([ct_dim[1], ct_dim[0], ct_dim[2]]) * -1000.0],
        }

    def add_box_target(
        self,
        name: str,
        dimensions: List[float],
        offset: Optional[List[float]] = None,
        objectives: Optional[list] = None,
        HU: float = 0.0,
    ):
        """
        Add a box-shaped target VOI.

        Parameters
        ----------
        name : str
        dimensions : list [x, y, z] in voxels
        offset : list [x, y, z] offset from center in voxels
        objectives : list of objective dicts
        HU : float
            Hounsfield unit of the volume
        """
        voi = PhantomVOIBox(
            name, "TARGET", dimensions,
            offset=offset, objectives=objectives, HU=HU
        )
        self.volumes.append(voi)
        self._update_cst()

    def add_spherical_target(
        self,
        name: str,
        radius: float,
        offset: Optional[List[float]] = None,
        objectives: Optional[list] = None,
        HU: float = 0.0,
    ):
        """
        Add a spherical target VOI.

        Parameters
        ----------
        name : str
        radius : float
            Radius in voxels
        offset : list [x, y, z] offset from center in voxels
        objectives : list
        HU : float
        """
        voi = PhantomVOISphere(
            name, "TARGET", radius,
            offset=offset, objectives=objectives, HU=HU
        )
        self.volumes.append(voi)
        self._update_cst()

    def add_box_oar(
        self,
        name: str,
        dimensions: List[float],
        offset: Optional[List[float]] = None,
        objectives: Optional[list] = None,
        HU: float = 0.0,
    ):
        """Add a box-shaped OAR VOI."""
        voi = PhantomVOIBox(
            name, "OAR", dimensions,
            offset=offset, objectives=objectives, HU=HU
        )
        self.volumes.append(voi)
        self._update_cst()

    def add_spherical_oar(
        self,
        name: str,
        radius: float,
        offset: Optional[List[float]] = None,
        objectives: Optional[list] = None,
        HU: float = 0.0,
    ):
        """Add a spherical OAR VOI."""
        voi = PhantomVOISphere(
            name, "OAR", radius,
            offset=offset, objectives=objectives, HU=HU
        )
        self.volumes.append(voi)
        self._update_cst()

    def get_ct_cst(self) -> Tuple[dict, list]:
        """
        Get the CT and CST structures.

        Initializes HU values in reverse order of definition (first defined
        has highest priority in case of overlaps).

        Returns
        -------
        ct : dict
        cst : list
        """
        # Initialize HU in reverse order (last defined has lowest priority)
        n = len(self._cst)
        for i in range(n):
            voi_idx = n - 1 - i  # reverse order
            vox_indices = self._cst[voi_idx][3]
            if isinstance(vox_indices, list) and len(vox_indices) > 0:
                ix = vox_indices[0]
            else:
                ix = vox_indices

            if len(ix) == 0:
                continue

            # Convert 1-based Fortran linear indices to Python indexing
            ix_0based = np.asarray(ix, dtype=np.int64) - 1
            # Use Fortran order (column-major) to match MATLAB
            flat = self._ct["cubeHU"][0].ravel(order="F")
            flat[ix_0based] = self.volumes[voi_idx].HU
            self._ct["cubeHU"][0] = flat.reshape(self._ct["cubeDim"], order="F")

        return self._ct, self._cst

    def _update_cst(self):
        """Update cst by re-initializing the last volume."""
        self._cst = self.volumes[-1].initialize_parameters(self._ct, self._cst)
