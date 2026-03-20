"""
pyMatRad GUI using matplotlib.

Port of matRadGUI.m - provides interactive visualization of treatment planning results.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, RadioButtons, Button
from typing import Optional, Dict, List


class MatRadGUI:
    """
    Interactive treatment planning GUI using matplotlib.

    Provides:
    - CT slice viewer with dose overlay
    - DVH plot
    - Structure visibility controls
    """

    def __init__(
        self,
        ct: Optional[dict] = None,
        cst: Optional[list] = None,
        result: Optional[dict] = None,
        stf: Optional[list] = None,
    ):
        self.ct = ct
        self.cst = cst
        self.result = result
        self.stf = stf

        self._fig = None
        self._axes = {}
        self._current_plane = 2   # 1=sagittal, 2=coronal, 3=axial
        self._current_slice = 0
        self._dose_alpha = 0.5

    def update(
        self,
        ct: Optional[dict] = None,
        cst: Optional[list] = None,
        result: Optional[dict] = None,
        stf: Optional[list] = None,
    ):
        """Update GUI with new data."""
        if ct is not None:
            self.ct = ct
        if cst is not None:
            self.cst = cst
        if result is not None:
            self.result = result
        if stf is not None:
            self.stf = stf

        if self._fig is not None and plt.fignum_exists(self._fig.number):
            self._refresh()
        else:
            self.show()

    def show(self):
        """Create and show the GUI window."""
        plt.ion()
        self._fig = plt.figure(figsize=(14, 8))
        self._fig.suptitle("pyMatRad Treatment Planning System", fontsize=12)

        gs = gridspec.GridSpec(2, 3, figure=self._fig, hspace=0.4, wspace=0.3)

        # Main CT/dose viewer
        self._axes["ct_dose"] = self._fig.add_subplot(gs[:, 0:2])
        self._axes["dvh"] = self._fig.add_subplot(gs[0, 2])
        self._axes["info"] = self._fig.add_subplot(gs[1, 2])
        self._axes["info"].axis("off")

        self._setup_controls()
        self._refresh()

        plt.show()

    def _setup_controls(self):
        """Add interactive controls."""
        # Slice slider
        ax_slider = plt.axes([0.1, 0.02, 0.5, 0.02])
        n_slices = self._get_n_slices()
        self._slider = Slider(ax_slider, "Slice", 0, max(1, n_slices - 1),
                               valinit=n_slices // 2, valstep=1)
        self._slider.on_changed(self._on_slice_change)

        # Alpha slider for dose overlay
        ax_alpha = plt.axes([0.1, 0.05, 0.5, 0.02])
        self._alpha_slider = Slider(ax_alpha, "Dose alpha", 0, 1,
                                    valinit=self._dose_alpha)
        self._alpha_slider.on_changed(self._on_alpha_change)

    def _get_n_slices(self) -> int:
        if self.ct is None:
            return 1
        dims = self.ct.get("cubeDim", self.ct.get("dimensions", [1, 1, 1]))
        if self._current_plane == 1:
            return dims[1]  # Nx (sagittal)
        elif self._current_plane == 2:
            return dims[0]  # Ny (coronal)
        else:
            return dims[2]  # Nz (axial)

    def _on_slice_change(self, val):
        self._current_slice = int(val)
        self._refresh()

    def _on_alpha_change(self, val):
        self._dose_alpha = float(val)
        self._refresh()

    def _refresh(self):
        """Redraw all plots."""
        for ax in self._axes.values():
            ax.cla()

        self._draw_ct_dose()
        self._draw_dvh()
        self._draw_info()

        self._fig.canvas.draw_idle()

    def _draw_ct_dose(self):
        """Draw CT slice with dose overlay."""
        ax = self._axes["ct_dose"]

        if self.ct is None:
            ax.text(0.5, 0.5, "No CT loaded", ha="center", va="center",
                    transform=ax.transAxes)
            return

        # Get CT slice
        dims = self.ct.get("cubeDim", [100, 100, 50])
        Ny, Nx, Nz = dims[0], dims[1], dims[2]

        cube_hu = self.ct.get("cubeHU", [None])
        if cube_hu and cube_hu[0] is not None:
            ct_cube = cube_hu[0]
        else:
            ct_cube = np.zeros((Ny, Nx, Nz))

        slice_idx = min(self._current_slice, self._get_n_slices() - 1)

        if self._current_plane == 1:  # Sagittal (x fixed)
            j = min(slice_idx, Nx - 1)
            ct_slice = ct_cube[:, j, :].T  # [Nz, Ny]
            xlabel, ylabel = "x (voxel)", "z (voxel)"
        elif self._current_plane == 2:  # Coronal (y fixed)
            i = min(slice_idx, Ny - 1)
            ct_slice = ct_cube[i, :, :].T  # [Nz, Nx]
            xlabel, ylabel = "x (voxel)", "z (voxel)"
        else:  # Axial (z fixed)
            k = min(slice_idx, Nz - 1)
            ct_slice = ct_cube[:, :, k]  # [Ny, Nx]
            xlabel, ylabel = "x (voxel)", "y (voxel)"

        # Display CT
        vmin, vmax = -1000, 500
        ax.imshow(ct_slice, cmap="gray", vmin=vmin, vmax=vmax,
                  origin="lower", aspect="equal")

        # Overlay dose
        if self.result is not None and "physicalDose" in self.result:
            dose_cube = self.result["physicalDose"]
            if self._current_plane == 1:
                dose_slice = dose_cube[:, j, :].T if dose_cube.shape[1] > j else np.zeros_like(ct_slice)
            elif self._current_plane == 2:
                dose_slice = dose_cube[i, :, :].T if dose_cube.shape[0] > i else np.zeros_like(ct_slice)
            else:
                dose_slice = dose_cube[:, :, k] if dose_cube.shape[2] > k else np.zeros_like(ct_slice)

            if dose_cube.shape == (Ny, Nx, Nz):
                max_dose = np.max(dose_cube)
                if max_dose > 0:
                    dose_img = ax.imshow(
                        dose_slice, cmap="jet", vmin=0, vmax=max_dose,
                        alpha=self._dose_alpha, origin="lower", aspect="equal"
                    )

        # Draw contours
        if self.cst is not None:
            colors = plt.cm.get_cmap("tab20")(np.linspace(0, 1, len(self.cst)))
            for roi_idx, row in enumerate(self.cst):
                vox_list = row[3]
                if isinstance(vox_list, list) and len(vox_list) > 0:
                    vox_ix = np.asarray(vox_list[0], dtype=np.int64) - 1
                else:
                    vox_ix = np.asarray(vox_list, dtype=np.int64) - 1

                if len(vox_ix) == 0:
                    continue

                # Create mask
                mask = np.zeros((Ny, Nx, Nz), dtype=bool)
                ix_0 = vox_ix[vox_ix < Ny * Nx * Nz]
                mask.ravel(order="F")[ix_0] = True

                if self._current_plane == 1:
                    slice_mask = mask[:, min(slice_idx, Nx-1), :].T
                elif self._current_plane == 2:
                    slice_mask = mask[min(slice_idx, Ny-1), :, :].T
                else:
                    slice_mask = mask[:, :, min(slice_idx, Nz-1)]

                if np.any(slice_mask):
                    ax.contour(
                        slice_mask.astype(float), levels=[0.5],
                        colors=[colors[roi_idx][:3]], linewidths=1.5
                    )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plane_names = {1: "Sagittal", 2: "Coronal", 3: "Axial"}
        ax.set_title(f"{plane_names.get(self._current_plane, 'Axial')} slice {slice_idx}")
        ax.set_aspect("auto")

    def _draw_dvh(self):
        """Draw DVH plot."""
        ax = self._axes["dvh"]

        if self.result is None or "dvh" not in self.result:
            ax.text(0.5, 0.5, "No DVH data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("DVH")
            return

        dvh_list = self.result["dvh"]
        colors = plt.cm.get_cmap("tab20")(np.linspace(0, 1, len(dvh_list)))

        for i, dvh in enumerate(dvh_list):
            if dvh.get("doseValues") is None or len(dvh.get("doseValues", [])) == 0:
                continue
            dose_vals = dvh["doseValues"]
            vol_pts = dvh["volumePoints"]
            ax.plot(dose_vals, vol_pts, color=colors[i][:3],
                    label=dvh.get("name", f"VOI {i+1}"), linewidth=1.5)

        ax.set_xlabel("Dose (Gy)")
        ax.set_ylabel("Volume (%)")
        ax.set_title("DVH")
        ax.set_xlim(left=0)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    def _draw_info(self):
        """Draw quality indicators."""
        ax = self._axes["info"]
        ax.axis("off")

        if self.result is None or "qi" not in self.result:
            return

        qi_list = self.result["qi"]
        lines = ["Quality Indicators\n"]
        for qi in qi_list:
            name = qi.get("name", "?")
            d_mean = qi.get("D_mean", 0)
            d_95 = qi.get("D_95", 0)
            lines.append(f"{name}: Mean={d_mean:.1f}, D95={d_95:.1f}\n")

        ax.text(0.05, 0.95, "".join(lines), transform=ax.transAxes,
                fontsize=7, verticalalignment="top", fontfamily="monospace")


def launch_gui(
    ct: Optional[dict] = None,
    cst: Optional[list] = None,
    result: Optional[dict] = None,
    stf: Optional[list] = None,
) -> MatRadGUI:
    """
    Launch the pyMatRad GUI.

    Parameters
    ----------
    ct : dict, optional
    cst : list, optional
    result : dict, optional
    stf : list, optional

    Returns
    -------
    MatRadGUI
        The GUI instance (can be updated later)
    """
    gui = MatRadGUI(ct=ct, cst=cst, result=result, stf=stf)
    gui.show()
    return gui


def plot_slice(
    ct: dict,
    cst: Optional[list] = None,
    dose: Optional[np.ndarray] = None,
    plane: int = 3,
    slice_idx: Optional[int] = None,
    dose_alpha: float = 0.5,
    dose_window: Optional[list] = None,
    title: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot a single CT/dose slice.

    Port of matRad_plotSlice.m

    Parameters
    ----------
    ct : dict
    cst : list, optional
    dose : np.ndarray, optional
    plane : int
        1=sagittal, 2=coronal, 3=axial
    slice_idx : int, optional
        Slice index (defaults to middle)
    dose_alpha : float
    dose_window : list [min, max], optional
    title : str
    ax : matplotlib Axes, optional

    Returns
    -------
    Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    dims = ct.get("cubeDim", ct.get("dimensions", [100, 100, 50]))
    Ny, Nx, Nz = dims[0], dims[1], dims[2]

    cube_hu = ct.get("cubeHU", [None])
    ct_cube = cube_hu[0] if cube_hu and cube_hu[0] is not None else np.zeros((Ny, Nx, Nz))

    # Default slice
    if slice_idx is None:
        if plane == 1:
            slice_idx = Nx // 2
        elif plane == 2:
            slice_idx = Ny // 2
        else:
            slice_idx = Nz // 2

    # Extract slice
    if plane == 1:  # Sagittal
        j = min(slice_idx, Nx - 1)
        ct_slice = ct_cube[:, j, :].T
    elif plane == 2:  # Coronal
        i = min(slice_idx, Ny - 1)
        ct_slice = ct_cube[i, :, :].T
    else:  # Axial
        k = min(slice_idx, Nz - 1)
        ct_slice = ct_cube[:, :, k]

    # Display CT
    ax.imshow(ct_slice, cmap="gray", vmin=-1000, vmax=500, origin="lower")

    # Overlay dose
    if dose is not None and dose.shape == (Ny, Nx, Nz):
        if plane == 1:
            dose_slice = dose[:, j, :].T
        elif plane == 2:
            dose_slice = dose[i, :, :].T
        else:
            dose_slice = dose[:, :, k]

        if dose_window is None:
            dose_window = [0, np.max(dose)]

        im = ax.imshow(
            dose_slice, cmap="jet",
            vmin=dose_window[0], vmax=dose_window[1],
            alpha=dose_alpha, origin="lower"
        )
        plt.colorbar(im, ax=ax, label="Dose (Gy)")

    # Draw contours
    if cst is not None:
        colors = plt.cm.get_cmap("tab20")(np.linspace(0, 1, len(cst)))
        for roi_idx, row in enumerate(cst):
            vox_list = row[3]
            if isinstance(vox_list, list) and len(vox_list) > 0:
                vox_ix = np.asarray(vox_list[0], dtype=np.int64) - 1
            else:
                vox_ix = np.asarray(vox_list, dtype=np.int64) - 1

            if len(vox_ix) == 0:
                continue

            mask = np.zeros((Ny, Nx, Nz), dtype=bool)
            ix_0 = vox_ix[vox_ix < Ny * Nx * Nz]
            mask.ravel(order="F")[ix_0] = True

            if plane == 1:
                slice_mask = mask[:, j, :].T
            elif plane == 2:
                slice_mask = mask[i, :, :].T
            else:
                slice_mask = mask[:, :, k]

            if np.any(slice_mask):
                ax.contour(
                    slice_mask.astype(float), levels=[0.5],
                    colors=[colors[roi_idx][:3]],
                    linewidths=1.5, linestyles="-"
                )

    if title:
        ax.set_title(title)

    return fig
