"""
Example 2: Photon Treatment Plan.

Python port of matRad_example2_photons.m

This example demonstrates:
(i)  How to load patient data (TG119 phantom)
(ii) How to set up a photon dose calculation
(iii) Inverse optimization of beamlet intensities
(iv) Visual and quantitative evaluation
"""

import numpy as np
import sys
import os

# Add pyMatRad to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_tg119():
    """
    Load TG119 phantom from the matRad data files.

    Returns ct, cst dicts.
    """
    import scipy.io as sio

    # Look for TG119.mat in matRad's phantoms directory
    matrad_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "matRad", "matRad", "phantoms"
    )
    mat_file = os.path.join(matrad_root, "TG119.mat")

    if not os.path.isfile(mat_file):
        raise FileNotFoundError(
            f"TG119.mat not found at {mat_file}. "
            f"Please ensure matRad is installed at the expected location."
        )

    # Load the file
    raw = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)

    # Convert ct struct
    ct_raw = raw["ct"]
    ct = _matlab_struct_to_dict(ct_raw)

    # Convert cst cell array
    cst_raw = raw["cst"]
    cst = _matlab_cst_to_list(cst_raw)

    return ct, cst


def _matlab_struct_to_dict(obj):
    """Convert MATLAB struct to Python dict."""
    import scipy.io as sio
    if isinstance(obj, sio.matlab.mat_struct):
        result = {}
        for key in obj._fieldnames:
            val = getattr(obj, key)
            result[key] = _matlab_struct_to_dict(val)
        return result
    elif isinstance(obj, np.ndarray):
        if obj.dtype.names:
            result = {}
            for name in obj.dtype.names:
                result[name] = _matlab_struct_to_dict(obj[name])
            return result
        elif obj.dtype == object:
            if obj.ndim == 0:
                return _matlab_struct_to_dict(obj.item())
            return [_matlab_struct_to_dict(v) for v in obj.flat]
        else:
            if obj.ndim == 0:
                return obj.item()
            if obj.size == 1:
                return float(obj.flat[0])
            return obj
    else:
        return obj


def _matlab_cst_to_list(cst_raw: np.ndarray) -> list:
    """Convert MATLAB cst cell array to Python list of lists."""
    if cst_raw.ndim == 1:
        n_rows = cst_raw.shape[0]
        n_cols = 1
    else:
        n_rows, n_cols = cst_raw.shape

    cst = []
    for i in range(n_rows):
        row = []
        for j in range(n_cols if cst_raw.ndim > 1 else 1):
            if cst_raw.ndim > 1:
                cell = cst_raw[i, j]
            else:
                cell = cst_raw[i]

            if isinstance(cell, np.ndarray):
                if cell.dtype.names:
                    val = _matlab_struct_to_dict(cell)
                elif cell.dtype == object:
                    val = [_matlab_struct_to_dict(v) for v in cell.flat]
                else:
                    val = cell
            else:
                val = cell
            row.append(val)
        cst.append(row)

    return cst


def run_example2():
    """Run photon treatment plan example."""
    from matRad.steering.stf_generator import generate_stf
    from matRad.doseCalc.calc_dose_influence import calc_dose_influence
    from matRad.optimization.fluence_optimization import fluence_optimization
    from matRad.planAnalysis.plan_analysis import plan_analysis
    from matRad.geometry.geometry import get_iso_center

    print("=" * 60)
    print("pyMatRad Example 2: Photon Treatment Plan (TG119)")
    print("=" * 60)

    # =========================================================================
    # Load patient data (TG119 phantom)
    # =========================================================================
    print("\nLoading TG119 phantom...")
    try:
        ct, cst = load_tg119()
        print(f"CT dimensions: {ct.get('cubeDim', 'unknown')}")
        print(f"CT resolution: {ct.get('resolution', 'unknown')}")
        print(f"Number of structures: {len(cst)}")
    except Exception as e:
        print(f"Could not load TG119.mat: {e}")
        print("Using synthetic phantom instead...")
        from matRad.phantoms.builder import PhantomBuilder
        from matRad.optimization.DoseObjectives import SquaredDeviation, SquaredOverdosing

        builder = PhantomBuilder([100, 100, 50], [3, 3, 3], 1)
        obj1 = SquaredDeviation(penalty=800, d_ref=2.0)  # 2 Gy/fraction = 60 Gy total
        builder.add_spherical_target("PTV", radius=20, objectives=[obj1.to_dict()], HU=0)
        builder.add_box_oar("OAR", [60, 40, 40], offset=[0, -15, 0], HU=0)
        ct, cst = builder.get_ct_cst()

    # =========================================================================
    # Define treatment plan
    # =========================================================================
    pln = {
        "radiationMode": "photons",
        "machine": "Generic",
        "numOfFractions": 30,
        "bioModel": "none",
        "multScen": "nomScen",
        "propStf": {
            "gantryAngles": list(range(0, 360, 40)),  # 9 beams, 40° spacing
            "couchAngles": [0] * len(list(range(0, 360, 40))),
            "bixelWidth": 5,
            "isoCenter": None,
            "visMode": 0,
            "addMargin": True,
        },
        "propOpt": {
            "runDAO": False,
            "runSequencing": True,
        },
        "propDoseCalc": {
            "doseGrid": {
                "resolution": {"x": 3, "y": 3, "z": 3}
            }
        },
    }

    print(f"\nRadiation mode: {pln['radiationMode']}")
    print(f"Machine: {pln['machine']}")
    print(f"Gantry angles: {pln['propStf']['gantryAngles']}")
    print(f"Number of fractions: {pln['numOfFractions']}")

    # =========================================================================
    # Generate STF
    # =========================================================================
    print("\nGenerating STF...")
    stf = generate_stf(ct, cst, pln)

    print(f"Generated {len(stf)} beams")
    print(f"Beam 1: {stf[0]['numOfRays']} rays, {stf[0]['totalNumOfBixels']} bixels")

    # =========================================================================
    # Dose Calculation
    # =========================================================================
    print("\nCalculating dose influence matrix...")
    dij = calc_dose_influence(ct, cst, stf, pln)
    print(f"DIJ: {dij['physicalDose'][0].shape}, {dij['physicalDose'][0].nnz} non-zeros")

    # =========================================================================
    # Optimization
    # =========================================================================
    print("\nOptimizing fluence (9-beam, 40° spacing)...")
    result = fluence_optimization(dij, cst, pln)
    print(f"Dose: min={result['physicalDose'].min():.3f}, max={result['physicalDose'].max():.3f} Gy/frac")

    # Plan analysis
    result = plan_analysis(result, ct, cst, stf, pln)

    # =========================================================================
    # Second plan with coarser beam spacing (50°)
    # =========================================================================
    print("\n--- Second plan: 50° spacing ---")
    pln_coarse = dict(pln)
    pln_coarse["propStf"] = dict(pln["propStf"])
    pln_coarse["propStf"]["gantryAngles"] = list(range(0, 360, 50))
    pln_coarse["propStf"]["couchAngles"] = [0] * len(list(range(0, 360, 50)))

    print(f"Generating STF ({len(pln_coarse['propStf']['gantryAngles'])} beams, 50° spacing)...")
    stf_coarse = generate_stf(ct, cst, pln_coarse)

    print("Calculating dose...")
    dij_coarse = calc_dose_influence(ct, cst, stf_coarse, pln_coarse)

    print("Optimizing...")
    result_coarse = fluence_optimization(dij_coarse, cst, pln_coarse)
    result_coarse = plan_analysis(result_coarse, ct, cst, stf_coarse, pln_coarse)

    # =========================================================================
    # Comparison
    # =========================================================================
    print("\n--- Plan Comparison ---")
    print("Fine beam spacing (40°) vs Coarse beam spacing (50°):")

    for struct_idx in range(min(len(result["qi"]), len(result_coarse["qi"]))):
        qi1 = result["qi"][struct_idx]
        qi2 = result_coarse["qi"][struct_idx]
        name = qi1.get("name", f"VOI {struct_idx+1}")
        print(f"\n  {name} ({qi1.get('type', '')}):")
        print(f"    Fine:   D_mean={qi1.get('D_mean', 0)*pln['numOfFractions']:.1f} Gy, "
              f"D_95={qi1.get('D_95', 0)*pln['numOfFractions']:.1f} Gy")
        print(f"    Coarse: D_mean={qi2.get('D_mean', 0)*pln['numOfFractions']:.1f} Gy, "
              f"D_95={qi2.get('D_95', 0)*pln['numOfFractions']:.1f} Gy")

    # =========================================================================
    # Visualization
    # =========================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from gui.matrad_gui import plot_slice

        iso = get_iso_center(cst, ct)
        from matRad.geometry.geometry import world_to_cube_index
        iso_idx = world_to_cube_index(np.atleast_2d(iso), ct)[0]
        slice_z = int(iso_idx[2])

        dose1 = result["physicalDose"] * pln["numOfFractions"]
        dose2 = result_coarse["physicalDose"] * pln["numOfFractions"]
        max_dose = max(float(dose1.max()), float(dose2.max()))
        dose_window = [0, max_dose]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("pyMatRad Example 2 - Photon Treatment Plans", fontsize=12)

        plot_slice(ct, cst=cst, dose=dose1, plane=3, slice_idx=slice_z,
                   dose_alpha=0.7, dose_window=dose_window,
                   title="Fine beam spacing (40°)", ax=axes[0])

        plot_slice(ct, cst=cst, dose=dose2, plane=3, slice_idx=slice_z,
                   dose_alpha=0.7, dose_window=dose_window,
                   title="Coarse beam spacing (50°)", ax=axes[1])

        # DVH comparison
        ax_dvh = axes[2]
        colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(result["dvh"])))
        for i, dvh in enumerate(result["dvh"]):
            if not dvh.get("doseValues", []) is None and len(dvh.get("doseValues", [])) > 0:
                ax_dvh.plot(
                    dvh["doseValues"] * pln["numOfFractions"],
                    dvh["volumePoints"],
                    color=colors[i][:3], linestyle="-",
                    label=f"{dvh.get('name', f'VOI {i+1}')} (fine)", linewidth=2
                )

        for i, dvh in enumerate(result_coarse["dvh"]):
            if not dvh.get("doseValues", []) is None and len(dvh.get("doseValues", [])) > 0:
                ax_dvh.plot(
                    dvh["doseValues"] * pln["numOfFractions"],
                    dvh["volumePoints"],
                    color=colors[i][:3], linestyle="--",
                    label=f"{dvh.get('name', f'VOI {i+1}')} (coarse)", linewidth=1.5
                )

        ax_dvh.set_xlabel("Dose (Gy)")
        ax_dvh.set_ylabel("Volume (%)")
        ax_dvh.set_title("DVH Comparison")
        ax_dvh.legend(fontsize=7)
        ax_dvh.grid(True, alpha=0.3)
        ax_dvh.set_ylim(0, 105)

        out_file = "/tmp/pyMatRad_example2.png"
        plt.tight_layout()
        plt.savefig(out_file, dpi=100, bbox_inches="tight")
        print(f"\nFigure saved to {out_file}")
        plt.close()

    except Exception as e:
        print(f"\nVisualization skipped: {e}")

    print("\nExample 2 complete!")
    return result, result_coarse


if __name__ == "__main__":
    result, result_coarse = run_example2()
