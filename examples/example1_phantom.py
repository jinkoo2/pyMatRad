"""
Example 1: Generate a phantom geometry and run a treatment plan.

Python port of matRad_example1_phantom.m

This example demonstrates:
(i)  How to create arbitrary CT data (resolution, CT numbers)
(ii) How to create a CST structure containing volumes of interest
(iii) Generate a treatment plan for this phantom
"""

import numpy as np
import sys
import os

# Add pyMatRad to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_example1():
    """Run phantom example."""
    from matRad.phantoms.builder import PhantomBuilder
    from matRad.optimization.DoseObjectives import SquaredDeviation, SquaredOverdosing
    from matRad.steering.stf_generator import generate_stf
    from matRad.doseCalc.calc_dose_influence import calc_dose_influence
    from matRad.optimization.fluence_optimization import fluence_optimization
    from matRad.planAnalysis.plan_analysis import plan_analysis
    from matRad.geometry.geometry import get_iso_center, world_to_cube_index

    print("=" * 60)
    print("pyMatRad Example 1: Phantom Treatment Plan")
    print("=" * 60)

    # =========================================================================
    # Create a CT image series
    # =========================================================================
    ct_dim = [200, 200, 100]    # [x, y, z] dimensions in voxels
    ct_resolution = [2, 2, 3]  # [x, y, z] resolution in mm/voxel

    builder = PhantomBuilder(ct_dim, ct_resolution, num_of_ct_scen=1)

    # =========================================================================
    # Define objectives for VOIs
    # =========================================================================
    objective1 = SquaredDeviation(penalty=800, d_ref=45)   # Target: hit 45 Gy
    objective2 = SquaredOverdosing(penalty=400, d_ref=0)   # OAR1: min overdose
    objective3 = SquaredOverdosing(penalty=10, d_ref=0)    # OAR2: mild overdose constraint

    # Add volumes
    builder.add_spherical_target("Volume1", radius=20, objectives=[objective1.to_dict()], HU=0)
    builder.add_box_oar("Volume2", [60, 30, 60], offset=[0, -15, 0],
                         objectives=[objective2.to_dict()], HU=0)
    builder.add_box_oar("Volume3", [60, 30, 60], offset=[0, 15, 0],
                         objectives=[objective3.to_dict()], HU=0)

    # Get CT and CST
    ct, cst = builder.get_ct_cst()

    print(f"CT dimensions: {ct['cubeDim']}")
    print(f"CT resolution: {ct['resolution']}")
    print(f"Number of structures: {len(cst)}")
    for row in cst:
        n_vox = len(row[3][0]) if isinstance(row[3], list) else len(row[3])
        print(f"  - {row[1]} ({row[2]}): {n_vox} voxels")

    # =========================================================================
    # Define treatment plan
    # =========================================================================
    pln = {
        "radiationMode": "photons",
        "machine": "Generic",
        "bioModel": "none",
        "multScen": "nomScen",
        "numOfFractions": 30,
        "propStf": {
            "gantryAngles": list(range(0, 360, 70)),  # [0, 70, 140, 210, 280, 350]
            "couchAngles": [0] * len(list(range(0, 360, 70))),
            "bixelWidth": 5,
            "isoCenter": None,  # Will be auto-computed
            "visMode": 0,
            "addMargin": True,
            "fillEmptyBixels": False,
        },
        "propOpt": {
            "runDAO": False,
            "runSequencing": False,
        },
        "propDoseCalc": {
            "doseGrid": {
                "resolution": {"x": 3, "y": 3, "z": 3}
            }
        },
    }

    print(f"\nGantry angles: {pln['propStf']['gantryAngles']}")
    print(f"Bixel width: {pln['propStf']['bixelWidth']} mm")
    print(f"Num fractions: {pln['numOfFractions']}")

    # =========================================================================
    # Generate beam geometry (STF)
    # =========================================================================
    print("\nGenerating beam geometry (STF)...")
    stf = generate_stf(ct, cst, pln)

    print(f"Generated {len(stf)} beams:")
    total_bixels = sum(b["totalNumOfBixels"] for b in stf)
    for i, beam in enumerate(stf):
        print(f"  Beam {i+1}: gantry={beam['gantryAngle']}°, "
              f"rays={beam['numOfRays']}, bixels={beam['totalNumOfBixels']}")
    print(f"Total bixels: {total_bixels}")

    # =========================================================================
    # Dose Calculation
    # =========================================================================
    print("\nCalculating dose influence matrix...")
    dij = calc_dose_influence(ct, cst, stf, pln)

    print(f"DIJ shape: {dij['physicalDose'][0].shape}")
    print(f"DIJ non-zeros: {dij['physicalDose'][0].nnz}")

    # =========================================================================
    # Inverse Optimization
    # =========================================================================
    print("\nRunning fluence optimization...")
    result = fluence_optimization(dij, cst, pln)

    print(f"Optimized bixel weights: min={result['w'].min():.4f}, "
          f"max={result['w'].max():.4f}, "
          f"mean={result['w'].mean():.4f}")

    dose = result["physicalDose"]
    print(f"Dose cube: shape={dose.shape}, "
          f"min={dose.min():.2f}, max={dose.max():.2f} Gy/fraction")

    # =========================================================================
    # Plan Analysis
    # =========================================================================
    print("\nComputing plan analysis...")
    result = plan_analysis(result, ct, cst, stf, pln)

    print("\nQuality Indicators:")
    for qi in result["qi"]:
        print(f"  {qi['name']} ({qi['type']}):")
        print(f"    D_mean = {qi.get('D_mean', 0):.2f}, "
              f"D_95 = {qi.get('D_95', 0):.2f}, "
              f"D_5 = {qi.get('D_5', 0):.2f}")

    # =========================================================================
    # Visualization (skipping GUI in batch mode)
    # =========================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        from gui.matrad_gui import plot_slice

        # Find isocenter slice
        iso_center = get_iso_center(cst, ct)
        from matRad.geometry.geometry import world_to_cube_index
        iso_idx = world_to_cube_index(np.atleast_2d(iso_center), ct)[0]
        slice_z = int(iso_idx[2])

        # Normalize dose per total (not per fraction) for display
        dose_total = dose * pln["numOfFractions"]
        dose_window = [0, float(dose_total.max())]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("pyMatRad Example 1 - Phantom Plan", fontsize=12)

        plot_slice(ct, cst=cst, dose=dose_total, plane=3, slice_idx=slice_z,
                   dose_alpha=0.7, dose_window=dose_window,
                   title="Axial slice (iso-center)", ax=axes[0])

        # DVH
        ax_dvh = axes[1]
        colors = plt.colormaps["tab10"](np.linspace(0, 1, len(result["dvh"])))
        for i, dvh in enumerate(result["dvh"]):
            if dvh.get("doseValues") is None:
                continue
            ax_dvh.plot(dvh["doseValues"] * pln["numOfFractions"],
                       dvh["volumePoints"],
                       color=colors[i][:3], label=dvh.get("name", f"VOI {i+1}"),
                       linewidth=2)
        ax_dvh.set_xlabel("Dose (Gy)")
        ax_dvh.set_ylabel("Volume (%)")
        ax_dvh.set_title("DVH")
        ax_dvh.legend(fontsize=9)
        ax_dvh.grid(True, alpha=0.3)
        ax_dvh.set_ylim(0, 105)

        out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyMatRad_example1.png")
        plt.tight_layout()
        plt.savefig(out_file, dpi=100, bbox_inches="tight")
        print(f"\nFigure saved to {out_file}")
        plt.close()

    except Exception as e:
        print(f"\nVisualization skipped: {e}")

    print("\nExample 1 complete!")
    return ct, cst, stf, dij, result, pln


if __name__ == "__main__":
    ct, cst, stf, dij, result, pln = run_example1()
