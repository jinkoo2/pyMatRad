"""
Example 8: Photon VMAT Treatment Plan.

Python port of matRad_example8_photonsVMAT.m

This example demonstrates:
(i)   How to configure a VMAT arc plan (anchor angles → fine/DAO/FMO grids)
(ii)  Arc STF generation (StfGeneratorPhotonVMAT)
(iii) Dose influence calculation over all fine-angle arc beams
(iv)  Fluence Map Optimisation (FMO) on arc beams
(v)   Plan analysis and DVH visualisation

NOTE: The full VMAT pipeline in MATLAB continues with leaf sequencing
(matRad_arcSequencing) and Direct Aperture Optimisation (matRad_directApertureOptimization),
which enforce arc delivery constraints (leaf speed, gantry speed, MU rate).
These steps are NOT yet ported to pyMatRad.  The optimised result here is a
dense-arc fluence plan, not a deliverable VMAT plan.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Patient data loader (TG119, reused from example2)
# ---------------------------------------------------------------------------

def load_tg119():
    import scipy.io as sio

    matrad_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "matRad", "matRad", "phantoms",
    )
    mat_file = os.path.join(matrad_root, "TG119.mat")
    if not os.path.isfile(mat_file):
        raise FileNotFoundError(f"TG119.mat not found at {mat_file}")

    raw = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    ct = _struct_to_dict(raw["ct"])
    cst = _cst_to_list(raw["cst"])
    return ct, cst


def _struct_to_dict(obj):
    import scipy.io as sio
    if isinstance(obj, sio.matlab.mat_struct):
        return {k: _struct_to_dict(getattr(obj, k)) for k in obj._fieldnames}
    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            return {n: _struct_to_dict(obj[n]) for n in obj.dtype.names}
        if obj.dtype == object:
            if obj.ndim == 0:
                return _struct_to_dict(obj.item())
            return [_struct_to_dict(v) for v in obj.flat]
        if obj.ndim == 0:
            return obj.item()
        if obj.size == 1:
            return float(obj.flat[0])
        return obj
    return obj


def _cst_to_list(cst_raw: np.ndarray) -> list:
    n_rows = cst_raw.shape[0]
    n_cols = cst_raw.shape[1] if cst_raw.ndim > 1 else 1
    cst = []
    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            cell = cst_raw[i, j] if cst_raw.ndim > 1 else cst_raw[i]
            if isinstance(cell, np.ndarray):
                if cell.dtype.names:
                    val = _struct_to_dict(cell)
                elif cell.dtype == object:
                    val = [_struct_to_dict(v) for v in cell.flat]
                else:
                    val = cell
            else:
                val = cell
            row.append(val)
        cst.append(row)
    return cst


# ---------------------------------------------------------------------------
# Main example
# ---------------------------------------------------------------------------

def run_example8():
    from matRad.steering.stf_generator import generate_stf
    from matRad.doseCalc.calc_dose_influence import calc_dose_influence
    from matRad.optimization.fluence_optimization import fluence_optimization
    from matRad.planAnalysis.plan_analysis import plan_analysis
    from matRad.geometry.geometry import get_iso_center

    print("=" * 60)
    print("pyMatRad Example 8: Photon VMAT Treatment Plan (TG119)")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Patient data
    # -----------------------------------------------------------------------
    print("\nLoading TG119 phantom...")
    try:
        ct, cst = load_tg119()
        print(f"  CT dimensions : {ct.get('cubeDim', 'unknown')}")
        print(f"  CT resolution : {ct.get('resolution', 'unknown')}")
        print(f"  Structures    : {len(cst)}")
    except Exception as e:
        print(f"  TG119.mat not found ({e}), using synthetic phantom.")
        from matRad.phantoms.builder import PhantomBuilder
        from matRad.optimization.DoseObjectives import SquaredDeviation, SquaredOverdosing
        builder = PhantomBuilder([100, 100, 50], [3, 3, 3], 1)
        obj1 = SquaredDeviation(penalty=800, d_ref=2.0)
        builder.add_spherical_target("PTV", radius=20, objectives=[obj1.to_dict()], HU=0)
        builder.add_box_oar("OAR", [60, 40, 40], offset=[0, -15, 0], HU=0)
        ct, cst = builder.get_ct_cst()

    # -----------------------------------------------------------------------
    # 2. Plan definition
    #
    #   gantryAngles  : arc anchor points; [-180, 180] = full 360° arc
    #   maxGantryAngleSpacing    : fine dose-calc beam spacing [deg]
    #   maxDAOGantryAngleSpacing : DAO control-point spacing [deg]
    #   maxFMOGantryAngleSpacing : FMO control-point spacing [deg]
    #   generator     : 'PhotonVMAT' selects the arc STF generator
    # -----------------------------------------------------------------------
    pln = {
        "radiationMode": "photons",
        "machine": "Generic",
        "numOfFractions": 30,
        "bioModel": "none",
        "multScen": "nomScen",
        "propStf": {
            "gantryAngles": [-180, 180],      # arc anchor points
            "couchAngles": [0, 0],
            "bixelWidth": 5,
            "isoCenter": None,
            "visMode": 0,
            "addMargin": True,
            "generator": "PhotonVMAT",        # select VMAT arc generator
            "maxGantryAngleSpacing": 15,       # fine beam spacing [deg]
            "maxDAOGantryAngleSpacing": 30,    # DAO control-point spacing [deg]
            "maxFMOGantryAngleSpacing": 45,    # FMO control-point spacing [deg]
            "continuousAperture": False,       # step-and-shoot mode
        },
        "propOpt": {
            "runDAO": False,
            "runSequencing": False,
            "runVMAT": True,
        },
        "propDoseCalc": {
            "doseGrid": {
                "resolution": {"x": 3, "y": 3, "z": 3},
            },
            "enableDijSampling": False,
        },
    }

    print(f"\nArc anchors : {pln['propStf']['gantryAngles']} deg")
    print(f"Fine spacing: {pln['propStf']['maxGantryAngleSpacing']} deg")
    print(f"DAO spacing : {pln['propStf']['maxDAOGantryAngleSpacing']} deg")
    print(f"FMO spacing : {pln['propStf']['maxFMOGantryAngleSpacing']} deg")

    # -----------------------------------------------------------------------
    # 3. Arc STF generation
    # -----------------------------------------------------------------------
    print("\nGenerating arc STF...")
    stf = generate_stf(ct, cst, pln)
    print(f"  Total arc beams : {len(stf)}")
    print(f"  Fine angles     : {[b['gantryAngle'] for b in stf]}")

    # Summarise beam types
    n_fmo = sum(1 for b in stf if b.get("propVMAT", {}).get("FMOBeam", False))
    n_dao = sum(1 for b in stf if b.get("propVMAT", {}).get("DAOBeam", False))
    n_rays = stf[0]["numOfRays"]
    print(f"  FMO beams       : {n_fmo}")
    print(f"  DAO beams       : {n_dao}")
    print(f"  Rays per beam   : {n_rays} (master ray set, uniform across arc)")

    # Sample propVMAT for the first beam
    b0 = stf[0]
    vmat0 = b0.get("propVMAT", {})
    print(f"\n  Beam 0 ({b0['gantryAngle']}°) propVMAT:")
    print(f"    FMOBeam          : {vmat0.get('FMOBeam')}")
    print(f"    DAOBeam          : {vmat0.get('DAOBeam')}")
    print(f"    doseAngleBorders : {vmat0.get('doseAngleBorders')}")
    if vmat0.get("DAOBeam"):
        print(f"    DAOAngleBorders  : {vmat0.get('DAOAngleBorders')}")
        print(f"    timeFacCurr      : {vmat0.get('timeFacCurr'):.4f}")
        print(f"    timeFac          : {vmat0.get('timeFac')}")

    # -----------------------------------------------------------------------
    # 4. Dose influence calculation
    # -----------------------------------------------------------------------
    print("\nCalculating dose influence matrix (all arc beams)...")
    dij = calc_dose_influence(ct, cst, stf, pln)
    print(f"  DIJ shape : {dij['physicalDose'][0].shape}")
    print(f"  Non-zeros : {dij['physicalDose'][0].nnz}")

    # -----------------------------------------------------------------------
    # 5. Fluence optimisation
    #
    #    NOTE: This runs fluence optimisation over ALL arc beams (dense arc
    #    IMRT), NOT the proper VMAT pipeline.  A full VMAT plan would require:
    #      a) FMO on FMO beams only
    #      b) Arc leaf sequencing (matRad_arcSequencing)
    #      c) Direct Aperture Optimisation (matRad_directApertureOptimization)
    #    Steps (b) and (c) are not yet ported to pyMatRad.
    # -----------------------------------------------------------------------
    print("\nOptimising fluence (dense-arc FMO, all beams)...")
    result = fluence_optimization(dij, cst, pln)
    d_min = result["physicalDose"].min()
    d_max = result["physicalDose"].max()
    print(f"  Dose: min={d_min:.3f}  max={d_max:.3f}  Gy/frac")

    # -----------------------------------------------------------------------
    # 6. Plan analysis
    # -----------------------------------------------------------------------
    result = plan_analysis(result, ct, cst, stf, pln)

    print("\n--- Quality Indicators ---")
    nfx = pln["numOfFractions"]
    for qi in result.get("qi", []):
        name = qi.get("name", "VOI")
        vtype = qi.get("type", "")
        d_mean = qi.get("D_mean", 0) * nfx
        d_95 = qi.get("D_95", 0) * nfx
        print(f"  {name:20s} ({vtype:8s})  D_mean={d_mean:.1f} Gy  D_95={d_95:.1f} Gy")

    # -----------------------------------------------------------------------
    # 7. Visualisation
    # -----------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from gui.matrad_gui import plot_slice
        from matRad.geometry.geometry import world_to_cube_index

        iso = get_iso_center(cst, ct)
        iso_idx = world_to_cube_index(np.atleast_2d(iso), ct)[0]
        slice_z = int(iso_idx[2])
        dose_total = result["physicalDose"] * nfx
        dose_window = [0, float(dose_total.max())]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("pyMatRad Example 8 – Photon VMAT (dense-arc FMO)", fontsize=12)

        plot_slice(ct, cst=cst, dose=dose_total, plane=3, slice_idx=slice_z,
                   dose_alpha=0.7, dose_window=dose_window,
                   title=f"Axial slice  z={slice_z}", ax=axes[0])

        # DVH
        ax_dvh = axes[1]
        colors = plt.colormaps["tab10"](np.linspace(0, 1, len(result.get("dvh", []))))
        for i, dvh in enumerate(result.get("dvh", [])):
            dv = dvh.get("doseValues", [])
            vp = dvh.get("volumePoints", [])
            if dv is not None and len(dv) > 0:
                ax_dvh.plot(np.asarray(dv) * nfx, vp, color=colors[i][:3],
                            label=dvh.get("name", f"VOI {i+1}"), linewidth=2)
        ax_dvh.set_xlabel("Dose (Gy)")
        ax_dvh.set_ylabel("Volume (%)")
        ax_dvh.set_title("DVH")
        ax_dvh.legend(fontsize=8)
        ax_dvh.grid(True, alpha=0.3)
        ax_dvh.set_ylim(0, 105)

        out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyMatRad_example8.png")
        plt.tight_layout()
        plt.savefig(out_file, dpi=100, bbox_inches="tight")
        print(f"\nFigure saved to {out_file}")
        plt.close()

    except Exception as e:
        print(f"\nVisualisation skipped: {e}")

    print("\nExample 8 complete.")
    return result, stf


if __name__ == "__main__":
    result, stf = run_example8()
