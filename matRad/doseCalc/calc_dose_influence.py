"""
Dose influence matrix calculation.
Port of matRad_calcDoseInfluence.m
"""


def calc_dose_influence(ct: dict, cst: list, stf: list, pln: dict) -> dict:
    """
    Calculate dose influence matrix.

    Python port of matRad_calcDoseInfluence.m

    Parameters
    ----------
    ct : dict
    cst : list
    stf : list
    pln : dict

    Returns
    -------
    dict
        dij struct
    """
    from .DoseEngines.dose_engine_base import DoseEngineBase
    engine = DoseEngineBase.get_engine_from_pln(pln)
    return engine.calc_dose_influence(ct, cst, stf)
