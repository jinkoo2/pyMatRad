"""
Scenario models for robustness analysis.

Python port of matRad scenario models (NominalScenario, etc.)
"""

import numpy as np


class NominalScenario:
    """
    Nominal scenario only - no uncertainty.
    Port of matRad_NominalScenario / 'nomScen'.
    """

    def __init__(self, ct=None):
        self.num_of_ct_scen = 1
        self.tot_num_shift_scen = 1
        self.tot_num_range_scen = 1
        self.tot_num_scen = 1

        # Scenario mask: [ctScen, shiftScen, rangeScen]
        self.scen_mask = np.array([[[True]]])
        self.linear_mask = np.array([[1, 1, 1]])  # [ctScen, shiftScen, rangeScen]

        # Shifts
        self.iso_shift = np.zeros((1, 3))
        self.range_shift_rel = np.zeros(1)
        self.range_shift_abs = np.zeros(1)

    def sub2scen_ix(self, ct_scen: int, shift_scen: int, range_scen: int) -> int:
        """Get linear scenario index from subscript indices."""
        return 1  # Only one scenario

    @classmethod
    def from_pln(cls, pln_scen_name: str, ct=None):
        """Create scenario model from pln.multScen string."""
        if pln_scen_name == "nomScen" or pln_scen_name is None:
            return cls(ct)
        else:
            # Default to nominal for unknown scenario models
            print(f"Warning: Scenario model '{pln_scen_name}' not implemented, using nominal.")
            return cls(ct)
