"""
Dose objective functions for inverse planning.

Python port of matRad dose objective classes:
- matRad_DoseObjective.m
- matRad_SquaredDeviation.m
- matRad_SquaredOverdosing.m
- matRad_SquaredUnderdosing.m
- matRad_MeanDose.m
- matRad_MaxDVH.m
- matRad_MinDVH.m
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class DoseObjective(ABC):
    """
    Abstract base class for dose objectives.
    Port of matRad_DoseObjective.m
    """

    name = "Abstract Dose Objective"
    parameter_names = []
    parameter_types = []

    def __init__(self, penalty: float = 1.0):
        self.penalty = float(penalty)
        self.parameters = []

    @abstractmethod
    def compute_dose_objective_function(self, dose: np.ndarray) -> float:
        """Compute objective function value."""
        pass

    @abstractmethod
    def compute_dose_objective_gradient(self, dose: np.ndarray) -> np.ndarray:
        """Compute gradient of objective function."""
        pass

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "class": self.__class__.__name__,
            "penalty": self.penalty,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DoseObjective":
        """Deserialize from dict."""
        obj = cls(d.get("penalty", 1.0))
        obj.parameters = d.get("parameters", [])
        return obj


class SquaredDeviation(DoseObjective):
    """
    Penalized least squares objective.
    Port of matRad_SquaredDeviation.m

    f(dose) = (1/N) * ||dose - d_ref||^2
    grad = (2/N) * (dose - d_ref)
    """

    name = "Squared Deviation"
    parameter_names = ["d_ref"]
    parameter_types = ["dose"]

    def __init__(self, penalty: float = 1.0, d_ref: float = 60.0):
        super().__init__(penalty)
        self.parameters = [float(d_ref)]

    @property
    def d_ref(self) -> float:
        return self.parameters[0]

    def compute_dose_objective_function(self, dose: np.ndarray) -> float:
        deviation = dose - self.d_ref
        return float(np.dot(deviation, deviation) / len(dose))

    def compute_dose_objective_gradient(self, dose: np.ndarray) -> np.ndarray:
        deviation = dose - self.d_ref
        return 2.0 / len(dose) * deviation


class SquaredOverdosing(DoseObjective):
    """
    Penalized overdosing objective.
    Port of matRad_SquaredOverdosing.m

    Only penalizes dose > d_ref.
    f(dose) = (1/N) * ||max(0, dose - d_ref)||^2
    """

    name = "Squared Overdosing"
    parameter_names = ["d_ref"]
    parameter_types = ["dose"]

    def __init__(self, penalty: float = 1.0, d_ref: float = 0.0):
        super().__init__(penalty)
        self.parameters = [float(d_ref)]

    @property
    def d_ref(self) -> float:
        return self.parameters[0]

    def compute_dose_objective_function(self, dose: np.ndarray) -> float:
        overdose = np.maximum(0.0, dose - self.d_ref)
        return float(np.dot(overdose, overdose) / len(dose))

    def compute_dose_objective_gradient(self, dose: np.ndarray) -> np.ndarray:
        overdose = np.maximum(0.0, dose - self.d_ref)
        return 2.0 / len(dose) * overdose


class SquaredUnderdosing(DoseObjective):
    """
    Penalized underdosing objective.
    Port of matRad_SquaredUnderdosing.m

    Only penalizes dose < d_ref.
    f(dose) = (1/N) * ||min(0, dose - d_ref)||^2
    """

    name = "Squared Underdosing"
    parameter_names = ["d_ref"]
    parameter_types = ["dose"]

    def __init__(self, penalty: float = 1.0, d_ref: float = 60.0):
        super().__init__(penalty)
        self.parameters = [float(d_ref)]

    @property
    def d_ref(self) -> float:
        return self.parameters[0]

    def compute_dose_objective_function(self, dose: np.ndarray) -> float:
        underdose = np.minimum(0.0, dose - self.d_ref)
        return float(np.dot(underdose, underdose) / len(dose))

    def compute_dose_objective_gradient(self, dose: np.ndarray) -> np.ndarray:
        underdose = np.minimum(0.0, dose - self.d_ref)
        return 2.0 / len(dose) * underdose


class MeanDose(DoseObjective):
    """
    Mean dose objective.
    Port of matRad_MeanDose.m

    Minimizes |mean(dose) - d_ref|.
    """

    name = "Mean Dose"
    parameter_names = ["d_ref"]
    parameter_types = ["dose"]

    def __init__(self, penalty: float = 1.0, d_ref: float = 0.0):
        super().__init__(penalty)
        self.parameters = [float(d_ref)]

    @property
    def d_ref(self) -> float:
        return self.parameters[0]

    def compute_dose_objective_function(self, dose: np.ndarray) -> float:
        d_mean = np.mean(dose)
        return float((d_mean - self.d_ref) ** 2)

    def compute_dose_objective_gradient(self, dose: np.ndarray) -> np.ndarray:
        d_mean = np.mean(dose)
        return 2.0 * (d_mean - self.d_ref) * np.ones_like(dose) / len(dose)


class MaxDVH(DoseObjective):
    """
    Maximum DVH constraint objective.
    Port of matRad_MaxDVH.m

    Penalizes doses exceeding d_ref at volume volume_threshold.
    """

    name = "Max DVH"
    parameter_names = ["d_ref", "volume_threshold"]
    parameter_types = ["dose", "volume"]

    def __init__(self, penalty: float = 1.0, d_ref: float = 50.0, vol_threshold: float = 5.0):
        super().__init__(penalty)
        self.parameters = [float(d_ref), float(vol_threshold)]

    @property
    def d_ref(self) -> float:
        return self.parameters[0]

    @property
    def vol_threshold(self) -> float:
        return self.parameters[1]

    def compute_dose_objective_function(self, dose: np.ndarray) -> float:
        # Find dose to vol_threshold% of structure
        vol_frac = self.vol_threshold / 100.0
        sorted_dose = np.sort(dose)[::-1]  # descending
        ix = max(0, int(np.round(vol_frac * len(dose))) - 1)
        d_vt = sorted_dose[ix] if len(sorted_dose) > 0 else 0.0
        overdose = max(0.0, d_vt - self.d_ref)
        return float(overdose ** 2)

    def compute_dose_objective_gradient(self, dose: np.ndarray) -> np.ndarray:
        # Smooth approximation using soft max
        vol_frac = self.vol_threshold / 100.0
        sorted_dose = np.sort(dose)[::-1]
        ix = max(0, int(np.round(vol_frac * len(dose))) - 1)
        d_vt = sorted_dose[ix] if len(sorted_dose) > 0 else 0.0

        grad = np.zeros_like(dose)
        overdose = max(0.0, d_vt - self.d_ref)
        if overdose > 0:
            # Gradient: only voxels with dose >= d_vt contribute
            mask = dose >= d_vt
            if np.any(mask):
                grad[mask] = 2.0 * overdose / np.sum(mask)
        return grad


class MinDVH(DoseObjective):
    """
    Minimum DVH constraint objective.
    Port of matRad_MinDVH.m

    Penalizes doses below d_ref at volume volume_threshold.
    """

    name = "Min DVH"
    parameter_names = ["d_ref", "volume_threshold"]
    parameter_types = ["dose", "volume"]

    def __init__(self, penalty: float = 1.0, d_ref: float = 50.0, vol_threshold: float = 95.0):
        super().__init__(penalty)
        self.parameters = [float(d_ref), float(vol_threshold)]

    @property
    def d_ref(self) -> float:
        return self.parameters[0]

    @property
    def vol_threshold(self) -> float:
        return self.parameters[1]

    def compute_dose_objective_function(self, dose: np.ndarray) -> float:
        vol_frac = self.vol_threshold / 100.0
        sorted_dose = np.sort(dose)  # ascending
        ix = min(len(sorted_dose) - 1, int(np.round((1 - vol_frac) * len(dose))))
        d_vt = sorted_dose[ix] if len(sorted_dose) > 0 else 0.0
        underdose = max(0.0, self.d_ref - d_vt)
        return float(underdose ** 2)

    def compute_dose_objective_gradient(self, dose: np.ndarray) -> np.ndarray:
        vol_frac = self.vol_threshold / 100.0
        sorted_dose = np.sort(dose)
        ix = min(len(sorted_dose) - 1, int(np.round((1 - vol_frac) * len(dose))))
        d_vt = sorted_dose[ix] if len(sorted_dose) > 0 else 0.0

        grad = np.zeros_like(dose)
        underdose = max(0.0, self.d_ref - d_vt)
        if underdose > 0:
            mask = dose <= d_vt
            if np.any(mask):
                grad[mask] = -2.0 * underdose / np.sum(mask)
        return grad
