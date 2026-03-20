"""
MatRad_Config - Global configuration singleton for pyMatRad.

Python port of MatRad_Config.m
"""

import os
import logging


class MatRad_Config:
    """
    Singleton configuration class for pyMatRad.
    Manages global defaults, logging, and paths.
    """

    _instance = None

    # Default properties for dose calculation
    _defaults = {
        "propDoseCalc": {
            "geometricLateralCutOff": 50,       # mm
            "dosimetricLateralCutOff": 0.995,
            "ssdDensityThreshold": 0.05,
            "useGivenEqDensityCube": False,
            "ignoreOutsideDensities": False,
            "useCustomPrimaryPhotonFluence": False,
            "kernelCutOff": 20,                 # mm
        },
        "propStf": {
            "bixelWidth": 5,                    # mm
            "gantryAngles": [0],
            "couchAngles": [0],
            "visMode": 0,
            "isoCenter": None,
            "addMargin": True,
            "generator": None,
            "centered": True,
            "fillEmptyBixels": False,
        },
        "propOpt": {
            "optimizer": "scipy",
            "runDAO": False,
            "runSequencing": False,
            "facets": False,
        },
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Logging level: 1=errors only, 2=+warnings, 3=+info, 4=+deprecation, 5=debug
        self.log_level = 3
        self.keep_log = False
        self.write_log = False
        self.disable_gui = False
        self.dev_mode = False
        self.edu_mode = False

        # Set up Python logger
        self._logger = logging.getLogger("pyMatRad")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[pyMatRad] %(levelname)s: %(message)s"))
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)

        # Paths
        self.matrad_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.user_folders = [os.path.join(self.matrad_root, "userdata")]

    @classmethod
    def instance(cls):
        """Get singleton instance (matches MATLAB interface)."""
        return cls()

    @property
    def defaults(self):
        """Return nested defaults object with attribute-style access."""
        return _DictAsAttr(self._defaults)

    @property
    def primary_user_folder(self):
        """Returns first user folder."""
        if self.user_folders:
            return self.user_folders[0]
        return os.path.expanduser("~")

    def disp_info(self, msg, *args):
        """Display info message (log level >= 3)."""
        if self.log_level >= 3:
            if args:
                print(msg % args, end="")
            else:
                print(msg, end="")

    def disp_warning(self, msg, *args):
        """Display warning message (log level >= 2)."""
        if self.log_level >= 2:
            if args:
                self._logger.warning(msg % args)
            else:
                self._logger.warning(msg)

    def disp_error(self, msg, *args):
        """Raise error."""
        if args:
            raise RuntimeError(msg % args)
        raise RuntimeError(msg)

    def disp_deprecation_warning(self, msg, *args):
        """Display deprecation warning (log level >= 4)."""
        if self.log_level >= 4:
            if args:
                self._logger.warning("DEPRECATED: " + msg % args)
            else:
                self._logger.warning("DEPRECATED: " + msg)


class _DictAsAttr:
    """Helper to allow dict['key'] and dict.key style access."""

    def __init__(self, d):
        self._d = d

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._d:
            val = self._d[name]
            if isinstance(val, dict):
                return _DictAsAttr(val)
            return val
        raise AttributeError(f"No config attribute '{name}'")

    def __getitem__(self, key):
        return self.__getattr__(key)
