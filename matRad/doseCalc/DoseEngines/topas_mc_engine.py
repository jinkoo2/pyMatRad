"""
TOPAS MC dose engine (Python port).

Python port of matRad_TopasMCEngine.m.  Generates TOPAS parameter files for
the CT geometry and photon beam setup, optionally executes the TOPAS binary,
and reads the scored binary dose back into a pyMatRad dij structure.

Modes
-----
externalCalculation = 'off'     — write files and execute locally (default)
externalCalculation = 'write'   — write files only (for cluster submission)
externalCalculation = <folder>  — read previously computed results from folder

Requirements
------------
TOPAS >= 3.7 must be installed and the executable path set via
  pln['propDoseCalc']['topasExec']
or the environment variable TOPAS_EXEC.

For the dij matrix the engine runs one TOPAS simulation per beam with all
beamlets at uniform weight 1.  The resulting dose cube is split uniformly
across bixels within that beam (approximation; sufficient for validation).
For a full per-bixel dij, set pln['propDoseCalc']['calcDij'] = True — this
runs one TOPAS simulation per bixel (much slower).
"""

import io
import os
import struct
import subprocess
import tempfile
import shutil
import time
import zipfile
import numpy as np
import scipy.sparse as sp
from typing import Optional

from .dose_engine_base import DoseEngineBase
from ...config import MatRad_Config
from ...geometry import get_world_axes
from ...geometry.geometry import get_rotation_matrix, world_to_cube_coords


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TOPAS_EXEC    = "topas"          # override with env TOPAS_EXEC
DEFAULT_N_HISTORIES   = 100_000          # per beam (increase for clinical use)
DEFAULT_N_THREADS     = 0               # 0 = all logical cores
DEFAULT_ENERGY_MV     = 6.0             # effective photon energy [MV] for beam label


# ---------------------------------------------------------------------------
# Helper: write binary CT cube
# ---------------------------------------------------------------------------

def _hu_to_material_tag(hu_cube: np.ndarray) -> np.ndarray:
    """
    Convert HU values to integer material tags for TOPAS ByTagNumber converter.
    Tags: 0=air, 1=lung, 2=water/soft-tissue, 3=bone
    """
    hu = np.asarray(hu_cube)
    tags = np.full(hu.shape, 2, dtype=np.int16)  # default: water
    tags[hu < -700]  = 1   # lung
    tags[hu < -950]  = 0   # air
    tags[hu >= 101]  = 3   # bone
    return tags


def _write_ct_binary(path: str, hu_cube: np.ndarray):
    """Write CT material-tag binary in Fortran order (TOPAS ImageCube ByTagNumber)."""
    arr = _hu_to_material_tag(hu_cube)
    # TOPAS reads Z-fastest (Fortran col-major = MATLAB order)
    with open(path, "wb") as f:
        f.write(arr.ravel(order="F").tobytes())


def _read_topas_dose_binary(header_path: str, data_path: str) -> Optional[np.ndarray]:
    """
    Read a TOPAS scored binary dose file.

    TOPAS binary scorer output: float32 array, dimensions from .binheader.
    """
    if not os.path.exists(header_path) or not os.path.exists(data_path):
        return None

    nx = ny = nz = None
    with open(header_path, "r") as fh:
        for line in fh:
            l = line.strip()
            if l.startswith("Bins In X:"):
                nx = int(l.split(":")[1])
            elif l.startswith("Bins In Y:"):
                ny = int(l.split(":")[1])
            elif l.startswith("Bins In Z:"):
                nz = int(l.split(":")[1])

    if None in (nx, ny, nz):
        return None

    n_vox = nx * ny * nz
    with open(data_path, "rb") as fd:
        data = np.frombuffer(fd.read(n_vox * 4), dtype=np.float32)

    if len(data) < n_vox:
        return None

    # TOPAS writes in C order (X-fastest → reshape as (Nz,Ny,Nx) then transpose)
    cube = data[:n_vox].reshape((nz, ny, nx)).transpose(2, 1, 0)  # → (Nx,Ny,Nz)
    # Convert to matRad order (Ny,Nx,Nz)
    return cube.transpose(1, 0, 2)   # (Ny, Nx, Nz)


# ---------------------------------------------------------------------------
# Engine class
# ---------------------------------------------------------------------------

class TopasMCEngine(DoseEngineBase):
    """
    TOPAS MC photon dose engine (Python port of matRad_TopasMCEngine.m).

    Generates TOPAS parameter files from the matRad plan, calls the TOPAS
    binary, and reads the scored dose cube back into a dij sparse matrix.

    Parameters (pln['propDoseCalc'])
    --------------------------------
    topasExec          : str   — path to TOPAS executable (local mode)
    workingDir         : str   — folder for TOPAS files  (default: tmp dir)
    externalCalculation: str   — 'off' | 'write' | <result folder>
    numHistories       : int   — histories per beam
    numThreads         : int   — CPU threads (0 = auto)
    calcDij            : bool  — per-bixel dij (slow) vs total dose only
    topasApiUrl        : str   — base URL of OpenTOPAS API server
                                 (e.g. 'http://localhost:7778'); when set,
                                 simulations run on the remote server instead
                                 of a local TOPAS installation
    topasApiToken      : str   — Bearer token for the API server
    """

    name       = "TOPAS"
    short_name = "TOPAS"
    possible_radiation_modes = ["photons"]

    def __init__(self, pln: Optional[dict] = None):
        super().__init__(pln)

        prop = (pln or {}).get("propDoseCalc", {})

        self.topas_exec = (
            prop.get("topasExec") or
            os.environ.get("TOPAS_EXEC", DEFAULT_TOPAS_EXEC)
        )
        self.g4_data_dir = (
            prop.get("g4DataDirectory") or
            os.environ.get("TOPAS_G4_DATA_DIR", "")
        )
        self.working_dir          = prop.get("workingDir", "")
        self.external_calculation = prop.get("externalCalculation", "off")
        self.num_histories        = int(prop.get("numHistories", DEFAULT_N_HISTORIES))
        self.num_threads          = int(prop.get("numThreads",   DEFAULT_N_THREADS))
        self.calc_dij             = bool(prop.get("calcDij",     False))
        self.topas_api_url        = prop.get("topasApiUrl", "").rstrip("/")
        self.topas_api_token      = prop.get("topasApiToken", "")

        self._tmp_dir_created = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_dose_calc(self, ct: dict, cst: list, stf: list) -> dict:
        dij = super()._init_dose_calc(ct, cst, stf)

        # Prepare working directory
        if not self.working_dir:
            self.working_dir = tempfile.mkdtemp(prefix="pymatrad_topas_")
            self._tmp_dir_created = True
        os.makedirs(self.working_dir, exist_ok=True)

        # Load machine for SAD / SCD
        pln = self._pln or {}
        from ...basedata import load_machine
        self.machine = load_machine(pln)

        # Density cube
        ct = self._calc_water_eq_density(ct)
        self._cube_wed = ct.get("cube", [])

        n_vox    = dij["doseGrid"]["numOfVoxels"]
        n_bixels = dij["totalNumOfBixels"]
        dij["physicalDose"] = [sp.lil_matrix((n_vox, n_bixels))]
        return dij

    def _calc_water_eq_density(self, ct: dict) -> dict:
        ct = get_world_axes(ct)
        if "cube" in ct and ct["cube"] is not None:
            if not isinstance(ct["cube"], list):
                ct["cube"] = [np.asarray(ct["cube"])]
            return ct
        num_scen     = ct.get("numOfCtScen", 1)
        cube_hu_list = ct.get("cubeHU", [])
        ct["cube"]   = []
        for s in range(num_scen):
            hu  = (cube_hu_list[s] if s < len(cube_hu_list) else cube_hu_list[0]).astype(float)
            red = np.where(hu <= -1000, 0.0, 1.0 + hu / 1000.0)
            ct["cube"].append(np.clip(red, 0.0, 3.0))
        return ct

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _calc_dose(self, ct: dict, cst: list, stf: list, dij: dict) -> dict:
        cfg = MatRad_Config.instance()

        ct = get_world_axes(ct)
        ct = self._calc_water_eq_density(ct)

        # ---- validate TOPAS is reachable (unless read-only or API mode) ----
        mode = self.external_calculation
        if mode not in ("write",) and not os.path.isdir(str(mode)) \
                and not self.topas_api_url:
            topas_ok = shutil.which(self.topas_exec) is not None
            if not topas_ok:
                raise RuntimeError(
                    f"TOPAS executable not found: '{self.topas_exec}'.\n"
                    f"Install TOPAS (https://topas.mgh.harvard.edu/) and set\n"
                    f"  pln['propDoseCalc']['topasExec'] = '/path/to/topas'\n"
                    f"or set the environment variable TOPAS_EXEC.\n"
                    f"Alternatively use externalCalculation='write' to generate\n"
                    f"input files only."
                )

        if os.path.isdir(str(mode)):
            # Read-only: load existing results
            cfg.disp_info(f"TOPAS: reading results from '{mode}'\n")
            return self._read_results(ct, stf, dij, result_dir=str(mode))

        # ---- write CT geometry -------------------------------------------
        cfg.disp_info(f"TOPAS: writing files to '{self.working_dir}'\n")
        hu_cube = ct["cubeHU"][0] if isinstance(ct["cubeHU"], list) else ct["cubeHU"]
        self._write_patient(ct, hu_cube)

        # ---- run per-beam (or per-bixel) simulations ---------------------
        beam_dose_cubes = {}
        for beam_idx, beam in enumerate(stf):
            cfg.disp_info(f"  TOPAS beam {beam_idx+1}/{len(stf)}: "
                          f"gantry={beam['gantryAngle']}°\n")

            if self.calc_dij:
                # Per-bixel: one simulation per beamlet (slow)
                bixel_cubes = self._run_per_bixel(ct, beam, beam_idx)
                beam_dose_cubes[beam_idx] = bixel_cubes
            else:
                # Total dose: all bixels in one simulation
                cube = self._run_beam(ct, beam, beam_idx)
                beam_dose_cubes[beam_idx] = cube

            if mode == "write":
                cfg.disp_info("  TOPAS (write-only mode): files generated.\n")

        if mode == "write":
            cfg.disp_info(f"TOPAS: input files written to '{self.working_dir}'.\n"
                          "  Submit to cluster and re-run with "
                          "externalCalculation='<result_folder>'.\n")
            # Return empty dij
            dij["physicalDose"] = [sp.csc_matrix(
                (dij["doseGrid"]["numOfVoxels"], dij["totalNumOfBixels"])
            )]
            return dij

        return self._assemble_dij(ct, stf, dij, beam_dose_cubes)

    # ------------------------------------------------------------------
    # File writing
    # ------------------------------------------------------------------

    def _write_patient(self, ct: dict, hu_cube: np.ndarray):
        """Write CT binary data file (matRad_cube.dat).

        CT geometry parameters are now inlined per-beam in _write_beam_file
        to avoid the includeFile path resolution issues on Windows.
        """
        dat_path = os.path.join(self.working_dir, "matRad_cube.dat")
        _write_ct_binary(dat_path, hu_cube)

    def _write_beam_file(self, beam: dict, beam_idx: int, out_path: str,
                         result_prefix: str, ct: dict, n_histories: int,
                         use_relative_paths: bool = False):
        """Write a self-contained TOPAS parameter file for one beam.

        All CT geometry parameters are inlined (no includeFile) because
        TOPAS on Windows cannot reliably resolve includeFile paths.
        String parameters use forward slashes (TOPAS handles these in
        quoted string values); only OutputFile uses a relative name.
        """
        ga   = float(beam["gantryAngle"])
        ca   = float(beam.get("couchAngle", 0.0))
        iso  = np.asarray(beam["isoCenter"])
        rays = beam["ray"]
        n_bixels = len(rays)

        hu_cube = ct["cubeHU"][0] if isinstance(ct["cubeHU"], list) else ct["cubeHU"]
        ny, nx, nz = hu_cube.shape
        rx = ct["resolution"]["x"]
        ry = ct["resolution"]["y"]
        rz = ct["resolution"]["z"]
        x0 = float(ct["x"][0])
        y0 = float(ct["y"][0])
        z0 = float(ct["z"][0])

        src_world = (np.asarray(beam["sourcePoint"]) - iso)

        # TOPAS string parameters accept forward slashes on Windows.
        wd_fwd = self.working_dir.replace("\\", "/")
        # In API mode all files are co-located in the server's job dir; use "./"
        ct_dir = "./" if use_relative_paths else f"{wd_fwd}/"
        # Output file: relative name (TOPAS writes to its cwd = self.working_dir)
        rp_rel = os.path.basename(result_prefix)

        lines = [
            f"# TOPAS beam {beam_idx+1}: gantry={ga}°  couch={ca}°",
            "",
            "# ---- World --------------------------------------------------",
            "s:Ge/World/Material = \"G4_AIR\"",
            "d:Ge/World/HLX      = 3000 mm",
            "d:Ge/World/HLY      = 3000 mm",
            "d:Ge/World/HLZ      = 3000 mm",
            "",
            "# ---- Patient (CT geometry) -----------------------------------",
            "s:Ge/Patient/Type               = \"TsImageCube\"",
            "s:Ge/Patient/Parent             = \"World\"",
            f"s:Ge/Patient/InputDirectory     = \"{ct_dir}\"",
            "s:Ge/Patient/InputFile          = \"matRad_cube.dat\"",
            "s:Ge/Patient/ImagingToMaterialConverter = \"MaterialTagNumber\"",
            "iv:Ge/Patient/MaterialTagNumbers = 4 0 1 2 3",
            "sv:Ge/Patient/MaterialNames      = 4 "
            "\"G4_AIR\" \"G4_LUNG_ICRP\" \"G4_WATER\" \"G4_BONE_COMPACT_ICRU\"",
            f"i:Ge/Patient/NumberOfVoxelsX    = {nx}",
            f"i:Ge/Patient/NumberOfVoxelsY    = {ny}",
            f"i:Ge/Patient/NumberOfVoxelsZ    = {nz}",
            f"d:Ge/Patient/VoxelSizeX         = {rx:.4f} mm",
            f"d:Ge/Patient/VoxelSizeY         = {ry:.4f} mm",
            f"d:Ge/Patient/VoxelSizeZ         = {rz:.4f} mm",
            f"d:Ge/Patient/TransX             = {x0 + nx*rx/2.0:.4f} mm",
            f"d:Ge/Patient/TransY             = {y0 + ny*ry/2.0:.4f} mm",
            f"d:Ge/Patient/TransZ             = {z0 + nz*rz/2.0:.4f} mm",
            "",
            "# ---- Beam nozzle (virtual source) ----------------------------",
            "s:Ge/Nozzle/Type      = \"Group\"",
            "s:Ge/Nozzle/Parent    = \"World\"",
            f"d:Ge/Nozzle/TransX    = {src_world[0]:.4f} mm",
            f"d:Ge/Nozzle/TransY    = {src_world[1]:.4f} mm",
            f"d:Ge/Nozzle/TransZ    = {src_world[2]:.4f} mm",
            f"d:Ge/Nozzle/RotX      = {ga:.4f} deg",
            f"d:Ge/Nozzle/RotY      = {ca:.4f} deg",
            f"d:Ge/Nozzle/RotZ      = 0 deg",
            "",
            "# ---- Source: flat photon beam (one field covering all bixels) -",
            "s:So/Photon/Type            = \"Beam\"",
            "s:So/Photon/Component       = \"Nozzle\"",
            "s:So/Photon/BeamParticle    = \"gamma\"",
            f"d:So/Photon/BeamEnergy      = {DEFAULT_ENERGY_MV:.1f} MeV",
            "u:So/Photon/BeamEnergySpread = 0",
            "s:So/Photon/BeamShape       = \"Rectangle\"",
            f"d:So/Photon/BeamFlatteningHalfWidth = {n_bixels * 5.0 / 2.0:.1f} mm",
            f"d:So/Photon/BeamFlatteningHalfHeight = {n_bixels * 5.0 / 2.0:.1f} mm",
            f"i:So/Photon/NumberOfHistoriesInRun = {n_histories}",
            "",
            "# ---- Scorer --------------------------------------------------",
            "s:Sc/Dose/Quantity       = \"DoseToMedium\"",
            "s:Sc/Dose/Component      = \"Patient\"",
            "b:Sc/Dose/OutputToConsole = \"False\"",
            "s:Sc/Dose/OutputType     = \"Binary\"",
            "s:Sc/Dose/IfOutputFileAlreadyExists = \"Overwrite\"",
            f"s:Sc/Dose/OutputFile     = \"{rp_rel}\"",
            "",
            "# ---- Physics -------------------------------------------------",
            "sv:Ph/Default/Modules = 1 \"g4em-standard_opt4\"",
            "",
            "# ---- Run control ---------------------------------------------",
            "i:Ts/Seed = 97",
            # Geant4 prebuilt is single-threaded; clamp to 1 to avoid MT RunManager
            f"i:Ts/NumberOfThreads = {max(self.num_threads, 1)}",
            "b:Ts/ShowCPUTime = \"True\"",
        ]
        # Prepend G4 data directory so TOPAS sets all G4xxxDATA env vars
        if self.g4_data_dir:
            g4_dir = self.g4_data_dir.replace("\\", "/")
            lines.insert(0, f"s:Ts/G4DataDirectory = \"{g4_dir}\"")
            lines.insert(1, "")
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _run_beam(self, ct: dict, beam: dict, beam_idx: int) -> Optional[np.ndarray]:
        """Write and run a TOPAS simulation for one beam; return dose cube."""
        cfg = MatRad_Config.instance()

        param_file    = os.path.join(self.working_dir,
                                     f"beam_{beam_idx:03d}.txt")
        result_prefix = os.path.join(self.working_dir,
                                     f"dose_beam{beam_idx:03d}")

        self._write_beam_file(beam, beam_idx, param_file, result_prefix,
                              ct, self.num_histories)

        if self.external_calculation == "write":
            return None

        if self.topas_api_url:
            return self._run_beam_via_api(ct, beam, beam_idx)

        cfg.disp_info(f"    Running TOPAS for beam {beam_idx+1}...\n")
        cmd = [self.topas_exec, param_file]

        # On Windows, add Geant4/GDCM DLL directories to PATH so topas.exe
        # can find its shared libraries when launched from Python subprocess.
        run_env = dict(os.environ)
        if os.name == "nt":
            topas_dir = os.path.dirname(os.path.abspath(self.topas_exec))
            extra_paths = [topas_dir]
            # Auto-discover common sibling install locations
            for candidate in [
                os.path.join(topas_dir, "..", "..", "geant4", "bin"),
                os.path.join(topas_dir, "..", "..", "geant4",
                             "WIN32-VC168-10", "Geant4-10.7.4-Windows", "bin"),
                r"C:\Program Files\GDCM 3.2\bin",
                r"C:\Users\jkim20\Desktop\projects\tps\geant4\WIN32-VC168-10\Geant4-10.7.4-Windows\bin",
            ]:
                candidate = os.path.normpath(candidate)
                if os.path.isdir(candidate):
                    extra_paths.append(candidate)
            run_env["PATH"] = os.pathsep.join(extra_paths) + os.pathsep + run_env.get("PATH", "")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600,
                cwd=self.working_dir, env=run_env,
            )
            if result.returncode != 0 or result.stderr.strip():
                cfg.disp_warning(
                    f"TOPAS exited with code {result.returncode}:\n"
                    f"STDOUT: {result.stdout[-1000:]}\n"
                    f"STDERR: {result.stderr[-2000:]}"
                )
            if result.returncode != 0:
                return None
        except FileNotFoundError:
            raise RuntimeError(
                f"TOPAS executable not found: '{self.topas_exec}'. "
                "Set pln['propDoseCalc']['topasExec'] or env TOPAS_EXEC."
            )

        # Read output
        hdr = result_prefix + ".binheader"
        dat = result_prefix + ".bin"
        return _read_topas_dose_binary(hdr, dat)

    def _run_per_bixel(self, ct: dict, beam: dict, beam_idx: int):
        """Run one TOPAS simulation per bixel; returns list of dose cubes."""
        cfg  = MatRad_Config.instance()
        rays = beam["ray"]
        cubes = []
        for ri, ray in enumerate(rays):
            single_beam = dict(beam)
            single_beam["ray"] = [ray]
            single_beam["totalNumOfBixels"] = 1
            single_beam["numOfRays"] = 1
            cube = self._run_beam(ct, single_beam, beam_idx * 10000 + ri)
            cubes.append(cube)
        return cubes

    # ------------------------------------------------------------------
    # API execution
    # ------------------------------------------------------------------

    def _run_beam_via_api(self, ct: dict, beam: dict,
                          beam_idx: int) -> Optional[np.ndarray]:
        """Submit a beam simulation to the OpenTOPAS REST API, wait for
        completion, download the results zip, and return the dose cube."""
        try:
            import requests as _requests
        except ImportError:
            raise RuntimeError(
                "The 'requests' package is required for API mode. "
                "Install it with: pip install requests"
            )

        cfg = MatRad_Config.instance()

        param_file    = os.path.join(self.working_dir, f"beam_{beam_idx:03d}.txt")
        result_prefix = f"dose_beam{beam_idx:03d}"
        ct_file       = os.path.join(self.working_dir, "matRad_cube.dat")

        # Write param file with relative InputDirectory (files co-located on server)
        self._write_beam_file(beam, beam_idx, param_file, result_prefix,
                              ct, self.num_histories, use_relative_paths=True)

        headers = {}
        if self.topas_api_token:
            headers["Authorization"] = f"Bearer {self.topas_api_token}"

        # ── 1. Submit job ───────────────────────────────────────────────
        cfg.disp_info(f"    Submitting beam {beam_idx+1} to {self.topas_api_url}...\n")
        with open(param_file, "rb") as pf, open(ct_file, "rb") as cf:
            resp = _requests.post(
                f"{self.topas_api_url}/jobs",
                files=[
                    ("param_file",  (os.path.basename(param_file), pf, "text/plain")),
                    ("input_files", ("matRad_cube.dat", cf, "application/octet-stream")),
                ],
                headers=headers,
                timeout=60,
            )
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
        cfg.disp_info(f"    Job {job_id} queued.\n")

        # ── 2. Poll until done ──────────────────────────────────────────
        while True:
            time.sleep(5)
            poll = _requests.get(
                f"{self.topas_api_url}/jobs/{job_id}",
                headers=headers,
                timeout=10,
            )
            poll.raise_for_status()
            status = poll.json()["status"]
            cfg.disp_info(f"    Job {job_id}: {status}\n")
            if status in ("done", "failed", "cancelled"):
                break

        if status != "done":
            cfg.disp_warning(
                f"TOPAS API job {job_id} ended with status '{status}'."
            )
            return None

        # ── 3. Download results zip to a temp file ──────────────────────
        cfg.disp_info(f"    Downloading results for job {job_id}...\n")
        dl = _requests.get(
            f"{self.topas_api_url}/jobs/{job_id}/results",
            headers=headers,
            timeout=300,
            stream=True,
        )
        dl.raise_for_status()

        tmp_fd, tmp_zip = tempfile.mkstemp(suffix=".zip")
        os.close(tmp_fd)
        try:
            with open(tmp_zip, "wb") as f:
                for chunk in dl.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            with zipfile.ZipFile(tmp_zip) as zf:
                for member in zf.namelist():
                    if member.endswith((".bin", ".binheader", ".log")):
                        zf.extract(member, self.working_dir)
        finally:
            os.unlink(tmp_zip)

        # ── 4. Read dose ────────────────────────────────────────────────
        hdr = os.path.join(self.working_dir, result_prefix + ".binheader")
        dat = os.path.join(self.working_dir, result_prefix + ".bin")
        return _read_topas_dose_binary(hdr, dat)

    # ------------------------------------------------------------------
    # Result reading (external calculation)
    # ------------------------------------------------------------------

    def _read_results(self, ct: dict, stf: list, dij: dict,
                      result_dir: str) -> dict:
        """Read TOPAS output from a previously computed result directory."""
        cfg = MatRad_Config.instance()
        beam_cubes = {}
        for beam_idx in range(len(stf)):
            prefix = os.path.join(result_dir, f"dose_beam{beam_idx:03d}")
            cube = _read_topas_dose_binary(prefix + ".binheader", prefix + ".bin")
            if cube is None:
                cfg.disp_warning(f"  No TOPAS result found for beam {beam_idx+1}.")
            beam_cubes[beam_idx] = cube
        return self._assemble_dij(ct, stf, dij, beam_cubes)

    # ------------------------------------------------------------------
    # DIJ assembly
    # ------------------------------------------------------------------

    def _assemble_dij(self, ct: dict, stf: list, dij: dict,
                      beam_dose_cubes: dict) -> dict:
        """
        Convert TOPAS dose cubes to a sparse dij matrix.

        For uniform-weight total-dose mode, each bixel in the beam receives
        1/n_bixels of the beam's total dose (uniform approximation).
        For per-bixel mode, each column gets its individual cube.
        """
        cfg = MatRad_Config.instance()
        dose_grid = dij["doseGrid"]
        n_vox     = dose_grid["numOfVoxels"]
        n_bixels  = dij["totalNumOfBixels"]

        coo_rows, coo_cols, coo_data = [], [], []

        bixel_offset = 0
        for beam_idx, beam in enumerate(stf):
            n_bixels_beam = beam["totalNumOfBixels"]
            cube_or_list  = beam_dose_cubes.get(beam_idx)

            if cube_or_list is None:
                bixel_offset += n_bixels_beam
                continue

            if self.calc_dij and isinstance(cube_or_list, list):
                # Per-bixel mode
                for bi, cube in enumerate(cube_or_list):
                    if cube is None:
                        continue
                    dose_flat = self._resample_to_dose_grid(cube, ct, dose_grid)
                    nz = dose_flat > 0
                    if np.any(nz):
                        coo_rows.append(self._V_dose_grid[nz] - 1)
                        coo_cols.append(np.full(int(nz.sum()),
                                                bixel_offset + bi, dtype=np.int32))
                        coo_data.append(dose_flat[nz])
            else:
                # Uniform mode: distribute total dose equally across bixels
                cube = cube_or_list
                dose_flat = self._resample_to_dose_grid(cube, ct, dose_grid)
                dose_per_bixel = dose_flat / max(n_bixels_beam, 1)
                nz = dose_per_bixel > 0
                if np.any(nz):
                    rows = (self._V_dose_grid[nz] - 1).astype(np.int32)
                    vals = dose_per_bixel[nz]
                    for bi in range(n_bixels_beam):
                        coo_rows.append(rows)
                        coo_cols.append(np.full(len(rows),
                                                bixel_offset + bi, dtype=np.int32))
                        coo_data.append(vals)

            # Fill bookkeeping
            for bi in range(n_bixels_beam):
                col = bixel_offset + bi
                dij["bixelNum"][col] = bi + 1
                dij["rayNum"][col]   = bi + 1
                dij["beamNum"][col]  = beam_idx + 1

            bixel_offset += n_bixels_beam

        if coo_rows:
            all_r = np.concatenate(coo_rows)
            all_c = np.concatenate(coo_cols)
            all_d = np.concatenate(coo_data)
            dose_csc = sp.coo_matrix(
                (all_d, (all_r, all_c)), shape=(n_vox, n_bixels)
            ).tocsc()
        else:
            dose_csc = sp.csc_matrix((n_vox, n_bixels))

        dij["physicalDose"] = [dose_csc]
        return dij

    # ------------------------------------------------------------------
    # Grid resampling
    # ------------------------------------------------------------------

    def _resample_to_dose_grid(self, cube: np.ndarray, ct: dict,
                               dose_grid: dict) -> np.ndarray:
        """
        Resample a (Ny,Nx,Nz) dose cube from CT grid to dose grid,
        then extract valid voxels (matching V_dose_grid order).
        """
        from scipy.interpolate import RegularGridInterpolator

        if cube is None:
            return np.zeros(len(self._V_dose_grid))

        # CT grid axes
        ny_ct, nx_ct, nz_ct = cube.shape
        y_ct = ct["y"][:ny_ct]
        x_ct = ct["x"][:nx_ct]
        z_ct = ct["z"][:nz_ct]

        # Trim if needed
        cube = cube[:len(y_ct), :len(x_ct), :len(z_ct)]

        interp = RegularGridInterpolator(
            (y_ct, x_ct, z_ct), cube,
            method="linear", bounds_error=False, fill_value=0.0,
        )

        # Sample at dose-grid valid voxel world coordinates
        pts = self._vox_world_coords_dose_grid   # (N, 3)
        dose_at_vox = interp(pts)
        return np.maximum(dose_at_vox, 0.0)
