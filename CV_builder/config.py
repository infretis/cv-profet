"""
config.py — Central configuration dataclasses for CV_builder.

All structured settings live here so that builder.py, simulationrunner.py,
and the aggregation stage import from one place rather than defining
their own inline classes.

Classes:
    Paths                   — resolved filesystem paths
    SimSettings             — [simulation] block
    CVMatSettings           — [cvmat] block
    GridDefinition          — pre-computed grid indices + OP values
    PredictivePowerSettings — [predictive_power] block
    OutputConfig            — HDF5 write control (compression, CV filter, debug)
    AggregatorConfig        — sorted-H5 aggregation stage control
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    """Resolved filesystem paths used throughout the pipeline."""
    load_dir:   Path    # Directory containing trajectory step folders
    data_file:  Path    # infretis_data.txt used for WHAM
    toml_file:  Path    # TOML settings file
    output_dir: Path    # Root output directory (e.g. CVs/)
    h5_input:   Optional[Path] = None   # Optional HDF5 input with {step}/traj.txt and {step}/order.txt datasets

    @classmethod
    def from_args(cls, args) -> "Paths":
        """
        Resolve all paths, anchoring defaults to the TOML file's parent directory
        (the *project root*) so the program works correctly regardless of CWD.

        Layout assumed by defaults:
            <project_root>/
                infretis.toml       ← always provided explicitly
                load/               ← RETIS step directories
                infretis_data.txt   ← WHAM data
                CVs/                ← HDF5 output
                CV_builder/         ← this program
        """
        try:
            toml_path = Path(args.toml).resolve(strict=True)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"TOML file does not exist: {e.args[0]}") from e

        # Project root = directory that contains the TOML file.
        # All other paths default to siblings of the TOML file.
        root = toml_path.parent

        load_arg = getattr(args, "load_dir", None)
        data_arg = getattr(args, "data",     None)
        out_arg  = getattr(args, "output_dir", None)
        h5_arg   = getattr(args, "h5_input", None)

        load_path = Path(load_arg) if load_arg else root / "load"
        data_path = Path(data_arg) if data_arg else root / "infretis_data.txt"
        out_path  = Path(out_arg)  if out_arg  else root / "CVs"
        h5_path   = Path(h5_arg)   if h5_arg   else None

        check_active_mode = bool(getattr(args, "check_active", False))

        try:
            # In HDF5-input mode, load_dir may legitimately be absent.
            load = load_path.resolve(strict=not bool(h5_path))
            # check-active mode does not use WHAM/data_file.
            data = data_path.resolve(strict=not check_active_mode)
            h5   = h5_path.resolve(strict=True) if h5_path else None
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Path does not exist: {e.args[0]}") from e

        out = out_path.resolve()   # output dir need not exist yet
        return cls(load_dir=load, data_file=data, toml_file=toml_path, output_dir=out, h5_input=h5)

    def log_summary(self) -> None:
        log.info("  load_dir   : %s", self.load_dir)
        log.info("  data_file  : %s", self.data_file)
        log.info("  toml_file  : %s", self.toml_file)
        log.info("  output_dir : %s", self.output_dir)
        if self.h5_input is not None:
            log.info("  h5_input   : %s", self.h5_input)


# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimSettings:
    """Settings read from [simulation] in the TOML."""
    interfaces: List[float]

    @classmethod
    def from_dict(cls, block: dict) -> "SimSettings":
        try:
            return cls(interfaces=block["interfaces"])
        except KeyError as e:
            raise KeyError(f"[simulation] missing required key: {e.args[0]}") from e


# ---------------------------------------------------------------------------
# CVmat settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CVMatSettings:
    """Settings read from [cvmat] in the TOML."""
    n_grid:     int
    selection:  List[float]
    start_grid: int
    end_grid:   int
    step_size:  int
    nskip:      Union[int, float] = 0

    @classmethod
    def from_dict(cls, block: dict) -> "CVMatSettings":
        try:
            return cls(
                n_grid=block["n_grid"],
                selection=block["selection"],
                start_grid=block["start_grid"],
                end_grid=block["end_grid"],
                step_size=block["step_size"],
                nskip=block.get("nskip", 0),
            )
        except KeyError as e:
            raise KeyError(f"[cvmat] missing required key: {e.args[0]}") from e


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridDefinition:
    """Pre-computed grid indices and physical OP values for this run."""
    grid_idx: np.ndarray   # (K,) int   — indices into the n_grid-point grid
    grid_pts: np.ndarray   # (K,) float — physical OP values at those indices

    @classmethod
    def from_cvmat_settings(
        cls,
        args,
        cvmat: CVMatSettings,
        interfaces: List[float],
    ) -> "GridDefinition":
        n_grid    = int(cvmat.n_grid)
        full_grid = np.linspace(min(interfaces), max(interfaces), n_grid)
        all_idx   = np.arange(n_grid)
        mode      = args.grid_mode

        if mode == "selection":
            if not cvmat.selection:
                raise KeyError("grid_mode='selection' but [cvmat].selection is empty.")
            grid_idx = np.array(cvmat.selection, dtype=int)

        elif mode == "range":
            grid_idx = all_idx[int(cvmat.start_grid) : int(cvmat.end_grid) : int(cvmat.step_size)]

        else:
            raise ValueError(f"Unknown grid_mode '{mode}'.")

        grid_idx = np.clip(grid_idx, 0, n_grid - 1)
        return cls(grid_idx=grid_idx, grid_pts=full_grid[grid_idx])




# ---------------------------------------------------------------------------
# Predictive power settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictivePowerSettings:
    """Settings read from [predictive_power] in the TOML."""
    chemical_system:   str
    n_cvar_lists:      List[int]
    n_best:            int
    optimizer:         str
    optimizer_steps:   int
    parallel_procs:    int
    h5_input:          str
    calculate_missing: bool
    recalc_lincomb:    bool

    @classmethod
    def from_dict(cls, block: dict) -> "PredictivePowerSettings":
        try:
            return cls(
                chemical_system=block["chemical_system"],
                n_cvar_lists=block["n_cvar_list"],
                n_best=block["n_best"],
                optimizer=block["optimizer"],
                optimizer_steps=block["optimizer_steps"],
                parallel_procs=block["parallel_procs"],
                h5_input=block["h5_input"],
                calculate_missing=block["calculate_missing"],
                recalc_lincomb=block["recalc_lincomb"],
            )
        except KeyError as e:
            raise KeyError(f"[predictive_power] missing required key: {e.args[0]}") from e

    def __post_init__(self) -> None:
        if self.n_best < 1:
            raise ValueError("n_best must be ≥ 1")
        if self.optimizer_steps < 1:
            raise ValueError("optimizer_steps must be ≥ 1")
        if self.parallel_procs < 1:
            raise ValueError("parallel_procs must be ≥ 1")


# ---------------------------------------------------------------------------
# Output / HDF5 control
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OutputConfig:
    """
    Controls what gets written to HDF5 and how.

    include_cvs:        If set, only CVs whose name matches an entry are written.
                        None means "write all".
    exclude_cvs:        CV names to skip (applied after include_cvs filter).
    compress_level:     gzip compression level (0 = off, 9 = max; default 4).
    checkpoint_every:   Flush the HDF5 file to disk every N RETIS steps.
    debug_cv_output:    When True, log shape/range/NaN stats for every CV
                        at DEBUG level.
    """
    include_cvs:       Optional[List[str]] = None
    exclude_cvs:       List[str]           = field(default_factory=list)
    compress_level:    int                 = 4
    checkpoint_every:  int                 = 50
    debug_cv_output:   bool                = False

    def should_write(self, cv_name: str) -> bool:
        """Return True if this CV should be written to HDF5."""
        if self.include_cvs is not None and cv_name not in self.include_cvs:
            return False
        if cv_name in self.exclude_cvs:
            return False
        return True

    @classmethod
    def from_args(cls, args) -> "OutputConfig":
        return cls(
            include_cvs=getattr(args, "h5_include_cvs", None) or None,
            exclude_cvs=getattr(args, "h5_exclude_cvs", None) or [],
            compress_level=getattr(args, "h5_compress", 4),
            debug_cv_output=getattr(args, "debug", False),
        )


# ---------------------------------------------------------------------------
# Aggregation (sorted-H5 stage)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AggregatorConfig:
    """
    Controls the sorted-H5 aggregation stage that merges per-step HDF5 files
    into a sorted matrix file consumable by PPA.

    steps_dir:       Directory containing step_*.h5 files  (e.g. CVs/890/steps/).
    out_file:        Base path for part files → part files land at
                     <out_file>.part.<rank>.h5  (e.g. CVs/890/sorted/CVmat.h5).
    merged_file:     Final merged output path (e.g. CVs/890/CVmat.h5).
    all_grids:       If True, auto-discover all numeric grid groups.
    grid_ids:        Explicit list of grid IDs (overrides all_grids when non-empty).
    modules:         CV module names to include; None = auto-discover all.
    float_dtype:     numpy dtype for output cv matrices ('f4' or 'f8').
    shard_mode:      How to split step files across workers ('stride' or 'block').
    n_workers:       Number of parallel workers on Mac (0 = os.cpu_count()).
    resume:          Skip writing a part file if it already exists.
    force_overwrite: Always overwrite existing part files.
    """
    steps_dir:       Path
    out_file:        Path          # part-file base  (lives in sorted/)
    merged_file:     Path          # final CVmat.h5  (lives at ngrid root)
    interfaces:      List[float]     = field(default_factory=list)
    cvmat_n_grid:    int             = 0
    all_grids:       bool            = True
    grid_ids:        List[int]       = field(default_factory=list)
    modules:         Optional[List[str]] = None
    float_dtype:     str             = "f4"
    shard_mode:      str             = "stride"
    n_workers:       int             = 0
    resume:          bool            = False
    force_overwrite: bool            = False

    @classmethod
    def from_args(
        cls,
        args,
        interfaces:   list[float] | None = None,
        cvmat_n_grid: int | None  = None,
        output_dir:   Path | None = None,
    ) -> "AggregatorConfig":
        """
        Build from parsed CLI args.

        `interfaces`   — physical lambda boundaries from [simulation] TOML block.
        `cvmat_n_grid` — passed from builder so we can derive the default steps_dir.
        `output_dir`   — the already-resolved output Path from Paths.from_args();
                         used as fallback when --agg-steps-dir is not provided.
        """
        # ── steps_dir ──────────────────────────────────────────────────────
        if getattr(args, "agg_steps_dir", None):
            steps_dir = Path(args.agg_steps_dir).resolve()
        else:
            base = output_dir or (Path(args.output_dir) if args.output_dir else None)
            if base is None:
                raise ValueError(
                    "Cannot determine steps_dir: provide --agg-steps-dir or --output-dir"
                )
            sub = base / str(cvmat_n_grid) if cvmat_n_grid is not None else base
            steps_dir = (sub / "steps").resolve()

        # cvs/{ngrid}/  root
        cv_root = steps_dir.parent  # strip 'steps/'

        # ── part-file base (lives in sorted/) ──────────────────────────────
        if getattr(args, "agg_out", None):
            out_file = Path(args.agg_out).resolve()
        else:
            out_file = (cv_root / "sorted" / "CVmat.h5").resolve()

        # ── final merged file (lives at ngrid root) ─────────────────────────
        merged_file = (cv_root / "CVmat.h5").resolve()

        return cls(
            steps_dir=steps_dir,
            out_file=out_file,
            merged_file=merged_file,
            interfaces=interfaces or [],
            cvmat_n_grid=cvmat_n_grid or 0,
            all_grids=not bool(getattr(args, "agg_grids", None)),
            grid_ids=list(getattr(args, "agg_grids", None) or []),
            modules=getattr(args, "agg_modules", None) or None,
            float_dtype=getattr(args, "agg_dtype", "f4"),
            shard_mode=getattr(args, "agg_shard_mode", "stride"),
            n_workers=getattr(args, "agg_workers", 0) or 0,
            resume=getattr(args, "agg_resume", False),
            force_overwrite=getattr(args, "agg_force_overwrite", False),
        )
