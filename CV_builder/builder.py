"""
builder.py — Orchestrates application construction from CLI args + TOML.

Responsibilities:
  1. Load and validate the TOML config.
  2. Construct all typed settings objects (via config.py).
  3. Discover + filter RETIS step directories.
  4. Compute WHAM weights  (weights.retis_steps is the authoritative
     filtered step list — steps without a weight are dropped here).
  5. Return a fully-initialised SimulationRunner ready to call .run().
"""
from __future__ import annotations

import glob
import importlib
import importlib.util
import logging
import os
from dataclasses import replace
from pathlib import Path

import h5py
import toml

from config import (
    GridDefinition,
    OutputConfig,
    Paths,
    PredictivePowerSettings,
    SimSettings,
    CVMatSettings,
)
from topology.base import MoleculeSystem  # type: ignore
from CV_manager.base import CVSuite  # type: ignore
from simulationrunner import SimulationRunner
from wham_weights import TrajWeights

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_runner(args) -> SimulationRunner:
    """
    Build and return a SimulationRunner from parsed CLI arguments.

    All heavy I/O (WHAM weight computation, reference-frame loading) is
    deferred to SimulationRunner.__init__ / .run().
    """
    # ── 1. Paths & config ──────────────────────────────────────────────────
    paths  = Paths.from_args(args)
    config = _load_toml(paths.toml_file)
    is_main_proc = "CV_RANK" not in os.environ
    if is_main_proc:
        paths.log_summary()

    # ── 2. Structured settings from TOML ───────────────────────────────────
    sim       = SimSettings.from_dict(config["simulation"])
    cvmat     = CVMatSettings.from_dict(config["cvmat"])
    pred_power = PredictivePowerSettings.from_dict(config["predictive_power"])
    grid      = GridDefinition.from_cvmat_settings(args, cvmat, sim.interfaces)
    out_cfg   = OutputConfig.from_args(args)

    # ── 3. Collect CV types from TOML ─────────────────────────────────────
    all_types: set[str] = set()

    # ── 4. Overwrite output if requested ──────────────────────────────────
    is_main_proc = "CV_RANK" not in os.environ
    if getattr(args, "overwrite", False) and is_main_proc:
        cv_dir = paths.output_dir / str(cvmat.n_grid)
        if cv_dir.exists():
            log.info("Overwrite requested: deleting %s", cv_dir)
            import shutil
            shutil.rmtree(cv_dir, ignore_errors=True)

    # ── 5. Discover RETIS step directories ────────────────────────────────
    check_active_mode = bool(getattr(args, "check_active", False))
    if check_active_mode:
        raw_steps = _active_steps_from_config(config)
    else:
        raw_steps = _find_retis_steps(paths)
    print(len(raw_steps))
    if is_main_proc:
        log.info("Found %d step directories (first 10: %s)", len(raw_steps), raw_steps[:10])

    raw_steps = _drop_burn_in(raw_steps, nskip=cvmat.nskip, silent=not is_main_proc)


    # ── 5. WHAM weights ────────────────────────────────────────────────────
    # NOTE: TrajWeights.retis_steps is the *filtered* list — steps without
    # a valid WHAM weight are dropped inside TrajWeights.from_inputs().
    # The runner iterates only over weights.retis_steps, so every step
    # it processes is guaranteed to have an assigned weight.
    if check_active_mode:
        # check-active mode is CV-only: skip WHAM and use unit weights.
        active_steps = [int(s) for s in raw_steps]
        weights = TrajWeights(
            weights={s: 1.0 for s in active_steps},
            retis_steps=active_steps,
        )
        if is_main_proc:
            log.info(
                "check-active mode: using %d active step(s) from [current].active (unit weights, no WHAM)",
                len(weights.retis_steps),
            )
    else:
        wham_load_dir = paths.load_dir
        if paths.h5_input is not None and not wham_load_dir.exists():
            # In HDF5-input mode, cache next to the input file if load_dir is absent.
            wham_load_dir = paths.h5_input.parent

        weights = TrajWeights.from_inputs(
            raw_steps=raw_steps,
            load_dir=wham_load_dir,
            interfaces=sim.interfaces,
            data_file=paths.data_file,
            approximate_threshold=getattr(args, 'approximate_threshold', 1e-5),
            approximate_factor=getattr(args, 'approximate_factor', 100.0),
            silent=not is_main_proc,
        )
        if is_main_proc:
            log.info(
                "WHAM: %d steps with weights (from %d raw steps)",
                len(weights.retis_steps), len(raw_steps),
            )

    # ── 6. Molecule system from [topology] TOML section ───────────────────
    mol_system = _build_mol_system(config)

    # ── 7. CV Suite from [cv_modules] TOML section ────────────────────────
    cv_suite = _build_cv_suite(config)

    return SimulationRunner(
        paths=paths,
        config=config,
        interfaces=sim.interfaces,
        cvmat=cvmat,
        grid=grid,
        pred_power=pred_power,
        all_types=all_types,
        weights=weights,
        out_cfg=out_cfg,
        mol_system=mol_system,
        cv_suite=cv_suite,
        h5_input=paths.h5_input,
    )

# ---------------------------------------------------------------------------
# Molecule system factory (reads [topology] from TOML)
# ---------------------------------------------------------------------------

# Registry of built-in systems (name → import path, class name)
# "xyz_data" is the recommended starting point for new users.
_BUILTIN_SYSTEMS: dict[str, tuple[str, str]] = {
    "xyz_data":      ("topology.xyz_reader",             "XYZTopologyReader"),
    "lammps_data":   ("topology.lammps_reader",          "LAMMPSDataReader"),
    "cp2k_water_ion": ("topology.examples.cp2k_water_ion", "CP2KWaterIonSystem"),
    # Add more built-ins here as new MoleculeSystem implementations are contributed.
}

def _build_mol_system(config: dict) -> "MoleculeSystem | None":
    """
    Instantiate the MoleculeSystem from the [topology] TOML section.

    TOML examples
    -------------
    # Recommended: minimal XYZ topology (good starting point for new users):
    [topology]
    system    = "xyz_data"     # built-in XYZ reader
    file      = "initial.xyz" # path to your topology file
    cell_size = 12.5           # optional periodic box length in Å

    # Custom user-defined system (point to your Python file):
    [topology]
    module    = "my_project/my_system.py"
    class     = "MyCustomSystem"
    cell_size = 12.5

    See infretis_example.toml for a complete minimal configuration.
    """
    topo_cfg = config.get("topology", {})

    if not topo_cfg:
        raise ValueError(
            "[topology] section is missing from infretis.toml.\n"
            "At minimum, add:\n"
            "  [topology]\n"
            '  system = "xyz_data"\n'
            '  file    = "initial.xyz"\n'
            "See CV_builder/infretis_example.toml for a complete example."
        )

    cell_size  = topo_cfg.get("cell_size", config.get("simulation", {}).get("cell_size", 0.0))
    ion_symbol = topo_cfg.get("ion_symbol", None)
    # Accept both 'topology_file' (preferred) and 'file' (legacy/XYZTopologyReader)
    topo_file  = topo_cfg.get("topology_file") or topo_cfg.get("file", None)

    # ── Custom module path ─────────────────────────────────────────────────
    if "module" in topo_cfg:
        module_path = Path(topo_cfg["module"])
        class_name  = topo_cfg.get("class")
        if not class_name:
            raise ValueError("[topology] module requires a 'class' key (the MoleculeSystem subclass)")
        if not module_path.exists():
            raise FileNotFoundError(f"[topology] module not found: {module_path}")
        spec   = importlib.util.spec_from_file_location("user_topology", module_path)
        mod    = importlib.util.module_from_spec(spec)   # type: ignore
        spec.loader.exec_module(mod)                     # type: ignore
        cls    = getattr(mod, class_name)
        log.info("[topology] loaded custom class %s from %s", class_name, module_path)
        return cls(cell_size=cell_size, ion_symbol=ion_symbol, topology_file=topo_file)

    # ── Built-in system name ───────────────────────────────────────────────
    system_name = topo_cfg.get("system", "")
    if not system_name:
        raise ValueError(
            "[topology] section requires either 'system' (built-in name) or "
            "'module' (path to a custom MoleculeSystem).\n"
            f"Built-in options: {list(_BUILTIN_SYSTEMS)}"
        )
    if system_name not in _BUILTIN_SYSTEMS:
        raise ValueError(
            f"[topology] system='{system_name}' is not recognised. "
            f"Known built-ins: {list(_BUILTIN_SYSTEMS)}. "
            f"For a custom system, use 'module = path/to/your_system.py' instead."
        )
    import_path, class_name = _BUILTIN_SYSTEMS[system_name]
    mod = importlib.import_module(import_path)
    cls = getattr(mod, class_name)
    log.info("[topology] using built-in system '%s' (%s)", system_name, class_name)
    return cls(cell_size=cell_size, ion_symbol=ion_symbol, topology_file=topo_file)


# ---------------------------------------------------------------------------
# CV Suite factory (reads [cv_modules] from TOML)
# ---------------------------------------------------------------------------

def _build_cv_suite(config: dict) -> "CVSuite":
    cv_cfg = config.get("cv_modules", {})
    module_path_str = cv_cfg.get("module", "").strip()

    if not module_path_str:
        raise ValueError(
            "[cv_modules] section is missing or has no 'module =' key in infretis.toml.\n"
            "At minimum, add:\n"
            "  [cv_modules]\n"
            '  module = "CV_builder/CV_manager/minimal_suite.py"\n'
            '  class  = "MinimalCVSuite"\n'
            "See CV_builder/infretis_example.toml for a complete example."
        )

    module_path = Path(module_path_str)
    class_name = cv_cfg.get("class", "")

    if not class_name:
        raise ValueError("[cv_modules] requires a 'class' key (the CVSuite subclass)")
    if not module_path.exists():
        raise FileNotFoundError(f"[cv_modules] module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("user_cv_suite", module_path)
    mod = importlib.util.module_from_spec(spec)      # type: ignore
    spec.loader.exec_module(mod)                     # type: ignore
    cls = getattr(mod, class_name)
    log.info("[cv_modules] loaded suite %s from %s", class_name, module_path)
    suite_kwargs = {k: v for k, v in cv_cfg.items() if k not in {"module", "class"}}
    try:
        return cls(**suite_kwargs)
    except TypeError as exc:
        if suite_kwargs:
            raise TypeError(
                f"Failed to initialize CVSuite '{class_name}' with [cv_modules] keys "
                f"{sorted(suite_kwargs.keys())}: {exc}"
            ) from exc
        raise


def _load_toml(path: Path) -> dict:
    with open(path) as f:
        return toml.load(f)


def _active_steps_from_config(config: dict) -> list[str]:
    """Read and normalize step IDs from [current].active in TOML."""
    current = config.get("current", {})
    active = current.get("active", [])

    if not active:
        raise KeyError("[current].active is empty or missing; cannot run --check-active")

    out: list[str] = []
    for item in active:
        token = str(item).strip()
        if token.isdigit():
            out.append(token)
            continue

        p = Path(token)
        candidates = [p.name, p.parent.name]
        step = next((c for c in candidates if c.isdigit()), None)
        if step is None:
            raise ValueError(
                f"Could not parse active step from [current].active entry '{item}'. "
                "Use numeric step IDs or paths ending in a numeric step directory."
            )
        out.append(step)

    # Preserve order, remove duplicates.
    return list(dict.fromkeys(out))

def _find_retis_steps(paths: Paths) -> list[str]:
    """Return numerically sorted RETIS step IDs from directories or HDF5 groups."""
    if paths.h5_input is not None:
        with h5py.File(paths.h5_input, "r") as h5:
            groups = [name for name in h5.keys() if name.isdigit()]
        return sorted(groups, key=lambda s: int(s))

    dirs = [
        os.path.basename(d)
        for d in glob.glob(str(paths.load_dir / "*"))
        if os.path.isdir(d) and os.path.basename(d).isdigit()
    ]
    return sorted(dirs, key=lambda s: int(s))

def _get_shard_from_env() -> tuple[int, int]:
    """Detect Slurm rank/world size; falls back to (0, 1) for local runs."""
    rank  = int(os.environ.get("SLURM_PROCID", os.environ.get("CV_RANK", "0")))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("CV_WORLD",  "1")))
    return rank, world

def _shard_steps(steps: list[str], rank: int, world: int) -> list[str]:
    """Stride-based sharding for multi-process (Slurm) runs."""
    if world <= 1:
        return steps
    return steps[rank::world]

def _drop_burn_in(seq: list[str], nskip: float | int, silent: bool = False) -> list[str]:
    """
    Drop the first `nskip` entries as a burn-in phase.

    - 0 < nskip < 1  → treat as a fraction of total steps.
    - nskip >= 1     → treat as an absolute count (int).
    - nskip == 0     → no burn-in.
    """
    total = len(seq)
    if isinstance(nskip, float) and 0 < nskip < 1:
        n = int(round(nskip * total))
    elif isinstance(nskip, (int, float)) and nskip >= 1:
        n = int(nskip)
    else:
        n = 0
    n = max(0, min(n, total))
    if n and not silent:
        log.info("Burn-in: dropping first %d of %d steps", n, total)
    return seq[n:]
