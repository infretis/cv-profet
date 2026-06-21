"""
config_loader.py — Unified configuration loader for all PPA tools.

Reads ``infretis.toml`` and exposes a single ``PPAConfig`` dataclass that
resolves every path and setting needed by ``screener.py``, ``analyze.py``,
and ``CV_builder/main.py``.

Usage
-----
    from config_loader import load_ppa_config

    cfg = load_ppa_config(Path("infretis.toml"))
    print(cfg.cvmat_path)   # .../CVs/890/CVmat.h5
    print(cfg.lambda_pairs) # [(6, 20)]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import toml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class TMatConfig:
    """Settings for T-matrix generation (``[ppa.tmat]``)."""
    lambda_c_opt: int = 6
    lambda_r_opt: int = 20
    lambda_r_max: int = 30
    n_cvar_list: List[int] = field(default_factory=lambda: [1])
    top_k: int = 10
    calculate_missing: bool = False
    calc_full: bool = True


@dataclass
class SGConfig:
    """Savitzky-Golay + grid settings (``[ppa.sg]``)."""
    sg_grid: int = 1200
    sg_polyorder: int = 2
    sg_window_frac: float = 1.0 / 25.0
    grid_expand_factor: float = 1.5
    n_hist_bins: int = 50
    integer_bins: bool = False
    discrete_max_unique: int = 50   # auto-discrete if n_unique ≤ N and all-integer values


@dataclass
class ScreeningConfig:
    """Dual-annealing parameters for linear combination optimisation (``[ppa.screening]``)."""
    initial_temp: float = 26150.0
    maxiter_lin2: int = 200
    maxiter_lin3: int = 1000


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------

@dataclass
class PPAConfig:
    """Master configuration resolved from ``infretis.toml``."""

    # ── Resolved paths ────────────────────────────────────────────────────
    root_dir: Path = field(default_factory=Path)
    cv_dir: Path = field(default_factory=Path)
    cvmat_path: Path = field(default_factory=Path)
    screen_path: Path = field(default_factory=Path)
    t_mat_path: Path = field(default_factory=Path)
    steps_dir: Path = field(default_factory=Path)
    sorted_dir: Path = field(default_factory=Path)
    load_dir: Path = field(default_factory=Path)
    data_file: Path = field(default_factory=Path)
    toml_path: Path = field(default_factory=Path)

    # ── Simulation ────────────────────────────────────────────────────────
    interfaces: List[float] = field(default_factory=list)

    # ── CVMAT ─────────────────────────────────────────────────────────────
    n_grid: int = 890

    # ── PPA shared settings ───────────────────────────────────────────────
    output_dir_name: str = "CVs"
    workers: int = 8

    # Lambda pairs
    lambda_c: List[int] = field(default_factory=lambda: [6])
    lambda_r: List[int] = field(default_factory=lambda: [20])
    lambda_pairs: List[tuple] = field(default_factory=list)

    # Screening
    n_best_obo: int = 15
    n_best_lin: int = 10
    n_try_lin: int = 15
    n_cvar_list: List[int] = field(default_factory=lambda: [1])
    optimizer: str = "dual_annealing"

    # CV selection mode
    cv_mode: str = "n_best"  # "n_best" | "manual"
    obo_from_toml: List[str] = field(default_factory=list)
    lin2_from_toml: List[str] = field(default_factory=list)
    lin3_from_toml: List[str] = field(default_factory=list)

    # Reference lambda pair (shared by tmat, diagnostics, distribution)
    lambda_c_opt: int = 6
    lambda_r_opt: int = 20

    # Sub-configs
    tmat: TMatConfig = field(default_factory=TMatConfig)
    sg: SGConfig = field(default_factory=SGConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)

    def log_summary(self) -> None:
        """Log key resolved settings."""
        log.info("PPAConfig summary:")
        log.info("  root_dir    : %s", self.root_dir)
        log.info("  cv_dir      : %s", self.cv_dir)
        log.info("  cvmat_path  : %s", self.cvmat_path)
        log.info("  screen_path : %s", self.screen_path)
        log.info("  t_mat_path  : %s", self.t_mat_path)
        log.info("  load_dir    : %s", self.load_dir)
        log.info("  n_grid      : %d", self.n_grid)
        log.info("  workers     : %d", self.workers)
        log.info("  lambda_c    : %s", self.lambda_c)
        log.info("  lambda_r    : %s", self.lambda_r)
        log.info("  lambda_pairs: %s", self.lambda_pairs)
        log.info("  cv_mode     : %s", self.cv_mode)
        log.info("  n_best_obo  : %d", self.n_best_obo)
        log.info("  n_best_lin  : %d", self.n_best_lin)
        log.info("  n_try_lin   : %d", self.n_try_lin)
        log.info("  n_cvar_list : %s", self.n_cvar_list)
        log.info("  sg          : %s", self.sg)
        log.info("  screening   : %s", self.screening)


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def _build_lambda_pairs(lambda_c: List[int], lambda_r: List[int]) -> List[tuple]:
    """Generate all valid (lc, lr) pairs where lr > lc, sorted."""
    return sorted((lc, lr) for lc in lambda_c for lr in lambda_r if lr > lc)


def _coerce_fraction(val, default: float = 0.0) -> float:
    """Accept float, int, or fraction string like '1/16' and return float."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        if "/" in val:
            parts = val.split("/", 1)
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                log.warning("Cannot parse fraction '%s', using default %g", val, default)
                return default
        try:
            return float(val)
        except ValueError:
            log.warning("Cannot parse '%s' as float, using default %g", val, default)
            return default
    return default


def _coerce_int_list(val) -> List[int]:
    if isinstance(val, (int, float)):
        return [int(val)]
    return [int(x) for x in val]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_ppa_config(
    toml_path: Path,
    *,
    lambda_c: Optional[List[int]] = None,
    lambda_r: Optional[List[int]] = None,
    workers: Optional[int] = None,
    n_best: Optional[int] = None,
    cv_mode: Optional[str] = None,
) -> PPAConfig:
    """
    Load ``infretis.toml`` and return a fully resolved ``PPAConfig``.

    CLI overrides (keyword arguments) take precedence over TOML values.
    """
    toml_path = Path(toml_path).resolve()
    if not toml_path.exists():
        raise FileNotFoundError(f"TOML file not found: {toml_path}")

    raw = toml.load(toml_path)
    root_dir = toml_path.parent

    # ── Read sections ─────────────────────────────────────────────────────
    sim = raw["simulation"]
    cvmat = raw["cvmat"]
    ppa = raw["ppa"]
    tmat_raw = ppa.get("tmat", {})
    sg_raw = ppa.get("sg", {})
    scr_raw = ppa.get("screening", {})

    # ── Core values ───────────────────────────────────────────────────────
    n_grid = cvmat["n_grid"]
    output_dir_name = ppa["output_dir"]
    interfaces = sim["interfaces"]
    load_dir_name = sim["load_dir"]

    # ── Paths ─────────────────────────────────────────────────────────────
    cv_dir = root_dir / output_dir_name / str(n_grid)

    # ── Lambda (CLI override > TOML) ──────────────────────────────────────
    lc = _coerce_int_list(lambda_c if lambda_c is not None else ppa["lambda_c"])
    lr = _coerce_int_list(lambda_r if lambda_r is not None else ppa["lambda_r"])

    # ── Workers ───────────────────────────────────────────────────────────
    w = workers if workers is not None else ppa["workers"]

    # ── CV mode ───────────────────────────────────────────────────────────
    mode = cv_mode if cv_mode is not None else ppa["cv_mode"]

    # ── Screening settings ────────────────────────────────────────────────
    n_best_obo = n_best if n_best is not None else ppa["n_best_obo"]

    # ── Reference lambda pair ─────────────────────────────────────────────
    lc_opt = ppa["lambda_c_opt"]
    lr_opt = ppa["lambda_r_opt"]

    # ── Sub-configs ───────────────────────────────────────────────────────
    tmat_cfg = TMatConfig(
        lambda_c_opt=lc_opt,
        lambda_r_opt=lr_opt,
        lambda_r_max=tmat_raw["lambda_r_max"],
        n_cvar_list=tmat_raw["n_cvar_list"],
        top_k=tmat_raw["top_k"],
        calculate_missing=tmat_raw["calculate_missing"],
        calc_full=tmat_raw["calc_full"],
    )

    sg_cfg = SGConfig(
        sg_grid=sg_raw.get("sg_grid", 1200),
        sg_polyorder=sg_raw.get("sg_polyorder", 2),
        sg_window_frac=_coerce_fraction(sg_raw.get("sg_window_frac", 1.0 / 25.0)),
        grid_expand_factor=_coerce_fraction(sg_raw.get("grid_expand_factor", 1.5)),
        n_hist_bins=sg_raw.get("n_hist_bins", 50),
        integer_bins=bool(sg_raw.get("integer_bins", False)),
        discrete_max_unique=sg_raw.get("discrete_max_unique", 50),
    )

    scr_cfg = ScreeningConfig(
        initial_temp=scr_raw.get("initial_temp", 26150.0),
        maxiter_lin2=scr_raw.get("maxiter_lin2", 200),
        maxiter_lin3=scr_raw.get("maxiter_lin3", 1000),
    )

    return PPAConfig(
        root_dir=root_dir,
        cv_dir=cv_dir,
        cvmat_path=cv_dir / "CVmat.h5",
        screen_path=cv_dir / "screen.h5",
        t_mat_path=cv_dir / "T_mat.h5",
        steps_dir=cv_dir / "steps",
        sorted_dir=cv_dir / "sorted",
        load_dir=root_dir / load_dir_name,
        data_file=root_dir / "infretis_data.txt",
        toml_path=toml_path,
        interfaces=interfaces,
        n_grid=n_grid,
        output_dir_name=output_dir_name,
        workers=w,
        lambda_c=lc,
        lambda_r=lr,
        lambda_pairs=_build_lambda_pairs(lc, lr),
        n_best_obo=n_best_obo,
        n_best_lin=ppa["n_best_lin"],
        n_try_lin=ppa["n_try_lin"],
        n_cvar_list=ppa["n_cvar_list"],
        optimizer=ppa["optimizer"],
        cv_mode=mode,
        obo_from_toml=ppa.get("obo_from_toml", []),
        lin2_from_toml=ppa.get("lin2_from_toml", []),
        lin3_from_toml=ppa.get("lin3_from_toml", []),
        lambda_c_opt=lc_opt,
        lambda_r_opt=lr_opt,
        tmat=tmat_cfg,
        sg=sg_cfg,
        screening=scr_cfg,
    )
