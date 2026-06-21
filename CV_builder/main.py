"""
main.py — Entry point for the CV_builder pipeline.

Usage:
    python main.py --toml infretis.toml [options]

Run `python main.py --help` for the full option list.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from builder import build_runner
from config import AggregatorConfig
from storage.aggregator import merge_parts, run_aggregation
from storage.report import write_cvmat_report

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="cv_builder",
        description="Compute collective variables (CVs) from RETIS trajectories.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    paths = p.add_argument_group("Paths")
    paths.add_argument(
        "--toml",
        default=Path("infretis.toml"), type=Path,
        metavar="FILE",
        help="TOML settings file  (default: %(default)s)",
    )
    paths.add_argument(
        "--load-dir",
        default=None, type=Path,
        metavar="DIR",
        help="Directory containing RETIS step folders  (default: <toml-dir>/load)",
    )
    paths.add_argument(
        "--data",
        default=None, type=Path,
        metavar="FILE",
        help="infretis_data.txt for WHAM weights  (default: <toml-dir>/infretis_data.txt)",
    )
    paths.add_argument(
        "--output-dir",
        default=None, type=Path,
        metavar="DIR",
        help="Root output directory for HDF5 files  (default: <toml-dir>/CVs)",
    )
    paths.add_argument(
        "--h5-input",
        default=None, type=Path,
        metavar="FILE",
        help=(
            "Optional input HDF5 file containing per-step text datasets at "
            "{step}/traj.txt and {step}/order.txt"
        ),
    )

    # ── WHAM ───────────────────────────────────────────────────────────────
    wham = p.add_argument_group("WHAM / grid")
    wham.add_argument(
        "--grid-mode",
        choices=["selection", "range"],
        default="selection",
        help=(
            "How to pick grid indices to analyse:\n"
            "  selection  use [cvmat].selection from the TOML\n"
            "  range      use [cvmat].start_grid / end_grid / step_size"
        ),
    )
    wham.add_argument(
        "--selection",
        nargs="+", type=int, default=None,
        metavar="INT",
        help="Override TOML selection with these grid indices."
    )
    wham.add_argument(
        "--lamres", type=float, default=None, metavar="FLOAT",
        help="Hard-code the WHAM lamres step size (default: auto-selected)",
    )

    # ── HDF5 output ────────────────────────────────────────────────────────
    h5 = p.add_argument_group("HDF5 output")
    h5.add_argument(
        "--h5-compress",
        type=int, default=4, choices=range(10),
        metavar="0-9",
        help="gzip compression level for HDF5 datasets  (default: %(default)s)",
    )
    h5.add_argument(
        "--h5-include-cvs", nargs="*", type=str, default=None,
        metavar="NAME",
        help="Whitelist: only write these CV names to HDF5 (default: all)",
    )
    h5.add_argument(
        "--h5-exclude-cvs", nargs="*", type=str, default=None,
        metavar="NAME",
        help="Blacklist: skip these CV names when writing HDF5",
    )

    # ── Aggregation (sorted-H5 stage) ──────────────────────────────────────
    agg = p.add_argument_group(
        "Aggregation (optional — run sorted-H5 stage after CV computation)"
    )
    agg.add_argument(
        "--no-sort-h5", action="store_false", dest="sort_h5",
        help="Do NOT run sorted_h5 aggregation after CV computation (on by default)",
    )
    agg.add_argument(
        "--agg-grids", nargs="*", type=int, default=None, metavar="INT",
        help="Explicit grid IDs to aggregate (default: all)",
    )
    agg.add_argument(
        "--agg-dtype", default="f4", choices=["f4", "f8"],
        help="Float precision for output cv matrices  (default: %(default)s)",
    )
    agg.add_argument(
        "--agg-shard-mode", default="stride", choices=["stride", "block"],
        help="How to shard steps across workers  (default: %(default)s)",
    )
    agg.add_argument(
        "--agg-resume", action="store_true",
        help="Skip aggregation if this rank's part file already exists",
    )
    agg.add_argument(
        "--agg-force-overwrite", action="store_true",
        help="Always overwrite existing part files during aggregation",
    )

    # ── Runtime ────────────────────────────────────────────────────────────
    runtime = p.add_argument_group("Runtime")
    runtime.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity  (default: %(default)s)",
    )
    runtime.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable debug mode: forces log-level DEBUG, prints CV shapes/ranges\n"
            "for every computed frame."
        ),
    )
    runtime.add_argument(
        "--workers",
        type=int, default=None, metavar="N",
        help=(
            "Override [ppa].workers from TOML.\n"
            "Used for both CV step parallelism and aggregation.\n"
            "On HPC, use srun -n N instead (no flag needed)."
        ),
    )
    runtime.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete previous output before running.",
    )
    runtime.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-completed frames (resume a partial run).",
    )
    runtime.add_argument(
        "--check-active",
        action="store_true",
        help=(
            "CV-only mode: read [current].active from TOML and compute CVs only "
            "for those steps; skips WHAM weight calculation."
        ),
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level_name: str, *, debug: bool = False) -> None:
    """Configure root logger.  --debug forces DEBUG regardless of --log-level."""
    level = logging.DEBUG if debug else getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def log_startup_banner(args: argparse.Namespace, runner=None) -> None:
    """Print a concise startup summary at INFO level."""
    if "CV_RANK" in os.environ:
        return
        
    sep = "─" * 60
    log.info(sep)
    log.info("CV_builder  —  starting run")
    log.info(sep)
    log.info("  toml         : %s", Path(args.toml).resolve())
    log.info("  load_dir     : %s", args.load_dir or "(default: <toml-dir>/load)")
    log.info("  data         : %s", args.data     or "(default: <toml-dir>/infretis_data.txt)")
    log.info("  output_dir   : %s", args.output_dir or "(default: <toml-dir>/CVs)")
    log.info("  grid_mode    : %s", args.grid_mode)
    log.info("  h5_compress  : %s", args.h5_compress)
    log.info("  workers      : %s", args.workers or "(from TOML [ppa].workers)")
    log.info("  sort_h5      : %s", args.sort_h5)
    log.info("  debug        : %s", args.debug)
    
    if runner is not None:
        pts = runner.grid.grid_pts
        if len(pts) > 0:
            log.info("  grid_pts     : %d points [%.4f ... %.4f]", len(pts), min(pts), max(pts))
        else:
            log.info("  grid_pts     : 0 points")
            
    log.info(sep)


# ---------------------------------------------------------------------------
# Parallel CV step dispatch (Mac/local only)
# ---------------------------------------------------------------------------

def _run_cv_parallel(args: argparse.Namespace) -> None:
    """
    Spawn N independent subprocess copies of main.py, each with
    CV_RANK=r and CV_WORLD=N in env so simulationrunner shards steps.

    --sort-h5 is NOT forwarded to subprocesses; aggregation is always
    done in the parent process after all workers finish.
    """
    n = args.workers or 1
    log.info("Spawning %d parallel CV workers", n)

    # Strip flags that only the parent process should handle.
    # Workers do CV computation only; aggregation / reporting is parent-only.
    bool_drop  = {
        "--sort-h5",          # boolean flag
        "--agg-resume",
        "--agg-force-overwrite",
        "--overwrite",        # boolean flag
    }
    value_drop = {
        "--workers",          # flag + 1 value
        "--agg-grids",
        "--agg-dtype",
        "--agg-shard-mode",
    }
    argv_base = [sys.executable, sys.argv[0]] + sys.argv[1:]
    filtered  = []
    i = 0
    while i < len(argv_base):
        tok = argv_base[i]
        if tok in bool_drop:
            i += 1            # skip flag only
        elif tok in value_drop:
            i += 2            # skip flag + its value
        else:
            filtered.append(tok)
            i += 1

    procs = []
    for rank in range(n):
        env = os.environ.copy()
        env["CV_RANK"]  = str(rank)
        env["CV_WORLD"] = str(n)
        procs.append(subprocess.Popen(filtered, env=env))

    failed = 0
    for rank, proc in enumerate(procs):
        rc = proc.wait()
        if rc != 0:
            log.error("CV worker rank %d exited with code %d", rank, rc)
            failed += 1

    if failed:
        raise RuntimeError(f"{failed} CV worker(s) failed — check logs above")
    log.info("All %d CV workers completed successfully", n)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.perf_counter()
    exit_code = 0
    try:
        args = parse_args()
        setup_logging(args.log_level, debug=args.debug)
        runner = build_runner(args)
        args._runner = runner  # type: ignore[attr-defined]
        log_startup_banner(args, runner)

        in_slurm = os.environ.get("SLURM_PROCID") is not None
        n_workers = args.workers or 1
        if n_workers > 1 and not in_slurm:
            # Mac / local: spawn N subprocesses
            _run_cv_parallel(args)
        else:
            # Single-process (or inside srun rank)
            runner.run()

        if args.sort_h5:
            log.info("─" * 60)
            log.info("Starting sorted-H5 aggregation stage")
            log.info("─" * 60)
            # Prevent deleting the output we just computed!
            args.overwrite = False
            # Rebuild wham_n_grid for path derivation when we ran in parallel
            runner = getattr(args, "_runner", None) or build_runner(args)
            agg_cfg = AggregatorConfig.from_args(
                args,
                interfaces=runner.interfaces,
                cvmat_n_grid=runner.cvmat.n_grid,
                output_dir=runner.paths.output_dir
            )
            parts = run_aggregation(agg_cfg)
            log.info("Aggregation complete — %d part file(s) written:", len(parts))
            for p in parts:
                log.info("  %s", p)

            cvmat = merge_parts(
                agg_cfg.out_file,
                out_file=agg_cfg.merged_file,
                interfaces=agg_cfg.interfaces,
                cvmat_n_grid=agg_cfg.cvmat_n_grid
            )
            log.info("Merged → %s", cvmat)

            report = write_cvmat_report(cvmat)
            log.info("Report  → %s", report)

    except FileNotFoundError as e:
        log.error("File not found: %s", e)
        exit_code = 1
    except KeyError as e:
        log.error("Configuration error: %s", e)
        exit_code = 1
    except Exception as e:
        log.exception("Fatal error: %s", e)
        exit_code = 1
    finally:
        elapsed = time.perf_counter() - start
        log.info("Total computation time: %.2f s", elapsed)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
