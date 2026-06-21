"""
storage/aggregator.py — Backend-agnostic parallel aggregation engine.

Merges per-step HDF5 files (step_*.h5) into sorted matrix files for PPA.

Usage
-----
From main.py (Mac — uses multiprocessing):
    from storage.aggregator import run_aggregation
    run_aggregation(cfg)          # cfg.n_workers controls Pool size

From sorted_h5_shard.py (HPC — single-rank called by srun):
    from storage.aggregator import AggregationWorker
    rank, world = _get_slurm_env()
    AggregationWorker(cfg, rank, world).run()

The worker logic is identical in both cases; only the dispatch differs.
"""
from __future__ import annotations

import logging
import os
import re
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

from config import AggregatorConfig

log = logging.getLogger(__name__)

STEP_RE = re.compile(r"step_(\d+)\.h5$")


def _set_h5_env() -> None:
    """Disable HDF5 file locking (needed on macOS with parallel readers)."""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Disable at import time so the main process also benefits
_set_h5_env()



# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _get_slurm_env() -> tuple[int, int]:
    """Return (rank, world) from Slurm/MPI env variables; default (0, 1)."""
    rank  = int(os.getenv("SLURM_PROCID", os.getenv("PMI_RANK", "0")))
    world = int(os.getenv("SLURM_NTASKS", os.getenv("PMI_SIZE", os.getenv("WORLD_SIZE", "1"))))
    return rank, world


def _detect_backend() -> str:
    """Return 'slurm' if Slurm env is active, else 'local'."""
    if os.getenv("SLURM_PROCID") is not None or os.getenv("PMI_RANK") is not None:
        return "slurm"
    return "local"


# ---------------------------------------------------------------------------
# Step-file discovery & validation
# ---------------------------------------------------------------------------

def discover_step_files(steps_dir: Path) -> list[Path]:
    steps = sorted(
        (p for p in steps_dir.glob("step_*.h5") if STEP_RE.search(p.name)),
        key=lambda p: int(STEP_RE.search(p.name).group(1)),  # numeric sort
    )
    if not steps:
        raise FileNotFoundError(f"No step_*.h5 found in {steps_dir}")
    return steps


def validate_step_files(steps: list[Path]) -> tuple[list[Path], list[Path]]:
    """Split into (good, bad) by attempting a lightweight open of each file."""
    good, bad = [], []
    for p in steps:
        try:
            if h5py.is_hdf5(p):
                with h5py.File(p, "r", locking=False) as f:
                    _ = list(f.keys())
                good.append(p)
            else:
                bad.append(p)
        except Exception:
            bad.append(p)
    return good, bad


def shard_list(items: list, rank: int, world: int, mode: str = "stride") -> list:
    items = list(items)
    if world <= 1:
        return items
    if mode == "block":
        n  = len(items)
        b  = (n + world - 1) // world
        lo = rank * b
        hi = min(lo + b, n)
        return items[lo:hi]
    return items[rank::world]  # stride (better load balance)


# ---------------------------------------------------------------------------
# Grid / module discovery
# ---------------------------------------------------------------------------

def discover_all_grids(steps: list[Path]) -> list[int]:
    grids: set[int] = set()
    for p in steps:
        try:
            with h5py.File(p, "r", locking=False) as f:
                for k in f.keys():
                    try:
                        grids.add(int(k))
                    except ValueError:
                        pass
        except Exception:
            pass
    return sorted(grids)


_EXCLUDE_MODULES = {"pair_dist_all_atoms", "dataset", "standard"}


def discover_modules_for_grid(steps: list[Path], grid_id: int) -> list[str]:
    """Find all CV-module group names under /<grid_id>/ in the first valid step file."""
    for p in steps:
        try:
            with h5py.File(p, "r", locking=False) as f:
                if str(grid_id) in f:
                    return sorted(k for k in f[str(grid_id)].keys()
                                  if k not in _EXCLUDE_MODULES)
        except Exception:
            continue
    return []


def step_has_group(step: Path, grid_id: int, module: str) -> bool:
    try:
        with h5py.File(step, "r", locking=False) as f:
            return f"{grid_id}/{module}" in f
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Low-level read helpers
# ---------------------------------------------------------------------------

def _to_str_list(a: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in a]


def _to_sbytes(a: Sequence[str]) -> np.ndarray:
    return np.array([s.encode("utf-8") for s in a], dtype="S")


def _parse_step_id(p: Path) -> int:
    m = STEP_RE.search(p.name)
    if not m:
        raise ValueError(f"Cannot parse step id from filename: {p.name}")
    return int(m.group(1))


def _module_prefix(mod: str) -> str:
    """Short column prefix for a CV module name."""
    _MAP = {
        "neighbor_wannier_ranked":       "wannier",
        "neighbor_forces_projection":    "forces",
        "neighbor_mulliken_ranked":      "mulliken",
        "neighbor_pt_geometry_ranked":   "ptgeom",
        "neighbor_OO_ranked":            "OO",
        "neighbor_OH_ranked":            "OH",
        "neighbor_OH_orientation_ed":    "OHed",
        "neighbor_HOH_angle_ranked":     "HOH",
        "neighbor_IonO_ranked":          "IonO",
        "hb_wire_length":                "hb",
        "pair_dist_all_atoms":           "pairdist",
    }
    return _MAP.get(mod, mod)


def _read_labels(step: Path, grid_id: int, module: str) -> list[str]:
    with h5py.File(step, "r", locking=False) as f:
        g = f[f"{grid_id}/{module}"]
        if "labels" not in g:
            raise KeyError(f"{step}:{grid_id}/{module} has no 'labels' dataset")
        return _to_str_list(g["labels"][()])


def _read_values_row(step: Path, grid_id: int, module: str) -> np.ndarray:
    """Return the single-row CV value array for this step as a 1-D float64 array."""
    with h5py.File(step, "r", locking=False) as f:
        g = f[f"{grid_id}/{module}"]
        if "values" not in g:
            raise KeyError(f"{step}:{grid_id}/{module} has no 'values' dataset")
        vals = g["values"][()]
        if vals.ndim == 2:
            vals = vals[0]   # take first (and only) row
        return vals.astype(np.float64)


def _ensure_labels_consistent(steps: list[Path], grid_id: int, module: str,
                               ref_labels: list[str]) -> None:
    for p in steps:
        labs = _read_labels(p, grid_id, module)
        if labs != ref_labels:
            raise ValueError(
                f"Label mismatch for module '{module}' in {p}.\n"
                f"Expected {ref_labels[:6]}…, got {labs[:6]}…"
            )


# ---------------------------------------------------------------------------
# Matrix aggregation
# ---------------------------------------------------------------------------

def _aggregate_module(
    steps: list[Path], grid_id: int, module: str, dtype: str
) -> tuple[np.ndarray, list[str]]:
    """
    Build a 2-D matrix (n_steps, 1 + n_features) for one CV module
    across all eligible step files.  Column 0 = RETIS_step.
    """
    eligible = [p for p in steps if step_has_group(p, grid_id, module)]
    if not eligible:
        return np.empty((0, 0)), []

    ref_labels = _read_labels(eligible[0], grid_id, module)
    _ensure_labels_consistent(eligible, grid_id, module, ref_labels)

    prefix = _module_prefix(module)
    cols   = ["RETIS_step"] + [f"{prefix}::{lab}" for lab in ref_labels]

    rows = []
    for p in eligible:
        sid  = _parse_step_id(p)
        vals = _read_values_row(p, grid_id, module)
        rows.append(np.concatenate([[sid], vals]))

    rows.sort(key=lambda r: r[0])
    mat = np.vstack(rows).astype(np.float64)
    return mat, cols


def _concatenate_modules(
    mats_cols: list[tuple[np.ndarray, list[str]]]
) -> tuple[np.ndarray, list[str]]:
    """
    Horizontally concatenate per-module matrices.
    All matrices must share the same RETIS_step ordering (column 0).
    """
    if not mats_cols:
        raise ValueError("No module matrices to concatenate")

    base_mat, base_cols = mats_cols[0]
    all_feats = [base_mat[:, 1:]]
    all_cols  = list(base_cols)

    for mat, cols in mats_cols[1:]:
        if not np.array_equal(mat[:, 0], base_mat[:, 0]):
            raise ValueError("RETIS_step ordering mismatch between modules")
        all_feats.append(mat[:, 1:])
        all_cols.extend(cols[1:])

    full = np.concatenate([base_mat[:, 0:1]] + all_feats, axis=1)
    return full, all_cols


def _build_lambda_weight(steps: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    rows, skipped = [], 0
    for p in steps:
        try:
            sid = _parse_step_id(p)
            with h5py.File(p, "r", locking=False) as f:
                lam = float(f.attrs.get("lambda_max", np.nan))
                w   = float(f.attrs.get("weight",     np.nan))
            rows.append((sid, lam, w))
        except Exception:
            skipped += 1
    if skipped:
        log.warning("lambda_and_weight: skipped %d unreadable files", skipped)
    rows.sort(key=lambda r: r[0])
    return np.array(rows, dtype=np.float64), np.array(["RETIS_step", "lambda_max", "weight"], dtype=object)


def _write_group(outf: h5py.File, grid_id: int, name: str,
                 cv: np.ndarray, cols: list[str], dtype: str = "f4") -> None:
    gpath = f"{grid_id}/{name}"
    if gpath in outf:
        del outf[gpath]
    g = outf.create_group(gpath)
    g.create_dataset("cv",   data=cv.astype(np.dtype(dtype)), compression="gzip", shuffle=True)
    g.create_dataset("cols", data=_to_sbytes(cols),           compression="gzip", shuffle=True)


# ---------------------------------------------------------------------------
# AggregationWorker — one rank's work unit
# ---------------------------------------------------------------------------

class AggregationWorker:
    """
    Processes one rank's shard of step_*.h5 files and writes a single
    part file:  <out_file>.part.<rank>.h5
    """

    def __init__(self, cfg: AggregatorConfig, rank: int, world: int) -> None:
        self.cfg   = cfg
        self.rank  = rank
        self.world = world

    def run(self) -> Path | None:
        """Execute the aggregation for this rank.  Returns the part file path."""
        cfg   = self.cfg
        rank  = self.rank
        world = self.world

        # ── Discover + validate step files ────────────────────────────────
        all_steps = discover_step_files(cfg.steps_dir)
        all_steps, bad = validate_step_files(all_steps)
        if bad:
            log.warning("[rank %d] Skipping %d unreadable step files", rank, len(bad))
        if not all_steps:
            log.error("[rank %d] No readable step_*.h5 files found", rank)
            return None

        my_steps = shard_list(all_steps, rank, world, mode=cfg.shard_mode)
        log.info("[rank %d/%d] processing %d of %d step files",
                 rank, world, len(my_steps), len(all_steps))

        if not my_steps:
            log.info("[rank %d] no steps assigned — exiting cleanly", rank)
            return None

        # ── Determine grid IDs ─────────────────────────────────────────────
        if cfg.grid_ids:
            grid_ids = sorted(cfg.grid_ids)
        elif cfg.all_grids:
            grid_ids = discover_all_grids(all_steps)
            if not grid_ids:
                raise RuntimeError("No numeric grid groups found in any step file")
        else:
            raise ValueError("AggregatorConfig: specify grid_ids or set all_grids=True")

        # ── Part file path ─────────────────────────────────────────────────
        part_path = Path(f"{cfg.out_file}.part.{rank}.h5")
        tmp_path  = Path(f"{part_path}.tmp.{os.getpid():x}")
        part_path.parent.mkdir(parents=True, exist_ok=True)

        if part_path.exists() and cfg.resume and not cfg.force_overwrite:
            log.info("[rank %d] part file exists — resuming: %s", rank, part_path)
            return part_path

        # ── Read shared root attrs from any step file ──────────────────────
        with h5py.File(all_steps[0], "r", locking=False) as f0:
            shared_attrs = {k: f0.attrs[k] for k in ("wham_ngrid", "cell_size", "n_atoms")
                            if k in f0.attrs}

        # ── Write tmp → atomic rename ──────────────────────────────────────
        try:
            with h5py.File(tmp_path, "w") as outf:
                # copy root attrs
                for k, v in shared_attrs.items():
                    outf.attrs[k] = v

                # lambda_and_weight summary
                lamw, lamw_labels = _build_lambda_weight(my_steps)
                d = outf.create_dataset("lambda_and_weight", data=lamw,
                                        compression="gzip", shuffle=True)
                d.attrs["labels"] = lamw_labels

                # Copy order_txt from each step file
                order_grp = outf.create_group("order_txt")
                for p in my_steps:
                    try:
                        sid = _parse_step_id(p)
                        with h5py.File(p, "r", locking=False) as sf:
                            if "order_txt" in sf:
                                order_grp.create_dataset(
                                    str(sid), data=sf["order_txt"][()],
                                    compression="gzip",
                                )
                    except Exception:
                        pass
                log.info("[rank %d] order_txt: stored %d/%d steps", rank, len(order_grp), len(my_steps))

                for gid in grid_ids:
                    mods = cfg.modules or discover_modules_for_grid(all_steps, gid)
                    if not mods:
                        log.debug("[rank %d] grid %d: no modules found — skip", rank, gid)
                        continue

                    mats_cols: list[tuple[np.ndarray, list[str]]] = []
                    for mod in mods:
                        mat, cols = _aggregate_module(my_steps, gid, mod, cfg.float_dtype)
                        if mat.size == 0:
                            log.debug("[rank %d] grid %d / %s: no data — skip", rank, gid, mod)
                            continue
                        _write_group(outf, gid, mod, mat, cols, cfg.float_dtype)
                        mats_cols.append((mat, cols))
                        log.debug("[rank %d] grid %d / %s: wrote %d rows", rank, gid, mod, mat.shape[0])

                    if mats_cols:
                        full, full_cols = _concatenate_modules(mats_cols)
                        _write_group(outf, gid, "dataset", full, full_cols, cfg.float_dtype)
                        log.info("[rank %d] grid %d: dataset  shape=%s", rank, gid, full.shape)

            os.replace(tmp_path, part_path)
            log.info("[rank %d] wrote %s", rank, part_path)

        finally:
            if tmp_path.exists():
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        return part_path


# ---------------------------------------------------------------------------
# Top-level dispatcher  (Mac = multiprocessing, HPC = single-rank)
# ---------------------------------------------------------------------------

def _worker_fn(cfg: AggregatorConfig, rank: int, world: int) -> Path | None:
    """Module-level function so it can be pickled by multiprocessing.Pool."""
    # Disable HDF5 locking in the subprocess before importing h5py
    import os as _os
    _os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-8s [worker %(process)d] %(name)s: %(message)s")
    return AggregationWorker(cfg, rank, world).run()


def run_aggregation(cfg: AggregatorConfig) -> list[Path]:
    """
    Dispatch aggregation workers.

    - On HPC (Slurm env detected):  run as a single rank; parallelism is
      provided externally by `srun -n N`.
    - On Mac / local:  spawn `cfg.n_workers` processes via multiprocessing.Pool.

    Returns list of part-file paths written by this process (1 on HPC, N on Mac).
    """
    backend = _detect_backend()

    if backend == "slurm":
        rank, world = _get_slurm_env()
        log.info("AggregationWorker  backend=slurm  rank=%d/%d", rank, world)
        result = AggregationWorker(cfg, rank, world).run()
        return [result] if result is not None else []

    # Local multiprocessing
    world = cfg.n_workers or os.cpu_count() or 1
    log.info("AggregationWorker  backend=local  workers=%d", world)

    # For very small jobs (world=1) skip Pool overhead
    if world == 1:
        result = AggregationWorker(cfg, 0, 1).run()
        return [result] if result is not None else []

    with Pool(processes=world) as pool:
        results = pool.starmap(_worker_fn, [(cfg, r, world) for r in range(world)])

    return [p for p in results if p is not None]


# ---------------------------------------------------------------------------
# Sanitization Helpers
# ---------------------------------------------------------------------------

def drop_near_constant(mat: np.ndarray, cols: list[str], eps: float = 1e-12) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Remove columns from a CV matrix that have standard deviation < eps.
    Returns: (cleaned_mat, cleaned_cols, dropped_cols)
    Assumes column 0 is RETIS_step and is never dropped.
    """
    if mat.shape[1] < 2:
        return mat, cols, []

    dropped = []
    keep_idx = [0]
    for i in range(1, mat.shape[1]):
        std = np.std(mat[:, i])
        if std > eps:
            keep_idx.append(i)
        else:
            dropped.append(cols[i])

    if len(keep_idx) == mat.shape[1]:
        return mat, cols, dropped

    return mat[:, keep_idx], [cols[i] for i in keep_idx], dropped


def drop_duplicates_perfect_corr(mat: np.ndarray, cols: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Remove columns that are perfectly correlated to an earlier column.
    Returns: (cleaned_mat, cleaned_cols, dropped_cols)
    Assumes column 0 is RETIS_step and is never dropped.
    """
    if mat.shape[1] < 2:
        return mat, cols, []

    df = pd.DataFrame(mat, columns=cols)
    
    # We only compute correlation for feature columns (exclude RETIS_step)
    feat_cols = cols[1:]
    if len(feat_cols) < 2:
        return mat, cols, []
        
    corr_matrix = df[feat_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop_set = set()
    for column in upper_tri.columns:
        # Check if perfectly correlated or anti-correlated
        if any((upper_tri[column] > 0.999999) | (upper_tri[column] < -0.999999)):
            to_drop_set.add(column)

    if not to_drop_set:
        return mat, cols, []

    dropped = list(to_drop_set)
    keep_cols = [c for c in cols if c not in to_drop_set]
    keep_idx = [cols.index(c) for c in keep_cols]

    return mat[:, keep_idx], keep_cols, dropped


# ---------------------------------------------------------------------------
# Merge part files → CVmat.h5
# ---------------------------------------------------------------------------

def merge_parts(
    parts_base: Path,
    out_file:     Path | None = None,
    delete_parts: bool = False,
    interfaces: list[float] | None = None,
    cvmat_n_grid: int = 0,
) -> Path:
    """
    Merge all  <parts_base>.part.<rank>.h5  files into a single CVmat.h5.

    For each grid_id/module group the cv matrices are vertically concatenated
    and re-sorted by RETIS_step (column 0).  The merged file is written
    atomically (tmp → rename).

    Args:
        parts_base:    The base path used by AggregationWorker
                       (e.g. CVs/890/sorted/CVmat.h5).  Part files are
                       discovered as <parts_base>.part.*.h5 in the same dir.
        out_file:      Where to write the merged result.  Defaults to
                       <parts_base.parent>/CVmat.h5 (i.e. strips 'sorted/').
                       Pass explicitly to decouple merged location from parts.
        delete_parts:  If True, remove part files after a successful merge.

    Returns the path of the merged file.
    """
    parent   = parts_base.parent
    stem     = parts_base.name               # e.g. "CVmat.h5"
    parts    = sorted(parent.glob(f"{stem}.part.*.h5"))

    if not parts:
        raise FileNotFoundError(
            f"No part files found matching {parent}/{stem}.part.*.h5"
        )

    # Default merged output: one level up from sorted/ → CVs/890/CVmat.h5
    if out_file is None:
        out_file = (parent.parent / stem).resolve()

    log.info("Merging %d part files → %s", len(parts), out_file)

    tmp_path = out_file.parent / f"{stem}.merging.{os.getpid():x}.h5"
    out_file.parent.mkdir(parents=True, exist_ok=True)


    try:
        # ── Collect grid IDs and modules from part 0 ──────────────────────
        with h5py.File(parts[0], "r", locking=False) as f0:
            shared_attrs = {k: f0.attrs[k] for k in f0.attrs}
            # grid IDs are numeric top-level groups
            grid_ids = sorted(
                int(k) for k in f0.keys()
                if k not in ("lambda_and_weight",) and k.isdigit()
            )
            modules_for_grid = {
                gid: sorted(f0[str(gid)].keys())
                for gid in grid_ids
            }

        with h5py.File(tmp_path, "w", locking=False) as outf:
            # Root attrs
            for k, v in shared_attrs.items():
                outf.attrs[k] = v

            # lambda_and_weight: concat and sort
            lw_rows = []
            for p in parts:
                with h5py.File(p, "r", locking=False) as f:
                    if "lambda_and_weight" in f:
                        lw_rows.append(f["lambda_and_weight"][()])
            if lw_rows:
                lw_all = np.vstack(lw_rows)
                lw_all = lw_all[lw_all[:, 0].argsort()]
                d = outf.create_dataset(
                    "lambda_and_weight", data=lw_all,
                    compression="gzip", shuffle=True,
                )
                d.attrs["labels"] = np.array(
                    ["RETIS_step", "lambda_max", "weight"], dtype=object
                )
                d.attrs["col_names"] = np.array(
                    ["RETIS_step", "lambda_max", "weight"], dtype=object
                )

            # Pre-bake wham_grid and reactive_masks
            if interfaces and cvmat_n_grid > 0 and lw_rows:
                wham_grid = np.linspace(interfaces[0], interfaces[-1], cvmat_n_grid, dtype=np.float32)
                outf.create_dataset("wham_grid", data=wham_grid)

                # reactive_masks[i, j] = lambda_max[i] > wham_grid[j]
                lambda_max_arr = lw_all[:, 1:2]  # shape (N, 1)
                masks = lambda_max_arr > wham_grid  # broadcasts to (N, n_grid)
                outf.create_dataset(
                    "reactive_masks", data=masks,
                    compression="gzip", shuffle=True
                )

            # Merge order_txt groups from all parts
            order_grp = outf.create_group("order_txt")
            for p in parts:
                with h5py.File(p, "r", locking=False) as f:
                    if "order_txt" in f:
                        for step_id in f["order_txt"].keys():
                            if step_id not in order_grp:
                                order_grp.create_dataset(
                                    step_id,
                                    data=f[f"order_txt/{step_id}"][()],
                                    compression="gzip",
                                )
            log.info("order_txt: merged %d steps into CVmat.h5", len(order_grp))

            # Per-grid groups
            dropped_records = []
            for gid in grid_ids:
                for mod in modules_for_grid.get(gid, []):
                    if mod == "dataset":
                        continue
                        
                    all_cv, all_cols = [], None
                    for p in parts:
                        with h5py.File(p, "r", locking=False) as f:
                            gpath = f"{gid}/{mod}"
                            if gpath not in f:
                                continue
                            g = f[gpath]
                            if all_cols is None and "cols" in g:
                                raw   = g["cols"][()]
                                all_cols = [
                                    c.decode("utf-8") if isinstance(c, bytes) else str(c)
                                    for c in raw
                                ]
                            if "cv" in g:
                                all_cv.append(g["cv"][()])

                    if not all_cv or all_cols is None:
                        continue

                    mat = np.vstack(all_cv)
                    mat = mat[mat[:, 0].argsort()]   # sort by RETIS_step

                    # Sanitize
                    mat, all_cols, dropped_c = drop_near_constant(mat, all_cols)
                    if dropped_c:
                        dropped_records.append((gid, mod, "near_constant", dropped_c))
                        log.info("Grid %d mod %s dropped near-constant: %s", gid, mod, dropped_c)
                    print(mat, all_cols)
                    mat, all_cols, dropped_d = drop_duplicates_perfect_corr(mat, all_cols)
                    if dropped_d:
                        dropped_records.append((gid, mod, "duplicate_corr", dropped_d))
                        log.info("Grid %d mod %s dropped dupes: %s", gid, mod, dropped_d)

                    _write_group(outf, gid, mod, mat, all_cols)
                    log.debug("Merged %d/%s: shape=%s", gid, mod, mat.shape)

                log.info("Grid %d merged  (modules: %d)", gid, len(modules_for_grid.get(gid, [])))

            if dropped_records:
                san_grp = outf.create_group("sanitization")
                for gid, mod, reason, dropped_cols in dropped_records:
                    ds_name = f"{gid}/{mod}/{reason}"
                    san_grp.create_dataset(ds_name, data=np.array(dropped_cols, dtype=object))

        os.replace(tmp_path, out_file)
        log.info("CVmat.h5 written → %s", out_file)

        if delete_parts:
            for p in parts:
                try:
                    p.unlink()
                except Exception:
                    log.warning("Could not delete part file: %s", p)

    finally:
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return out_file
