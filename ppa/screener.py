"""
screener.py — PPA Screener: one-by-one and linear combination CV screening.

Reads settings from infretis.toml (via config_loader), writes results to
CVs/{n_grid}/screen.h5, and generates a screen_report.md summary.
"""
import argparse
import itertools
import logging
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import dual_annealing

from core import compute_predictive_capacity_from_scalar, ppa_objective
from config_loader import load_ppa_config


logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger("screener")

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_cli_args():
    p = argparse.ArgumentParser(description="PPA Screener for CVmat.h5")
    p.add_argument("--toml", default=Path("infretis.toml"), type=Path,
                   help="Path to infretis.toml (default: infretis.toml)")
    p.add_argument("--lambda-c", type=int, nargs="+", default=None,
                   help="Override lambda_c from TOML (list of grid IDs)")
    p.add_argument("--lambda-r", type=int, nargs="+", default=None,
                   help="Override lambda_r from TOML (list of grid IDs)")
    p.add_argument("--n-cvar", type=int, default=None,
                   help="Override n_cvar (1=OBO, 2=lin2, etc.)")
    p.add_argument("--n-best", type=int, default=None,
                   help="Override n_best_obo from TOML")
    p.add_argument("--workers", type=int, default=None,
                   help="Override workers from TOML")
    p.add_argument("--force", action="store_true",
                   help="Force recomputation even if previously screened")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Worker functions (top-level for multiprocessing pickling)
# ═══════════════════════════════════════════════════════════════════════════

def process_obo_task(cv_idx: int, cv_name: str, cv_data: np.ndarray,
                     weight: np.ndarray, reactive: np.ndarray) -> tuple[str, float, float]:
    """Worker task for One-By-One screening."""
    T_val, P_val, _ = compute_predictive_capacity_from_scalar(cv_data, weight, reactive)
    return cv_name, float(T_val), float(P_val)


def process_lin_task(cv_names: tuple[str, ...], feat_mat: np.ndarray,
                     weight: np.ndarray, reactive: np.ndarray,
                     maxiter: int, initial_temp: float) -> tuple[tuple[str, ...], float, np.ndarray]:
    """Worker task for linear combination screening."""
    n_cvar = feat_mat.shape[1]

    # Z-score features for optimizer stability
    mu = feat_mat.mean(axis=0)
    sigma = feat_mat.std(axis=0, ddof=0)
    sigma[sigma == 0.0] = 1.0
    feat_norm = (feat_mat - mu) / sigma

    bounds = [(-1.0, 1.0)] * n_cvar
    alpha0 = np.ones(n_cvar, dtype=float)

    result = dual_annealing(
        ppa_objective,
        bounds=bounds,
        args=(weight, reactive, feat_norm, True),
        maxiter=maxiter,
        initial_temp=initial_temp,
        no_local_search=True,
        x0=alpha0,
    )

    if result.success:
        T_val = 1.0 - float(result.fun)
    else:
        T_val, _, _ = ppa_objective(result.x, weight, reactive, feat_norm, optimize=False)
        T_val = float(T_val)

    return cv_names, T_val, np.asarray(result.x, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════

def write_screen_report(h5_path: Path) -> None:
    """Generate a markdown report summarising screen.h5 contents."""
    report_path = h5_path.with_name("screen_report.md")
    lines = [f"# Screen Report\n\n**File:** `{h5_path}`\n\n"]

    if not h5_path.exists():
        lines.append("No `screen.h5` file found.")
        report_path.write_text("".join(lines), encoding="utf-8")
        return

    with h5py.File(h5_path, "r", locking=False) as f:
        lc_keys = sorted([k for k in f.keys() if k.startswith("lambda_c=")],
                         key=lambda x: int(x.split("=")[1]))
        total_lc = len(lc_keys)
        total_lr = 0
        total_tests = 0

        for lc in lc_keys:
            lr_keys = [k for k in f[lc].keys() if k.startswith("lambda_r=")]
            total_lr += len(lr_keys)
            for lr in lr_keys:
                for mode in f[f"{lc}/{lr}"].keys():
                    g = f[f"{lc}/{lr}/{mode}"]
                    if "cv_names" in g:
                        total_tests += len(g["cv_names"])

        lines += [
            f"**Total `lambda_c` grids:** {total_lc}  \n",
            f"**Total `lambda_r` entries:** {total_lr}  \n",
            f"**Total CV combinations evaluated:** {total_tests}  \n\n",
        ]

        def _decode(item):
            if isinstance(item, np.ndarray):
                return " + ".join(c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in item)
            return item.decode("utf-8") if isinstance(item, bytes) else str(item)

        for lc in lc_keys:
            lines.append(f"## {lc}\n\n")
            lines.append("| lambda_r | type | total_tested | rank | top CV(s) | T_score |\n")
            lines.append("|---|---|---|---|---|---|\n")
            lr_keys_sorted = sorted([k for k in f[lc].keys() if k.startswith("lambda_r=")],
                                    key=lambda x: int(x.split("=")[1]))
            for lr in lr_keys_sorted:
                for mode in sorted(f[f"{lc}/{lr}"].keys()):
                    g = f[f"{lc}/{lr}/{mode}"]
                    if "cv_names" not in g or "T_scores" not in g:
                        continue
                    n_cvs = len(g["cv_names"])
                    if n_cvs == 0:
                        continue
                    lr_val = lr.split("=")[1]
                    for i in range(min(15, n_cvs)):
                        cv_str = _decode(g["cv_names"][i])
                        t_val = float(g["T_scores"][i])
                        if i == 0:
                            lines.append(f"| {lr_val} | {mode} | {n_cvs} | 1 | `{cv_str}` | {t_val:.4e} |\n")
                        else:
                            lines.append(f"| | | | {i+1} | `{cv_str}` | {t_val:.4e} |\n")
            lines.append("\n")

        # Global ranking
        mode_cv_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for lc in lc_keys:
            for lr in [k for k in f[lc].keys() if k.startswith("lambda_r=")]:
                for mode in f[f"{lc}/{lr}"].keys():
                    g = f[f"{lc}/{lr}/{mode}"]
                    if "cv_names" not in g or "T_scores" not in g:
                        continue
                    for i in range(len(g["cv_names"])):
                        mode_cv_scores[mode][_decode(g["cv_names"][i])].append(float(g["T_scores"][i]))

        if mode_cv_scores:
            lines.append("---\n\n# Global Ranking (mean T across all λ_c / λ_r pairs)\n\n")
            for mode in sorted(mode_cv_scores.keys()):
                cv_means = sorted(
                    [(cv, sum(s) / len(s), max(s), min(s), len(s)) for cv, s in mode_cv_scores[mode].items()],
                    key=lambda x: x[1], reverse=True,
                )
                mode_label = "one-by-one (OBO)" if mode == "obo" else mode
                lines.append(f"## Global {mode_label} ranking\n\n")
                lines.append("| rank | CV | n_pairs | <T> | range(T) | T_max | T_min |\n")
                lines.append("|---|---|---|---|---|---|---| \n")
                for rank, (cv_str, mean_t, max_t, min_t, n) in enumerate(cv_means[:15], 1):
                    lines.append(f"| {rank} | `{cv_str}` | {n} | {mean_t:.4e} | {max_t - min_t:.4e} | {max_t:.4e} | {min_t:.4e} |\n")
                lines.append("\n")

    report_path.write_text("".join(lines), encoding="utf-8")
    log.info("Screen report written → %s", report_path)


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _group_pairs_by_lc(pairs: list[tuple]) -> list[tuple[int, list[int]]]:
    """Group lambda pairs by lc: [(lc, [lr1, lr2, ...]), ...]"""
    d: dict[int, list[int]] = defaultdict(list)
    for lc, lr in pairs:
        d[lc].append(lr)
    return [(lc, sorted(lrs)) for lc, lrs in sorted(d.items())]


def _screen_one(cfg, force: bool, n_cvar, lc, lr, lam_val,
                feat_mat, all_names, weights, reactive,
                out_file, rank, world, num_workers):
    """Run OBO or lin screening for one (lc, lr, n_cvar) triplet."""
    scr = cfg.screening

    if n_cvar == 1:
        grp_path = f"lambda_c={lc}/lambda_r={lr}/obo"
        if out_file.exists() and not force:
            with h5py.File(out_file, "r") as outf:
                if grp_path in outf:
                    log.info("OBO for %s already exists. Skipping. (--force to recompute)", grp_path)
                    return

        tasks = [(i, all_names[i], feat_mat[:, i], weights, reactive) for i in range(len(all_names))]
        if world > 1 and "CV_RANK" in os.environ:
            tasks = tasks[rank::world]

        start = time.perf_counter()
        with Pool(processes=num_workers) as pool:
            results = list(pool.starmap(process_obo_task, tasks))
        log.info("OBO computed %d CVs in %.1fs", len(results), time.perf_counter() - start)

        results.sort(key=lambda x: x[1], reverse=True)
        if rank == 0:
            with h5py.File(out_file, "a") as outf:
                if grp_path in outf:
                    del outf[grp_path]
                grp = outf.create_group(grp_path)
                grp.create_dataset("cv_names", data=np.array([x[0] for x in results], dtype=object))
                grp.create_dataset("T_scores", data=np.array([x[1] for x in results], dtype=np.float32))
                grp.create_dataset("P_A", data=results[0][2] if results else 0.0)
                grp.create_dataset("wham_grid_pt", data=lam_val)

    else:
        grp_path = f"lambda_c={lc}/lambda_r={lr}/lin{n_cvar}"
        if out_file.exists() and not force:
            with h5py.File(out_file, "r") as outf:
                if grp_path in outf:
                    log.info("lin%d for %s already exists. Skipping.", n_cvar, grp_path)
                    return

        obo_path = f"lambda_c={lc}/lambda_r={lr}/obo"
        with h5py.File(out_file, "r") as outf:
            if obo_path not in outf:
                log.error("Cannot run lin%d before OBO! Run with --n-cvar 1 first.", n_cvar)
                return
            obo_names_raw = outf[f"{obo_path}/cv_names"][()]
            obo_names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in obo_names_raw]

        top_names = obo_names[:cfg.n_try_lin]
        log.info("Forming lin%d combinations from top %d OBO CVs (n_try_lin=%d, save best %d)",
                 n_cvar, len(top_names), cfg.n_try_lin, cfg.n_best_lin)

        combos = list(itertools.combinations(top_names, n_cvar))
        log.info("Formed %d combinations of size %d", len(combos), n_cvar)

        col_idx = {name: i for i, name in enumerate(all_names)}
        maxiter = scr.maxiter_lin2 if n_cvar == 2 else scr.maxiter_lin3
        initial_temp = scr.initial_temp

        tasks = []
        for combo in combos:
            idx = [col_idx[n] for n in combo if n in col_idx]
            if len(idx) != n_cvar:
                continue
            tasks.append((combo, feat_mat[:, idx], weights, reactive, maxiter, initial_temp))

        if world > 1 and "CV_RANK" in os.environ:
            tasks = tasks[rank::world]

        start = time.perf_counter()
        with Pool(processes=num_workers) as pool:
            results = list(pool.starmap(process_lin_task, tasks))
        log.info("lin%d computed %d combos in %.1fs", n_cvar, len(results), time.perf_counter() - start)

        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:cfg.n_best_lin]

        if rank == 0:
            with h5py.File(out_file, "a") as outf:
                if grp_path in outf:
                    del outf[grp_path]
                grp = outf.create_group(grp_path)
                grp.create_dataset("cv_names", data=np.array([x[0] for x in results], dtype=object))
                grp.create_dataset("T_scores", data=np.array([x[1] for x in results], dtype=np.float32))
                grp.create_dataset("alpha", data=np.vstack([x[2] for x in results]))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run_screening(
    toml_path: Path = Path("infretis.toml"),
    *,
    lambda_c=None,
    lambda_r=None,
    n_cvar=None,
    n_best=None,
    workers=None,
    force: bool = False,
    cv_mode=None,
):
    """Run CV screening. Can be called from analyze.py or directly."""
    cfg = load_ppa_config(
        toml_path,
        lambda_c=lambda_c,
        lambda_r=lambda_r,
        workers=workers,
        n_best=n_best,
        cv_mode=cv_mode,
    )
    cfg.log_summary()

    cvmat_path = cfg.cvmat_path
    out_file = cfg.screen_path

    rank = int(os.environ.get("CV_RANK", 0))
    world = int(os.environ.get("CV_WORLD", cfg.workers))
    num_workers = max(world, 1)

    n_cvar_list = [n_cvar] if n_cvar is not None else cfg.n_cvar_list

    if not cvmat_path.exists():
        log.error("CVmat.h5 not found at %s", cvmat_path)
        return

    log.info("Screening on %s → %s", cvmat_path, out_file)
    log.info("Workers: %d, n_cvar_list: %s", num_workers, n_cvar_list)

    lambda_pairs = cfg.lambda_pairs
    if not lambda_pairs:
        log.error("No valid lambda pairs (need lr > lc). Check [ppa] lambda_c / lambda_r.")
        return

    log.info("Lambda pairs to screen: %s", lambda_pairs)

    for lc, lr_list_for_lc in _group_pairs_by_lc(lambda_pairs):
        with h5py.File(cvmat_path, "r", locking=False) as f:
            lw = f["lambda_and_weight"][()]
            lw_labels = [l.decode("utf-8") if isinstance(l, bytes) else str(l)
                         for l in f["lambda_and_weight"].attrs["labels"]]
            step_col_lw = lw_labels.index("RETIS_step")
            weight_col_lw = lw_labels.index("weight")
            all_steps = lw[:, step_col_lw].astype(int)
            all_weights = lw[:, weight_col_lw]
            wham_grid = f["wham_grid"][()]

            gid = str(lc)
            if gid not in f:
                log.error("Grid ID %s not found in %s", gid, cvmat_path)
                continue

            mods = [m for m in sorted(f[gid].keys()) if "cv" in f[f"{gid}/{m}"]]
            all_features, all_names, cv_steps = [], [], None

            for mod in mods:
                cv_arr = f[f"{gid}/{mod}/cv"][()]
                cols = [c.decode("utf-8") if isinstance(c, bytes) else str(c)
                        for c in f[f"{gid}/{mod}/cols"][()]]
                if cv_steps is None and "RETIS_step" in cols:
                    cv_steps = cv_arr[:, cols.index("RETIS_step")].astype(int)
                for i, col in enumerate(cols):
                    if col == "RETIS_step":
                        continue
                    all_features.append(cv_arr[:, i])
                    all_names.append(col)

            feat_mat = np.column_stack(all_features)
            masks_full = f["reactive_masks"][()]

            if cv_steps is not None and len(cv_steps) != len(all_weights):
                step_set = set(cv_steps.tolist())
                keep_idx = np.array([i for i, s in enumerate(all_steps) if int(s) in step_set])
                weights = all_weights[keep_idx]
                masks_dset = masks_full[keep_idx]
                log.info("Aligned weights/masks: %d → %d rows (lambda_c=%s)", len(all_weights), len(weights), gid)
            else:
                weights, masks_dset = all_weights, masks_full

        log.info("Loaded %d features across %d modules", feat_mat.shape[1], len(mods))

        for lr in lr_list_for_lc:
            if lr < 0 or lr >= len(wham_grid):
                log.error("lambda_r %d out of bounds", lr)
                continue
            lam_val = float(wham_grid[lr])
            reactive = masks_dset[:, lr]
            log.info("── Screening lambda_c=%d, lambda_r=%d (λ=%.3f, react=%d/%d) ──",
                     lc, lr, lam_val, reactive.sum(), len(reactive))

            for n_cvar_i in n_cvar_list:
                _screen_one(cfg, force, n_cvar_i, lc, lr, lam_val,
                            feat_mat, all_names, weights, reactive,
                            out_file, rank, world, num_workers)

    log.info("Screening finished.")
    if rank == 0:
        write_screen_report(out_file)


if __name__ == "__main__":
    _a = _parse_cli_args()
    run_screening(
        toml_path=_a.toml,
        lambda_c=_a.lambda_c,
        lambda_r=_a.lambda_r,
        n_cvar=_a.n_cvar,
        n_best=_a.n_best,
        workers=_a.workers,
        force=_a.force,
    )
