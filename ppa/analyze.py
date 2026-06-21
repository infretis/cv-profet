"""
analyze.py — PPA Analyzer: T-Matrix, Post-Optimisation, and Diagnostics.

Modes (mutually exclusive):
  --tmat      Build T-matrix heatmaps across all (λ_c, λ_r) pairs.
  --optimize  Post-optimize alpha weights for a linear combination.
    --diagnose  Run full diagnostics (PDF/CDF/overlap plots) for top CVs.

All settings are read from infretis.toml. See config_loader.py.
"""
import argparse
import logging
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from core import (
    compute_predictive_capacity_from_scalar,
    compute_predictive_capacity_discrete,
    _is_discrete,
    ppa_objective,
)
from config_loader import load_ppa_config, PPAConfig
from plots import make_diag_plots, plot_t_matrix, plot_lincomb_joint_shapes, make_discrete_diag_plots
from cv_utils import decode_cv_item, short_name, sanitize_filename


logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
log = logging.getLogger("analyze")


# ═══════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_reactive_frames(
    pw_df: pd.DataFrame, interfaces: list[float], lambda_r_grid: list[int], max_grid: int
) -> pd.DataFrame:
    """
    Build a boolean mask DataFrame: True if a path is reactive w.r.t each λ_r.

    Column 0 (all-True) is always inserted for the baseline (no filtering).
    """
    lambda_r = np.asarray(lambda_r_grid, dtype=int)
    l_min, l_max = interfaces[0], interfaces[-1]
    lambda_values = l_min + (lambda_r / max_grid) * (l_max - l_min)

    r_dict: dict = {}
    if 0 not in lambda_r:
        r_dict[0] = np.ones(len(pw_df), dtype=bool)

    for l_r, val in zip(lambda_r, lambda_values):
        r_dict[int(l_r)] = pw_df["lambda_max"] > float(val)

    return pd.DataFrame(r_dict, index=pw_df.index)


def get_screen_winners(
    screen_h5: Path, lc: int, lr: int, mode: str, n_best: int
) -> list[str]:
    """
    Retrieve top n_best CV names from a previous screen.

    Parameters
    ----------
    mode : str
        'obo', 'lin2', or 'lin3'.
    """
    if not screen_h5.exists():
        log.warning("Screen file %s not found.", screen_h5)
        return []

    h5_mode = "obo" if mode == "obo" else f"lin{mode[-1]}"
    log.info("Extracting top %d %s winners from %s @ lc=%d/lr=%d", n_best, mode, screen_h5, lc, lr)

    with h5py.File(screen_h5, "r", locking=False) as f:
        path = f"lambda_c={lc}/lambda_r={lr}/{h5_mode}"
        if path not in f:
            log.warning("Path '%s' not found in screen file.", path)
            return []
        cv_names = f[path]["cv_names"]
        return [decode_cv_item(cv_names[i]) for i in range(min(n_best, len(cv_names)))]


def get_lin_winners_with_alpha(
    screen_h5: Path, lc: int, lr: int, n_cvar: int, n_best: int,
) -> list[tuple[list[str], np.ndarray]]:
    """Retrieve top n_best linear combo CVs AND their alpha weights from screen.h5."""
    if not screen_h5.exists():
        return []

    mode = f"lin{n_cvar}"
    with h5py.File(screen_h5, "r", locking=False) as f:
        path = f"lambda_c={lc}/lambda_r={lr}/{mode}"
        if path not in f:
            log.warning("Path '%s' not in screen.h5", path)
            return []

        grp = f[path]
        cv_names_raw = grp["cv_names"][()]
        alphas = grp["alpha"][()]
        top_n = min(n_best, len(cv_names_raw))

        results = []
        for i in range(top_n):
            names = cv_names_raw[i]
            if isinstance(names, np.ndarray):
                name_list = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in names]
            else:
                name_list = [names.decode("utf-8") if isinstance(names, bytes) else str(names)]
            results.append((name_list, alphas[i]))

    return results


def extract_features(
    cvmat_h5: Path, lc: int, cv_names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract feature matrix and path weights for a given λ_c.

    Returns
    -------
    df_feat : DataFrame — RETIS_step + requested CV columns
    df_pw   : DataFrame — path weights / lambda_max (aligned to df_feat)
    """
    with h5py.File(cvmat_h5, "r", locking=False) as f:
        pw_raw = f["lambda_and_weight"][()]
        labels = [l.decode("utf-8") if isinstance(l, bytes) else str(l) for l in f["lambda_and_weight"].attrs["labels"]]
        df_pw = pd.DataFrame(pw_raw, columns=labels)

        if str(lc) not in f:
            raise KeyError(f"lambda_c={lc} not found in CVmat.h5")

        lc_group = f[str(lc)]
        first_mod = list(lc_group.keys())[0]
        step_cols = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in lc_group[first_mod]["cols"][()]]
        steps = lc_group[first_mod]["cv"][()][:, step_cols.index("RETIS_step")]
        df_feat = pd.DataFrame({"RETIS_step": steps.astype(int)})

        for cv in cv_names:
            target_cv = cv
            target_mod = ""
            if "@" in cv:
                parts = cv.split("@")
                target_cv = parts[0]
                target_mod = parts[1] if len(parts) > 1 else ""

            found = False
            if target_mod and target_mod in lc_group:
                cols = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in lc_group[target_mod]["cols"][()]]
                if target_cv in cols:
                    df_feat[cv] = lc_group[target_mod]["cv"][()][:, cols.index(target_cv)]
                    found = True

            if not found:
                for mod_name in lc_group.keys():
                    cols = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in lc_group[mod_name]["cols"][()]]
                    if target_cv in cols:
                        df_feat[cv] = lc_group[mod_name]["cv"][()][:, cols.index(target_cv)]
                        found = True
                        break

            if not found:
                log.warning("Could not find CV %s in CVmat.h5 under lambda_c=%d", cv, lc)

        df_pw_aligned = df_pw[df_pw["RETIS_step"].isin(df_feat["RETIS_step"])].reset_index(drop=True)
        df_feat = df_feat.reset_index(drop=True)

    return df_feat, df_pw_aligned


# ═══════════════════════════════════════════════════════════════════════════
# Metadata writer
# ═══════════════════════════════════════════════════════════════════════════

def _write_metadata(outdir: Path, entries: list[dict], header_extra: str = "") -> None:
    """Write metadata.txt with ranked CVs and diagnostic numbers."""
    meta_path = outdir / "metadata.txt"
    with open(meta_path, "w") as f:
        f.write("# PPA Diagnostics Metadata\n")
        if header_extra:
            f.write(f"# {header_extra}\n")
        f.write(f"# {'Rank':<5} {'T':>8} {'S':>8} {'P_A':>12} "
                f"{'N_rea':>8} {'N_unr':>8} "
                f"{'W_rea':>12} {'W_unr':>12}  CV\n")
        f.write(f"# {'-'*5} {'-'*8} {'-'*8} {'-'*12} "
                f"{'-'*8} {'-'*8} "
                f"{'-'*12} {'-'*12}  {'─'*30}\n")
        for i, e in enumerate(entries, 1):
            f.write(
                f"  {i:<5d} {e['T']:8.4f} {e['S']:8.4f} {e['P_A']:12.4e} "
                f"{e['N_rea']:8d} {e['N_unr']:8d} "
                f"{e['W_rea']:12.4e} {e['W_unr']:12.4e}  {e['name']}\n"
            )
    log.info("Metadata written → %s", meta_path)


def _meta_entry(cv_name: str, T_val: float, P_A: float, dbg: dict, w: np.ndarray, reactive: np.ndarray) -> dict:
    """Build a single metadata entry dict from diagnostic results."""
    is_re = np.asarray(reactive, dtype=bool)
    return {
        "name": cv_name,
        "T": T_val,
        "S": float(dbg.get("S_val", 1.0 - T_val)),
        "P_A": P_A,
        "N_rea": int(np.sum(is_re)),
        "N_unr": int(np.sum(~is_re)),
        "W_rea": float(w[is_re].sum()),
        "W_unr": float(w[~is_re].sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# CV selection helpers
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_target_cvs(cfg: PPAConfig, n_best: int) -> list[str]:
    """
    Return the CV list to analyse, honouring cv_mode.

    manual → read obo_from_toml / lin2_from_toml / lin3_from_toml
    n_best  → auto-extract from screen.h5, respecting cfg.tmat.n_cvar_list

    Lin2/lin3 combinations are returned as "cv_a + cv_b" strings.
    Use _feature_cols_for_tmat() to get the flat individual-column list
    needed by extract_features.
    """
    sc = cfg.screen_path
    lc, lr = cfg.tmat.lambda_c_opt, cfg.tmat.lambda_r_opt

    if cfg.cv_mode == "manual":
        cvs = list(cfg.obo_from_toml)
        cvs.extend(cfg.lin2_from_toml)
        cvs.extend(cfg.lin3_from_toml)
        return cvs

    # n_best auto-extract — only fetch modes requested in [ppa.tmat]
    cvs: list[str] = []
    if 1 in cfg.tmat.n_cvar_list:
        cvs.extend(get_screen_winners(sc, lc, lr, "obo", n_best))
    for ncv in cfg.tmat.n_cvar_list:
        if ncv >= 2:
            cvs.extend(get_screen_winners(sc, lc, lr, f"lin{ncv}", n_best))
    return cvs


def _feature_cols_for_tmat(target_cvs: list[str]) -> list[str]:
    """
    Build the flat list of individual column names needed by extract_features.

    Lin2/lin3 combo strings like "cv_a + cv_b" are split into their components
    so that extract_features loads each component independently.  The combo
    string itself is NOT passed to extract_features (it cannot be looked up).
    """
    cols: list[str] = []
    seen: set[str] = set()
    for cv in target_cvs:
        parts = [c.strip() for c in cv.split("+")] if " + " in cv else [cv]
        for p in parts:
            if p not in seen:
                cols.append(p)
                seen.add(p)
    return cols


# ═══════════════════════════════════════════════════════════════════════════
# T-matrix mode
# ═══════════════════════════════════════════════════════════════════════════

def generate_t_heatmap(cfg: PPAConfig) -> None:
    """Build T-matrices mapping predictive capacities across (λ_c, λ_r) space."""
    cvmat_h5 = cfg.cvmat_path
    n_best = cfg.tmat.top_k
    lr_max = cfg.tmat.lambda_r_max
    grid_size = lr_max + 1

    target_cvs = _resolve_target_cvs(cfg, n_best)

    if not target_cvs:
        log.error("No CVs identified for T-matrix heatmap! Run screener first or specify obo_from_toml.")
        sys.exit(1)

    log.info("Target CVs to map (%d): %s", len(target_cvs), target_cvs)

    # expand combo strings → individual feature columns for loading
    feature_cols = _feature_cols_for_tmat(target_cvs)
    log.info("Feature columns to load from CVmat (%d): %s", len(feature_cols), feature_cols)

    out_dir = cvmat_h5.parent
    t_mat_path = out_dir / "T_mat.h5"
    lamb_range = list(range(grid_size))

    # SG keyword args from config
    sg_kw = dict(
        grid_expand_factor=cfg.sg.grid_expand_factor,
        sg_grid=cfg.sg.sg_grid,
        sg_polyorder=cfg.sg.sg_polyorder,
        sg_window_frac=cfg.sg.sg_window_frac,
        n_hist_bins=cfg.sg.n_hist_bins,
        integer_bins=cfg.sg.integer_bins,
    )

    with h5py.File(t_mat_path, "w") as out_f:
        with h5py.File(cvmat_h5, "r", locking=False) as cvf:
            lw = cvf["lambda_and_weight"][()]
            lw_labels = [l.decode("utf-8") if isinstance(l, bytes) else str(l) for l in cvf["lambda_and_weight"].attrs["labels"]]
            step_col = lw_labels.index("RETIS_step")
            weight_col = lw_labels.index("weight")
            all_steps = lw[:, step_col].astype(int)
            all_weights = lw[:, weight_col]
            masks_full = cvf["reactive_masks"][()]

        for lc in lamb_range:
            try:
                # Load individual feature columns (not the combo strings)
                df_feat, df_pw = extract_features(cvmat_h5, lc, feature_cols)
            except KeyError:
                log.warning("Skipping lambda_c=%d (not found in CVmat)", lc)
                continue

            cv_steps = df_feat.get("RETIS_step")
            if cv_steps is not None:
                step_set = set(cv_steps.astype(int).tolist())
                keep_idx = np.array([i for i, s in enumerate(all_steps) if int(s) in step_set])
                weights = all_weights[keep_idx]
                reactive_masks = masks_full[keep_idx]
            else:
                weights, reactive_masks = all_weights, masks_full

            for cv in target_cvs:
                if " + " in cv:
                    cols = [c.strip() for c in cv.split("+")]
                    try:
                        feat_arr = df_feat[cols].sum(axis=1).to_numpy()
                    except KeyError:
                        log.warning("Skipping %s at lc=%d: missing columns", cv, lc)
                        continue
                else:
                    if cv not in df_feat.columns:
                        log.warning("Skipping %s at lc=%d: missing column", cv, lc)
                        continue
                    feat_arr = df_feat[cv].to_numpy()

                dset_name = f"obo/{cv}" if " + " not in cv else f"lin_comb/{cv}"
                if dset_name not in out_f:
                    out_f.create_dataset(dset_name, shape=(grid_size, grid_size), dtype=np.float32, fillvalue=0.0)

                dset = out_f[dset_name]
                for lr in lamb_range:
                    if lr <= lc or lr >= reactive_masks.shape[1]:
                        continue
                    react = reactive_masks[:, lr]
                    T_val, _, _ = compute_predictive_capacity_from_scalar(feat_arr, weights, react, **sg_kw)
                    if isinstance(dset, h5py.Dataset):
                        dset[lr, lc] = float(T_val)

    log.info("T-matrices written → %s", t_mat_path)

    plot_t_matrix(t_mat_path, cfg.root_dir / "plots" / str(cfg.n_grid) / "T_mat")
    summarize_t_matrix(t_mat_path)


def summarize_t_matrix(t_mat_path: Path, sep: int = 5) -> None:
    """Generate a markdown summary report of T_mat.h5."""
    report_path = t_mat_path.with_name("T_mat_report.md")
    lines = [f"# T-Matrix Report\n\n**File:** `{t_mat_path}`\n\n"]

    with h5py.File(t_mat_path, "r", locking=False) as f:
        lines.append("## Structure\n\n| Group | Dataset | Shape |\n|---|---|---|\n")
        for grp_name in sorted(f.keys()):
            grp = f[grp_name]
            if isinstance(grp, h5py.Group):
                for ds_name in sorted(grp.keys()):
                    ds = grp[ds_name]
                    shape_str = "x".join(str(s) for s in ds.shape) if isinstance(ds, h5py.Dataset) else "group"
                    lines.append(f"| `{grp_name}` | `{ds_name}` | {shape_str} |\n")
        lines.append("\n")

        for grp_name in sorted(f.keys()):
            grp = f[grp_name]
            if not isinstance(grp, h5py.Group):
                continue
            cv_stats = []
            for ds_name in sorted(grp.keys()):
                ds = grp[ds_name]
                if not isinstance(ds, h5py.Dataset) or ds.ndim != 2:
                    continue
                mat = ds[()]
                grid_size = mat.shape[0]
                valid_vals, best_T, best_lc, best_lr = [], -1.0, 0, 0
                for lc in range(grid_size):
                    for lr in range(lc + sep, grid_size):
                        t = float(mat[lr, lc])
                        if t > 0.0:
                            valid_vals.append(t)
                            if t > best_T:
                                best_T, best_lc, best_lr = t, lc, lr
                if valid_vals:
                    arr = np.array(valid_vals)
                    cv_stats.append((ds_name, float(np.median(arr)), float(np.max(arr)), float(np.min(arr)), best_lc, best_lr, best_T, len(valid_vals)))

            if not cv_stats:
                continue
            cv_stats.sort(key=lambda x: x[1], reverse=True)
            grp_label = "one-by-one" if grp_name == "obo" else grp_name
            lines.append(f"## Global ranking by median T for {grp_label} CVs (λ_r ≥ λ_c + {sep})\n\n```\n")
            for rank, (name, med, mx, mn, blc, blr, bt, n) in enumerate(cv_stats[:10], 1):
                lines.append(f"{rank:>2}. {name:<70s} n={n:>4d}  <T>={med:.3e}  T_max={mx:.4e}  T_min={mn:.3e}  (best @ λ_c={blc}, λ_r={blr}, T={bt:.3e})\n")
            lines.append("```\n\n")

    report_path.write_text("".join(lines), encoding="utf-8")
    log.info("T-Matrix summary report → %s", report_path)


# ═══════════════════════════════════════════════════════════════════════════
# Optimize mode
# ═══════════════════════════════════════════════════════════════════════════

def post_optimize_linear_combination(cfg: PPAConfig) -> None:
    """Polish the alpha weights of a linear combination using L-BFGS-B."""
    manual_lin2 = cfg.lin2_from_toml
    if not manual_lin2:
        log.error("No linear combinations in `lin2_from_toml` to optimize!")
        sys.exit(1)

    combo = manual_lin2[0]
    if isinstance(combo, str) and " + " in combo:
        cols = [c.split("@")[0].strip() for c in combo.split("+")]
    elif isinstance(combo, list):
        cols = [str(c).split("@")[0].strip() for c in combo]
    else:
        log.error("Unrecognized format in lin2_from_toml: %s", combo)
        sys.exit(1)

    lc_opt, lr_opt = cfg.tmat.lambda_c_opt, cfg.tmat.lambda_r_opt
    log.info("Polishing alpha weights for: %s at lc=%d, lr=%d", cols, lc_opt, lr_opt)

    df_feat, df_pw = extract_features(cfg.cvmat_path, lc_opt, cols)
    grid_size = cfg.tmat.lambda_r_max + 1
    reactive_masks = get_reactive_frames(df_pw, cfg.interfaces, [lr_opt], grid_size - 1)
    react = reactive_masks[lr_opt].to_numpy()
    weights = df_pw["weight"].to_numpy()

    feat_mat = df_feat[cols].to_numpy()
    mu = feat_mat.mean(axis=0)
    sigma = feat_mat.std(axis=0, ddof=0)
    sigma[sigma == 0.0] = 1.0
    feat_norm = (feat_mat - mu) / sigma

    sg_kw = dict(
        grid_expand_factor=cfg.sg.grid_expand_factor,
        sg_grid=cfg.sg.sg_grid,
        sg_polyorder=cfg.sg.sg_polyorder,
        sg_window_frac=cfg.sg.sg_window_frac,
        n_hist_bins=cfg.sg.n_hist_bins,
        integer_bins=cfg.sg.integer_bins,
    )

    alpha0 = np.ones(len(cols), dtype=float)
    T_init = float(1.0 - ppa_objective(alpha0, weights, react, feat_norm))
    log.info("Initial T-score = %.5f (alpha = %s)", T_init, alpha0)

    bounds = [(-1.0, 1.0)] * len(cols)
    res = minimize(ppa_objective, alpha0, args=(weights, react, feat_norm, True),
                   method="L-BFGS-B", bounds=bounds, options={"disp": False})

    alpha_opt = res.x
    T_opt_raw, P_res, debug_res = ppa_objective(alpha_opt, weights, react, feat_norm, optimize=False)
    T_opt = float(T_opt_raw)
    log.info("Optimized T-score = %.5f (alpha = %s)", T_opt, alpha_opt)
    log.info("Improvement: +%.5f", T_opt - T_init)

    if "r_sg" not in debug_res:
        log.warning("No diagnostic data available (T=%.3f). Zero reactive paths? Skipping plots.", T_opt)
        return

    out_dir = cfg.root_dir / "plots" / str(cfg.n_grid) / "T_mat"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_name_clean = "_+".join(cols).replace("::", "-")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(debug_res["r_sg"], debug_res["sg_r"], label="Reactive CDF", color="red")
    axes[0].plot(debug_res["r_sg"], debug_res["sg_u"], label="Unreactive CDF", color="blue")
    axes[0].set_title(f"Smoothed CDFs (T={T_opt:.3f})")
    axes[0].legend()
    axes[1].plot(debug_res["r_sg"], debug_res["pdf_r"], label="Reactive PDF", color="red")
    axes[1].plot(debug_res["r_sg"], debug_res["pdf_u"], label="Unreactive PDF", color="blue")
    axes[1].fill_between(debug_res["r_sg"], 0, debug_res["overlap_q"], color="purple", alpha=0.3, label="Overlap (S)")
    axes[1].set_title("PDFs and Overlap")
    axes[1].legend()
    plt.suptitle(f"Post-Optimized: {cv_name_clean}")
    plt.tight_layout()
    plot_path = out_dir / f"opt_{cv_name_clean}.pdf"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    log.info("Optimization plots → %s", plot_path)


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostics mode
# ═══════════════════════════════════════════════════════════════════════════

def run_diagnostics(
    cfg: PPAConfig,
    log_plots: bool = False,
    discrete_mode: bool = False,
    discrete_max_unique: int = 50,
) -> None:
    """
    Run full diagnostics for top OBO and lin2 CVs.
    
    Output: plots/distribution/{lc}/{lr}/obo/ and lin2/
    discrete_mode : bool
        When True, auto-detect integer-valued CVs and generate extra
        DISCRETE/ bar-chart plots (PMF + committor) using the exact
        PMF overlap sum instead of SG smoothing.
    discrete_max_unique : int
        Max number of unique values before a CV is considered continuous.
    """
    lc_opt, lr_opt = cfg.lambda_c_opt, cfg.lambda_r_opt
    plots_root = cfg.root_dir / "plots" / str(cfg.n_grid) / "distribution"

    log.info("Running diagnostics (ref λc=%d, λr=%d)", lc_opt, lr_opt)

    sg_kw = dict(
        grid_expand_factor=cfg.sg.grid_expand_factor,
        sg_grid=cfg.sg.sg_grid,
        sg_polyorder=cfg.sg.sg_polyorder,
        sg_window_frac=cfg.sg.sg_window_frac,
        n_hist_bins=cfg.sg.n_hist_bins,
        integer_bins=cfg.sg.integer_bins,
    )

    if cfg.cv_mode == "manual":
        log.info("  cv_mode: manual")
        obo_names = cfg.obo_from_toml
        lin2_pairs = [
            (tuple(p.strip() for p in combo.split("+")), np.array([0.0, 0.0]))
            for combo in cfg.lin2_from_toml
            if " + " in combo
        ]
    else:
        log.info("  cv_mode: n_best")
        obo_names = get_screen_winners(cfg.screen_path, lc_opt, lr_opt, "obo", cfg.n_best_obo)
        lin2_pairs = get_lin_winners_with_alpha(cfg.screen_path, lc_opt, lr_opt, 2, cfg.n_best_lin)

    if not obo_names and not lin2_pairs:
        log.warning("No OBO or lin2 winners found. Nothing to do.")
        return

    log.info("  OBO winners:  %d", len(obo_names))
    log.info("  Lin2 winners: %d", len(lin2_pairs))

    all_cv_names = list(obo_names)
    for names, _ in lin2_pairs:
        for n in names:
            if n not in all_cv_names:
                all_cv_names.append(n)

    for lc, lr in cfg.lambda_pairs:
        log.info("\n%s\nDiagnostics for λc=%d, λr=%d\n%s", "=" * 60, lc, lr, "=" * 60)

        try:
            df_feat, df_pw = extract_features(cfg.cvmat_path, lc, all_cv_names)
        except KeyError as e:
            log.warning("Skipping (lc=%d): %s", lc, e)
            continue

        reactive_df = get_reactive_frames(df_pw, cfg.interfaces, [lr], cfg.n_grid - 1)
        reactive = reactive_df[lr].values.astype(bool)
        w = df_pw["weight"].values.astype(float)

        # ── OBO ──────────────────────────────────────────────────────────
        obo_dir = plots_root / str(lc) / str(lr) / "obo"
        obo_dir.mkdir(parents=True, exist_ok=True)
        obo_meta = []

        for cv_name in obo_names:
            if cv_name not in df_feat.columns:
                log.warning("  CV '%s' not found in CVmat.h5, skipping.", cv_name)
                continue
            x = df_feat[cv_name].values.astype(float)
            T_val, P_A, dbg = compute_predictive_capacity_from_scalar(x, w, reactive, **sg_kw)
            prefix = sanitize_filename(short_name(cv_name))
            make_diag_plots(
                dbg, obo_dir, prefix,
                cv_label=short_name(cv_name),
                title=f"{short_name(cv_name)}  λc={lc} λr={lr}  T={T_val:.3f}",
                log_plots=log_plots, is_obo=True,
            )
            if discrete_mode and _is_discrete(x, max_unique=discrete_max_unique):
                T_d, P_Ad, dbg_d = compute_predictive_capacity_discrete(x, w, reactive)
                log.info("  [DISCRETE] %s: T_disc=%.4f vs T_SG=%.4f (Δ=%+.4f)",
                         short_name(cv_name), T_d, T_val, T_d - T_val)
                make_discrete_diag_plots(
                    dbg_d, obo_dir, prefix,
                    cv_label=short_name(cv_name),
                    title=f"{short_name(cv_name)}  λc={lc} λr={lr}",
                    T_continuous=T_val,
                )
            obo_meta.append(_meta_entry(cv_name, T_val, P_A, dbg, w, reactive))
            log.info("  OBO %s: T=%.4f, P_A=%.4e", short_name(cv_name), T_val, P_A)

        obo_meta.sort(key=lambda e: e["T"], reverse=True)
        if obo_meta:
            _write_metadata(obo_dir, obo_meta, header_extra=f"λc={lc}  λr={lr}  OBO")

        # ── Lin2 ─────────────────────────────────────────────────────────
        lin2_dir = plots_root / str(lc) / str(lr) / "lin2"
        lin2_dir.mkdir(parents=True, exist_ok=True)
        lin2_meta = []

        for cv_names_pair, alpha in lin2_pairs:
            missing = [c for c in cv_names_pair if c not in df_feat.columns]
            if missing:
                log.warning("  Missing CVs %s for lin2, skipping.", missing)
                continue

            X_pair = df_feat[list(cv_names_pair)].values
            x_lin = X_pair @ alpha

            T_val, P_A, dbg = compute_predictive_capacity_from_scalar(x_lin, w, reactive, **sg_kw)
            short1, short2 = short_name(cv_names_pair[0]), short_name(cv_names_pair[1])
            combo = f"{short1}-{short2}"
            prefix = sanitize_filename(combo)
            title = f"{combo}  λc={lc} λr={lr}  T={T_val:.3f}"

            # Joint heatmap
            tmp_base = f"_tmp_{prefix}"
            plot_lincomb_joint_shapes(
                x1=df_feat[cv_names_pair[0]].values,
                x2=df_feat[cv_names_pair[1]].values,
                w=w, reactive=reactive,
                name1=cv_names_pair[0], name2=cv_names_pair[1],
                title=title, outdir=str(lin2_dir), basename=tmp_base,
            )

            log_dir = lin2_dir / "LOG"
            log_dir.mkdir(parents=True, exist_ok=True)

            for src_suffix, tgt_dir, tgt_name in [
                ("_joint_shapes.pdf",     lin2_dir, f"HEATMAP-{prefix}.pdf"),
                ("_joint_shapes_log.pdf", log_dir,  f"{prefix}-LOG.pdf"),
                ("_joint_raw_log.pdf",    log_dir,  f"{prefix}-LOG_raw.pdf"),
            ]:
                src = lin2_dir / f"{tmp_base}{src_suffix}"
                if src.exists():
                    src.rename(tgt_dir / tgt_name)

            # Remove 3d variant (not requested)
            src_3d = lin2_dir / f"{tmp_base}_joint_shapes_3d.pdf"
            if src_3d.exists():
                src_3d.unlink()

            make_diag_plots(
                dbg, lin2_dir, prefix,
                cv_label=combo, title=title,
                log_plots=log_plots, is_obo=False,
            )

            lin2_meta.append(_meta_entry(" + ".join(cv_names_pair), T_val, P_A, dbg, w, reactive))
            log.info("  LIN2 %s: T=%.4f, P_A=%.4e", combo, T_val, P_A)

        lin2_meta.sort(key=lambda e: e["T"], reverse=True)
        if lin2_meta:
            _write_metadata(lin2_dir, lin2_meta, header_extra=f"λc={lc}  λr={lr}  lin2")

    log.info("\nDiagnostics complete → %s", plots_root)


# ═══════════════════════════════════════════════════════════════════════════
# CLI — unified entry point for all PPA analysis modes
# ═══════════════════════════════════════════════════════════════════════════

def main():

    p = argparse.ArgumentParser(
        description="PPA Analyzer — unified entry point for screening, T-matrix, diagnostics, and optimisation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes (pick one):
  --screen    Screen CVs and write screen.h5           (runs screener.py)
  --tmat      Build T-matrix heatmaps from screen.h5
    --diagnose  Run PDF/CDF/overlap diagnostics for top CVs
  --optimize  Post-optimize alpha weights for a linear combination

Examples:
  python analyze.py --screen
  python analyze.py --screen --n-cvar 2 --workers 4
  python analyze.py --tmat
    python analyze.py --diagnose --log
  python analyze.py --optimize
""",
    )

    # ── shared ────────────────────────────────────────────────────────────
    p.add_argument("--toml", default=Path("infretis.toml"), type=Path,
                   help="Path to infretis.toml (default: infretis.toml)")
    p.add_argument("--cv-mode", choices=["n_best", "manual"], default=None,
                   help="Override cv_mode from TOML")

    # ── mode flags ────────────────────────────────────────────────────────
    modes = p.add_argument_group("modes (mutually exclusive)")
    modes.add_argument("--screen",   action="store_true", help="Run CV screening → screen.h5")
    modes.add_argument("--tmat",     action="store_true", help="Build T-matrix heatmaps")
    modes.add_argument("--diagnose", action="store_true", help="Run diagnostics (PDF/CDF/overlap plots)")
    modes.add_argument("--optimize", action="store_true", help="Post-optimize a linear combo")

    # ── --diagnose options ────────────────────────────────────────────────
    diagnose_grp = p.add_argument_group("--diagnose options")
    diagnose_grp.add_argument("--log", action="store_true",
                      help="Include log-scale plots alongside linear-scale ones")
    diagnose_grp.add_argument("--discrete", action="store_true",
                      help="Also run exact discrete PMF analysis for integer-valued CVs (--diagnose only)")
    diagnose_grp.add_argument("--discrete-max-unique", type=int, default=50, metavar="N",
                      help="CV treated as discrete if n_unique ≤ N and all values are integers (default: 50)")

    # ── --screen options ──────────────────────────────────────────────────
    scrn = p.add_argument_group("--screen options")
    scrn.add_argument("--lambda-c", type=int, nargs="+", default=None,
                      help="Override lambda_c grid IDs from TOML")
    scrn.add_argument("--lambda-r", type=int, nargs="+", default=None,
                      help="Override lambda_r grid IDs from TOML")
    scrn.add_argument("--n-cvar", type=int, default=None,
                      help="Run only this n_cvar (1=OBO, 2=lin2, …)")
    scrn.add_argument("--n-best", type=int, default=None,
                      help="Override n_best_obo from TOML")
    scrn.add_argument("--workers", type=int, default=None,
                      help="Override number of parallel workers from TOML")
    scrn.add_argument("--force", action="store_true",
                      help="Force recomputation even if already screened")

    args = p.parse_args()

    n_modes = sum([args.screen, args.tmat, args.diagnose, args.optimize])
    if n_modes == 0:
        p.print_help()
        sys.exit(0)
    if n_modes > 1:
        p.error("Specify exactly one mode: --screen, --tmat, --diagnose, or --optimize")

    if args.screen:
        # Delegate to screener.run_screening, forwarding relevant args
        from screener import run_screening
        run_screening(
            toml_path=args.toml,
            lambda_c=args.lambda_c,
            lambda_r=args.lambda_r,
            n_cvar=args.n_cvar,
            n_best=args.n_best,
            workers=args.workers,
            force=args.force,
            cv_mode=args.cv_mode,
        )
        return

    # For all other modes, load config once
    cfg = load_ppa_config(args.toml, cv_mode=args.cv_mode)
    cfg.log_summary()

    if args.tmat:
        generate_t_heatmap(cfg)
    elif args.diagnose:
        run_diagnostics(cfg, log_plots=args.log,
                        discrete_mode=args.discrete,
                        discrete_max_unique=args.discrete_max_unique)
    elif args.optimize:
        post_optimize_linear_combination(cfg)


if __name__ == "__main__":
    main()
