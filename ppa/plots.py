"""
plots.py — All matplotlib diagnostics and visualisation for the PPA pipeline.

Functions
---------
make_diag_plots         Standard OBO/lin2 diagnostic PDFs (OVERLAP, REA, UNR, CDF, optional LOG).
plot_t_matrix           Heatmap plots from T_mat.h5.
plot_lincomb_joint_shapes  Joint 2D density + 1D marginals for a lin2 CV pair.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
import numpy as np

from core import compute_predictive_capacity_from_scalar
from cv_utils import short_cv_label

# Publication-quality defaults for all PDF output
plt.rcParams.update({
    "savefig.dpi": 300,
    "figure.dpi": 150,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

log = logging.getLogger(__name__)


def _write_histogram_weights_txt(
    outdir: Path,
    prefix: str,
    hist_centers: np.ndarray,
    r_hist: np.ndarray,
    u_hist: np.ndarray,
) -> None:
    """Write histogram bin centers and weights to a text file."""
    centers = np.asarray(hist_centers, dtype=float).ravel()
    if centers.size == 0:
        return

    # Store the first column as integer bin labels.
    bin_labels = np.floor(centers).astype(int)

    r_vals = np.full(centers.shape, np.nan, dtype=float)
    u_vals = np.full(centers.shape, np.nan, dtype=float)

    r_hist_arr = np.asarray(r_hist, dtype=float).ravel()
    u_hist_arr = np.asarray(u_hist, dtype=float).ravel()
    if r_hist_arr.size == centers.size:
        r_vals = r_hist_arr
    if u_hist_arr.size == centers.size:
        u_vals = u_hist_arr

    keep_mask = ~((r_vals == 0.0) & (u_vals == 0.0))
    data = np.column_stack([bin_labels[keep_mask], r_vals[keep_mask], u_vals[keep_mask]])
    out_path = Path(outdir) / f"{prefix}-histogram.txt"
    np.savetxt(
        out_path,
        data,
        fmt=["%d", "%.18e", "%.18e"],
        header="bin_center_int r_hist u_hist",
        comments="",
    )
    log.info("Histogram bins/weights written -> %s", out_path)


# ---------------------------------------------------------------------------
# Shared style helper
# ---------------------------------------------------------------------------

def _style(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=9, fontweight="bold")
    ax.tick_params(axis="both", labelsize=9)
    # ax.grid(True, linestyle="--", alpha=0.4)


# ---------------------------------------------------------------------------
# Diagnostic plots — OBO and lin2
# ---------------------------------------------------------------------------

def make_diag_plots(
    debug: dict,
    outdir: Path,
    prefix: str,
    cv_label: str,
    title: str,
    log_plots: bool = False,
    is_obo: bool = False,
) -> None:
    """
    Generate the standard set of diagnostic plots from a debug dict.

    Naming convention:
      OBO:  OVERLAP-{cv}.pdf (best viz as prefix),  {cv}-CDF.pdf, {cv}-REA.pdf, {cv}-UNR.pdf
      Lin2: {combo}-OVERLAP.pdf (suffix),           {combo}-CDF.pdf, etc.
    Log plots go in LOG/ subdirectory.
    """
    outdir = Path(outdir)

    # Guard: refuse to draw SG line plots for discrete CVs
    if debug.get("mode") == "discrete":
        log.warning(
            "make_diag_plots called with a discrete debug dict for '%s'. "
            "Use make_discrete_diag_plots instead. Skipping.", prefix
        )
        return

    x_grid = np.asarray(debug.get("x_grid"))
    dx = float(debug.get("dx", 1.0))
    R_int_raw = np.asarray(debug.get("R_int_raw"))
    U_int_raw = np.asarray(debug.get("U_int_raw"))
    R_int_sg = np.asarray(debug.get("R_int_sg"))
    U_int_sg = np.asarray(debug.get("U_int_sg"))
    r_sg = np.asarray(debug.get("r_sg"))
    u_sg = np.asarray(debug.get("u_sg"))
    P_A = float(debug.get("P_A", 0.0))
    T_val = float(debug.get("T", 0.0))
    hist_centers = np.asarray(debug.get("hist_bin_centers"))
    print(hist_centers)
    r_hist = np.asarray(debug.get("r_hist"))
    u_hist = np.asarray(debug.get("u_hist"))
    overlap_q = np.asarray(debug.get("overlap_q"))

    _write_histogram_weights_txt(outdir, prefix, hist_centers, r_hist, u_hist)

    tiny = 1e-16
    PA_safe = max(P_A, tiny)
    uA_safe = max(1.0 - P_A, tiny)
    r_shape = r_sg / PA_safe
    u_shape = u_sg / uA_safe

    q_shape = overlap_q.copy() if overlap_q.size == x_grid.size else np.zeros_like(x_grid)

    overlap_name = f"OVERLAP-{prefix}.pdf" if is_obo else f"{prefix}-OVERLAP.pdf"

    # ── 1) OVERLAP: shape-only PDFs + overlap integrand ─────────────────────
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(x_grid, r_shape, color="#C81919", lw=1.5, label=r"$r_{\rm shape}$")
    ax.plot(x_grid, u_shape, color="#034793", lw=1.5, label=r"$u_{\rm shape}$")
    ax.plot(x_grid, q_shape, color="black", ls=":", lw=1.5, label=r"$S_A$")
    _style(ax, "Shape-only probability density")
    ax.legend(loc="best", fontsize=10, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(outdir / overlap_name, bbox_inches="tight")
    plt.close(fig)

    # ── 2) REA: reactive PDF ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.suptitle(f"{title}\nReactive probability density", fontsize=10)
    ax.plot(x_grid, r_sg, color="#C81919", lw=1.5, label=r"$r_{\rm SG}$")
    if hist_centers.size > 0 and r_hist.size == hist_centers.size:
        bw = (hist_centers[1] - hist_centers[0]) * 0.8 if len(hist_centers) > 1 else 1.0
        ax.bar(hist_centers, r_hist, width=bw, alpha=0.3, color="#C81919", label=r"$r_{\rm hist}$")
    _style(ax, "Probability density")
    ax.legend(loc="best", fontsize=9, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}-REA.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── 3) UNR: unreactive PDF ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.suptitle(f"{title}\nUnreactive probability density", fontsize=10)
    ax.plot(x_grid, u_sg, color="#034793", lw=1.5, label=r"$u_{\rm SG}$")
    if hist_centers.size > 0 and u_hist.size == hist_centers.size:
        bw = (hist_centers[1] - hist_centers[0]) * 0.8 if len(hist_centers) > 1 else 1.0
        ax.bar(hist_centers, u_hist, width=bw, alpha=0.3, color="#034793", label=r"$u_{\rm hist}$")
    _style(ax, "Probability density")
    ax.legend(loc="best", fontsize=9, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}-UNR.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── 4) CDF ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.suptitle(f"{title}\nCDF diagnostics", fontsize=10)
    ax.plot(x_grid, R_int_raw, color="#C81919", lw=1.7, label=r"$R_{\rm raw}$")
    ax.plot(x_grid, R_int_sg, color="tab:green", lw=1.5, ls=(0, (1, 2)), alpha=0.8, label=r"$R_{\rm SG}$")
    ax.plot(x_grid, U_int_raw, color="#034793", lw=1.7, label=r"$U_{\rm raw}$")
    ax.plot(x_grid, U_int_sg, color="tab:purple", lw=1.5, ls=(0, (1, 2)), alpha=0.8, label=r"$U_{\rm SG}$")
    if overlap_q.size == x_grid.size:
        q_nonneg = np.clip(overlap_q, 0.0, None)
        S_cum = np.cumsum(q_nonneg) * dx
        S_total = float(debug.get("S_val", S_cum[-1]))
        if S_cum[-1] > 0:
            S_cum *= S_total / S_cum[-1]
        S_cum = np.clip(S_cum, 0.0, 1.0)
        ax.plot(x_grid, S_cum, color="black", ls=":", lw=1.5, label=r"$S(\psi)$")
    ax.set_ylim(-0.02, 1.02)
    _style(ax, "Cumulative probability / overlap")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}-CDF.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── 5) Log plots ─────────────────────────────────────────────────────────
    if not log_plots:
        return

    log_dir = outdir / "LOG"
    log_dir.mkdir(parents=True, exist_ok=True)

    r_log = np.clip(r_shape, tiny, None)
    u_log = np.clip(u_shape, tiny, None)
    q_log_shape = np.clip(q_shape, tiny, None)
    q_log_unscaled = np.clip(q_shape * P_A, tiny, None)

    # OVERLAP LOG
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(x_grid, r_log, color="#C81919", lw=1.5, label=r"$r_{\rm shape}$")
    ax.plot(x_grid, u_log, color="#034793", lw=1.5, label=r"$u_{\rm shape}$")
    ax.plot(x_grid, q_log_shape, color="black", ls=":", lw=1.5, label=r"$S_A$")
    ax.set_yscale("log")
    _style(ax, "Shape-only probability density (log)")
    ax.legend(loc="best", fontsize=9, framealpha=0.8)
    fig.tight_layout()
    fname = f"OVERLAP-{prefix}-LOG.pdf" if is_obo else f"{prefix}-OVERLAP-LOG.pdf"
    fig.savefig(log_dir / fname, bbox_inches="tight")
    plt.close(fig)

    # REA LOG
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.suptitle(f"{title}\nReactive probability density (log)", fontsize=10)
    ax.plot(x_grid, np.clip(r_sg, tiny, None), color="#C81919", lw=1.5, label=r"$r_{\rm SG}$")
    if hist_centers.size > 0 and r_hist.size == hist_centers.size:
        bw = (hist_centers[1] - hist_centers[0]) * 0.8 if len(hist_centers) > 1 else 1.0
        ax.bar(hist_centers, r_hist, width=bw, alpha=0.3, color="#C81919", label=r"$r_{\rm hist}$")
    ax.set_yscale("log")
    _style(ax, "Probability density (log)")
    ax.legend(loc="best", fontsize=9, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(log_dir / f"{prefix}-REA-LOG.pdf", bbox_inches="tight")
    plt.close(fig)

    # UNR LOG
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.suptitle(f"{title}\nUnreactive probability density (log)", fontsize=10)
    ax.plot(x_grid, np.clip(u_sg, tiny, None), color="#034793", lw=1.5, label=r"$u_{\rm SG}$")
    if hist_centers.size > 0 and u_hist.size == hist_centers.size:
        bw = (hist_centers[1] - hist_centers[0]) * 0.8 if len(hist_centers) > 1 else 1.0
        ax.bar(hist_centers, u_hist, width=bw, alpha=0.3, color="#034793", label=r"$u_{\rm hist}$")
    ax.set_yscale("log")
    _style(ax, "Probability density (log)")
    ax.legend(loc="best", fontsize=9, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(log_dir / f"{prefix}-UNR-LOG.pdf", bbox_inches="tight")
    plt.close(fig)

    # CDF LOG
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.suptitle(f"{title}\nCDF diagnostics (log)", fontsize=10)
    ax.plot(x_grid, np.clip(R_int_raw, tiny, None), color="#C81919", lw=1.7, label=r"$R_{\rm raw}$")
    ax.plot(x_grid, np.clip(R_int_sg, tiny, None), color="tab:green", lw=1.5, ls=(0, (1, 2)), alpha=0.8, label=r"$R_{\rm SG}$")
    ax.plot(x_grid, np.clip(U_int_raw, tiny, None), color="#034793", lw=1.7, label=r"$U_{\rm raw}$")
    ax.plot(x_grid, np.clip(U_int_sg, tiny, None), color="tab:purple", lw=1.5, ls=(0, (1, 2)), alpha=0.8, label=r"$U_{\rm SG}$")
    if overlap_q.size == x_grid.size:
        q_nonneg = np.clip(overlap_q, 0.0, None)
        S_cum = np.cumsum(q_nonneg) * dx
        S_total = float(debug.get("S_val", S_cum[-1]))
        if S_cum[-1] > 0:
            S_cum *= S_total / S_cum[-1]
        S_cum = np.clip(S_cum, tiny, 1.0)
        ax.plot(x_grid, S_cum, color="black", ls=":", lw=1.5, label=r"$S(\psi)$")
    ax.set_yscale("log")
    _style(ax, "Cumulative probability / overlap (log)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(log_dir / f"{prefix}-CDF-LOG.pdf", bbox_inches="tight")
    plt.close(fig)

    # RAW PDFs (log, no rescaling)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.suptitle(f"{title}\nRaw PDFs (log, no rescaling)", fontsize=10)
    ax.plot(x_grid, np.clip(r_sg, tiny, None), color="#C81919", lw=1.5, label=r"$r(\psi)$")
    ax.plot(x_grid, np.clip(u_sg, tiny, None), color="#034793", lw=1.5, label=r"$u(\psi)$")
    ax.plot(x_grid, q_log_unscaled, color="black", ls=":", lw=1.5, label=r"$q(\psi)$")
    ax.set_yscale("log")
    _style(ax, "Density (log, unscaled)")
    ax.legend(loc="best", fontsize=9, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(log_dir / f"{prefix}-LOG_raw.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# T-matrix heatmap
# ---------------------------------------------------------------------------

def plot_t_matrix(t_mat_path: Path, out_dir: Path) -> None:
    """Read T_mat.h5 and generate heatmap plots for each CV matrix."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(t_mat_path, "r") as f:
        for mode in f.keys():
            if not isinstance(f[mode], h5py.Group):
                continue
            for cv in f[mode].keys():
                mat = f[mode][cv][()]
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(mat, origin="lower", cmap="viridis", aspect="auto")
                fig.colorbar(im, ax=ax, label="Predictive Capacity (T)")
                ax.set_xlabel("lambda_c index")
                ax.set_ylabel("lambda_r index")
                ax.set_title(f"{mode} / {cv}")
                cv_clean = cv.replace(" + ", "_").replace("::", "-")
                fig.savefig(out_dir / f"{mode}_{cv_clean}.pdf", dpi=300, bbox_inches="tight")
                plt.close(fig)

    log.info("T-Matrix heatmaps plotted → %s", out_dir)


# ---------------------------------------------------------------------------
# Joint 2D density plot for lin2 pairs
# ---------------------------------------------------------------------------

def plot_lincomb_joint_shapes(
    x1,
    x2,
    w,
    reactive,
    name1: str,
    name2: str,
    title: str,
    outdir: str,
    basename: str,
    show: bool = False,
) -> None:
    """
    Joint-plot for a linear combination of two CVs:
      - centre: 2D weighted density (reactive vs unreactive)
      - top:    shape-only 1D PDFs along x1
      - right:  shape-only 1D PDFs along x2

    Saves three files:
      {basename}_joint_shapes.pdf     — linear density scale
      {basename}_joint_shapes_log.pdf — log density scale
      {basename}_joint_raw_log.pdf    — raw (unscaled) log density
    """
    os.makedirs(outdir, exist_ok=True)

    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    w = np.asarray(w, dtype=float)
    reactive = np.asarray(reactive, dtype=bool)

    T1, P_A1, dbg1 = compute_predictive_capacity_from_scalar(x1, w, reactive)
    T2, P_A2, dbg2 = compute_predictive_capacity_from_scalar(x2, w, reactive)

    P_A = float(P_A1)
    tiny = 1e-16
    PA_safe = max(P_A, tiny)
    uA_safe = max(1.0 - P_A, tiny)

    x1_grid = np.asarray(dbg1.get("x_grid"))
    r1_sg = np.asarray(dbg1.get("r_sg"))
    u1_sg = np.asarray(dbg1.get("u_sg"))
    r1_shape = r1_sg / PA_safe
    u1_shape = u1_sg / uA_safe

    x2_grid = np.asarray(dbg2.get("x_grid"))
    r2_sg = np.asarray(dbg2.get("r_sg"))
    u2_sg = np.asarray(dbg2.get("u_sg"))
    r2_shape = r2_sg / PA_safe
    u2_shape = u2_sg / uA_safe

    is_re = reactive
    is_un = ~reactive

    # Build 2D grid
    margin = 0.05
    x1_min, x1_max = np.nanmin(x1), np.nanmax(x1)
    x2_min, x2_max = np.nanmin(x2), np.nanmax(x2)
    dx1 = (x1_max - x1_min) * margin
    dx2 = (x2_max - x2_min) * margin
    x1_lin = np.linspace(x1_min - dx1, x1_max + dx1, 80)
    x2_lin = np.linspace(x2_min - dx2, x2_max + dx2, 80)
    X, Y = np.meshgrid(x1_lin, x2_lin)
    grid_points = np.vstack([X.ravel(), Y.ravel()])

    try:
        from scipy.stats import gaussian_kde
        Z_un = gaussian_kde(np.vstack([x1[is_un], x2[is_un]]), weights=w[is_un])(grid_points).reshape(X.shape) if np.any(is_un) else np.zeros_like(X)
        Z_re = gaussian_kde(np.vstack([x1[is_re], x2[is_re]]), weights=w[is_re])(grid_points).reshape(X.shape) if np.any(is_re) else np.zeros_like(X)
    except Exception:
        n_bins = 50
        rng = [[x1_lin[0], x1_lin[-1]], [x2_lin[0], x2_lin[-1]]]
        H_un, xe, ye = np.histogram2d(x1[is_un], x2[is_un], bins=[n_bins, n_bins], weights=w[is_un], range=rng)
        H_re, _, _ = np.histogram2d(x1[is_re], x2[is_re], bins=[n_bins, n_bins], weights=w[is_re], range=rng)
        X, Y = np.meshgrid(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]))
        Z_un, Z_re = H_un.T, H_re.T

    Z_un_norm = Z_un / np.max(Z_un) if np.max(Z_un) > 0 else Z_un
    Z_re_norm = Z_re / np.max(Z_re) if np.max(Z_re) > 0 else Z_re

    handles = [
        Patch(facecolor="#034793", edgecolor="#034793", alpha=0.5, label="unreactive density"),
        Patch(facecolor="#C81919", edgecolor="#C81919", alpha=0.5, label="reactive density"),
    ]

    def _joint_figure(Z_un_plot, Z_re_plot, levels, norm=None, log_scale=False):
        """Build the gridspec figure with joint density + marginals."""
        fig = plt.figure(figsize=(6, 6), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_joint = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_joint)

        kw = dict(norm=norm) if norm is not None else {}
        ax_joint.contourf(X, Y, Z_un_plot, levels=levels, cmap="#034793", alpha=0.5, **kw)
        ax_joint.contourf(X, Y, Z_re_plot, levels=levels, cmap="#C81919", alpha=0.5, **kw)
        ax_joint.contour(X, Y, Z_un_plot, levels=levels, colors="#034793", linewidths=0.6, alpha=0.8, **kw)
        ax_joint.contour(X, Y, Z_re_plot, levels=levels, colors="#C81919", linewidths=0.6, alpha=0.8, **kw)
        ax_joint.set_xlabel(short_cv_label(name1), fontsize=9, fontweight="bold")
        ax_joint.set_ylabel(short_cv_label(name2), fontsize=9, fontweight="bold")
        ax_joint.tick_params(axis="both", labelsize=9)
        # ax_joint.grid(True, linestyle="--", alpha=0.3)
        ax_joint.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8)

        # Top marginal
        ax_top.plot(x1_grid, r1_shape, color="#C81919", lw=1.3)
        ax_top.plot(x1_grid, u1_shape, color="#034793", lw=1.3)
        ax_top.set_ylabel("Shape-only PDF", fontsize=9, fontweight="bold")
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.tick_params(axis="y", labelsize=9)
        # ax_top.grid(True, linestyle="--", alpha=0.4)
        if log_scale:
            ax_top.set_yscale("log")

        # Right marginal
        ax_right.plot(r2_shape, x2_grid, color="#C81919", lw=1.3)
        ax_right.plot(u2_shape, x2_grid, color="#034793", lw=1.3)
        ax_right.set_xlabel("Shape-only PDF", fontsize=9, fontweight="bold")
        ax_right.tick_params(axis="x", labelsize=9)
        ax_right.tick_params(axis="y", labelleft=False)
        # ax_right.grid(True, linestyle="--", alpha=0.4)
        if log_scale:
            ax_right.set_xscale("log")

        return fig

    # Linear scale
    levels = np.linspace(0.1, 1.0, 6)
    fig = _joint_figure(Z_un_norm, Z_re_norm, levels)
    fig.savefig(os.path.join(outdir, basename + "_joint_shapes.pdf"), bbox_inches="tight", dpi=300)
    if not show:
        plt.close(fig)

    # Log scale (shape-normalised)
    eps = 1e-3
    Z_un_log = np.clip(Z_un_norm, eps, 1.0)
    Z_re_log = np.clip(Z_re_norm, eps, 1.0)
    levels_log = np.geomspace(eps, 1.0, 6)
    norm_log = LogNorm(vmin=eps, vmax=1.0)

    fig_log = _joint_figure(Z_un_log, Z_re_log, levels_log, norm=norm_log, log_scale=True)
    fig_log.suptitle(f"{title}\nJoint density (log scale)", fontsize=10)
    fig_log.savefig(os.path.join(outdir, basename + "_joint_shapes_log.pdf"), bbox_inches="tight", dpi=300)
    if not show:
        plt.close(fig_log)

    # Raw log scale (unscaled densities)
    Z_un_raw = np.asarray(Z_un, dtype=float)
    Z_re_raw = np.asarray(Z_re, dtype=float)
    pos_all = np.concatenate([Z_un_raw[Z_un_raw > 0], Z_re_raw[Z_re_raw > 0]])

    if pos_all.size > 0:
        vmin = max(np.min(pos_all), tiny)
        vmax = np.max(pos_all)
        levels_raw = np.geomspace(vmin, vmax, 6)
        norm_raw = LogNorm(vmin=vmin, vmax=vmax)
        Z_un_raw_plot = np.clip(Z_un_raw, vmin, vmax)
        Z_re_raw_plot = np.clip(Z_re_raw, vmin, vmax)

        fig_raw = plt.figure(figsize=(6, 6), constrained_layout=True)
        gs_raw = fig_raw.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
        ax_top_r = fig_raw.add_subplot(gs_raw[0, 0])
        ax_joint_r = fig_raw.add_subplot(gs_raw[1, 0])
        ax_right_r = fig_raw.add_subplot(gs_raw[1, 1], sharey=ax_joint_r)

        fig_raw.suptitle(f"{title}\nRAW joint density (log scale)", fontsize=10)
        ax_joint_r.contourf(X, Y, Z_un_raw_plot, levels=levels_raw, cmap="Oranges", alpha=0.5, norm=norm_raw)
        ax_joint_r.contourf(X, Y, Z_re_raw_plot, levels=levels_raw, cmap="Blues", alpha=0.5, norm=norm_raw)
        ax_joint_r.contour(X, Y, Z_un_raw_plot, levels=levels_raw, colors="#034793", linewidths=0.6, alpha=0.8, norm=norm_raw)
        ax_joint_r.contour(X, Y, Z_re_raw_plot, levels=levels_raw, colors="#C81919", linewidths=0.6, alpha=0.8, norm=norm_raw)
        ax_joint_r.set_xlabel(name1, fontsize=9, fontweight="bold")
        ax_joint_r.set_ylabel(name2, fontsize=9, fontweight="bold")
        ax_joint_r.tick_params(axis="both", labelsize=9)
        # ax_joint_r.grid(True, linestyle="--", alpha=0.3)
        ax_joint_r.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8)

        for (vals, x_g, color, label), ax, vert in [
            ((np.clip(r1_sg, tiny, None), x1_grid, "#C81919", r"$r_{\rm raw}(\psi_1)$"), ax_top_r, False),
            ((np.clip(u1_sg, tiny, None), x1_grid, "#034793", r"$u_{\rm raw}(\psi_1)$"), ax_top_r, False),
            ((np.clip(r2_sg, tiny, None), x2_grid, "#C81919", r"$r_{\rm raw}(\psi_2)$"), ax_right_r, True),
            ((np.clip(u2_sg, tiny, None), x2_grid, "#034793", r"$u_{\rm raw}(\psi_2)$"), ax_right_r, True),
        ]:
            if vert:
                ax.plot(vals, x_g, color=color, lw=1.3, label=label)
            else:
                ax.plot(x_g, vals, color=color, lw=1.3, label=label)

        ax_top_r.set_yscale("log")
        ax_top_r.set_ylabel("Raw PDF (log)", fontsize=9, fontweight="bold")
        ax_top_r.tick_params(axis="x", labelbottom=False)
        ax_top_r.tick_params(axis="y", labelsize=9)
        # ax_top_r.grid(True, linestyle="--", alpha=0.4)

        ax_right_r.set_xscale("log")
        ax_right_r.set_xlabel("Raw PDF (log)", fontsize=9, fontweight="bold")
        ax_right_r.tick_params(axis="x", labelsize=9)
        ax_right_r.tick_params(axis="y", labelleft=False)
        # ax_right_r.grid(True, linestyle="--", alpha=0.4)

        fig_raw.savefig(os.path.join(outdir, basename + "_joint_raw_log.pdf"), bbox_inches="tight", dpi=300)
        if not show:
            plt.close(fig_raw)


# ---------------------------------------------------------------------------
# Discrete CV diagnostics
# ---------------------------------------------------------------------------

def make_discrete_diag_plots(
    dbg_disc: dict,
    outdir: Path,
    prefix: str,
    cv_label: str,
    title: str,
    T_continuous: float | None = None,
) -> None:
    """
    Bar-chart diagnostics for a discrete (integer-valued) CV.

    Generates:
      DISCRETE/{prefix}-PMF.pdf     — r_pmf and u_pmf per integer state
      DISCRETE/{prefix}-COMMITTOR.pdf — per-state committor p(reactive|k)

    If `T_continuous` is provided a text box comparing T_discrete vs T_continuous
    is added to the PMF figure.

    Parameters
    ----------
    dbg_disc : dict
        Output from `compute_predictive_capacity_discrete`.
    outdir : Path
        Directory in which to create the DISCRETE/ sub-folder.
    prefix : str
        File-name prefix (e.g. sanitized CV name).
    cv_label : str
        LaTeX-friendly CV label for axis/legend.
    title : str
        Figure suptitle.
    T_continuous : float or None
        T value from the SG pipeline (for side-by-side comparison).
    """
    disc_dir = Path(outdir) / "DISCRETE"
    disc_dir.mkdir(parents=True, exist_ok=True)

    vals = np.asarray(dbg_disc.get("x_vals", []))
    r = np.asarray(dbg_disc.get("r_pmf", []))
    u = np.asarray(dbg_disc.get("u_pmf", []))
    P_PP = np.asarray(dbg_disc.get("P_PP", []))
    T_disc = float(dbg_disc.get("T", 0.0))
    P_A = float(dbg_disc.get("P_A", 0.0))
    S_A = float(dbg_disc.get("S_val", 1.0 - T_disc))

    if vals.size == 0:
        log.warning("make_discrete_diag_plots: empty debug dict for %s", prefix)
        return

    x_ticks = np.arange(len(vals))
    width = 0.4

    # ── 1) PMF bar chart ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(5, len(vals) * 0.5 + 2), 4.5))
    ax.bar(x_ticks - width / 2, r, width=width, color="#C81919", alpha=0.8, label=r"$r_k$ (reactive)")
    ax.bar(x_ticks + width / 2, u, width=width, color="#034793", alpha=0.8, label=r"$u_k$ (unreactive)")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(v)) if np.isclose(v, round(v)) else f"{v:.2g}" for v in vals], fontsize=8)
    ax.set_xlabel(cv_label, fontsize=10, fontweight="bold")
    ax.set_ylabel("PMF (weight / W)", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.8)
    # ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Comparison box
    info_lines = [
        f"$T_{{\\rm disc}}$ = {T_disc:.4f}",
        f"$S_{{\\rm disc}}$ = {S_A:.4f}",
        f"$P_A$ = {P_A:.3e}",
        f"n states = {len(vals)}",
    ]
    if T_continuous is not None:
        info_lines.insert(1, f"$T_{{\\rm SG}}$ = {T_continuous:.4f}   (Δ = {T_disc - T_continuous:+.4f})")
    ax.text(0.98, 0.97, "\n".join(info_lines), transform=ax.transAxes,
            fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))
    fig.suptitle(f"{title}\nDisc. PMF", fontsize=10)
    fig.tight_layout()
    fig.savefig(disc_dir / f"{prefix}-PMF.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── 2) Per-state committor ─────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(max(5, len(vals) * 0.5 + 2), 4.0))
    ax2.bar(x_ticks, P_PP, width=0.6, color="tab:purple", alpha=0.8, label=r"$p_{\rm comm}(k)$")
    ax2.axhline(P_A, color="black", ls="--", lw=1.2, label=f"$P_A$ = {P_A:.3e}")
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([str(int(v)) if np.isclose(v, round(v)) else f"{v:.2g}" for v in vals], fontsize=8)
    ax2.set_xlabel(cv_label, fontsize=10, fontweight="bold")
    ax2.set_ylabel("p(reactive | k)", fontsize=10)
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(fontsize=9, framealpha=0.8)
    # ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig2.suptitle(f"{title}\nPer-state committor", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(disc_dir / f"{prefix}-COMMITTOR.pdf", bbox_inches="tight")
    plt.close(fig2)

    log.info("Discrete plots → %s", disc_dir)

