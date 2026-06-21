"""
core.py — PPA mathematical kernel.

Functions
---------
compute_predictive_capacity_from_scalar
    Main entry point: computes T_A and P_A for a scalar CV.
_compute_empirical_cdf
    Helper: weighted empirical CDF from raw path data.
_smooth_and_get_pdfs
    Helper: Savitzky-Golay smoothing → smoothed CDFs + derivative PDFs.
"""
import logging

import numpy as np
from scipy.signal import savgol_filter

log = logging.getLogger(__name__)


def _compute_empirical_cdf(
    x_cls: np.ndarray, w_cls: np.ndarray, W_tot: float
) -> tuple[np.ndarray, np.ndarray]:
    """Sorts data, aggregates weights, and computes unique-value weighted CDF."""
    if x_cls.size == 0 or W_tot <= 0:
        return np.array([]), np.array([])

    sort_idx = np.argsort(x_cls)
    x_sort = x_cls[sort_idx]
    cum_w = np.cumsum(w_cls[sort_idx]) / float(W_tot)

    # Collapse duplicates: keep last cumulative value for each unique x
    np_unique, idx_first = np.unique(x_sort, return_index=True)
    idx_last = np.r_[idx_first[1:] - 1, len(x_sort) - 1]
    cdf_unique = np.clip(cum_w[idx_last], 0.0, 1.0)

    return np_unique, cdf_unique


def _smooth_and_get_pdfs(
    x_grid: np.ndarray,
    dx: float,
    R_int: np.ndarray,
    U_int: np.ndarray,
    sg_polyorder: int,
    sg_window_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Applies Savitzky-Golay to get smoothed CDFs and their derivatives (PDFs)."""
    sg_grid = x_grid.size
    win_len = int(round(sg_grid * sg_window_frac))
    if win_len % 2 == 0:
        win_len += 1
    min_win = sg_polyorder + 3
    if win_len < min_win:
        win_len = min_win
    if win_len > sg_grid - 1:
        win_len = sg_grid - 1
        if win_len % 2 == 0:
            win_len -= 1
    if win_len < 5:
        win_len = 5

    # 0th derivative (Smoothed CDFs)
    R_sg = savgol_filter(R_int, window_length=win_len, polyorder=sg_polyorder, deriv=0, delta=dx, mode="nearest")
    U_sg = savgol_filter(U_int, window_length=win_len, polyorder=sg_polyorder, deriv=0, delta=dx, mode="nearest")

    # Enforce strictly monotonic, non-negative visual CDFs to prevent
    # log-scale polynomial ringing undershoots (Runge's phenomenon visually).
    # This purely affects the CDF plots, it does not touch the math.
#    R_sg = np.maximum.accumulate(np.clip(R_sg, 0.0, None))
#    U_sg = np.maximum.accumulate(np.clip(U_sg, 0.0, None))

    # 1st derivative (PDFs)
    r_sg = savgol_filter(R_sg, window_length=win_len, polyorder=sg_polyorder, deriv=1, delta=dx, mode="nearest")
    u_sg = savgol_filter(U_sg, window_length=win_len, polyorder=sg_polyorder, deriv=1, delta=dx, mode="nearest")
    #r_sg = np.gradient(R_sg, dx)
    #u_sg = np.gradient(U_sg, dx)



    # Clip small negative wiggles (ringing effects)
    r_sg = np.clip(r_sg, 0.0, None)
    u_sg = np.clip(u_sg, 0.0, None)

    return R_sg, U_sg, r_sg, u_sg


def compute_predictive_capacity_from_scalar(
    x,
    w,
    reactive,
    *,
    grid_expand_factor: float = 1.5,
    sg_grid: int = 1200,
    sg_polyorder: int = 2,
    sg_window_frac: float = 1.0 / 25.0,
    n_hist_bins: int = 50,
    integer_bins: bool = False,
) -> tuple[float, float, dict]:
    """
    Compute predictive capacity T_A and crossing probability P_A
    for a scalar CV (per path), following the PPA + Savitzky–Golay
    pipeline. Also return a rich `debug` dict for diagnostics/plotting.

    Parameters
    ----------
    x : array-like, shape (N,)
        Scalar CV values for each path (evaluated at lambda_c crossing).
    w : array-like, shape (N,)
        WHAM / path weights (NOT pre-normalized).
    reactive : array-like of bool, shape (N,)
        True for reactive paths (reach lambda_r before returning to A).
    grid_expand_factor : float
        Grid extends this factor beyond [xmin, xmax].
    sg_grid : int
        Number of uniform interpolation grid points.
    sg_polyorder : int
        Savitzky-Golay polynomial order.
    sg_window_frac : float
        SG window as fraction of sg_grid.
    n_hist_bins : int
        Number of bins for the raw histogram PDF overlay.
    integer_bins : bool
        If True, force histogram bins to be centered on integers, with
        edges at half-integers [k-0.5, k+0.5).

    Returns
    -------
    T_A : float
        Predictive capacity (1 - overlap S_A). In [0, 1].
    P_A : float
        Crossing probability P_A(lambda_r | lambda_c).
    debug : dict
        Diagnostic data for plotting.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    reactive = np.asarray(reactive, dtype=bool)

    debug: dict = {}

    # --- Basic sanity checks ---
    mask_finite = np.isfinite(x) & np.isfinite(w)
    x = x[mask_finite]
    w = w[mask_finite]
    reactive = reactive[mask_finite]

    if x.size == 0 or w.sum() <= 0.0:
        log.debug("compute_pp early-return: empty or zero-weight (x.size=%d, w.sum()=%g)", x.size, w.sum())
        return 0.0, 0.0, debug

    is_re = reactive
    is_un = ~reactive

    w_re, w_un = w[is_re], w[is_un]
    x_re, x_un = x[is_re], x[is_un]

    W = w.sum()
    W_re = w_re.sum()
    W_un = w_un.sum()

    P_A = W_re / W if W > 0.0 else 0.0
    debug["P_A"] = P_A

    if W_re <= 0.0 or W_un <= 0.0:
        log.debug("compute_pp early-return: single-class (W_re=%g, W_un=%g)", W_re, W_un)
        debug["reason"] = "single-class or zero-weight"
        return 0.0, P_A, debug

    # --- Build empirical CDFs ---
    x_re_unique, R_unique = _compute_empirical_cdf(x_re, w_re, W)
    x_un_unique, U_unique = _compute_empirical_cdf(x_un, w_un, W)

    # --- Set up uniform grid with extended range ---
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax <= xmin:
        debug["reason"] = "cv_constant"
        debug["x_grid"] = np.array([xmin])
        debug["dx"] = 1.0
        return P_A, P_A, debug

    center = 0.5 * (xmin + xmax)
    half_range = 0.5 * (xmax - xmin)
    ext_half = half_range * grid_expand_factor
    extxmin, extxmax = center - ext_half, center + ext_half

    if not np.isfinite(extxmin) or not np.isfinite(extxmax) or extxmax <= extxmin:
        extxmin, extxmax = xmin, xmax

    sg_grid = max(int(sg_grid), 32)
    dx = (extxmax - extxmin) / (sg_grid - 1)
    x_grid = extxmin + dx * np.arange(sg_grid)

    # --- Interpolate raw CDFs onto the grid (Right-Continuous Step) ---
    # np.interp creates linear ramps across gaps. For sparse path data, a wide gap
    # between two observed paths will create a constant slope in the CDF, leading
    # to an artificial, non-zero probability density plateau in the PDF.
    # We use a pure step function (right-continuous empirical CDF) instead.
    def _step_interpolate(x_grid, x_unique, cdf_unique):
        if len(x_unique) == 0:
            return np.zeros_like(x_grid)
        idx = np.searchsorted(x_unique, x_grid, side="right") - 1
        res = np.zeros_like(x_grid)
        valid = idx >= 0
        res[valid] = cdf_unique[idx[valid]]
        return res

    R_int = _step_interpolate(x_grid, x_re_unique, R_unique)
    U_int = _step_interpolate(x_grid, x_un_unique, U_unique)

    # Accumulate maximum to ensure rigorous monotonicity despite floating point noise
    R_int = np.maximum.accumulate(R_int)
    U_int = np.maximum.accumulate(U_int)

    debug["x_grid"], debug["dx"] = x_grid, dx
    debug["R_int_raw"], debug["U_int_raw"] = R_int.copy(), U_int.copy()

    # --- Savitzky–Golay smoothing + derivative ---
    R_sg, U_sg, r_sg, u_sg = _smooth_and_get_pdfs(
        x_grid, dx, R_int, U_int, sg_polyorder, sg_window_frac
    )

    # --- Prevent edge extrapolation artifacts ---
    # savgol_filter(mode="interp") extrapolates constant tails via a local polynomial.
    # Force PDFs to exactly 0.0 outside the truly observed domain [xmin, xmax].
    mask_tails = (x_grid < xmin) | (x_grid > xmax)
    r_sg[mask_tails] = 0.0
    u_sg[mask_tails] = 0.0

    t_sg = r_sg + u_sg

    debug["R_int_sg"], debug["U_int_sg"] = R_sg.copy(), U_sg.copy()
    debug["r_sg"], debug["u_sg"] = r_sg.copy(), u_sg.copy()

    # --- Overlap integral S_A and predictive capacity T_A ---
    overlap_integrand = np.zeros_like(t_sg)
    mask_t = t_sg > 0.0
    overlap_integrand[mask_t] = (r_sg[mask_t] * u_sg[mask_t]) / t_sg[mask_t]

    S_A = (1.0 / P_A) * np.sum(overlap_integrand * dx) if P_A > 0.0 else 1.0
    if not np.isfinite(S_A):
        S_A = 1.0

    T_A = max(0.0, min(1.0, 1.0 - S_A))

    debug["S_val"], debug["T"] = S_A, T_A
    debug["overlap_q"] = (overlap_integrand / max(P_A, 1e-300)).copy()

    # --- Histogram-based "raw" PDFs ---
    if integer_bins:
        left_edge = int(np.floor(xmin))
        right_edge = int(np.ceil(xmax))
        if right_edge < left_edge:
            right_edge = left_edge
        bin_centers = np.arange(left_edge, right_edge + 1, dtype=float)
        bin_edges = np.arange(left_edge - 0.5, right_edge + 1.5, 1.0)
    else:
        bin_edges = np.linspace(xmin, xmax, n_hist_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    r_hist_counts, _ = np.histogram(x_re, bins=bin_edges, range=(xmin, xmax), weights=w_re / W)
    u_hist_counts, _ = np.histogram(x_un, bins=bin_edges, range=(xmin, xmax), weights=w_un / W)

    r_hist = r_hist_counts / bin_width
    u_hist = u_hist_counts / bin_width

    debug["hist_bin_centers"] = bin_centers
    debug["r_hist"], debug["u_hist"] = r_hist, u_hist

    # --- Committors: PP (SG-based) and RAW (hist-based) ---
    P_PP = np.zeros_like(x_grid)
    P_PP[mask_t] = r_sg[mask_t] / t_sg[mask_t]
    debug["P_PP"] = P_PP

    denom_hist = r_hist + u_hist
    P_raw = np.zeros_like(bin_centers)
    mask_hist = denom_hist > 0.0
    P_raw[mask_hist] = r_hist[mask_hist] / denom_hist[mask_hist]
    debug["P_raw"] = P_raw

    debug["x_raw"], debug["w_raw"], debug["reactive_raw"] = x.copy(), w.copy(), reactive.copy()

    return T_A, P_A, debug


def ppa_objective(alpha, weight, reactive, feat_mat, optimize=True):
    """
    Objective function for scipy/dual_annealing optimisers.

    Maps alpha coefficients → T_val.  Returns 1 - T_val when `optimize=True`
    (minimisation target), or (T_val, P_val, debug) when `optimize=False`.
    Always uses the SG-based path: linear combination outputs are float, never
    integer-valued, so the discrete path is never appropriate here.
    """
    x = np.dot(feat_mat, alpha)
    T_val, P_val, debug = compute_predictive_capacity_from_scalar(x, weight, reactive)
    if optimize:
        return 1.0 - float(T_val)
    return float(T_val), float(P_val), debug


def compute_predictive_capacity(
    x,
    w,
    reactive,
    *,
    auto_discrete: bool = True,
    max_unique: int = 50,
    # SG kwargs forwarded when path is continuous:
    grid_expand_factor: float = 1.5,
    sg_grid: int = 1200,
    sg_polyorder: int = 2,
    sg_window_frac: float = 1.0 / 25.0,
    n_hist_bins: int = 50,
    integer_bins: bool = False,
) -> tuple[float, float, dict]:
    """
    **Unified dispatcher** — the single entry point for all single-CV
    predictive capacity calculations.

    Routes to the exact discrete PMF path when the data qualifies, and to the
    SG-smoothed continuous path otherwise.

    A CV qualifies as discrete when:
      (a) it has at most `max_unique` distinct values, AND
      (b) all values are (close to) integers.

    The returned ``debug`` dict carries ``debug["mode"] = "discrete"`` when
    the discrete path was taken, so callers can branch their plotting
    accordingly.  When ``auto_discrete=False`` the SG path is always used.

    Parameters
    ----------
    x, w, reactive : array-like
        CV values, path weights, and reactive-flag — same as for the two
        underlying functions.
    auto_discrete : bool
        Enable automatic routing to the discrete path (default: True).
    max_unique : int
        Threshold for discrete detection (passed to `_is_discrete`).
    grid_expand_factor, sg_grid, sg_polyorder, sg_window_frac, n_hist_bins, integer_bins :
        SG-pipeline keyword arguments, forwarded when the continuous path is
        taken (ignored for the discrete path).
    """
    x_arr = np.asarray(x, dtype=float)
    if auto_discrete and _is_discrete(x_arr, max_unique=max_unique):
        return compute_predictive_capacity_discrete(x_arr, w, reactive)
    return compute_predictive_capacity_from_scalar(
        x_arr, w, reactive,
        grid_expand_factor=grid_expand_factor,
        sg_grid=sg_grid,
        sg_polyorder=sg_polyorder,
        sg_window_frac=sg_window_frac,
        n_hist_bins=n_hist_bins,
        integer_bins=integer_bins,
    )




# ---------------------------------------------------------------------------
# Discrete CV path — exact PMF sum (no SG smoothing)
# ---------------------------------------------------------------------------

def _is_discrete(x: np.ndarray, max_unique: int = 50) -> bool:
    """
    Auto-detect whether a CV should be treated as discrete.

    A CV is considered discrete if:
      (a) it has at most `max_unique` distinct values, AND
      (b) all values are (close to) integers.

    Parameters
    ----------
    x : array-like
        Raw CV values (finite, float).
    max_unique : int
        Maximum number of unique values to still call it discrete.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return False
    unique_vals = np.unique(x)
    if len(unique_vals) > max_unique:
        return False
    return bool(np.allclose(unique_vals, np.round(unique_vals), atol=1e-6))


def compute_predictive_capacity_discrete(
    x,
    w,
    reactive,
) -> tuple[float, float, dict]:
    """
    Compute predictive capacity T_A for a **discrete** (integer-valued) CV
    using an exact PMF overlap sum — no SG smoothing, no grid artefacts.

    For a discrete CV taking values {k}, the PMFs are:

        r_k = Pr(reactive, X=k) / W        (normalised by total weight)
        u_k = Pr(unreactive, X=k) / W

    The overlap is:

        S_A = (1/P_A) * Σ_k  r_k * u_k / (r_k + u_k)
        T_A = 1 - S_A

    Parameters
    ----------
    x : array-like, shape (N,)
        Integer-valued CV (may be stored as float).
    w : array-like, shape (N,)
        WHAM path weights (not pre-normalised).
    reactive : array-like of bool, shape (N_)
        True for reactive paths.

    Returns
    -------
    T_A : float
    P_A : float
    debug : dict
        Keys: x_vals, r_pmf, u_pmf, P_PP (per-state committor),
        P_A, T, S_val, mode="discrete".
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    reactive = np.asarray(reactive, dtype=bool)

    mask = np.isfinite(x) & np.isfinite(w)
    x, w, reactive = x[mask], w[mask], reactive[mask]

    debug: dict = {"mode": "discrete"}

    W = w.sum()
    if W <= 0 or x.size == 0:
        return 0.0, 0.0, debug

    vals, inv = np.unique(x, return_inverse=True)

    w_re = w * reactive
    w_un = w * (~reactive)

    # PMFs normalised by total weight W (so r + u sums to 1)
    r = np.bincount(inv, weights=w_re, minlength=len(vals)) / W
    u = np.bincount(inv, weights=w_un, minlength=len(vals)) / W

    P_A = float(r.sum())
    debug["P_A"] = P_A

    if P_A <= 0.0 or (1.0 - P_A) <= 0.0:
        debug["reason"] = "single-class"
        return 0.0, P_A, debug

    denom = r + u
    active = denom > 0.0

    S_A = (1.0 / P_A) * float(np.sum((r[active] * u[active]) / denom[active]))
    if not np.isfinite(S_A):
        S_A = 1.0

    T_A = max(0.0, min(1.0, 1.0 - S_A))

    # Per-state committor p(reactive | X=k)
    P_PP = np.zeros(len(vals))
    P_PP[active] = r[active] / denom[active]

    debug.update({
        "x_vals": vals,
        "r_pmf": r,
        "u_pmf": u,
        "P_PP": P_PP,
        "S_val": S_A,
        "T": T_A,
    })

    log.debug("discrete T_A=%.4f, P_A=%.4e, n_states=%d", T_A, P_A, len(vals))
    return T_A, P_A, debug
