# PPA Analysis — `profet`

**The Predictive Power Analysis (PPA) pipeline finds which collective variables (CVs)**
**best separate reactive from unreactive paths in your RETIS simulation.**

All settings live in `infretis.toml`. The pipeline is run via `profet` from the repository root:

```bash
cd /path/to/CV_ana
```

---

## Workflow Overview

```
1. Build CVs          →  profet build
                              ↓
                         CVs/<n_grid>/  (h5 files per step)
                         CVs/<n_grid>/sorted/  (aggregated cvmat.h5)

2. Screen CVs         →  profet screen
                              ↓
                         CVs/<n_grid>/sorted/screen.h5

3. T-matrix (opt.)    →  profet tmat
                              ↓
                         CVs/<n_grid>/sorted/T_mat.h5
                         plots/<n_grid>/T_mat/  (heatmaps)

4. Diagnostics        →  profet diagnose
                              ↓
                         plots/<n_grid>/distribution/  (PDF/CDF/overlap)
```

Steps 1 (build) and 2–4 (analysis) are independent.  
You only need to re-run `profet build` if your CV definitions or data change.

---

## Step-by-Step Guide

### 1. Build the CV matrix
```bash
profet build [--overwrite]
```
Reads `infretis.toml` → `[topology]`, `[cv_modules]`, `[cvmat]`, `[ppa]`.  
Writes one `.h5` file per RETIS step and sorts them into `cvmat.h5`.

---

### 2. Screen CVs (`--screen`)

Ranks every CV by its predictive capacity T at the specified (λ_c, λ_r) pairs.  
Results saved to `screen.h5` and can be inspected to find top performers.

```bash
profet screen

# override TOML settings on the fly:
profet screen --lambda-c 6 --lambda-r 26 --n-cvar 1 --workers 8
profet screen --n-cvar 2   # screen linear combos (slower)
profet screen --force      # force rerun even if screen.h5 exists
```

**What is (λ_c, λ_r)?**
- **λ_c** (committed interface) — interface that a path must cross to be committed.
- **λ_r** (reactive interface) — interface used to define reactive vs. unreactive paths.  
  T = ∫|P_reactive(x) − P_unreactive(x)|dx measures how well a CV separates the two.

---

### 3. Build T-matrix (`--tmat`)

Sweeps all (λ_c, λ_r) pairs for the top CVs selected in step 2.  
Produces a 2D heatmap showing where T is highest across interface space.

```bash
profet tmat
```

CVs to analyse are controlled by `cv_mode` in the TOML:
- `cv_mode = "manual"` → uses the `obo_from_toml` / `lin2_from_toml` lists you specify.
- `cv_mode = "n_best"` → auto-reads top CVs from `screen.h5`.

---

### 4. Run diagnostics (`--diagnose`)

Generates detailed PDF/CDF/overlap plots for each selected CV.  
Optionally runs exact discrete PMF analysis for count-type CVs.

```bash
profet diagnose
profet diagnose --log          # also generate log-scale plots
profet diagnose --discrete     # extra bar-chart plots for integer CVs
```

---

### 5. Polish a linear combination (`--optimize`)

If you have a promising linear combination in `lin2_from_toml`, this refines
the α weights with L-BFGS-B optimisation.

```bash
profet optimize
```

---

## TOML Settings Reference (`infretis.toml`)

### `[ppa]` — shared settings

| Key | Description |
|-----|-------------|
| `output_dir` | Root output folder (e.g. `"CVs"`). Results go to `output_dir/<n_grid>/`. |
| `workers` | Number of parallel processes used by screener and CV_builder. |
| `lambda_c` | List of committed-interface grid IDs to screen (e.g. `[6]`). |
| `lambda_r` | List of reactive-interface grid IDs to screen (e.g. `[26]`). |
| `n_best_obo` | How many top one-by-one (OBO) CVs to save in `screen.h5`. Also controls diagnostics output. |
| `n_best_lin` | How many top linear combination CVs to save. |
| `n_try_lin` | How many of the top OBO CVs to try combining (for lin2/lin3 screening). |
| `n_cvar_list` | Which combination sizes to screen: `[1]`=OBO, `[1,2]`=OBO+lin2, etc. |
| `optimizer` | Optimiser for linear combinations: `"dual_annealing"` (default). |

### `[ppa]` — CV selection (used by `--tmat` and `--diagnose`)

| Key | Description |
|-----|-------------|
| `cv_mode` | `"manual"` — use the lists below. `"n_best"` — auto-pick from `screen.h5`. |
| `lambda_c_opt` | Reference λ_c for picking top CVs from `screen.h5` (when `cv_mode = "n_best"`). |
| `lambda_r_opt` | Reference λ_r for picking top CVs from `screen.h5` (when `cv_mode = "n_best"`). |
| `obo_from_toml` | Explicit list of OBO CVs for `cv_mode = "manual"`. Format: `"module::column"`. |
| `lin2_from_toml` | Explicit list of linear combos, e.g. `"cv_a + cv_b"`. |
| `lin3_from_toml` | Three-way combos (rarely needed). |

### `[ppa.tmat]` — T-matrix settings

| Key | Description |
|-----|-------------|
| `lambda_r_max` | Maximum λ_r grid index to sweep in the T-matrix. |
| `n_cvar_list` | Which modes to include in T-matrix: `[1]`=OBO only, `[1,2]`=OBO+lin2. |
| `top_k` | How many CVs per mode to include (if `cv_mode = "n_best"`). |
| `calculate_missing` | If `true`, re-run `CV_builder/main.py` for missing grid points. |

### `[ppa.sg]` — Savitzky-Golay smoothing

Controls how the smoothed PDF/CDF is computed for the T metric.  
Rarely needs changing; defaults work well for most systems.

| Key | Description |
|-----|-------------|
| `sg_grid` | Number of uniform interpolation points (default 2000). |
| `sg_polyorder` | SG polynomial order (default 2). |
| `sg_window_frac` | SG window as fraction of CV range (e.g. `"1/16"`). Wider = smoother. |
| `grid_expand_factor` | Extend the x-axis beyond [min, max] by this factor. |
| `n_hist_bins` | Bins for raw histogram overlay in diagnostic plots. |
| `integer_bins` | If `true`, force raw histogram bins to be centered on integers (edges at half-integers). |
| `discrete_max_unique` | CVs with ≤ N unique integer values are treated as discrete. |

Notes:
- When `integer_bins = true`, histogram bins are `[k-0.5, k+0.5)` so bar centers lie exactly on integer `k`; `n_hist_bins` is ignored for that overlay.

### `[ppa.screening]` — dual-annealing parameters

Only relevant when screening linear combinations (`--screen --n-cvar 2`).

| Key | Description |
|-----|-------------|
| `initial_temp` | Initial temperature for dual-annealing exploration. |
| `maxiter_lin2` | Max iterations for two-CV combinations. |
| `maxiter_lin3` | Max iterations for three-CV combinations (much slower). |

### `[cvmat]` — CV matrix grid

Controls which grid points are stored and processed.

| Key | Description |
|-----|-------------|
| `n_grid` | Total number of order-parameter grid points (sets output folder name). |
| `step_size` | Step size along the grid. |
| `start_grid` / `end_grid` | Grid range for screening. |
| `selection` | Explicit list of grid IDs (used when `grid_mode = "selection"`). Leave empty `[]` to use `start_grid`→`end_grid`. |
| `nskip` | Burn-in: fraction (e.g. `0.1` = drop first 10%) or absolute step count. |

---

## Typical Fast Workflow

```bash
# 1. Build CVs (first time, or after changing CV definitions)
python CV_builder/main.py --overwrite

# 2. Quick screen at one (lc, lr) pair
python ppa/analyze.py --screen

# 3. Check results — top CVs will appear in screen.h5.
#    Set cv_mode = "manual" and fill obo_from_toml with the winners.

# 4. Run diagnostics to inspect those CVs
python ppa/analyze.py --diagnose

# 5. If a linear combo looks interesting, compute T-matrix for it
python ppa/analyze.py --tmat
```

---

## Output Locations

| Output | Location |
|--------|----------|
| Per-step CV data | `CVs/<n_grid>/steps/step_XXXX.h5` |
| Aggregated CVmat | `CVs/<n_grid>/sorted/cvmat.h5` |
| Screen results | `CVs/<n_grid>/sorted/screen.h5` |
| T-matrix | `CVs/<n_grid>/sorted/T_mat.h5` |
| T-matrix plots | `plots/<n_grid>/T_mat/` |
| Diagnostic plots | `plots/<n_grid>/distribution/<lc>/<lr>/obo/` |
| Diagnostic metadata | `plots/<n_grid>/distribution/<lc>/<lr>/obo/metadata.txt` |
