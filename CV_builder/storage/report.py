"""
storage/report.py — Human-readable summary of CVmat.h5.

Generates a markdown document next to CVmat.h5 describing:
  - File structure (root datasets, groups, modules)
  - Column names per module
  - Per-CV statistics: min, mean, max across all RETIS steps
"""
from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

log = logging.getLogger(__name__)


def _decode(b) -> str:
    return b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else str(b)


def write_cvmat_report(cvmat_path: Path) -> Path:
    """
    Read CVmat.h5 and write a human-readable markdown report alongside it.

    Returns the path of the written report.
    """
    report_path = cvmat_path.with_name("cvmat_report.md")

    with h5py.File(cvmat_path, "r", locking=False) as f:
        grid_ids = sorted(k for k in f.keys() if str(k).isdigit())
        has_lw   = "lambda_and_weight" in f
        n_steps  = 0

        # ── lambda_and_weight summary ──────────────────────────────────────
        lw_summary = ""
        if has_lw:
            lw = f["lambda_and_weight"][()].astype(float)
            n_steps = lw.shape[0]
            fin_lam = lw[np.isfinite(lw[:, 1]), 1]
            fin_w   = lw[np.isfinite(lw[:, 2]), 2]
            lw_summary = (
                f"| n_steps | RETIS_step range | λ_max range | weight range |\n"
                f"|---|---|---|---|\n"
                f"| {n_steps} "
                f"| [{int(lw[:,0].min())} – {int(lw[:,0].max())}] "
                f"| [{fin_lam.min():.4f} – {fin_lam.max():.4f}] "
                f"| [{fin_w.min():.6f} – {fin_w.max():.6f}] |\n"
            )

        # ── Sanitization summary ──────────────────────────────────────────
        san_summary = ""
        if "sanitization" in f:
            # Collect: (module, reason, cv_name) -> set of grid IDs
            from collections import defaultdict
            drop_map: dict[tuple[str, str, str], set[str]] = defaultdict(set)
            all_san_gids: set[str] = set()
            
            for gid, ggrp in f["sanitization"].items():
                all_san_gids.add(str(gid))
                for mod, mgrp in ggrp.items():
                    for reason, ds in mgrp.items():
                        dropped = ds[()]
                        for c in dropped:
                            c_str = _decode(c)
                            drop_map[(str(mod), str(reason), c_str)].add(str(gid))
            
            if drop_map:
                n_total_gids = len(all_san_gids)
                
                # Split into consistent (dropped everywhere) vs irregular
                consistent = []
                irregular = []
                for (mod, reason, cv), gids in sorted(drop_map.items()):
                    if len(gids) == n_total_gids:
                        consistent.append((mod, reason, cv))
                    else:
                        irregular.append((mod, reason, cv, gids))
                
                san_lines = [
                    "## Sanitization\n\n",
                    f"Columns removed across all {n_total_gids} grid points:\n\n",
                    "| Module | Reason | CV name |\n",
                    "|---|---|---|\n",
                ]
                for mod, reason, cv in consistent:
                    san_lines.append(f"| `{mod}` | `{reason}` | `{cv}` |\n")
                
                if irregular:
                    san_lines.append(f"\n### Irregularities\n\n")
                    san_lines.append(
                        "The following CVs were **not** consistently removed across all grid points:\n\n"
                    )
                    san_lines.append("| Module | Reason | CV name | Removed at grid IDs |\n")
                    san_lines.append("|---|---|---|---|\n")
                    for mod, reason, cv, gids in irregular:
                        gid_str = ", ".join(sorted(gids, key=int))
                        san_lines.append(
                            f"| `{mod}` | `{reason}` | `{cv}` | {gid_str} |\n"
                        )
                
                san_summary = "".join(san_lines) + "\n---\n\n"
            else:
                san_summary = "## Sanitization\n\nNo columns were dropped.\n\n---\n\n"

        # ── Reactive masks summary ────────────────────────────────────────
        mask_summary = ""
        if "reactive_masks" in f and "wham_grid" in f and has_lw:
            wham_grid = f["wham_grid"][()]
            masks = f["reactive_masks"][()]  # shape (N, n_grid)
            n_tot = masks.shape[0]
            mask_lines = [
                "## Reactive masks\n\n",
                f"WHAM grid: {len(wham_grid)} points, λ ∈ [{wham_grid[0]:.3f}, {wham_grid[-1]:.3f}] Å\n\n",
                "Reactive fraction (sampled indices):\n\n",
                "| lambda_r index | λ value (Å) | n_reactive / N | fraction |\n",
                "|---|---|---|---|\n"
            ]
            indices = np.linspace(0, len(wham_grid)-1, 10, dtype=int)
            for j in indices:
                lam_val = wham_grid[j]
                n_react = np.sum(masks[:, j])
                frac = n_react / n_tot if n_tot > 0 else 0.0
                mask_lines.append(f"| {j} | {lam_val:.3f} | {n_react} / {n_tot} | {frac:.3f} |\n")
            
            mask_summary = "".join(mask_lines) + "\n---\n\n"

        # ── Per-grid module data ───────────────────────────────────────────
        # Show one representative grid in detail, then a compact summary
        grid_sections: list[str] = []
        ref_gid = grid_ids[0] if grid_ids else None

        if ref_gid is not None:
            mods = sorted(f[ref_gid].keys())
            n_mods = len(mods)

            # Module overview table
            mod_rows = []
            for mod in mods:
                g = f[f"{ref_gid}/{mod}"]
                if "cv" not in g:
                    continue
                shape = g["cv"].shape
                n_feat = shape[1] - 1
                cols = ([_decode(c) for c in g["cols"][()]] if "cols" in g else [])
                feat_cols = [c for c in cols if not c.startswith("RETIS")]
                preview = ", ".join(feat_cols[:4])
                if len(feat_cols) > 4:
                    preview += f" … (+{len(feat_cols)-4} more)"
                mod_rows.append(
                    f"| `{mod}` | {shape[0]} × {shape[1]} | {n_feat} | {preview} |"
                )

            mod_table = (
                "| Module | Shape | n_features | Columns (preview) |\n"
                "|---|---|---|---|\n"
                + "\n".join(mod_rows)
            )

            # Per-CV stats table (from the reference grid)
            stat_rows: list[str] = []
            for mod in mods:
                g = f[f"{ref_gid}/{mod}"]
                if "cv" not in g or "cols" not in g:
                    continue
                cv   = g["cv"][()].astype(float)
                cols = [_decode(c) for c in g["cols"][()]]
                for i, col in enumerate(cols):
                    if col == "RETIS_step":
                        continue
                    data = cv[:, i]
                    fin  = data[np.isfinite(data)]
                    if fin.size == 0:
                        stat_rows.append(
                            f"| `{mod}` | `{col}` | n/a | n/a | n/a |"
                        )
                    else:
                        stat_rows.append(
                            f"| `{mod}` | `{col}` "
                            f"| {fin.min():.4f} | {fin.mean():.4f} | {fin.max():.4f} |"
                        )

            stat_table = (
                "| Module | Column | min | mean | max |\n"
                "|---|---|---|---|---|\n"
                + "\n".join(stat_rows)
            )

            grid_sections.append(
                f"## CV Modules (reference grid `{ref_gid}`, {n_mods} modules, {n_steps} steps)\n\n"
                f"### Module overview\n\n{mod_table}\n\n"
                f"### Per-CV statistics\n\n{stat_table}\n"
            )

            # Compact summary for remaining grids
            if len(grid_ids) > 1:
                summary_rows = []
                for gid in grid_ids:
                    gid_mods = sorted(f[gid].keys())
                    n_m = len(gid_mods)
                    # Get row count from first module
                    n_rows = "?"
                    for m in gid_mods:
                        if "cv" in f[f"{gid}/{m}"]:
                            n_rows = str(f[f"{gid}/{m}/cv"].shape[0])
                            break
                    summary_rows.append(f"| `{gid}` | {n_m} | {n_rows} |")
                
                grid_summary = (
                    "## All grid points summary\n\n"
                    "| Grid ID | n_modules | n_steps |\n"
                    "|---|---|---|\n"
                    + "\n".join(summary_rows)
                )
                grid_sections.append(grid_summary + "\n")

        # ── order_txt summary ──────────────────────────────────────────────
        order_summary = ""
        if "order_txt" in f:
            order_grp = f["order_txt"]
            n_order = len(order_grp)
            # Get shape from first entry
            sample_shape = ""
            for k in list(order_grp.keys())[:1]:
                sample_shape = f", shape per step: {order_grp[k].shape}"
            order_summary = (
                f"## `order_txt`\n\n"
                f"**{n_order}** RETIS steps stored{sample_shape}\n\n"
                f"Access pattern: `h5f[\"order_txt/<step_id>\"]` → 2D array "
                f"(columns: Time, Orderp, n_dissociation, atom1, atom2, atom3)\n\n"
            )

    # ── Assemble report ────────────────────────────────────────────────────
    lines = [
        f"# CVmat.h5 — Data Summary\n",
        f"**File:** `{cvmat_path}`  \n",
        f"**RETIS steps:** {n_steps}  \n",
        f"**Grid IDs:** {', '.join(f'`{g}`' for g in grid_ids)}  \n",
        f"\n---\n",
        f"## Root structure\n",
        f"```\n"
        f"CVmat.h5\n"
        f"├── lambda_and_weight          # dataset: (n_steps, 3)  [RETIS_step, λ_max, weight]\n"
        f"├── wham_grid                  # dataset: (n_grid,)     physical λ values\n"
        f"├── reactive_masks             # dataset: (n_steps, n_grid)  bool array: λ_max > wham_grid\n"
        f"├── order_txt/                 # group: one dataset per RETIS step\n"
        f"│   └── <step_id>             # dataset: (n_frames, 6)  raw order.txt data\n"
        f"└── <grid_id>/\n"
        f"    └── <module>/\n"
        f"        ├── cv                 # 2D float32 matrix: (n_steps, 1 + n_features)\n"
        f"        └── cols               # string array: [\"RETIS_step\", \"mod::feat\", ...]\n"
        f"```\n",
        f"## `lambda_and_weight`\n\n{lw_summary}\n",
        f"---\n\n",
        san_summary,
        mask_summary,
        order_summary,
    ]
    lines.extend(s + "\n---\n\n" for s in grid_sections)

    report_path.write_text("".join(lines), encoding="utf-8")
    log.info("CVmat report written → %s", report_path)
    return report_path
