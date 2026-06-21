"""
cv_utils.py — CV name helpers used across analysis tools.

All string utilities for parsing, shortening, sanitising, and decoding
CV names from HDF5 datasets live here.
"""
from __future__ import annotations

import re

import numpy as np


# ---------------------------------------------------------------------------
# Decoding (HDF5 bytes → Python str)
# ---------------------------------------------------------------------------

def decode_cv_item(item) -> str:
    """Decode a single HDF5 cv_names entry (bytes, ndarray, or str) → str."""
    if isinstance(item, np.ndarray):
        return " + ".join(
            c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in item
        )
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


# ---------------------------------------------------------------------------
# Name stripping
# ---------------------------------------------------------------------------

def base_cv_name(full: str) -> str:
    """
    Strip module prefix and group suffix from a full CV name.

    'group::name@dataset'  →  'name'
    'IonO::IonO6@dataset'  →  'IonO6'
    'zundel_coordinate::R_OdOa' →  'R_OdOa'
    """
    name = full
    if "::" in name:
        name = name.split("::", 1)[1]
    if "@" in name:
        name = name.split("@", 1)[0]
    return name


def short_name(full_cv_name: str) -> str:
    """
    Shorten a full CV name for use in filenames / axes labels (plain text).

    Strips @group suffix and module:: prefix.
    """
    name = full_cv_name
    if "@" in name:
        name = name.split("@")[0]
    if "::" in name:
        name = name.split("::", 1)[1]
    return name


def sanitize_filename(s: str) -> str:
    """Make a string safe for filenames (keep letters, digits, _, ., -)."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_.")


# ---------------------------------------------------------------------------
# LaTeX labels (used by plots)
# ---------------------------------------------------------------------------

def short_cv_label(full_name: str) -> str:
    """
    Return a short, pretty LaTeX legend label for a CV name.

    Falls back to `base_cv_name(full_name)` for unknown patterns.
    """
    name = base_cv_name(full_name)

    # Ion distances: IonOXX → $\mathrm{O}_{XX}$
    m = re.match(r"IonO(\d+)$", name)
    if m:
        return rf"$\mathrm{{O}}_{{{m.group(1)}}}$"

    # Solvation shell bins: S_r_binNN
    m = re.match(r"S_r_bin(\d+)$", name)
    if m:
        return rf"$s_{{{int(m.group(1))}}}^\mathrm{{shell}}$"

    # Zundel / proton-transfer CVs
    zundel_map = {
        "N_elec_a":  r"$N_\mathrm{elec}^{(a)}$",
        "N_elec_d":  r"$N_\mathrm{elec}^{(d)}$",
        "asym_N":    r"$\Delta N_\mathrm{elec}$",
        "mean_s":    r"$\langle s \rangle$",
        "R_OaH":     r"$\mathrm{R}_{\mathrm{O_aH_s}}$",
        "R_OdH":     r"$\mathrm{R}_{\mathrm{O_dH_s}}$",
        "R_OdOa":    r"$\mathrm{R}_{\mathrm{O_dO_a}}$",
        "delta":     r"$\delta$",
    }
    if name in zundel_map:
        return zundel_map[name]

    # Multi-proton wire features
    m = re.match(r"Sigma_R_OO_L(\d+)$", name)
    if m:
        return rf"$w_{{{m.group(1)}}}$"

    m = re.match(r"R_OO_first_L(\d+)$", name)
    if m:
        return rf"$w_{{{m.group(1)}}}^{{(1)}}$"

    m = re.match(r"q_cos_L(\d+)$", name)
    if m:
        return rf"$w_{{{m.group(1)}}}^{{(\theta)}}$"

    m = re.match(r"sigma_OO_L(\d+)$", name)
    if m:
        return rf"$w_{{{m.group(1)}}}^{{(\sigma)}}$"

    if name == "Sigma_delta_L4":
        return r"$\Sigma\,\delta^{(4)}$"

    # Wannier RC features
    wannier_map = {
        "accept_center_r_mean":    r"$\langle r_\mathrm{W}^{(a)} \rangle$",
        "accept_center_r_std":     r"$\sigma(r_\mathrm{W}^{(a)})$",
        "donor_center_r_mean":     r"$\langle r_\mathrm{W}^{(d)} \rangle$",
        "donor_center_r_std":      r"$\sigma(r_\mathrm{W}^{(d)})$",
        "donor_proj_max_along_OHstar": r"$p_{\max}^{(d)}$",
        "donor_proj_min_along_OHstar": r"$p_{\min}^{(d)}$",
    }
    if name in wannier_map:
        return wannier_map[name]

    # Local density / HB / tetrahedrality
    misc_map = {
        "N_lp_like":           r"$N_\mathrm{lp}^{(a)}$",
        "lp_mean_cos_axis":    r"$\langle \cos\theta_\mathrm{lp}^{(a)} \rangle$",
        "N_O_within_R":        r"$N_\mathrm{O}^{(R)}$",
        "HB_strength":         r"$S_\mathrm{HB}$",
        "q_tetra":             r"$q_\mathrm{tetra}$",
        "min_N_elec_second":   r"$N_{\min,\mathrm{HOMO}}^{(2)}$",
        "N1_elec":             r"$N_\mathrm{HOMO}^{(1,a)}$",
        "N2_elec":             r"$N_\mathrm{HOMO}^{(2)}$",
        "D1_over_D3":          r"$D_1/D_3$",
        "D1_over_D2":          r"$D_1/D_2$",
        "E_Od_NN":             r"$E_{\parallel}^{(\mathrm{d}\to\mathrm{NN})}$",
        "E_Oa_NN":             r"$E_{\parallel}^{(\mathrm{a}\to\mathrm{NN})}$",
        "E_OdOa":              r"$E_{\parallel}^{(\mathrm{d}\to\mathrm{a})}$",
        "rho_like":            r"$\rho_{\mathrm{H}^*}^{(\mathrm{HOMO})}$",
        "E_parallel_like":     r"$E_{\parallel}^{(\mathrm{HOMO})}$",
        "F_Hstar_parallel":    r"$F_{\parallel}^{(\mathrm{H}^*)}$",
        "don_accept_ratio":    r"$N_\mathrm{don}/N_\mathrm{acc}$",
    }
    if name in misc_map:
        return misc_map[name]

    return name
