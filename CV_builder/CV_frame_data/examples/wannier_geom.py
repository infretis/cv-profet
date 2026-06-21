# CV_manager/wannier_geom.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import math
import re
import glob
from CV_frame_data.math_utils import mic_delta

# ---------- PBC helpers ----------
def pbc_delta(d: np.ndarray, L) -> np.ndarray:
    return mic_delta(d, L)

def mic_dist2(a: np.ndarray, b: np.ndarray, L: Optional[float]) -> np.ndarray:
    d = pbc_delta(b - a, L)
    return np.einsum("...i,...i->...", d, d)

# ---------- file finder (your strict naming) ----------
def find_homo_centers_file(traj_path: Path, ase_idx: int) -> Optional[Path]:
    """
    Given .../accepted/<prefix>.xyz and ase_idx, try to find:
    <prefix>-HOMO_centers_s1-<ase_idx>.data
    <prefix>-HOMO_centers_s1-<ase_idx>_0.data
    and, if not found, glob a bit.
    """
    parent = traj_path.parent
    base   = traj_path.stem  # e.g. "044_814741_238_trajB"
    homo_path = parent / f"{base}-HOMO_centers_s1-1_{ase_idx}.data"
    if homo_path.exists():
        return homo_path

    return None

def load_centers_xyz(path: Path) -> np.ndarray:
    """
    Parse center coordinates from file; very lenient:
    grabs the first 3 floats from lines after a short header.
    Returns (M,3) in Å.
    """
    coords: List[List[float]] = []
    with path.open("r") as f:
        lines = f.readlines()

    # skip header until we hit a line with >=3 floats
    start = 0
    for i in range(min(4, len(lines))):
        floats = re.findall(r"[-+]?\d*\.\d+|\d+", lines[i])
        if len(floats) >= 3:
            start = i
            break
    for line in lines[start:]:
        floats = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        if len(floats) < 3:
            continue
        x, y, z = map(float, floats[:3])
        coords.append([x, y, z])

    if not coords:
        raise ValueError(f"No center coordinates parsed from {path}")
    return np.array(coords, dtype=float)

@dataclass(slots=True)
class WannierGeomCV:
    """
    Emits 8 Wannier-based features per frame, focused on the proton-transfer context:
        donor_center_r_mean, donor_center_r_std,
        donor_proj_max_along_OHstar, donor_proj_min_along_OHstar,
        accept_center_r_mean, accept_center_r_std,
        accept_proj_max_along_OaHstar, accept_proj_min_along_OaHstar
    """
    name: str = "wannier_geom"
    labels: np.ndarray = field(default_factory=lambda: np.array([
        "donor_center_r_mean",
        "donor_center_r_std",
        "donor_proj_max_along_OHstar",
        "donor_proj_min_along_OHstar",
        "accept_center_r_mean",
        "accept_center_r_std",
        "accept_proj_max_along_OaHstar",
        "accept_proj_min_along_OaHstar",
    ], dtype=object))

    # static topology in reference order (must be provided when constructing the CV):
    oxygen_indices: Optional[np.ndarray] = None   # (nO,) absolute row indices of O atoms
    ho_pairs: Optional[np.ndarray] = None         # (2*nO,2) pairs (O,H) in ref order

    # behavior:
    require_exact_four: bool = False

    provides_triplet_meta: bool = True
    last_triplet: tuple[int,int,int] | None = None
    
    def compute(self, inputs) -> np.ndarray:
        coords    = inputs.coords
        L         = inputs.box
        key       = inputs.key
        traj_path = getattr(inputs, "traj_path", None)
        ase_idx   = int(getattr(key, "ase_idx", -1)) if key is not None else -1

        if traj_path is None:
            raise ValueError("WannierGeomCV.compute: inputs.traj_path is required")
        if ase_idx < 0:
            return np.full((self.labels.shape[0],), np.nan, dtype=float)
        
        rc = inputs.data.get("reaction")
        assert rc is not None, "ReactionCenter missing in CVInputs.data['reaction']"
        O_d, Hs, O_a = rc.O_d, rc.Hs, rc.O_a
        e_d, e_a     = rc.e_d, rc.e_a
        
        self.last_triplet = (int(O_d), int(Hs), int(O_a))


        u_d = pbc_delta(coords[Hs] - coords[O_d], L); nd = np.linalg.norm(u_d) or 1.0; e_d = u_d / nd
        u_a = pbc_delta(coords[Hs] - coords[O_a], L); na = np.linalg.norm(u_a) or 1.0; e_a = u_a / na
        
        O_all = self.oxygen_indices


        # ---- load centers for this frame ----
        cpath = find_homo_centers_file(traj_path, ase_idx)
        if cpath is None or not cpath.exists():
            return np.full((self.labels.shape[0],), np.nan, dtype=float)
        C = load_centers_xyz(cpath)  # (M,3)

        # ---- assign centers to nearest oxygen (MIC) ----
        O_pos = coords[O_all]                                    # (nO,3)
        d2 = mic_dist2(C[:, None, :], O_pos[None, :, :], L)      # (M,nO)
        nearest_O_idx = np.argmin(d2, axis=1)                    # (M,)

        # index of O_d and O_a inside the O_all array
        try:
            k_d = int(np.where(O_all == O_d)[0][0])
            k_a = int(np.where(O_all == O_a)[0][0])
        except IndexError:
            return np.full((self.labels.shape[0],), np.nan, dtype=float)

        C_d = C[nearest_O_idx == k_d]
        C_a = C[nearest_O_idx == k_a]

        def pick_four(centers: np.ndarray, O_xyz: np.ndarray) -> np.ndarray:
            if centers.shape[0] == 4:
                return centers
            if self.require_exact_four:
                return np.full((4,3), np.nan, dtype=float)
            if centers.shape[0] == 0:
                return np.full((4,3), np.nan, dtype=float)
            d2loc = mic_dist2(np.asarray(centers), np.broadcast_to(O_xyz, centers.shape), L)
            order = np.argsort(d2loc)[:4]
            sel = centers[order]
            if sel.shape[0] < 4:
                pad = np.full((4 - sel.shape[0], 3), np.nan, dtype=float)
                sel = np.vstack([sel, pad])
            return sel

        C_d4 = pick_four(C_d, coords[O_d])
        C_a4 = pick_four(C_a, coords[O_a])

        def center_feats(C4: np.ndarray, O_xyz: np.ndarray, edir: np.ndarray) -> Tuple[float,float,float,float]:
            ok = ~np.isnan(C4).any(axis=1)
            if not ok.any():
                return (np.nan, np.nan, np.nan, np.nan)
            V = pbc_delta(C4[ok] - O_xyz, L)
            r = np.linalg.norm(V, axis=1)
            proj = V @ edir
            return (float(np.mean(r)), float(np.std(r)), float(np.max(proj)), float(np.min(proj)))

        don_mean, don_std, don_pmax, don_pmin = center_feats(C_d4, coords[O_d], e_d)
        acc_mean, acc_std, acc_pmax, acc_pmin = center_feats(C_a4, coords[O_a], e_a)

        return np.array([
            don_mean, don_std, don_pmax, don_pmin,
            acc_mean, acc_std, acc_pmax, acc_pmin
        ], dtype=float)
