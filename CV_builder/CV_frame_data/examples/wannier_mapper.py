"""
wannier_mapper.py — Core logic for mapping Wannier Centers to molecules.

Provides robust assignment of Wannier centers (WCs) to heavy atoms (O, Cl) 
under periodic boundary conditions, including repair logic to guarantee
exactly 4 WCs per heavy atom. Also provides utilities to compute
molecule-local unwrapped coordinates for stable CV features.
"""

import numpy as np
from CV_frame_data.math_utils import mic_delta

def pbc_delta(d: np.ndarray, L) -> np.ndarray:
    """Minimum image convention (MIC) displacement vector."""
    return mic_delta(d, L)


def mic_dist2_matrix(A: np.ndarray, B: np.ndarray, L) -> np.ndarray:
    """
    Compute pairwise squared distances between N points in A and M points in B.
    A: (N, 3), B: (M, 3)
    Returns: (N, M) matrix of squared distances.
    """
    # A[:, None, :] -> (N, 1, 3)
    # B[None, :, :] -> (1, M, 3)
    d = A[:, None, :] - B[None, :, :]
    d = pbc_delta(d, L)
    return np.sum(d**2, axis=-1)


def assign_wcs_to_heavy_atoms(
    coords: np.ndarray, 
    heavy_indices: np.ndarray, 
    centers: np.ndarray, 
    L
) -> dict[int, list[int]]:
    """
    Raw assignment of each WC to the nearest heavy atom.
    
    Args:
        coords: (N_atoms, 3) all atom coordinates.
        heavy_indices: (N_heavy,) absolute indices of O/Cl atoms.
        centers: (N_wc, 3) Wannier center coordinates.
        L: Box dimension for PBC.
        
    Returns:
        Mapping from heavy atom absolute index -> list of WC indices assigned to it.
    """
    heavy_coords = coords[heavy_indices]
    
    # D2 shape: (N_wc, N_heavy)
    D2 = mic_dist2_matrix(centers, heavy_coords, L)
    
    # nearest_idx shape: (N_wc,) containing indices 0...N_heavy-1
    nearest_idx = np.argmin(D2, axis=1)
    
    wc_groups: dict[int, list[int]] = {h: [] for h in heavy_indices}
    
    for wc_i, h_local_idx in enumerate(nearest_idx):
        h_abs = heavy_indices[h_local_idx]
        wc_groups[h_abs].append(wc_i)
        
    return wc_groups


def repair_wc_assignments(
    wc_groups: dict[int, list[int]], 
    coords: np.ndarray, 
    centers: np.ndarray, 
    L
) -> dict[int, list[int]]:
    """
    Ensures exactly 4 WCs per heavy atom by moving WCs from donors (>4) to receivers (<4).
    
    Args:
        wc_groups: The raw assignment map from assign_wcs_to_heavy_atoms.
        coords: (N_atoms, 3) atom coordinates.
        centers: (N_wc, 3) Wannier center coordinates.
        L: Box dimension for PBC.
        
    Returns:
        A new mapping dict with exactly 4 WCs per heavy atom.
    """
    # Deep copy the lists to avoid mutating the original
    groups = {k: list(v) for k, v in wc_groups.items()}
    
    # Iterative repair
    while True:
        receivers = [h for h, wcs in groups.items() if len(wcs) < 4]
        donors = [h for h, wcs in groups.items() if len(wcs) > 4]
        
        if not receivers or not donors:
            break
            
        # Pick the first receiver to fix
        r = receivers[0]
        r_coord = coords[r]
        
        # Collect all candidate WCs currently owned by *any* donor
        candidate_wcs = []
        for d in donors:
            candidate_wcs.extend(groups[d])
            
        if not candidate_wcs:
            break
            
        # Find the candidate WC physically closest to the receiver r
        cand_coords = centers[candidate_wcs]
        
        # D2 shape: (N_candidates, 1) -> (N_candidates,)
        D2 = mic_dist2_matrix(cand_coords, r_coord[None, :], L).ravel()
        best_cand_local_idx = np.argmin(D2)
        best_wc = candidate_wcs[best_cand_local_idx]
        
        # Move best_wc from its current donor to the receiver r
        for d in donors:
            if best_wc in groups[d]:
                groups[d].remove(best_wc)
                break
                
        groups[r].append(best_wc)
        
    return groups


def classify_water_wcs(
    O_idx: int,
    H_indices: list[int],
    wc_indices: list[int],
    coords: np.ndarray,
    centers: np.ndarray,
    L
) -> tuple[list[int], list[int]]:
    """
    Classifies the 4 WCs of a water molecule into 2 lone-pair WCs and 2 bonding WCs.
    
    Returns:
        (lp_wcs, bond_wcs): Two lists of WC absolute indices.
    """
    if len(wc_indices) != 4 or len(H_indices) != 2:
        return [], []
        
    O_pos = coords[O_idx]
    v1 = pbc_delta(coords[H_indices[0]] - O_pos, L)
    v2 = pbc_delta(coords[H_indices[1]] - O_pos, L)
    v1_hat = v1 / (np.linalg.norm(v1) + 1e-12)
    v2_hat = v2 / (np.linalg.norm(v2) + 1e-12)
    
    wcs_pos = centers[wc_indices]
    u = pbc_delta(wcs_pos - O_pos, L)
    u_hat = u / (np.linalg.norm(u, axis=1)[:, None] + 1e-12)
    
    cos1 = u_hat @ v1_hat
    cos2 = u_hat @ v2_hat
    # How well does each WC align with its closest O-H bond?
    max_cos = np.maximum(cos1, cos2)
    
    # Sort by alignment: the two lowest are lone pairs (pointing away from H bonds)
    # The two highest are the bond WCs
    sorted_order = np.argsort(max_cos)
    lp_wcs = [wc_indices[i] for i in sorted_order[:2]]
    bond_wcs = [wc_indices[i] for i in sorted_order[2:]]
    
    return lp_wcs, bond_wcs


class MoleculeLocalFrame:
    """
    Utility to compute properties in a molecule-local, unwrapped frame.
    Prevents PBC artifacts when computing dipole proxies, centroids, etc.
    """
    def __init__(
        self, 
        anchor_idx: int, 
        coords: np.ndarray, 
        L
    ):
        """
        Anchor is usually the Heavy atom (O or Cl).
        """
        self.anchor_idx = anchor_idx
        self.anchor_pos = coords[anchor_idx]
        self.L = L

    def unwrap_atom(self, atom_idx: int, coords: np.ndarray) -> np.ndarray:
        """Returns relative vector from anchor to atom_idx."""
        return pbc_delta(coords[atom_idx] - self.anchor_pos, self.L)

    def unwrap_point(self, point_pos: np.ndarray) -> np.ndarray:
        """Returns relative vector from anchor to an arbitrary 3D point (e.g. WC)."""
        return pbc_delta(point_pos - self.anchor_pos, self.L)

    def compute_electronic_centroid(self, wc_indices: list[int], centers: np.ndarray) -> np.ndarray:
        """Mean relative position of the 4 WCs."""
        if not wc_indices:
            return np.zeros(3)
        wc_pos = centers[wc_indices]
        rel_pos = pbc_delta(wc_pos - self.anchor_pos, self.L)
        return np.mean(rel_pos, axis=0)

    def compute_dipole_proxy(
        self, 
        h_indices: list[int], 
        wc_indices: list[int], 
        coords: np.ndarray, 
        centers: np.ndarray
    ) -> np.ndarray:
        """
        Computes mu proxy ≈ sum_i(Z_i * r_i) - 2 * sum_wc(r_wc)
        in the anchor-local frame.
        Z for Oxygen = 6 (valence)
        Z for Hydrogen = 1
        """
        # Anchor (Oxygen) is at 0,0,0 in the local frame, so its Z*r term is 0.
        
        # H nuclei contribution (Z=1)
        mu_nuc = np.zeros(3)
        for h in h_indices:
            r_h = self.unwrap_atom(h, coords)
            mu_nuc += 1.0 * r_h
            
        # WC electrons (e=2 per WC center)
        mu_elec = np.zeros(3)
        if wc_indices:
            wc_pos = centers[wc_indices]
            r_wc = pbc_delta(wc_pos - self.anchor_pos, self.L)
            mu_elec = 2.0 * np.sum(r_wc, axis=0)
            
        return mu_nuc - mu_elec
