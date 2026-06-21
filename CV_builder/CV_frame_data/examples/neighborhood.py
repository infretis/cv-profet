# CV_manager/neighborhood.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np
from CV_frame_data.math_utils import mic_delta, mic_dist

from .reaction_center import ReactionCenter  # type: ignore


def _mic_delta(a: np.ndarray, b: np.ndarray, box) -> np.ndarray:
    return mic_delta(b - a, box)


def _mic_dist(a: np.ndarray, b: np.ndarray, box) -> float:
    return mic_dist(a, b, box)


@dataclass(slots=True)
class ReactionNeighborhood:
    """
    Per-frame neighborhood centered on the reaction-center donor oxygen O_d.
    All indices are in the **invariant** index space (aligned with names_ref).
    """
    center_O: int            # O_d (donor oxygen)
    Hs: int                  # H* index
    O_indices: np.ndarray    # (K,) mol-center indices by ascending distance
    O_labels: np.ndarray     # (K,) human-readable labels
    H_pairs: np.ndarray      # (K, nC) companion atoms per neighbor (unordered)
    H_pairs_by_rule: np.ndarray  # (K, nC) companions sorted by distance rule
    d_OO: np.ndarray         # (K,) distances center → neighbor center (MIC)

    def rank_map(self) -> np.ndarray:
        """Return the rank → center-atom index mapping (shape K,)."""
        return self.O_indices

    def rank_labels(self) -> np.ndarray:
        """Return the rank → human label mapping (shape K,)."""
        return self.O_labels


def build_neighborhood(
    coords: np.ndarray,
    topo,                                     # SystemTopology
    rc: ReactionCenter,
    K: int = 12,
    box=None,
    include_self: bool = True,
    metric: Literal["OO"] = "OO",
    h_rank_mode: Literal["to_Hstar", "to_Od"] = "to_Hstar",
    tie_tol: float = 1e-10,
    # Legacy keyword kept for call-site backward compatibility (ignored)
    water_triplets: Optional[np.ndarray] = None,
    names_ref: Optional[np.ndarray] = None,
) -> ReactionNeighborhood:
    """Select the K nearest **molecule centers** to the donor oxygen O_d.

    System-agnostic: uses ``topo.mol_centers`` (representative atom per molecule,
    e.g. O for water) and ``topo.mol_members`` (companion atoms, e.g. [H1, H2]).

    For **water** (default with CP2KWaterIonSystem):
        mol_centers = O per water molecule
        mol_members = [H1, H2] per water molecule

    For **other systems**: set mol_centers/mol_members in your MoleculeSystem.build_topology().

    Parameters
    ----------
    coords : (N, 3)
        Current frame coordinates in topology sort order.
    topo : SystemTopology
        Provides mol_centers, mol_members, and names_ref.
        Falls back to water_triplets columns if mol_centers is None.
    rc : ReactionCenter
        Current reaction centre (O_d, Hs).
    K : int
        Number of neighbours to return.
        **TOML-compatible**: wire to ``[CVs] k_neighbors`` later.
    box : float | None
        PBC box length (Å).
    include_self : bool
        Include the donor molecule at rank 0.
    metric : "OO"
        Ranking metric. Currently only centre-to-centre distance supported.
        **TOML-compatible**: wire to ``[CVs] neighbor_metric`` later.
    h_rank_mode : "to_Hstar" | "to_Od"
        How to sort companion atoms within each neighbour.
        **TOML-compatible**: wire to ``[CVs] h_rank_mode`` later.
    """
    O_d, Hs = int(rc.O_d), int(rc.Hs)

    # ── Resolve mol_centers / mol_members from topology ───────────────────
    mol_centers = getattr(topo, "mol_centers", None)
    mol_members = getattr(topo, "mol_members", None)

    # Backward-compat fallback: derive from water_triplets if needed
    if mol_centers is None:
        wt = getattr(topo, "water_triplets", None)
        if wt is not None:
            mol_centers = np.asarray(wt, dtype=int)[:, 1]
            mol_members = np.asarray(wt, dtype=int)[:, [0, 2]]

    if mol_centers is None:
        raise ValueError(
            "build_neighborhood() requires topo.mol_centers. "
            "Set it in your MoleculeSystem.build_topology(), or ensure "
            "topo.molecule_groups['water'] is available for the backward-compat fallback."
        )

    center_atoms = np.asarray(mol_centers, dtype=int)          # (nMol,)
    companion = (np.asarray(mol_members, dtype=int)            # (nMol, nC)
                 if mol_members is not None else None)

    # ── Candidate centres (with optional self-inclusion) ──────────────────
    candidates = center_atoms.copy()
    if not include_self:
        candidates = candidates[candidates != O_d]

    # ── Distance-ranked sort (stable tie-break by index) ──────────────────
    dists = np.array([_mic_dist(coords[O_d], coords[o], box) for o in candidates])
    sorted_triples = sorted(
        zip(
            [round(d / max(tie_tol, 1e-20)) for d in dists],
            candidates.tolist(),
            dists.tolist(),
        ),
        key=lambda t: (t[0], t[1]),
    )
    use   = sorted_triples[:K] if (K and K > 0) else sorted_triples
    O_sel = np.array([c for _, c, _ in use], dtype=int)
    d_sel = np.array([d for _, _, d in use], dtype=float)

    # ── Companion atoms for each selected centre ───────────────────────────
    if companion is not None:
        idx_lookup = {int(c): i for i, c in enumerate(center_atoms)}
        H_for_O = np.array(
            [companion[idx_lookup[int(o)]] for o in O_sel], dtype=int
        )
        # Sort companions by distance to H* or O_d
        ref_pos = coords[Hs] if h_rank_mode == "to_Hstar" else coords[O_d]
        a   = np.broadcast_to(ref_pos, H_for_O.shape + (3,))
        b   = coords[H_for_O]
        dH  = np.linalg.norm(_mic_delta(a, b, box), axis=2)
        rows    = np.arange(H_for_O.shape[0])[:, None]
        H_sorted = H_for_O[rows, np.argsort(dH, axis=1)]
    else:
        H_for_O  = np.empty((len(O_sel), 0), dtype=int)
        H_sorted = H_for_O.copy()

    # ── Human-readable labels ─────────────────────────────────────────────
    nr = names_ref if names_ref is not None else getattr(topo, "names_ref", None)
    if nr is not None:
        O_labels = np.asarray(nr, dtype=object)[O_sel].astype(str)
    else:
        O_labels = np.array([str(o) for o in O_sel], dtype=object)

    return ReactionNeighborhood(
        center_O=O_d, Hs=Hs,
        O_indices=O_sel,
        O_labels=O_labels,
        H_pairs=H_for_O,
        H_pairs_by_rule=H_sorted,
        d_OO=d_sel,
    )