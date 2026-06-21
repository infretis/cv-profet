"""
topology.py — SystemTopology: the output of MoleculeSystem.build_topology().

This dataclass is computed ONCE from a reference frame and reused for every
subsequent trajectory frame. It contains all atom-index and pair arrays needed
by CV modules, in terms of the stable CP2K nuclear IDs (orig_idx).

NOTE FOR NEW USERS
------------------
If you are adapting this pipeline to a new chemical system, you need to provide
a MoleculeSystem subclass that produces a valid SystemTopology from your reference
frame. Run `debug_nuclear_ids.py` afterwards to verify that the bond distances
and molecule assignments are physically sensible (e.g., no O-H distance > 1.5 Å).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class SystemTopology:
    """
    Atom-index and pair arrays for one chemical system, in terms of stable
    CP2K nuclear IDs (orig_idx = 0-based position in the xyz file).

    Arrays use orig_idx throughout; no per-frame permutation is needed because
    CP2K preserves nuclear IDs across all trajectories of the same run.

    Fields
    ------
    n_atoms : int
        Total number of atoms in the system.
    O_indices : np.ndarray, shape (nO,)
        orig_idx of all oxygen atoms.
    H_indices : np.ndarray, shape (nH,)
        orig_idx of all hydrogen atoms.
    ion_index : int | None
        orig_idx of the single non-H/O ion, or None for a pure water system.
    ion_label : str | None
        Chemical symbol of the ion (e.g. "Cl", "Na"), or None.
    water_triplets : np.ndarray, shape (nO, 3)
        Each row is [H1_origidx, O_origidx, H2_origidx] for one water molecule.
        H1 and H2 are the two nearest hydrogens to the oxygen in the reference
        frame, ordered deterministically.
    water_triplet_labels : np.ndarray, shape (nO,)
        Human-readable labels like "H64-H65-O0" for each water molecule.
    ho_pairs : np.ndarray, shape (2*nO, 2)
        All O-H pairs from water_triplets: [O_idx, H_idx].
    hh_pairs : np.ndarray, shape (nO, 2)
        The geminal H-H pairs within each water molecule: [H1_idx, H2_idx].
    names_ref : np.ndarray, shape (n_atoms,)
        Canonical atom name indexed by orig_idx, e.g. names_ref[0] = "O0",
        names_ref[64] = "H64", names_ref[192] = "Cl192".
    """
    n_atoms:              int
    names_ref:            np.ndarray

    # ── Generic Groupings mapped by name ─────────────────────────────────
    # e.g., atom_groups["O"] = [0, 3, 6], atom_groups["H"] = [1, 2, 4, 5]
    # molecule_groups["water"] = [[1, 0, 2], [4, 3, 5]]
    atom_groups:          dict[str, np.ndarray] = field(default_factory=dict)
    molecule_groups:      dict[str, np.ndarray] = field(default_factory=dict)
    molecule_groups_labels: dict[str, list[str]] = field(default_factory=dict)
    
    # ── Engine-specific Custom Metadata ──────────────────────────────────
    # For complex custom algorithms (e.g., mapping O to H indices in CP2K)
    custom:               dict[str, Any] = field(default_factory=dict)
    
    cell_size:            float = 0.0

    # ── Neighbourhood helpers (optional but recommended) ───────────────────────
    # These two fields drive build_neighborhood() in a system-agnostic way.
    # They have sensible defaults for water but any MoleculeSystem can override them.
    #
    # mol_centers: for each molecule, the index of the atom used as the
    #   neighbourhood sorting atom ("center" of that molecule).
    #   Water default: the O atom of each H2O → water_triplets[:, 1]
    #   e.g. for Na⁺ solvation: the O of each water in the first shell.
    #
    # mol_members: for each molecule, the companion atom indices (used to rank
    #   the H atoms within each neighbor, e.g. for H-bond analysis).
    #   Water default: [H1, H2] of each H2O → water_triplets[:, [0, 2]]
    #   Other systems may set this to None if companion ranking isn't needed.
    #
    # TOML compatibility: build_neighborhood() reads K, metric, h_rank_mode from
    # arguments (passed by SimulationRunner). These can be wired to TOML keys
    # [CVs] k_neighbors, [CVs] neighbor_metric, [CVs] h_rank_mode in the future
    # without changing the topology format.
    #
    # For a completely custom neighbourhood scheme, override build_neighborhood()
    # in your MoleculeSystem subclass rather than trying to express it in TOML.
    mol_centers: np.ndarray | None = None   # (nMol,)     representative atom per molecule
    mol_members: np.ndarray | None = None   # (nMol, nC)  companion atoms per molecule

    def describe(self, max_waters: int = 5) -> str:
        """Return a human-readable summary for logging/debugging."""
        lines = [f"SystemTopology: {self.n_atoms} atoms"]
        
        for k, v in self.atom_groups.items():
            lines.append(f"  {k} atoms : {len(v)}  (orig_idx {v[0]}..{v[-1]} if sorted)")
            
        for k, v in self.molecule_groups.items():
            lines.append(f"  Molecule '{k}': {len(v)} instances, shape {v.shape}")
                
        return "\n".join(lines)
