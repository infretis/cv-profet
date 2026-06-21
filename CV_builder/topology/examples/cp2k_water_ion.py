"""
cp2k_water_ion.py — MoleculeSystem for a CP2K water box with one monovalent ion.

This reader accepts any standard .xyz topology file — the same format as
XYZTopologyReader — but additionally assigns H atoms to their O atoms via
nearest-neighbour search under PBC, and detects the ion. This gives the
pipeline molecule-level groupings ('water triplets', H→O map, ion index)
that CV modules such as reaction_center and neighborhood need.

System layout (193 atoms, 0-indexed):
  O     : orig_idx  0 .. 63   (64 oxygen atoms)
  H     : orig_idx 64 .. 191  (128 hydrogen atoms)
  Cl    : orig_idx 192        (single ion, always last)

Adapting for a different system
--------------------------------
If your system has a different layout (e.g. Na⁺ instead of Cl⁻, or no ion,
or more water molecules), either:
  a) Adjust the detect_ion() logic below, or
  b) Write a new MoleculeSystem subclass for your case.

Either way, run debug_nuclear_ids.py first to verify the topology is physical.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read as ase_read
from ase import Atoms

from topology.base import MoleculeSystem
from topology.topology import SystemTopology

log = logging.getLogger(__name__)

# Maximum physically reasonable O-H bond length (Å).
# Water O-H bonds are ~0.96 Å equilibrium; proton-transfer intermediates
# may stretch to ~1.4 Å. Above this threshold, the water assignment is wrong.
OH_BOND_WARN_ANGSTROM = 1.5


def _pbc_delta(delta: np.ndarray, L: float | None) -> np.ndarray:
    """Minimum-image displacement vector(s) under PBC."""
    if L is None or L <= 0:
        return delta
    return delta - L * np.round(delta / L)


def _two_nearest_h_per_o(
    O_pos: np.ndarray,
    H_pos: np.ndarray,
    box: float | None,
) -> np.ndarray:
    """
    For each oxygen (nO, 3), return indices of the 2 nearest hydrogens (nH, 3)
    under PBC. Returns shape (nO, 2) — indices into H_pos (0-based within H block).
    """
    d = O_pos[:, None, :] - H_pos[None, :, :]          # (nO, nH, 3)
    d = _pbc_delta(d, box)
    d2 = np.sum(d * d, axis=2)                           # (nO, nH)
    return np.argsort(d2, axis=1)[:, :2]                 # (nO, 2)


def _order_two_h_deterministically(
    O_pos: np.ndarray,
    H_pair: np.ndarray,
    box: float | None,
) -> np.ndarray:
    """
    Deterministically order the two H atoms around one O by projecting
    onto the golden-ratio unit vector (breaks ties without ambiguity).
    Returns order indices (length 2) into H_pair.
    """
    _PHI = (1.0 + 5.0 ** 0.5) / 2.0
    U = np.array([1.0, _PHI, _PHI**2])
    U /= np.linalg.norm(U)

    OH = _pbc_delta(H_pair - O_pos[None, :], box)  # (2, 3)
    proj = OH @ U
    return np.argsort(proj)[::-1]  # high → low


class CP2KWaterIonSystem(MoleculeSystem):
    """
    Topology builder for a CP2K simulation of liquid water with one ion.

    The element ordering in the xyz files is assumed to be stable across all
    frames and all trajectories: O-block first, H-block second, ion last.
    This has been verified empirically for the Cl⁻/water system.

    Parameters
    ----------
    cell_size : float | None
        Periodic box length in Å (cubic). Required for correct H→O assignment
        under PBC. Use None only for non-periodic systems.
    ion_symbol : str | None
        Expected ion element symbol (e.g. "Cl", "Na"). If None, auto-detect
        the first non-H/O atom. Set to None for pure water systems.
    oh_warn_threshold : float
        Warn if any assigned O-H distance exceeds this (Å). Default 1.5 Å.
    """

    def __init__(
        self,
        topology_file: str | Path,
        cell_size: float | None = None,
        ion_symbol: str | None = None,
        oh_warn_threshold: float = OH_BOND_WARN_ANGSTROM,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        topology_file : str | Path
            Path to a standard .xyz topology file (e.g. initial.xyz).
            Only the element types and atom order are used — positions are
            read from the trajectory by SimulationRunner each frame.
        cell_size : float | None
            Periodic box length in Å (cubic). Required for correct H→O assignment
            under PBC. Use None only for non-periodic systems.
        ion_symbol : str | None
            Expected ion element symbol (e.g. "Cl", "Na"). If None, auto-detect
            the first non-H/O atom. Set to None for pure water systems.
        oh_warn_threshold : float
            Warn if any assigned O-H distance exceeds this (Å). Default 1.5 Å.
        """
        self.topology_file = Path(topology_file)
        self.cell_size = float(cell_size) if cell_size is not None else None
        self.ion_symbol = ion_symbol
        self.oh_warn_threshold = float(oh_warn_threshold)

    def describe(self) -> str:
        return (
            f"CP2KWaterIonSystem(file={self.topology_file.name}, "
            f"cell_size={self.cell_size}, ion={self.ion_symbol or 'auto-detect'})"
        )

    @staticmethod
    def _load_atoms(xyz_path: str | Path, ase_idx: int = 0) -> pd.DataFrame:
        """Load atomic data from an .xyz file into a DataFrame preserving original indexing."""
        atoms = ase_read(str(xyz_path), index=ase_idx)
        assert isinstance(atoms, Atoms)
        elements = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        return pd.DataFrame({
            'element': elements,
            'orig_idx': list(range(len(elements))),
            'x': pos[:, 0],
            'y': pos[:, 1],
            'z': pos[:, 2],
        })

    def build_topology(self, ref_frame=None) -> SystemTopology:
        """
        Build the full SystemTopology from the topology file.

        Reads the .xyz topology file to get element types and positions for
        the H→O nearest-neighbour assignment. Positions in the topology file
        are only used here (once at startup) for molecule assignment — they
        are NOT stored in the topology and do not affect trajectory reading.

        ref_frame is accepted for interface compatibility but ignored;
        the topology file is always used instead.
        """
        log.info("CP2KWaterIonSystem: reading topology from %s", self.topology_file)
        ref_frame = self._load_atoms(self.topology_file, ase_idx=0)

        elements = ref_frame["element"].to_numpy()
        orig_idx = ref_frame["orig_idx"].to_numpy()
        xyz      = ref_frame[["x", "y", "z"]].to_numpy()
        n_atoms  = len(elements)


        # ── Identify O, H, ion by element ─────────────────────────────────
        mask_O   = elements == "O"
        mask_H   = elements == "H"
        mask_ion = ~(mask_O | mask_H)

        abs_O = np.flatnonzero(mask_O)   # row positions in ref_frame (= orig_idx here)
        abs_H = np.flatnonzero(mask_H)
        abs_ion = np.flatnonzero(mask_ion)

        O_pos = xyz[abs_O]
        H_pos = xyz[abs_H]

        # ── Ion detection ──────────────────────────────────────────────────
        ion_index: int | None = None
        ion_label: str | None = None
        if abs_ion.size == 1:
            ion_row = int(abs_ion[0])
            ion_index = int(orig_idx[ion_row])
            ion_label = str(elements[ion_row])
            if self.ion_symbol is not None and ion_label != self.ion_symbol:
                log.warning(
                    "Expected ion '%s' but found '%s' at orig_idx=%d",
                    self.ion_symbol, ion_label, ion_index,
                )
            log.info("Ion detected: %s at orig_idx=%d", ion_label, ion_index)
        elif abs_ion.size == 0:
            log.info("No ion detected — pure water system.")
        else:
            log.warning(
                "Found %d non-H/O atoms — topology builder expects exactly 1 ion. "
                "Only the first will be used.",
                abs_ion.size,
            )
            ion_row   = int(abs_ion[0])
            ion_index = int(orig_idx[ion_row])
            ion_label = str(elements[ion_row])

        # ── Water molecule assignment: 2 nearest H per O ───────────────────
        nn_H = _two_nearest_h_per_o(O_pos, H_pos, self.cell_size)  # (nO, 2), indices into H_pos

        nO = len(abs_O)
        water_triplets = np.empty((nO, 3), dtype=int)        # [H1, O, H2]
        water_triplet_labels = []
        ho_pairs_list = []
        hh_pairs_list = []

        names_ref = np.array(
            [f"{el}{int(i)}" for el, i in zip(elements, orig_idx)]
        )

        bad_oh_count = 0

        for o_i in range(nO):
            h_pair_local = nn_H[o_i]         # (2,) indices into H_pos
            H_pair_xyz   = H_pos[h_pair_local]

            order = _order_two_h_deterministically(O_pos[o_i], H_pair_xyz, self.cell_size)
            h1_local, h2_local = h_pair_local[order]

            # Absolute orig_idx values
            o_abs  = int(orig_idx[abs_O[o_i]])
            h1_abs = int(orig_idx[abs_H[h1_local]])
            h2_abs = int(orig_idx[abs_H[h2_local]])

            # Bond distance check for physical sanity
            for h_abs, h_local in [(h1_abs, h1_local), (h2_abs, h2_local)]:
                d_oh = float(np.linalg.norm(
                    _pbc_delta(H_pos[h_local] - O_pos[o_i], self.cell_size)
                ))
                if d_oh > self.oh_warn_threshold:
                    log.warning(
                        "Non-physical O-H distance: O%d--H%d = %.3f Å "
                        "(threshold=%.2f Å). Check your molecule assignment.",
                        o_abs, h_abs, d_oh, self.oh_warn_threshold,
                    )
                    bad_oh_count += 1

            water_triplets[o_i] = [h1_abs, o_abs, h2_abs]
            water_triplet_labels.append(
                f"{names_ref[h1_abs]}-{names_ref[h2_abs]}-{names_ref[o_abs]}"
            )
            ho_pairs_list += [[o_abs, h1_abs], [o_abs, h2_abs]]
            hh_pairs_list.append([h1_abs, h2_abs])

        if bad_oh_count:
            log.error(
                "%d non-physical O-H assignments found in topology. "
                "Run debug_nuclear_ids.py for details.",
                bad_oh_count,
            )


        water_H_of_O = {
            t[1]: [t[0], t[2]] for t in water_triplets
        }

        topology = SystemTopology(
            n_atoms=n_atoms,
            names_ref=names_ref,
            atom_groups={
                "O": orig_idx[abs_O],
                "H": orig_idx[abs_H],
                "ion": np.array([ion_index]) if ion_index is not None else np.array([]),
            },
            molecule_groups={
                "water": water_triplets,
                "ho_pairs": np.array(ho_pairs_list, dtype=int),
                "hh_pairs": np.array(hh_pairs_list, dtype=int),
            },
            molecule_groups_labels={
                "water": water_triplet_labels,
            },
            custom={
                "water_H_of_O": water_H_of_O,
                "ion_label": ion_label,
            },
            # Neighbourhood helpers — O is the representative centre of each water;
            # [H1, H2] are companions used for H-sorting within each neighbor.
            mol_centers=water_triplets[:, 1].copy(),       # (nO,)    O per water
            mol_members=water_triplets[:, [0, 2]].copy(),  # (nO, 2)  [H1, H2] per water
            cell_size=self.cell_size if self.cell_size is not None else 0.0,
        )

        log.info("Topology built: %d waters, ion=%s@%s",
                 nO, ion_label, ion_index)
        log.debug("\n%s", topology.describe())
        return topology
