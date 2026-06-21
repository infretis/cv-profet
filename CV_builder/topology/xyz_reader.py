"""
xyz_reader.py — Minimal MoleculeSystem that reads a standard .xyz topology file.

This is the recommended starting point for new users.

What this reader does
---------------------
It reads your topology file ONCE at startup to learn:
  1. How many atoms are in the system.
  2. The element type and canonical name for each atom (e.g. "O0", "H1", "C5").
  3. Atom groups by element type (e.g. atom_groups["O"] = [0, 2, 5, ...]).

It does NOT read positions from the topology file — positions change every
frame and are read from the trajectory files directly by the SimulationRunner.
The topology file only defines the stable atom identity: which element is at
which index, constant across ALL trajectories of the simulation.

What belongs in MoleculeSystem vs. CVSuite
-------------------------------------------
  MoleculeSystem.build_topology()   ← STATIC STRUCTURE (computed once)
      - Atom element types and canonical names
      - Atom groups by element (e.g. all oxygens, all carbons)
      - Molecule definitions if relevant (e.g. which H atoms belong to which O)
        → Needs reference positions for nearest-neighbor assignment;
          see topology/examples/cp2k_water_ion.py for an example.

  CVSuite.enrich_inputs()           ← DYNAMIC PER-FRAME DATA
      - Reaction center (may change every frame)
      - Neighbor lists (frame-specific geometry)
      - Auxiliary file data: forces, charges, dipoles (trajectory-specific)

Defining molecules (advanced)
------------------------------
XYZTopologyReader builds only element-based atom groups. If your CVs need
molecule-level groupings (e.g. "which H atoms belong to which O in water"),
you have two options:

  Option A: Override build_topology() in a subclass of XYZTopologyReader.
    Read the XYZ file, then apply your own nearest-neighbor logic to assign
    H atoms to O atoms and populate molecule_groups["water"], etc.
    Use topology/examples/cp2k_water_ion.py as a reference.

  Option B: Compute molecule membership in CVSuite.enrich_inputs() (fine for
    one-off or prototype work, but less efficient for large systems since it
    runs every frame).

TOML configuration
------------------
    [topology]
    system    = "xyz_data"         # selects this reader
    file      = "initial.xyz"      # path to your topology file (relative to TOML)
    cell_size = 12.5               # optional: periodic box length in Å (0 = non-periodic)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from topology.base import MoleculeSystem
from topology.topology import SystemTopology

log = logging.getLogger(__name__)


class XYZTopologyReader(MoleculeSystem):
    """
    Constructs a minimal SystemTopology from a standard .xyz file.

    Only the element type and atom index information is extracted —
    no positions are stored, since positions are frame-specific and
    are read from the trajectory by SimulationRunner.

    Parameters
    ----------
    topology_file : str | Path | None
        Path to the .xyz topology file (e.g. initial.xyz, equilibrium.xyz).
        Can be set via TOML: [topology] file = "initial.xyz"
    cell_size : float | None
        Periodic box length in Å (cubic). Use 0 or None for non-periodic.
    """

    def __init__(
        self,
        cell_size: float | None = None,
        topology_file: str | Path | None = None,
        **kwargs,
    ) -> None:
        self.cell_size = float(cell_size) if cell_size is not None else 0.0
        self.topology_file = Path(topology_file) if topology_file else None

    def describe(self) -> str:
        return f"XYZTopologyReader(file={self.topology_file}, cell_size={self.cell_size})"

    def build_topology(self, ref_frame: Optional[pd.DataFrame] = None) -> SystemTopology:
        """
        Parse the .xyz topology file and return a SystemTopology.

        Parameters
        ----------
        ref_frame : pd.DataFrame | None
            Ignored by this reader — topology is read entirely from the
            xyz file, not from a reference trajectory frame.

        Returns
        -------
        SystemTopology with:
          - n_atoms
          - names_ref : canonical name per atom, e.g. ["O0", "H1", "H2", ...]
          - atom_groups : {"O": array([0,...]), "H": array([1,2,...]), ...}
          - cell_size
        """
        if self.topology_file is None or not self.topology_file.exists():
            raise FileNotFoundError(
                f"XYZTopologyReader requires a valid topology file, missing: {self.topology_file}\n"
                "Set [topology] file = 'path/to/initial.xyz' in your infretis.toml."
            )

        log.info("XYZTopologyReader: parsing topology from %s", self.topology_file)

        with open(self.topology_file, "r") as f:
            lines = f.readlines()

        if len(lines) < 3:
            raise ValueError(
                f"XYZ file {self.topology_file} has only {len(lines)} lines "
                f"(expected: 1 count line + 1 comment + N atom lines)."
            )

        n_atoms = int(lines[0].strip())

        # Only the element column is needed — positions are NOT stored here.
        # Positions change every frame; the topology file just tells us the
        # stable atom ordering and element types.
        names_ref: list[str] = []
        element_counts: dict[str, int] = defaultdict(int)  # for unique per-element index
        groups: dict[str, list[int]] = defaultdict(list)

        for orig_idx, line in enumerate(lines[2 : 2 + n_atoms]):
            parts = line.split()
            if not parts:
                continue
            element = parts[0]
            names_ref.append(f"{element}{orig_idx}")
            groups[element].append(orig_idx)

        if len(names_ref) != n_atoms:
            log.warning(
                "XYZ file declared %d atoms but only %d atom lines were parsed.",
                n_atoms, len(names_ref),
            )
            n_atoms = len(names_ref)

        atom_groups = {el: np.array(idx_list, dtype=int) for el, idx_list in groups.items()}

        log.info(
            "Topology loaded: %d atoms, element groups: %s",
            n_atoms,
            {el: len(v) for el, v in atom_groups.items()},
        )

        return SystemTopology(
            n_atoms=n_atoms,
            names_ref=np.array(names_ref),
            atom_groups=atom_groups,
            cell_size=self.cell_size,
        )
