"""
base.py — MoleculeSystem: abstract base class for chemical-system topology builders.

PURPOSE
-------
This abstraction separates all knowledge about "what molecules are in the box
and how to identify them" from the SimulationRunner, which handles I/O and CV
computation. New users with different chemical systems only need to implement
this interface — they do NOT need to modify SimulationRunner.

HOW TO IMPLEMENT A NEW SYSTEM
------------------------------
1. Create a new file, e.g. `topology/my_system.py`.
2. Subclass MoleculeSystem and implement `build_topology()`.
3. In `build_topology()`, inspect the reference frame DataFrame and return a
   fully populated SystemTopology.
4. Run `topology/debug_nuclear_ids.py` to verify your topology produces
   physically sensible bond distances (e.g., O-H < 1.5 Å for all water pairs).
5. Pass `mol_system=MySystem(...)` to SimulationRunner in main.py.

CRITICAL: Verify your topology before running a full CV computation.
          A wrong molecule mapping will silently produce incorrect CVs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd

from .topology import SystemTopology


class MoleculeSystem(ABC):
    """
    Knows how to interpret a CP2K trajectory frame as a collection of molecules.

    build_topology() is called ONCE with the reference frame DataFrame and must
    return a SystemTopology that is valid for ALL frames in the simulation.
    This is possible because CP2K preserves nuclear IDs (atom ordering) across
    all trajectories of the same run.

    Parameters
    ----------
    ref_frame : pd.DataFrame
        First frame from the simulation, with columns:
        ['element', 'orig_idx', 'x', 'y', 'z']
    cell_size : float | None
        Periodic boundary condition box length in Å. Required for correct
        nearest-neighbour H→O assignment under PBC.
    """

    @abstractmethod
    def build_topology(self, ref_frame: "pd.DataFrame | None" = None) -> SystemTopology:
        """
        Build and return the SystemTopology for this chemical system.

        The ``ref_frame`` parameter is optional. Subclasses that derive topology
        from a reference trajectory frame (e.g. CP2KWaterIonSystem) should use it;
        subclasses that read from a static file (e.g. XYZTopologyReader) should
        ignore it. SimulationRunner always calls this with no arguments.

        Must be deterministic: calling with the same inputs always returns
        the same topology (same molecule groupings, same atom ordering).
        """
        ...

    def describe(self) -> str:
        """Short human-readable description of this MoleculeSystem class."""
        return self.__class__.__name__
