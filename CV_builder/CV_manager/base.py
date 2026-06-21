"""
base.py — Abstract base class for defining sets of Collective Variables (CVs).

This ABC defines the interface for creating suites of CVs to calculate during
data aggregation. New users writing CVs for differing chemical systems
should subclass CVSuite here and point `infretis.toml` to their file!
"""

from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path
from topology.topology import SystemTopology
from CV_frame_data.context import CVContext, CVInputs

class CVSuite(ABC):
    """
    Base class for a collection of CV modules.
    Subclass this to define the CVs and input enrichments for your specific system.

    CROSS-MODULE DEPENDENCIES:
    If a CV module needs results from another module calculated in the same frame
    (e.g., for ratios or derived quantities), it can signal this by setting:
    `requires_results_cache = True`.
    The `SimulationRunner` will then populate `inputs.results` with the outputs
    of all preceding modules, enabling efficient on-the-fly calculations without recomputing.
    """

    @abstractmethod
    def build_cv_modules(self, topo: SystemTopology) -> list:
        """
        Return the list of CV module objects to compute for each frame.
        The system topology is passed so that suites can conditionally add/remove 
        CVs depending on factors like whether an ion is present, number of molecules, etc.
        """
        ...
    
    def on_trajectory_opened(self, traj_path: Path, n_atoms: int) -> None:
        """
        Optional hook called exactly once when a new trajectory file is opened.
        Use this to cache massive trajectory-level auxiliary files (like forces or Mulliken charges)
        into memory, preventing the need to re-read them parsing every frame.
        """
        pass

    def enrich_inputs(self, ctx: CVContext) -> CVInputs:
        """
        Convert the raw CVContext into CVInputs for your CV modules.

        The default implementation is a clean pass-through: it passes
        geometry, topology, and metadata directly with an empty data dict.

        Override this in your CVSuite subclass to inject system-specific
        per-frame data (e.g. reaction center, neighborhood, Mulliken charges,
        forces, solvation shell, etc.) into CVInputs.data.

        Example override:
            def enrich_inputs(self, ctx: CVContext) -> CVInputs:
                rc = select_reaction_center(ctx.coords, ctx.topo, ctx.box)
                neigh = build_neighborhood(ctx.coords, ctx.topo, rc, K=12, box=ctx.box)
                return CVInputs(
                    coords=ctx.coords, topo=ctx.topo, key=ctx.key,
                    box=ctx.box, traj_path=ctx.traj_path,
                    data={"reaction": rc, "neighborhood": neigh},
                )
        """
        return CVInputs(
            coords=ctx.coords,
            topo=ctx.topo,
            key=ctx.key,
            box=ctx.box,
            traj_path=ctx.traj_path,
            data={},
        )


    def get_debug_log(self, cv_inputs: CVInputs) -> str:
        """
        Return a custom string to be logged per-frame at DEBUG level.
        """
        return ""
