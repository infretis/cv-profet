"""
cv_distance.py — Simple atom-pair distance CV.

This is the minimal building block for new users. It computes the Euclidean
distance between one or more pairs of atoms and returns them as a 1D array.

No topology assumptions are made — atom index pairs are specified explicitly
at construction time.

How to add this CV to your suite
---------------------------------
1. Import: from CV_manager.cvs.cv_distance import AtomPairDistanceCV
2. Instantiate: AtomPairDistanceCV(pairs=[(0, 1), (0, 2)])
3. Add to the list returned by your CVSuite.build_cv_modules()

How to write your own CV
-------------------------
Any CV must be a class with:
  - a `name` attribute (str): unique identifier used in HDF5 output
  - a `compute(cv_inputs: CVInputs) -> np.ndarray` method

The `compute` method receives a CVInputs object with:
  - cv_inputs.coords   : (N, 3) float array of atomic positions (Å)
  - cv_inputs.topo     : SystemTopology (atom indices, groups, etc.)
  - cv_inputs.box      : cell size in Å (for PBC), or None
  - cv_inputs.data     : dict with system-specific enrichments from CVSuite

Return a numpy array of shape (K,) where K is the number of scalar values
this CV produces per frame.
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional


class AtomPairDistanceCV:
    """
    Computes the Euclidean distance between one or more atom pairs.

    Parameters
    ----------
    pairs : list of (int, int)
        Each tuple is (atom_A_index, atom_B_index) using the 0-based
        original topology index (orig_idx). These indices must be stable
        across all trajectory frames; they are determined once from the
        topology file (e.g. initial.xyz) and do not change.
    use_pbc : bool
        If True, apply minimum-image convention using cv_inputs.box.
        Set to False for non-periodic systems or gas-phase simulations.
    name : str | None
        Override the auto-generated CV name. By default, the name is
        constructed from the atom pair indices.
    label : str | None
        Human-readable label used for plotting and reporting. Defaults to
        the value of `name` if not provided.

    Example
    -------
    # Distance between atom 0 and atom 1:
    cv = AtomPairDistanceCV(pairs=[(0, 1)])

    # Multiple distances in one CV (one value per pair):
    cv = AtomPairDistanceCV(pairs=[(0, 1), (0, 2), (1, 2)])
    """

    def __init__(
        self,
        pairs: List[Tuple[int, int]],
        use_pbc: bool = True,
        name: Optional[str] = None,
        labels: Optional[str] = None,
    ) -> None:
        if not pairs:
            raise ValueError("AtomPairDistanceCV requires at least one atom pair.")
        self.pairs = pairs
        self.use_pbc = use_pbc
        # Auto-generate a human-readable name from the pair list
        self.name = name or "distance_" + "_".join(f"{a}-{b}" for a, b in pairs)
        if labels is not None:
            self.labels = list(labels) if not isinstance(labels, list) else labels
        else:
            self.labels = [f"{a}-{b}" for a, b in pairs]

    def compute(self, cv_inputs) -> np.ndarray:
        """
        Compute distances for all pairs and return as a 1D array.

        Parameters
        ----------
        cv_inputs : CVInputs
            Per-frame input from the simulation runner.

        Returns
        -------
        np.ndarray, shape (len(pairs),)
            Distances in Å, one value per atom pair.
        """
        coords = cv_inputs.coords  # (N, 3)
        box = cv_inputs.box if self.use_pbc else None

        distances = np.empty(len(self.pairs), dtype=np.float64)
        for i, (a, b) in enumerate(self.pairs):
            delta = coords[a] - coords[b]
            if box is not None and box > 0:
                # Minimum-image convention for cubic periodic box
                delta -= box * np.round(delta / box)
            distances[i] = float(np.linalg.norm(delta))

        return distances
