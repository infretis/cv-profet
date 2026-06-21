"""
minimal_suite.py — The simplest possible CVSuite for new users.

This file is your starting point. It computes one CV (atom-pair distance)
and is fully system-agnostic. To add your own CVs, follow the instructions
below and in topology/README.md.

Minimum requirements to run the full pipeline:
    - infretis.toml          (see infretis_example.toml)
    - load/                  (your RETIS trajectory folders)
    - infretis_data.txt      (WHAM weights from infretis)
    - A topology file        (e.g. initial.xyz)

TOML configuration
------------------
Point your infretis.toml to this file:

    [cv_modules]
    module = "CV_builder/CV_manager/minimal_suite.py"
    class  = "MinimalCVSuite"

To configure which atom pair(s) to compute, edit build_cv_modules() below.

───────────────────────────────────────────────────────────────────────────
HOW TO ADD MORE CVs
───────────────────────────────────────────────────────────────────────────
1. Write a new CV class in CV_manager/cvs/ (use cv_distance.py as template).
   Every CV needs:
     - a `name` attribute (str): unique, used in HDF5 output filenames.
     - a `compute(cv_inputs: CVInputs) -> np.ndarray` method.

2. Import your new CV at the top of this file.

3. Add an instance of it to the list returned by build_cv_modules().

That's it — no changes to the core pipeline are needed.

───────────────────────────────────────────────────────────────────────────
NOTE ON CV_MODULE: STATE-DEPENDENT VARIABLES
───────────────────────────────────────────────────────────────────────────
For more advanced CVs that depend on system state (e.g. Mulliken charges,
forces, Wannier centers), you can use the CV_frame_data layer:

1. In your CVSuite subclass, override `on_trajectory_opened(traj_path, n_atoms)`
   to load trajectory-level auxiliary files ONCE per trajectory and cache them.

2. Override `enrich_inputs(ctx: CVContext) -> CVInputs` to compute per-frame
   derived quantities (reaction center, neighbor lists, etc.) and inject them
   into CVInputs.data. Your CV's compute() then pulls from cv_inputs.data.

This separation is efficient: heavy file I/O happens once per trajectory,
while light per-frame computation happens in enrich_inputs().

───────────────────────────────────────────────────────────────────────────
THE REACTION CENTER: AN ANCHOR FOR ALL CVs
───────────────────────────────────────────────────────────────────────────
When analyzing RETIS trajectories, it is critical that CVs from different
paths and time steps describe the *same physical feature*, not arbitrary
atom numbers. A Reaction Center (RC) serves as an "anchor" for this purpose.

  SIMPLE (FIXED) ANCHOR
  ─────────────────────
  If your system has a specific atom that is constant across all steps and
  trajectories — such as an ion, a tagged carbon, a metal center, or a
  ligand — this is a natural fixed RC. All CVs computed relative to this
  anchor (e.g. distance to nearest solvent, coordination number from ion)
  are directly comparable across all paths.

  Example: In a proton-transfer reaction with a Cl⁻ ion, the ion is the
  natural anchor. All CVs reference the ion's position.

  DYNAMIC ANCHOR
  ──────────────
  More complex systems may have no fixed atom of interest. Instead, the RC
  is identified by some geometric or electronic criterion that alternates
  between frames and trajectories. Common examples:
    - The atom with the longest O-H bond (proton transfer transition state)
    - The most electronegative atom in a given region
    - The atom with the highest Mulliken charge
    - The center of a hydrogen-bond wire

  Even though the specific atom changes between frames, defining CVs
  *relative to the current RC* gives physically comparable values, because
  they always describe the same geometric feature (e.g. "distance from
  the proton-receiving oxygen to the nearest acceptor") regardless of which
  oxygen is assigned as the RC.

  HOW TO IMPLEMENT
  ────────────────
  Compute the RC in enrich_inputs() or on_trajectory_opened(), then inject
  it into CVInputs.data["reaction"]. Your CV's compute() pulls it from there:

      rc = cv_inputs.data.get("reaction")   # your RC object
      rc_position = cv_inputs.coords[rc.anchor_index]

  See CV_manager/examples/cp2k_water_ion_suite.py for a full reference
  implementation of a dynamic RC (the proton being transferred in water).
"""
from __future__ import annotations

from typing import Iterable, Sequence

from CV_manager.base import CVSuite
from CV_manager.cvs.cv_distance import AtomPairDistanceCV
from topology.topology import SystemTopology


class MinimalCVSuite(CVSuite):
    """
    Minimal CVSuite: computes one simple atom-pair distance.

    Edit build_cv_modules() to:
      - Change which atom pair(s) to compute
      - Add more CVs (see instructions above)
      - Conditionally include CVs based on topology (e.g. ion present)

    TOML-driven configuration (recommended)
    ---------------------------------------
    You can set the atom pairs directly in [cv_modules]:

      [cv_modules]
      module = "CV_builder/CV_manager/minimal_suite.py"
      class  = "MinimalCVSuite"
      pairs  = [[0, 1], [0, 2]]
      use_pbc = true
      name = "my_distances"          # optional
      labels = "my_distances"        # optional
    """

    def __init__(
      self,
      pairs: Sequence[Sequence[int]] | None = None,
      use_pbc: bool = True,
      name: str | None = None,
      labels: str | None = None,
    ) -> None:
      self.pairs = self._normalize_pairs(pairs if pairs is not None else [(0, 1)])
      self.use_pbc = bool(use_pbc)
      self.name = name
      self.labels = labels

    @staticmethod
    def _normalize_pairs(pairs: Iterable[Sequence[int]]) -> list[tuple[int, int]]:
      norm: list[tuple[int, int]] = []
      for pair in pairs:
        if len(pair) != 2:
          raise ValueError(
            "Each entry in [cv_modules].pairs must contain exactly 2 atom indices, "
            f"got: {pair}"
          )
        a, b = int(pair[0]), int(pair[1])
        norm.append((a, b))
      if not norm:
        raise ValueError("[cv_modules].pairs must contain at least one atom pair")
      return norm

    def build_cv_modules(self, topo: SystemTopology) -> list:
        """
        Return the list of CV objects to compute for every trajectory frame.

        The topology is passed so you can conditionally include CVs based
        on system composition (e.g. add an ion-distance CV only if an ion
        is detected in the topology).

        Edit the pairs below to match your system.
        Atom indices are 0-based and refer to orig_idx in the topology file.
        """
        cvs = [
          # Configured via [cv_modules] pairs/use_pbc/name/labels in TOML.
          AtomPairDistanceCV(
            pairs=self.pairs,
            use_pbc=self.use_pbc,
            name=self.name,
            labels=self.labels,
          ),

            # ── Add more CVs here ────────────────────────────────────────
            # from CV_manager.cvs.cv_distance import AtomPairDistanceCV
            # AtomPairDistanceCV(pairs=[(2, 5)], name="my_bond_length"),
            # MyCustomCV(param=42),
        ]
        return cvs
