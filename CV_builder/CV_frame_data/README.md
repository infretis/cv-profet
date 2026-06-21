# cv_frame_data — The Per-Frame Data Layer

> **On the name:** This folder is currently called `CV_frame_data`. A clearer name would be **`cv_frame_data`** or **`cv_core`** — it is the data plumbing layer that connects `SimulationRunner` to your CV code. It does not contain CV logic itself. We recommend renaming it in a future clean-up.

---

## What This Module Does

This module owns two things:

1. **The data containers** that flow through the pipeline for every frame (`CVContext`, `CVInputs`)
2. **General math utilities** used across CV modules (`math_utils.py`)

It is **not** where CV logic lives — that belongs in `CV_manager/`. And it is **not** where topology reading lives — that belongs in `topology/`.

Think of it as the "envelope" that carries all frame-level information from the runner to your CV code.

---

## The Data Flow (One Frame)

```
SimulationRunner reads one trajectory frame
    │
    ▼
CVContext  ← raw frame data (coords, box, topology, metadata)
    │
    ▼  CVSuite.enrich_inputs(ctx)   ← YOUR CODE adds derived data
    │
    ▼
CVInputs   ← rich dataset (coords + topology + your custom data)
    │
    ├──▶ cv1.compute(inputs)  →  np.ndarray
    ├──▶ cv2.compute(inputs)  →  np.ndarray
    └──▶ cv3.compute(inputs)  →  np.ndarray
```

---

## `CVContext` — Raw Frame from the Runner

`CVContext` is created by `SimulationRunner` and passed to `CVSuite.enrich_inputs()`. You **read** from it, you don't create it.

```python
@dataclass
class CVContext:
    coords:    np.ndarray       # (N_atoms, 3) — atom positions in Å
    box:       float | None     # periodic box length in Å, or None
    key:       FrameKey         # step index, order parameter value, frame index
    topo:      SystemTopology   # molecule groups, atom types, cell info
    traj_path: Path | None      # path to the source .xyz trajectory file
```

**`FrameKey`** is a lightweight named tuple:
```python
@dataclass(frozen=True)
class FrameKey:
    step:    int    # RETIS step index (e.g. 1050)
    op:      float  # order parameter value for this frame
    ase_idx: int    # frame index within its trajectory file
```

---

## `CVInputs` — Rich Per-Frame Dataset for CVs

`CVInputs` is what `enrich_inputs()` builds and what every `cv.compute()` receives. It carries everything a CV might need.

```python
@dataclass
class CVInputs:
    # Always present:
    coords: np.ndarray          # (N_atoms, 3) — same as CVContext.coords
    topo:   SystemTopology      # same as CVContext.topo

    # Frame metadata:
    key:       FrameKey | None  # step, op, ase_idx
    box:       float | None     # cell size in Å
    traj_path: Path | None      # source trajectory file

    # ── Your custom data goes here ────────────────────────────────────────
    data: dict[str, Any]        # anything from enrich_inputs() lives here
    # e.g. data["reaction"]   → your ReactionCenter object
    #      data["neighborhood"] → your NeighborList object
    #      data["mulliken_charges"] → np.ndarray of partial charges

    # Results cache (advanced, opt-in):
    results: dict[str, np.ndarray]  # inter-CV dependencies (Ratio CVs etc.)
    flags:   dict[str, Any]         # quality flags from CV modules
```

### Accessing custom data in a CV

```python
def compute(self, inputs: CVInputs) -> np.ndarray:
    rc = inputs.data.get("reaction")   # None if not injected
    if rc is None:
        return np.array([np.nan])
    # use rc.O_d, rc.O_a, rc.Hs, etc.
```

---

## What to put in `data` vs. `topo`

| Quantity | Where it lives | Why |
|---|---|---|
| Atom types, atom groups by element | `topo.atom_groups` | Static — same for all frames |
| Molecule groupings (water triplets, H→O map) | `topo.molecule_groups` | Static — same for all frames |
| Cell size | `topo.cell_size` also `inputs.box` | Static, set at startup |
| Reaction center (changes per frame) | `inputs.data["reaction"]` | Dynamic — computed in `enrich_inputs()` |
| Neighbor list | `inputs.data["neighborhood"]` | Dynamic — expensive, computed once per frame |
| Mulliken charges | `inputs.data["mulliken_charges"]` | Dynamic — from aux file per frame |
| Wannier/HOMO centers | `inputs.data["homo_centers"]` | Dynamic — from aux file per frame |

---

## `math_utils.py` — General Utilities

Contains functions useful across many CV implementations:

```python
from CV_frame_data.math_utils import rational_switch

# Smooth switching function:  1 when x < x0,  0 when x >> x0
values = rational_switch(distances, x0=3.5, n=16, m=56)
```

---

## `examples/` — System-Specific Infrastructure

The `examples/` subfolder contains utilities that are specific to CP2K and the water+ion system. New users building a different system do not need these.

| File | Purpose |
|---|---|
| `reaction_center.py` | Finds the active O–H bond (longest bond) in a water system |
| `neighborhood.py` | Builds a neighbor list of water molecules around the RC |
| `wannier_mapper.py` | Assigns Wannier centers to heavy atoms under PBC |
| `load_cp2k_aux.py` | Parses CP2K auxiliary files (Mulliken, forces, dipole, HOMO centers) |

These are **reference implementations** — copy and adapt them for your own system's derived quantities.

---

## Adding Your Own Derived Quantities

You do **not** add new files to this folder unless you are adding a new general utility. System-specific derived quantities (equivalent to `reaction_center.py` for your system) go in `CV_frame_data/examples/my_system/` or alongside your suite file.

The pipeline does not read from this folder directly — your `CVSuite.enrich_inputs()` is the entry point that calls whatever utilities you need.
