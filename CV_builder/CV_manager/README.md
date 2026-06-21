# CV_manager — Define Your Collective Variables

This is one of **two places** you need to adapt for a new system. `CV_manager` is where you define:
- **Which CVs to compute** (the list of CV objects)
- **What per-frame derived data to inject** before CVs are computed (reaction center, neighbor lists, auxiliary files)

---

## How It Works

For every trajectory frame, the pipeline calls:

```
SimulationRunner
  └── CVSuite.enrich_inputs(ctx)   ← YOUR CODE: compute shared per-frame data
        └── CVSuite.build_cv_modules()  ← YOUR CODE: list of CV objects
              └── cv.compute(inputs)   ← called for every CV you registered
```

The result of `enrich_inputs()` is a `CVInputs` object — a rich data container holding coordinates, topology, and any custom data your CVs need. See [`cv_frame_data/README.md`](../CV_frame_data/README.md) for what `CVInputs` contains.

---

## Getting Started (New Users)

**Step 1.** Copy `minimal_suite.py` to your own file, for example `my_suite.py`.

**Step 2.** Edit `build_cv_modules()` to return the CVs you want:

```python
def build_cv_modules(self, topo: SystemTopology) -> list:
    return [
        AtomPairDistanceCV(name="ion_water_dist", pairs=[(42, 0)]),
    ]
```

**Step 3.** Point to it in `infretis.toml`:

```toml
[cv_modules]
module = "CV_builder/CV_manager/my_suite.py"
class  = "MySuite"
```

That's it. You don't need to touch anything else.

---

## Folder Structure

```
CV_manager/
├── base.py              ← CVSuite abstract base class. Read this first.
├── minimal_suite.py     ← Minimal template for new users. START HERE.
│
├── cvs/                 ← Generic, system-agnostic CV building blocks.
│   ├── cv_distance.py   ← Atom-pair distance CV. Copy as a template.
│   └── __init__.py
│
└── examples/            ← Full CP2K water+ion reference implementation.
    ├── cp2k_water_ion_suite.py  ← Full CVSuite for CP2K water+ion system.
    ├── cv_smooth_cn.py          ← Smooth coordination number CV.
    ├── cv_wannier_complex.py    ← Wannier-center based CVs.
    ├── cv_smart_wire.py         ← Smart H-bond wire CV.
    ├── cv_wire_derived.py       ← Wire compression and ratio CVs.
    ├── cv_ratios.py             ← Symmetric ratio CVs.
    ├── evaluate_wire.py         ← Wire analysis script.
    ├── debug_wannier_mapper.py  ← Wannier center debugging.
    ├── test_smart_wire.py       ← Wire CV tests.
    └── legacy_CVs/              ← Older CV implementations (kept for reference).
```

---

## The CVSuite Interface

Your `CVSuite` subclass must implement two methods:

### `build_cv_modules(topo) → list`

Called **once** at startup. Returns the list of CV objects that will be computed for every frame.

```python
def build_cv_modules(self, topo: SystemTopology) -> list:
    return [
        AtomPairDistanceCV(name="bond_length", pairs=[(0, 1)]),
    ]
```

Each CV object must have:
- `name: str` — unique identifier, used as the HDF5 dataset name
- `labels: tuple[str, ...]` — column names for the output array
- `compute(inputs: CVInputs) → np.ndarray` — returns a 1D float array

### `enrich_inputs(ctx: CVContext) → CVInputs`

Called **once per frame** before any CV is computed. Use this to calculate expensive shared quantities (neighbor lists, reaction centers, auxiliary data) that multiple CVs need.

```python
def enrich_inputs(self, ctx: CVContext) -> CVInputs:
    # ctx has: ctx.coords, ctx.box, ctx.topo, ctx.key, ctx.traj_path
    
    my_rc = compute_my_reaction_center(ctx.coords, ctx.topo)
    
    return CVInputs(
        coords=ctx.coords,
        topo=ctx.topo,
        key=ctx.key,
        box=ctx.box,
        traj_path=ctx.traj_path,
        data={
            "reaction": my_rc,   # CVs access this via inputs.data["reaction"]
        }
    )
```

If your CVs don't need any extra data, the default base implementation just passes `ctx` through. You don't need to override `enrich_inputs()` in that case.

---

## Writing a New CV

Copy `cvs/cv_distance.py` as a template. Every CV is a simple class:

```python
from dataclasses import dataclass
import numpy as np
from CV_frame_data.context import CVInputs

@dataclass
class MyCV:
    name: str = "my_cv"
    labels: tuple = ("my_value",)

    def compute(self, inputs: CVInputs) -> np.ndarray:
        # inputs.coords   → (N_atoms, 3) float64 array
        # inputs.topo     → SystemTopology with atom_groups, etc.
        # inputs.data     → dict of extra per-frame data from enrich_inputs()
        # inputs.box      → cell size in Å (for PBC), or None
        
        value = float(np.linalg.norm(inputs.coords[0] - inputs.coords[1]))
        return np.array([value])
```

Register it in your suite:

```python
def build_cv_modules(self, topo):
    return [MyCV()]
```

---

## The Reaction Center — An Anchor for Comparable CVs

In RETIS, trajectories from different paths all describe the same rare event but may involve different specific atoms. A **Reaction Center (RC)** is a per-frame anchor — a reference atom or group selected by a geometric/electronic criterion — that makes CVs from different trajectories physically comparable.

**Example:** In a proton-transfer reaction, the RC is always the water triplet (O_donor, H*, O_acceptor) with the longest O–H bond. All CVs (distances, coordination numbers, Wannier CV) are computed relative to these atoms. Even though the specific atoms change between trajectories, the CV always describes the same physical event.

Inject it in `enrich_inputs()`:

```python
data={"reaction": my_rc_object}
```

Pull it in your CV:

```python
rc = inputs.data.get("reaction")
if rc is None:
    return np.array([np.nan])
```

See `examples/cp2k_water_ion_suite.py` for a full worked example.

---

## Optional: Per-Trajectory Caching

If you need to parse large auxiliary files (forces, charges, Wannier centers), use `on_trajectory_opened()`:

```python
def on_trajectory_opened(self, traj_path: Path, n_atoms: int) -> None:
    """Called once when a new trajectory file is opened."""
    self._cached_charges = parse_mulliken(traj_path)
```

The cache is reset automatically for each new trajectory. Individual frame data is then served from the cache inside `enrich_inputs()`.
