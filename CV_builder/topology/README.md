# topology — Define Your Atomic System

This is one of **two places** you need to adapt for a new chemical system. `topology` tells the pipeline what atoms your system contains, how they are grouped into molecules, and any structural relationships needed by your CVs.

---

## Minimum Requirements for New Users

For most systems, you just need a topology file and three TOML lines:

```toml
[topology]
system    = "xyz_data"       # use the built-in XYZ reader
file      = "initial.xyz"    # path to a standard .xyz file
cell_size = 12.5             # periodic box length in Å (optional)
```

The built-in `XYZTopologyReader` reads your `.xyz` file and creates:
- A list of all atom types and indices
- Groups of atoms by element (`atom_groups["O"]`, `atom_groups["H"]`, etc.)

That's all the information a simple distance-based CV needs.

> **Does topology need coordinates?**
> No. `XYZTopologyReader` only reads the element type per atom — not positions. Positions change every frame and are read from the trajectory. The topology just gives the stable atom ordering.

---

## Folder Structure

```
topology/
├── base.py         ← MoleculeSystem abstract base class. START HERE if building a custom system.
├── topology.py     ← SystemTopology dataclass — the output of build_topology().
├── xyz_reader.py   ← Built-in reader for .xyz files. Recommended for new users.
│
└── examples/       ← System-specific implementations (not needed by most users).
    ├── cp2k_water_ion.py     ← CP2K water box with one monovalent ion.
    ├── ref_loader.py         ← Loads an equilibrium reference frame from a CP2K trajectory.
    ├── lammps_reader.py      ← LAMMPS data file reader (skeleton).
    └── debug_nuclear_ids.py  ← Verification script for topology stability.
```

---

## What `SystemTopology` Contains

`build_topology()` must return a `SystemTopology` object. At minimum it needs:

```python
SystemTopology(
    n_atoms   = 193,
    names_ref = np.array(["O0", "H1", "H2", ...]),     # unique name per atom
    atom_groups = {
        "O": np.array([0, 3, 6, ...]),   # indices of all oxygen atoms
        "H": np.array([1, 2, 4, 5, ...]),
    },
    cell_size = 12.5,   # optional
)
```

More complex systems can also add:
- `molecule_groups` — e.g. water triplets `(H1, O, H2)` or ion index
- `molecule_groups_labels` — human-readable labels for each group
- `custom` — free dict for any other system-specific info

See `topology.py` for all fields.

---

## Building a Custom System

If you need more than element grouping — for example, identifying which H atoms belong to which O, tracking a specific ion, or grouping atoms into residues — subclass `MoleculeSystem`:

**Step 1.** Create a new file, e.g. `topology/my_system.py`:

```python
from topology.base import MoleculeSystem
from topology.topology import SystemTopology
import numpy as np

class MySystem(MoleculeSystem):

    def __init__(self, topology_file: str, cell_size: float = 0.0):
        self.topology_file = topology_file
        self.cell_size = cell_size

    def build_topology(self, ref_frame=None) -> SystemTopology:
        # Parse your file / ref_frame here
        # Return a SystemTopology
        return SystemTopology(
            n_atoms=...,
            names_ref=np.array([...]),
            atom_groups={"C": np.array([...])},
            cell_size=self.cell_size,
        )

    def describe(self) -> str:
        return f"MySystem(file={self.topology_file})"
```

**Step 2.** Reference it in `infretis.toml`:

```toml
[topology]
module = "topology/my_system.py"
class  = "MySystem"
topology_file = "my_topology.dat"
cell_size     = 20.0
```

All keys inside `[topology]` (except `module` and `class`) are passed as keyword arguments to your class `__init__`.

---

## The `build_topology()` Contract

- Called **once** at startup, before any trajectory is read
- Must return a `SystemTopology` that is **valid for all frames** (atom ordering is stable for an ∞RETIS run)
- Use `ref_frame` (an optional reference DataFrame) if you need reference positions to assign internal structure (e.g. H→O nearest-neighbour assignment)
- `SimulationRunner` always calls `build_topology()` with no arguments — your class `__init__` is where you receive parameters

---

## Built-in Systems Reference

| TOML `system` key | Class | Notes |
|---|---|---|
| `xyz_data` | `XYZTopologyReader` | `.xyz` file, element groups only. **Start here.** |
| `lammps_data` | `LAMMPSDataReader` | LAMMPS data file (skeleton — adapt as needed). |
| `cp2k_water_ion` | `CP2KWaterIonSystem` | CP2K water box + monovalent ion. Full molecule grouping, H→O assignment. |

Custom systems: use `module = "path/to/my_system.py"` and `class = "MyClass"`.

---

## Verifying Your Topology

Run the debug script to check that your atom ordering is stable across trajectories:

```bash
cd CV_builder
python topology/examples/debug_nuclear_ids.py
```

This checks that O–H bond distances are physically sensible across multiple trajectory frames — a fast sanity check that your topology maps atoms to the right positions.
