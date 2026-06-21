# Collective Variable Predictions OF Ellusive Transitions

`cv-profet` is a modular pipeline that reads ∞RETIS trajectory data and computes Collective Variables (CVs) across all paths and grid points. It is designed from the ground up to be **system-agnostic** — the physics of your specific MD engine, molecule type, or trajectory format lives in two well-defined places that you control, while the core engine never needs to change.

---

## Minimum Requirements

To run a new analysis you need four things:

| File | Description |
|---|---|
| `infretis.toml` | Configuration file. Copy `CV_builder/infretis_example.toml` and edit it. |
| `load/` | Folder of RETIS trajectory steps produced by ∞RETIS. |
| `infretis_data.txt` | WHAM path weights from the ∞RETIS run. |
| A topology file | e.g. `initial.xyz` — just atom types, no positions needed. |

---

## Installation (pip)

Run these commands from the repository root (the folder that contains `CV_builder/`, `ppa/`, and `run_CV_builder.slurm`):

```bash
# Option A: one-command installer
bash install.sh

# Option B: direct pip install
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

This installs the public command-line launcher:

- `profet`

Run all workflows through `profet` from any directory. It auto-detects the `cv-profet` root.

If your environment is unusual (for example, HPC modules or wrappers), set:

```bash
export CV_PROFET_ROOT=/absolute/path/to/cv-profet
```

## Quick Start

```bash
# Run from the root of your analysis directory (where infretis.toml lives)
python CV_builder/main.py --toml infretis.toml

# Re-run and overwrite existing output
python CV_builder/main.py --toml infretis.toml --overwrite
```

Results are written to your configured output directory (default: `CVs/`).

If you installed with pip, run the same workflow with `profet`:

```bash
profet build --toml infretis.toml
profet build --toml infretis.toml --overwrite
```

## SLURM-style Run (matching run_CV_builder.slurm)

After installation, your run commands can be exactly this style:

```bash
profet build \
    --toml pp.toml \
    --h5-input simulation.h5 \
    --load-dir load \
    --data test.txt

profet screen --toml pp.toml
profet diagnose --toml pp.toml
```

This matches the command pattern used in `run_CV_builder.slurm`.

---

## Using Input HDF5 (order.txt + traj.txt)

If your RETIS step data is stored in a single HDF5 file, pass it with:

```bash
python CV_builder/main.py --toml infretis.toml --h5-input /path/to/input.h5
```

For your current layout, this is typically:

```bash
python CV_ana/CV_builder/main.py --toml pp.toml --h5-input 600K_int.h5 --load-dir load --data infretis_data.txt
```

Expected datasets inside the input HDF5 file:

```
<step>/order.txt
<step>/traj.txt
```

Where:

- `<step>` is a numeric step ID group (for example `0`, `1`, `2`, ...).
- `order.txt` stores order-parameter time series used to detect crossing frames.
- `traj.txt` stores frame mapping fields used to resolve trajectory file and ASE index.

Important:

- `--h5-input` provides `order.txt` and `traj.txt` lookup data.
- Actual trajectory files are still loaded from `--load-dir` (for example `load/<step>/accepted/...`).
- WHAM weights are still read from `--data` (default: `infretis_data.txt`).

---

## Folder Overview

```
CV_builder/
├── main.py                 ← Entry point. Reads TOML, builds runner, runs.
├── builder.py              ← Wires TOML config → MoleculeSystem + CVSuite.
├── simulationrunner.py    ← Core loop: iterate steps → read frames → compute CVs.
├── config.py               ← TOML config dataclasses (GridDefinition, Paths, …).
├── wham_weights.py         ← WHAM weight loading.
├── infretis_example.toml   ← Minimal example TOML to copy and adapt.
│
├── topology/        ← YOUR SYSTEM: atom topology and molecule definitions.
│   ├── README.md           ← Start here if you are adapting to a new system.
│   ├── base.py             ← MoleculeSystem abstract base class.
│   ├── topology.py         ← SystemTopology dataclass (output of build_topology).
│   ├── xyz_reader.py       ← Built-in reader for .xyz topology files (new users start here).
│   └── examples/           ← System-specific implementations (CP2K water+ion, LAMMPS, …).
│
├── CV_manager/              ← YOUR CVs: define which CVs to compute and how.
│   ├── README.md           ← Read this to add or modify CVs.
│   ├── base.py             ← CVSuite abstract base class.
│   ├── minimal_suite.py    ← Minimal template for new users. Start here.
│   ├── cvs/                ← Generic, system-agnostic CV building blocks.
│   │   └── cv_distance.py  ← Atom-pair distance CV (copy this as a template).
│   └── examples/           ← Full CP2K water+ion suite and Wannier CVs.
│
├── cv_frame_data/          ← Core data layer: CVContext, CVInputs, and utilities.
│   ├── README.md           ← Explains CVContext and CVInputs (read this!).
│   ├── context.py          ← CVContext and CVInputs dataclasses.
│   ├── math_utils.py       ← Rational switching functions, general math.
│   └── examples/           ← System-specific utilities (reaction center, Wannier mapper, …).
│
└── storage/                ← HDF5 output writer. No user changes needed.
```

> **Note on the name `CV_frame_data`:** The folder is currently named `CV_frame_data`. We suggest renaming it to `cv_frame_data` or `cv_core` to better reflect its purpose. See the README inside for details.

---

## What You Need to Adapt for Your System

You only ever need to touch **two files/classes**. Everything else is handled by the core engine.

### 1. `topology/` — Define Your Atoms

Create a class that tells the pipeline what atoms your system contains and how they are grouped.

- For most users, the built-in `XYZTopologyReader` is enough — just provide an `.xyz` topology file.
- For more complex systems (molecules with internal structure, ion tracking, etc.), subclass `MoleculeSystem` and implement `build_topology()`.

See → [`topology/README.md`](topology/README.md)

### 2. `CV_manager/` — Define Your CVs

Create a `CVSuite` subclass that lists the CVs you want to compute and optionally injects per-frame derived quantities (reaction center, neighbor lists, auxiliary data).

- Copy `CV_manager/minimal_suite.py` as a starting point.
- For advanced use (Wannier centers, reaction-center-anchored CVs), see `CV_manager/examples/`.

See → [`CV_manager/README.md`](CV_manager/README.md)

---

## TOML Configuration Reference

The `[topology]` section selects your `MoleculeSystem`:

```toml
[topology]
system    = "xyz_data"        # "xyz_data" | "lammps_data" | "cp2k_water_ion" | custom
file      = "initial.xyz"     # path to topology file  
cell_size = 12.5              # periodic box length in Å (optional)
```

The `[cv_modules]` section selects your `CVSuite`:

```toml
[cv_modules]
module = "CV_builder/CV_manager/minimal_suite.py"   # path to your CVSuite file
class  = "MinimalCVSuite"                          # class name inside that file
```

See `infretis_example.toml` for a fully annotated configuration.

---

## Output

Results are stored in HDF5 files under your output directory:

```
CVs/
└── <n_grid>/
    ├── .matrix_meta.ready        ← run metadata (JSON)
    └── steps/
        └── step_<N>.h5           ← per-step HDF5 file
            ├── <grid_id>/<cv_name>/values   ← (N_frames, K) float64
            ├── <grid_id>/<cv_name>/labels   ← CV column names
            └── <grid_id>/<cv_name>/meta     ← step, op, ase_idx, weight
```

Use the companion aggregation tools in `main.py` (`--agg-*` flags) to merge step files into a combined matrix for analysis.

---

## Extending the Pipeline

| What to add | Where to put it |
|---|---|
| New topology format (GROMACS, AMBER, …) | New file in `topology/examples/`, subclass `MoleculeSystem` |
| New CV | New file in `CV_manager/cvs/`, add to your `CVSuite.build_cv_modules()` |
| New per-frame derived quantity | Add to `CVSuite.enrich_inputs()` → inject into `CVInputs.data` |
| New auxiliary file parser | New file in `cv_frame_data/examples/` |

The core files (`simulationrunner.py`, `builder.py`, `storage/`) **never need to change** for a new system.
