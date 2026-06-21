"""
simulationrunner.py — Core runner for the CV_builder pipeline.

Iterates over RETIS steps and grid points, loads trajectory frames,
computes collective variables, and writes results to HDF5.

Control flow:
    SimulationRunner.run()
        └── _process_step(step)
                └── _process_grid_point(step, op, grid_id, writer)
                        ├── _load_frame(step_dir, frames)  → (ase_idx, traj_path)
                        ├── build CVInputs from SystemTopology
                        └── _compute_and_write(cv_inputs, writer, grid_id, op)

NOTE FOR NEW USERS
------------------
This runner is system-agnostic. All knowledge about molecule identities
(which H belongs to which O, where the ion is, etc.) lives in the
MoleculeSystem / SystemTopology objects passed at construction time.

To adapt this pipeline to a new chemical system:
  1. Subclass MoleculeSystem (in topology/base.py). TODO: implement topology file
  2. Implement build_topology() to return a valid SystemTopology.
  3. Run topology/debug_nuclear_ids.py to verify your topology.
  4. Pass mol_system=YourSystem(...) to SimulationRunner in config.py / infretis.toml
"""
from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read as ase_read

from config import (
    GridDefinition,
    OutputConfig,
    Paths,
    PredictivePowerSettings,
    CVMatSettings,
)
from CV_frame_data.context import CVContext, CVInputs, FrameKey
from CV_frame_data.math_utils import box_lengths
from CV_manager.base import CVSuite
from topology.base import MoleculeSystem
from topology.topology import SystemTopology
from storage.h5_writer import H5Writer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SimulationRunner
# ---------------------------------------------------------------------------

class SimulationRunner:
    """
    Iterates over RETIS steps and grid points, builds CVInputs per frame,
    runs all CV modules, and writes results to per-step HDF5 files.
    """

    def __init__(
        self,
        paths:      Paths,
        config:     dict,
        interfaces: List[float],
        cvmat:      CVMatSettings,
        grid:       GridDefinition,
        pred_power: PredictivePowerSettings,
        all_types:  set[str],
        weights,                           # TrajWeights
        mol_system: MoleculeSystem,
        out_cfg:    OutputConfig | None = None,
        cv_suite:   CVSuite | None = None,
        h5_input:   Path | None = None,
    ) -> None:
        self.paths      = paths
        self.config     = config
        self.interfaces = interfaces
        self.cvmat      = cvmat
        self.grid       = grid
        self.pred_power = pred_power
        self.all_types  = all_types
        self.weights    = weights
        self.out_cfg    = out_cfg or OutputConfig()
        self.cv_suite   = cv_suite
        self.h5_input   = h5_input or paths.h5_input
        self.accept_missing = bool(self.config.get("ppa", {}).get("accept_missing", False))

        # ── Molecule system & topology ────────────────────────────────────
        # The MoleculeSystem object is responsible for knowing how to
        # build a SystemTopology. This keeps SimulationRunner fully system-
        # agnostic: XYZTopologyReader builds from a .xyz file, CP2KWaterIonSystem
        # builds from a reference trajectory frame, etc. No topology logic lives here.
        self.mol_system: MoleculeSystem = mol_system
        log.info("MoleculeSystem: %s", self.mol_system.describe())
        self.topo: SystemTopology = mol_system.build_topology()
        log.info("\n%s", self.topo.describe())

        # ── CV modules ────────────────────────────────────────────────────
        self.cv_modules = self.cv_suite.build_cv_modules(self.topo)
        log.info("CVSuite defines %d modules", len(self.cv_modules))
        log.info("accept_missing: %s", self.accept_missing)

        # Check if any module requires the cross-module results cache
        self._needs_results_cache = any(
            getattr(cv, "requires_results_cache", False) for cv in self.cv_modules
        )
        self._warned_non_diagonal_cell = False

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def run(self) -> None:
        """
        Outer loop: iterate over all RETIS steps that have a valid WHAM weight.

        Parallelism:
            Set CV_RANK and CV_WORLD (or SLURM_PROCID / SLURM_NTASKS) to shard
            steps across workers.  Each worker writes to its own step_*.h5 files
            so there is no HDF5 contention.

        Output layout:
            <output_dir>/<n_grid>/
                steps/step_<N>.h5   ← per-step raw CV files
                sorted/             ← aggregated part files
                CVmat.h5            ← final merged matrix
        """
        from storage.aggregator import shard_list

        rank  = int(os.environ.get("CV_RANK",  os.environ.get("SLURM_PROCID",  "0")))
        world = int(os.environ.get("CV_WORLD", os.environ.get("SLURM_NTASKS",  "1")))

        cv_root   = self.paths.output_dir / str(self.cvmat.n_grid)
        steps_dir = cv_root / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)

        if rank == 0:
            self._write_matrix_meta(cv_root)

        all_steps = self.weights.retis_steps
        my_steps  = shard_list(all_steps, rank, world)
        n_mine    = len(my_steps)

        log.info("CV step loop: rank %d/%d → processing %d of %d steps",
                 rank, world, n_mine, len(all_steps))

        for i, step in enumerate(my_steps, 1):
            log.info("Processing step %s  (%d / %d)", step, i, n_mine)
            self._process_step(step, steps_dir)


    # -----------------------------------------------------------------------
    # Step-level processing
    # -----------------------------------------------------------------------

    def _write_matrix_meta(self, cv_root: Path) -> None:
        """Rank-0 writes the shared matrix-metadata HDF5; others wait for it."""
        rank     = int(os.environ.get("SLURM_PROCID", os.environ.get("PMI_RANK", "0")))
        sentinel = cv_root / ".matrix_meta.ready"

        # Use the first available molecule group labels (engine-agnostic)
        # Each user's MoleculeSystem populates molecule_groups_labels with its own naming scheme.
        pair_labels: list[str] = []
        for labels in self.topo.molecule_groups_labels.values():
            if labels:
                pair_labels = labels
                break

        if rank == 0:
            import json
            import numpy as _np

            class _NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, _np.integer):
                        return int(obj)
                    if isinstance(obj, _np.floating):
                        return float(obj)
                    return super().default(obj)

            meta = {
                "n_grid":      len(self.grid.grid_idx),
                "grid_ids":    list(self.grid.grid_idx),
                "pair_labels": pair_labels,
            }
            try:
                sentinel.write_text(json.dumps(meta, indent=2, cls=_NpEncoder))
                log.info("Wrote matrix meta sentinel: %s", sentinel)
            except Exception as exc:
                log.warning("Could not write matrix meta sentinel: %s", exc)
        else:
            for _ in range(40):  # up to ~20 s
                if sentinel.exists():
                    break
                time.sleep(0.5)

    def _process_step(self, step: int | str, cv_root: Path) -> None:
        """Process all grid points for one RETIS step."""
        writer: H5Writer | None = None

        try:
            for op, grid_id in zip(self.grid.grid_pts, self.grid.grid_idx):
                step_dir = self.paths.load_dir / str(step)
                frames, lambda_max = extract_frames(
                    step_dir,
                    grid_idx=grid_id,
                    grid_points=op,
                    h5_input=self.h5_input,
                    step=str(step),
                )
                if frames is None:
                    log.debug("Step %s, grid_id %s (OP=%.4f): no crossing — skip", step, grid_id, op)
                    continue

                writer = self._process_grid_point(
                    step=step, op=op, grid_id=grid_id,
                    step_dir=step_dir, frames=frames, lambda_max=lambda_max,
                    steps_dir=cv_root,
                    writer=writer,
                )
        finally:
            if writer is not None:
                # Store raw order.txt for Landau free energy calculations
                order_path = self.paths.load_dir / str(step) / "order.txt"
                order_exists = order_path.exists()
                if self.h5_input is not None:
                    with h5py.File(self.h5_input, "r") as h5f:
                        order_exists = f"{step}/order.txt" in h5f

                if order_exists:
                    try:
                        order_df = None
                        if self.h5_input is not None:
                            with h5py.File(self.h5_input, "r") as h5f:
                                if _step_exists_in_h5(h5f, str(step)):
                                    # Read as numpy array from h5: shape is (N, 2)
                                    order_array = _h5_read_dataset(h5f, f"{step}/order.txt")
                                    if order_array is not None:
                                        order_df = pd.DataFrame(order_array, columns=["Time", "Orderp"])
                        else:
                            order_df = pd.read_csv(
                                order_path, skiprows=2, sep=r"\s+", header=None,
                            )
                        
                        if order_df is not None:
                            writer._ensure_open()
                            if "order_txt" not in writer.h5:
                                writer.h5.create_dataset(
                                    "order_txt",
                                    data=order_df.to_numpy(dtype=np.float64),
                                    compression="gzip",
                                )
                    except Exception as e:
                        log.warning("Failed to store order.txt for step %s: %s", step, e)
                writer.close()

    def _process_grid_point(
        self,
        step:       int | str,
        op:         float,
        grid_id:    int,
        step_dir:   Path,
        frames:     list[int],
        lambda_max: float,
        steps_dir:  Path,
        writer:     H5Writer | None,
    ) -> H5Writer:
        """
        Process a single (step, grid_id) point:
          1. Load the trajectory frame.
          2. Build CVInputs.
          3. Compute and write all CVs.
        """

        frame_result = self._load_frame(step_dir, frames)
        if frame_result is None:
            return writer
        
        ase_idx, traj_path = frame_result

        if not traj_path.exists():
            msg = f"Trajectory file not found: {traj_path}"
            if self.accept_missing:
                log.warning("%s — skipping frame", msg)
                return writer
            raise FileNotFoundError(msg)

        try:
            atom_obj  = ase_read(traj_path, index=ase_idx)
        except FileNotFoundError:
            if self.accept_missing:
                log.warning("Trajectory file not found while reading %s — skipping frame", traj_path)
                return writer
            raise
        assert isinstance(atom_obj, Atoms)
        positions = atom_obj.get_positions()
        symbols   = atom_obj.get_chemical_symbols()
        frame_box = self._frame_box(atom_obj)

        # ── Raw unmapped coordinates ───────────────────────────────────────
        # We pass coordinates directly using original atom IDs to avoid mapping issues
        # during H+ jumps or permutations.
        coords = positions

        # ── CV Context & Enrichment ──────────────────────────────────────────
        key = FrameKey(step=step, op=float(op), ase_idx=ase_idx)
        ctx = CVContext(
            coords=coords,
            box=frame_box,
            key=key,
            topo=self.topo,
            traj_path=traj_path,
        )
        
        cv_inputs = self.cv_suite.enrich_inputs(ctx)

        # ── Per-frame debug log ───────────────────────────────────────────
        if log.isEnabledFor(logging.DEBUG):
            debug_str = self.cv_suite.get_debug_log(cv_inputs)
            log.debug(
                "[FRAME_DEBUG] step=%s op=%.4f ase_idx=%d\n"
                "  Coords  : n_atoms=%d (fixed topology, no ICP permutation)\n"
                "  %s",
                step, op, ase_idx,
                coords.shape[0], debug_str,
            )

        # ── Lazy-open writer and initialise step file ─────────────────────
        if writer is None:
            writer = H5Writer(steps_dir, step=step, out_cfg=self.out_cfg)
            writer.init_step_file(
                names_ref=self.topo.names_ref,
                wham_ngrid=int(self.cvmat.n_grid),
                cell_size=float(self._h5_cell_size(frame_box)),
                n_atoms=int(self.topo.n_atoms),
                weight=self.weights.weights.get(step, np.nan),
                extra_attrs={"lambda_max": lambda_max},
            )

        self._compute_and_write(cv_inputs, writer, grid_id=int(grid_id), op=op,
                                step=step, ase_idx=ase_idx, err=0.0, flags={}, jacc=1.0)
        return writer

    def _frame_box(self, atom_obj: Atoms) -> np.ndarray | float | None:
        """
        Return per-frame box from ASE Atoms.

        For periodic frames, returns only diagonal cell elements [a, b, c].
        If off-diagonal cell terms are non-zero, logs a warning once and still
        uses the diagonal elements.
        """
        fallback = float(self.topo.cell_size) if self.topo.cell_size and self.topo.cell_size > 0 else None

        pbc = np.asarray(atom_obj.get_pbc(), dtype=bool)
        if not pbc.any():
            return fallback

        cell = np.asarray(atom_obj.cell.array, dtype=float)
        if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
            return fallback

        diag = np.diag(cell).astype(float)
        off_diag = cell - np.diag(diag)
        if np.any(np.abs(off_diag) > 1e-12) and not self._warned_non_diagonal_cell:
            log.warning(
                "Non-zero off-diagonal cell terms detected; using only diagonal lengths [a,b,c]=%s.",
                diag.tolist(),
            )
            self._warned_non_diagonal_cell = True

        return diag

    def _h5_cell_size(self, frame_box: np.ndarray | float | None) -> float:
        lengths = box_lengths(frame_box)
        if lengths is not None and np.all(np.isfinite(lengths)):
            return float(np.mean(lengths))
        return float(self.topo.cell_size)



    # -----------------------------------------------------------------------
    # Frame I/O
    # -----------------------------------------------------------------------

    def _load_frame(
        self,
        step_dir: Path,
        frame_info: list[int],
    ) -> tuple[int, Path] | None:
        """
        Resolve (ase_idx, traj_path) from a [grid_idx, frame_idx] pair
        by reading the step's traj.txt.
        
        Returns (ase_idx, traj_path) or None if the data is not available.
        """
        traj_txt = step_dir / "traj.txt"
        
        try:
            if self.h5_input is not None:
                with h5py.File(self.h5_input, "r") as h5f:
                    # Check if step exists
                    if not _step_exists_in_h5(h5f, step_dir.name):
                        log.debug("Step %s not found in h5_input", step_dir.name)
                        return None
                    
                    # Read as structured array from h5: fields are (time, trajfile, index, vel)
                    traj_array = _h5_read_dataset(h5f, f"{step_dir.name}/traj.txt")
                    if traj_array is None:
                        log.debug("traj.txt not found for step %s in h5_input", step_dir.name)
                        return None
                    
                    # Convert structured array to DataFrame
                    traj_df = pd.DataFrame({
                        "Step": traj_array["time"],
                        "Filename": traj_array["trajfile"],
                        "index": traj_array["index"],
                        "vel": traj_array["vel"],
                    })
            else:
                if not traj_txt.exists():
                    log.debug("traj.txt not found in %s", step_dir)
                    return None
                
                traj_df = pd.read_csv(
                    traj_txt, skiprows=2, sep=r"\s+",
                    names=["Step", "Filename", "index", "vel"],
                )
        except (KeyError, OSError, pd.errors.EmptyDataError) as e:
            log.debug("Error reading traj.txt for step %s: %s", step_dir.name, e)
            return None
        
        log.debug("Frame info: %s", frame_info)
        _, frame_idx = frame_info

        row = traj_df[traj_df["Step"] == frame_idx]
        if row.empty:
            log.debug("Frame %d not found in traj.txt under %s", frame_idx, step_dir)
            return None

        raw = row["Filename"].iloc[0]
        filename  = raw.decode() if isinstance(raw, bytes) else str(raw)
        ase_index = int(row["index"].iloc[0])
        # Normalize away any '..' components (e.g. "../../10.traj" stored in
        # traj.txt is relative to the accepted/ subdir, but accepted/ may not
        # exist on disk — normpath resolves the '..' purely as string ops so
        # that the resulting path points directly to the actual file).
        traj_path = Path(os.path.normpath(step_dir / "accepted" / filename))
        log.debug("Frame info: %s  frame_idx=%d  filename=%s  ase_index=%d  traj_path=%s",
                  step_dir, frame_idx, filename, ase_index, traj_path)
        return ase_index, traj_path

    # -----------------------------------------------------------------------
    # CV compute + write
    # -----------------------------------------------------------------------

    def _compute_and_write(
        self,
        cv_inputs: CVInputs,
        writer:    H5Writer,
        grid_id:   int,
        op:        float,
        step:      int | str,
        ase_idx:   int,
        err:       float,
        flags:     dict,
        jacc:      float,
    ) -> None:
        """Run all CV modules and write outputs that pass the output filter."""
        for cv in self.cv_modules:
            if not self.out_cfg.should_write(cv.name):
                log.debug("CV '%s' excluded by output config — skipping", cv.name)
                continue

            if writer.has_frame(grid_id, cv, step=step, ase_idx=ase_idx):
                log.debug("CV '%s' grid_id=%s already written — skipping", cv.name, grid_id)
                continue

            writer.ensure_cv_group(grid_id, cv)
            vals_raw = cv.compute(cv_inputs)
            vals = np.asarray(vals_raw)
            if vals.ndim == 0:
                vals = vals.reshape(1)
            if vals.dtype == object or not np.issubdtype(vals.dtype, np.number):
                try:
                    vals = vals.astype(float)
                except Exception as exc:
                    raise TypeError(
                        f"CV '{cv.name}' returned non-numeric output of type {type(vals_raw).__name__}. "
                        "Expected a numeric array-like."
                    ) from exc

            # Conditional cache for downstream use (e.g. by RatioCV)
            if self._needs_results_cache:
                cv_inputs.results[cv.name] = vals

            if log.isEnabledFor(logging.DEBUG) or self.out_cfg.debug_cv_output:
                self._log_cv_debug(cv.name, vals)

            writer._ensure_dsets(grid_id, cv, K=vals.shape[0])
            writer.append(
                grid_id, cv, vals,
                meta={
                    "step": step, "op": float(op), "ase_idx": ase_idx,
                    "err": err, "flags": flags, "jaccard": jacc,
                },
            )

    @staticmethod
    def _log_cv_debug(name: str, vals: np.ndarray) -> None:
        """Log shape, range, and NaN/Inf statistics for a CV output."""
        arr = np.asarray(vals)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            rng = f"range [{finite.min():.4f}, {finite.max():.4f}]  mean={finite.mean():.4f}"
        else:
            rng = "range [all inf/NaN]"
        log.debug(
            "%-35s shape=%-15s  %s  NaNs=%d  Infs=%d",
            name, str(arr.shape), rng,
            int(np.isnan(arr).sum()), int(np.isinf(arr).sum()),
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _h5_text_dataset_as_buffer(h5_file: h5py.File, dataset_path: str) -> io.StringIO:
    """Return an in-memory text stream for a line-array text dataset."""
    if dataset_path not in h5_file:
        return None
    return h5_file[dataset_path][()]

def _h5_read_dataset(h5_file: h5py.File, dataset_path: str) -> Any:
    """
    Read a dataset from HDF5 file. Handles both structured arrays and regular arrays.
    Returns the raw numpy array or None if not found.
    """
    if dataset_path not in h5_file:
        return None
    return h5_file[dataset_path][()]

def _step_exists_in_h5(h5_file: h5py.File, step: str) -> bool:
    """Check if a step exists in the h5_input file."""
    return str(step) in h5_file

def extract_frames(
    directory: Union[str, Path],
    grid_idx:  Union[int, "np.integer"],
    grid_points: float,
    h5_input: Optional[Path] = None,
    step: Optional[str] = None,
) -> Tuple[Optional[list[int]], float]:
    """
    Find the first frame where the order parameter crosses `grid_points`.

    Reads `order.txt` from the given RETIS step directory or HDF5 file.
    Format: 2 header lines, then whitespace-separated columns.
      Column 0 = Time  (integer frame index)
      Column 1 = Orderp  (order parameter value)
      Columns 2+ = optional, engine-specific — ignored.

    Returns ([grid_idx, frame_idx], lambda_max) or (None, lambda_max).
    If step doesn't exist in h5_input, returns (None, 0.0) and logs a debug message.
    """
    order_file = os.path.join(directory, "order.txt")
    if h5_input is not None:
        if step is None:
            raise ValueError("extract_frames requires 'step' when h5_input is provided")
        try:
            with h5py.File(h5_input, "r") as h5f:
                # Check if step exists in h5_input
                if not _step_exists_in_h5(h5f, step):
                    log.debug("Step %s not found in h5_input — skipping", step)
                    return None, 0.0
                
                # Check if order.txt exists in step
                order_buffer = _h5_text_dataset_as_buffer(h5f, f"{step}/order.txt")
                if order_buffer is None:
                    log.debug("order.txt not found for step %s in h5_input — skipping", step)
                    return None, 0.0

                if not isinstance(order_buffer, np.ndarray):
                    if order_buffer.getvalue() == "":
                        log.debug("order.txt not found for step %s in h5_input — skipping", step)
                        return None, 0.0

                    order_df = pd.read_csv(
                        order_buffer,
                        skiprows=2,
                        usecols=[0, 1], names=["Time", "Orderp"],
                    )
                # Read as numpy array from h5: shape is (N, 2) with columns [time, orderp]
                order_array = _h5_read_dataset(h5f, f"{step}/order.txt")
                if order_array is None:
                    log.debug("order.txt not found for step %s in h5_input — skipping", step)
                    return None, 0.0
                
                # Convert to DataFrame: columns are time and orderp
                order_df = pd.DataFrame(order_array, columns=["Time", "Orderp"])
        except (KeyError, OSError, ValueError) as e:
            log.debug("Error reading step %s from h5_input: %s — skipping", step, e)
            return None, 0.0
    else:
        if not os.path.exists(order_file):
            log.debug("order.txt not found in %s — skipping", directory)
            return None, 0.0
        # Only read the first two columns — Time and Orderp — regardless of
        # how many extra engine-specific columns the file contains.
        order_df = pd.read_csv(
            order_file, skiprows=2, sep=r"\s+",
            usecols=[0, 1], names=["Time", "Orderp"],
        )

    if order_df.empty:
        log.debug("order.txt is empty for step %s — skipping", step)
        return None, 0.0
    
    lambda_max = float(order_df["Orderp"].max())
    log.debug("Processing OP=%.4f (grid_idx=%s) in %s", grid_points, grid_idx, directory)

    crossing = order_df[
        (order_df["Orderp"] > grid_points)
        & (order_df["Time"] != order_df["Time"].iloc[0])
        & (order_df["Time"] != order_df["Time"].iloc[-1])
    ]

    if crossing.empty:
        return None, lambda_max

    frame_idx = int(crossing["Time"].iloc[0])
    log.debug("  → frame=%d  lambda_max=%.4f", frame_idx, lambda_max)
    return [int(grid_idx), int(frame_idx)], float(lambda_max)
