# CV_frame_data/context.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import numpy as np

@dataclass(frozen=True)
class FrameKey:
    """Lightweight metadata identifying a specific frame passing through the pipeline."""
    step:    int
    op:      float
    ase_idx: int

@dataclass
class CVContext:
    """
    Raw simulation state provided by SimulationRunner for a single frame.
    Pass this into CVSuite.enrich_inputs() to build the full CVInputs.
    """
    coords:    np.ndarray       # (N, 3) geometry in reference topology order
    box:       Any              # scalar L, length-3, 3x3 cell vectors, or None
    key:       FrameKey         # step, op, ase_idx metadata
    topo:      Any              # SystemTopology instance 
    traj_path: Optional[Path]   # path to the source trajectory file

@dataclass
class CVInputs:
    """
    Rich per-frame input dataset passed to every CV module.
    Built by CVSuite.enrich_inputs(ctx) from the raw CVContext.
    """
    # ── Core geometry ──────────────────────────────────────────────────────
    coords: np.ndarray      # (N, 3) in topology sort order

    # ── Shared topology (immutable, same object for all frames) ─────────────
    topo: Any               # SystemTopology — molecule definitions, H→O map, ion info

    # ── Frame metadata ─────────────────────────────────────────────────────
    flags:     dict[str, Any]  = field(default_factory=dict)
    key:       object | None   = None   # FrameKey(step, op, ase_idx)
    box:       Any             = None   # scalar L, length-3, 3x3 cell vectors, or None
    traj_path: Optional[Path]  = None

    # ── Engine-specific or state-specific data ────────────────────────────
    # Contains everything that is context-specific, e.g. 'reaction', 'neighborhood',
    # 'mulliken_charges', 'forces_invariant', etc.
    data:      dict[str, Any]  = field(default_factory=dict)

    # ── Computed Results Cache (shared across modules for a single frame) ──
    # Note: This is an opt-in feature for on-the-fly cross-module dependencies.
    # To enable populating this cache, at least one registered CV module must
    # have the attribute `requires_results_cache = True`.
    # This is typically used by Wrapper CVs (like Ratio CVs) to pull results from
    # previously computed modules without re-calculating them.
    # Or history dependent CVs?
    results: dict[str, np.ndarray] = field(default_factory=dict)

