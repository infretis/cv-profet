"""
storage/h5_writer.py — HDF5 output writer for CV_builder.

Each RETIS step gets its own file:  <cv_root>/step_<step>.h5

Group layout:
    /names_ref              dataset: canonical reference atom names
    /<grid_id>/
        <cv_name>/
            labels          dataset: CV column labels
            values          dataset: (N_frames, K) float64
            meta            dataset: (N_frames,) structured (step, op, ase, err, jaccard, …)
            [indices]       optional: pair indices
            [triplet_idx]   optional: (Od, H*, Oa) triplets
            [wire_path_O]   optional: wire path atom indices
            [wire_path_labels] optional: wire path atom labels
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import h5py
import numpy as np

from config import OutputConfig

log = logging.getLogger(__name__)


class H5Writer:
    """
    Lazy-open HDF5 writer.  The file is not created until the first write
    call, so aborting early (e.g. no crossing for a step) leaves no artefact.
    """

    def __init__(
        self,
        out_dir: Path,
        step:    int | str,
        out_cfg: OutputConfig | None = None,
    ) -> None:
        self.out_dir  = out_dir
        self.path     = out_dir / f"step_{step}.h5"
        self.out_cfg  = out_cfg or OutputConfig()
        self.h5: h5py.File | None = None
        self._compress = self.out_cfg.compress_level

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self.h5 is None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.h5 = h5py.File(self.path, "a")
            log.debug("Opened HDF5 file: %s", self.path)

    def _gz(self, **kwargs) -> dict:
        """Convenience: inject compression kwargs if level > 0."""
        if self._compress > 0:
            return dict(compression="gzip", compression_opts=self._compress, **kwargs)
        return kwargs

    # -----------------------------------------------------------------------
    # Step-file initialisation
    # -----------------------------------------------------------------------

    def init_step_file(
        self,
        names_ref:   np.ndarray | None = None,
        wham_ngrid:  int | None = None,
        cell_size:   float | None = None,
        n_atoms:     int | None = None,
        weight:      float | None = None,
        extra_attrs: dict | None = None,
    ) -> None:
        """
        Open the file (lazy) and set all root-level metadata in one call.
          - Creates /names_ref once (if provided and not yet present).
          - Sets static attrs (wham_ngrid, cell_size, n_atoms) if absent.
          - Always updates weight if provided.
          - Writes any extra_attrs (e.g. lambda_max).
        """
        self._ensure_open()
        assert self.h5 is not None

        if names_ref is not None:
            self._ensure_names_ref(names_ref)

        def _cast(v):
            try:
                return v.item()
            except Exception:
                return v

        static = {"wham_ngrid": wham_ngrid, "cell_size": cell_size, "n_atoms": n_atoms}
        for key, val in static.items():
            if val is not None and key not in self.h5.attrs:
                self.h5.attrs[key] = _cast(val)

        if weight is not None:
            self.h5.attrs["weight"] = _cast(weight)

        for key, val in (extra_attrs or {}).items():
            if val is not None:
                self.h5.attrs[key] = _cast(val)

    # -----------------------------------------------------------------------
    # Group initialisation
    # -----------------------------------------------------------------------

    def ensure_cv_group(self, grid_id: int, cv) -> None:
        """Create/verify the HDF5 group for a given (grid_id, cv) combination."""
        self._ensure_open()
        assert self.h5 is not None
        g = self.h5.require_group(f"{grid_id}/{cv.name}")

        if "labels" not in g:
            str_dt = h5py.string_dtype(encoding="utf-8")
            data = np.asarray(cv.labels, dtype=object).astype(str_dt)
            if data.ndim == 0:
                # Scalar datasets cannot use chunking/compression filters in h5py.
                g.create_dataset("labels", data=data)
            else:
                g.create_dataset("labels", data=data, chunks=True, **self._gz())

        if hasattr(cv, "pairs") and "indices" not in g:
            g.create_dataset(
                "indices", data=np.asarray(cv.pairs, dtype=np.int32),
                chunks=True, **self._gz(),
            )

        if getattr(cv, "provides_triplet_meta", False) and "triplet_idx" not in g:
            g.create_dataset(
                "triplet_idx", shape=(0, 3), maxshape=(None, 3),
                dtype="i4", chunks=True, **self._gz(),
            )

        if hasattr(cv, "last_path_O") and "wire_path_O" not in g:
            g.create_dataset(
                "wire_path_O", shape=(0, 4), maxshape=(None, 4),
                dtype="i4", chunks=True, **self._gz(),
            )

        if hasattr(cv, "last_path_labels") and "wire_path_labels" not in g:
            g.create_dataset(
                "wire_path_labels", shape=(0, 4), maxshape=(None, 4),
                dtype=h5py.string_dtype(encoding="utf-8"),
                chunks=True, **self._gz(),
            )

    def _ensure_names_ref(self, names_ref: np.ndarray) -> None:
        assert self.h5 is not None
        if "names_ref" not in self.h5:
            dt = h5py.string_dtype(encoding="utf-8")
            self.h5.create_dataset(
                "names_ref",
                data=np.asarray(names_ref, dtype=object).astype(dt),
                chunks=True, **self._gz(),
            )

    def _ensure_dsets(self, grid_id: int, cv, K: int) -> None:
        """Create the values and meta datasets for a CV group if absent."""
        self._ensure_open()
        assert self.h5 is not None
        g = self.h5[f"{grid_id}/{cv.name}"]
        if not isinstance(g, h5py.Group):
            raise TypeError(f"Expected h5py.Group at {grid_id}/{cv.name}")

        if "values" not in g:
            row_chunk = max(64, min(512, 4096 // max(1, K)))
            g.create_dataset(
                "values", shape=(0, K), maxshape=(None, K),
                chunks=(row_chunk, K), dtype="f8", **self._gz(),
            )

        if "meta" not in g:
            meta_dt = np.dtype([
                ("step",       "i4"),
                ("op",         "f8"),
                ("ase",        "i4"),
                ("err",        "f8"),
                ("bad_err",    "b"),
                ("bad_h_cutoff", "b"),
                ("jaccard",    "f8"),
                ("flags_json", h5py.string_dtype(encoding="utf-8")),
            ])
            g.create_dataset(
                "meta", shape=(0,), maxshape=(None,),
                chunks=(512,), dtype=meta_dt, **self._gz(),
            )

    # -----------------------------------------------------------------------
    # Append
    # -----------------------------------------------------------------------

    def append(self, grid_id: int, cv, values: np.ndarray, meta: dict) -> None:
        """Append one frame's CV values + metadata to the HDF5 group."""
        self._ensure_open()
        assert self.h5 is not None
        g    = self.h5[f"{grid_id}/{cv.name}"]
        vals = g["values"]
        m    = g["meta"]

        if not isinstance(vals, h5py.Dataset):
            raise TypeError(f'"values" at {grid_id}/{cv.name} is not a Dataset')
        if not isinstance(m, h5py.Dataset):
            raise TypeError(f'"meta" at {grid_id}/{cv.name} is not a Dataset')

        n = vals.shape[0]
        vals.resize((n + 1, vals.shape[1]))
        m.resize((n + 1,))
        vals[n, :] = values

        flags = meta.get("flags", {}) or {}
        row              = np.zeros((), dtype=m.dtype)
        row["step"]      = int(meta.get("step", -1))
        row["op"]        = float(meta.get("op", float("nan")))
        row["ase"]       = int(meta.get("ase_idx", -1))
        row["err"]       = float(meta.get("err", float("nan")))
        row["bad_err"]   = bool(flags.get("bad_err", False))
        row["bad_h_cutoff"] = bool(flags.get("bad_h_cutoff", False))
        row["jaccard"]   = float(meta.get("jaccard", float("nan")))
        row["flags_json"] = json.dumps(flags)
        m[n] = row

        # Optional triplet metadata
        if getattr(cv, "provides_triplet_meta", False) and "triplet_idx" in g:
            trip  = getattr(cv, "last_triplet", None) or (-1, -1, -1)
            dtrip = g["triplet_idx"]
            dtrip.resize((n + 1, 3))
            dtrip[n, :] = np.asarray(trip, dtype="i4")

        # Optional wire-path indices
        if "wire_path_O" in g and hasattr(cv, "last_path_O"):
            ds  = g["wire_path_O"]
            ds.resize(ds.shape[0] + 1, axis=0)
            ds[ds.shape[0] - 1, :] = (
                cv.last_path_O if cv.last_path_O is not None
                else np.array([-1, -1, -1, -1], dtype=int)
            )

        # Optional wire-path labels
        if "wire_path_labels" in g and hasattr(cv, "last_path_labels"):
            ds  = g["wire_path_labels"]
            ds.resize(ds.shape[0] + 1, axis=0)
            ds[ds.shape[0] - 1, :] = (
                np.asarray(cv.last_path_labels, dtype=object)
                if cv.last_path_labels is not None
                else np.array(["", "", "", ""], dtype=object)
            )

    # -----------------------------------------------------------------------
    # Idempotency check
    # -----------------------------------------------------------------------

    def has_frame(self, grid_id: int, cv, step: int | str, ase_idx: int) -> bool:
        """
        Return True if a frame with the given (step, ase_idx) pair already
        exists in this file for the specified CV and grid point.
        """
        self._ensure_open()
        assert self.h5 is not None
        path = f"{grid_id}/{cv.name}"
        if path not in self.h5:
            return False
        g = self.h5[path]
        if "meta" not in g or g["meta"].shape[0] == 0:
            return False
        meta = g["meta"]
        # Check both step and ase_idx to avoid false positives on resumed runs.
        mask = (meta["ase"][:] == ase_idx) & (meta["step"][:] == int(step))
        return bool(np.any(mask))

    # -----------------------------------------------------------------------
    # Flush & close
    # -----------------------------------------------------------------------

    def checkpoint(self) -> None:
        """Flush in-memory HDF5 buffers to disk without closing the file."""
        if self.h5 is not None:
            self.h5.flush()
            log.debug("HDF5 checkpoint: %s", self.path)

    def close(self) -> None:
        if self.h5 is not None:
            try:
                self.h5.flush()
            finally:
                self.h5.close()
                self.h5 = None
                log.debug("Closed HDF5 file: %s", self.path)
