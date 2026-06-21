# CV_manager/load_cp2k_aux.py
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


# ------------------------------------------------------------
# Helper functions for locating CP2K-produced aux files
# ------------------------------------------------------------

def _find_exact(path: Path) -> Optional[Path]:
    """Return path if exists, else None."""
    return path if path.exists() else None


def find_mulliken_file(traj_path: Path) -> Optional[Path]:
    """
    Example filename:
       036_1525456_174_trajB-MULLI-1.mulliken
    """
    prefix = traj_path.stem
    cand = traj_path.parent / f"{prefix}-MULLI-1.mulliken"
    return _find_exact(cand)


def find_homo_centers_file(traj_path: Path, ase_idx: int) -> Optional[Path]:
    """
    Example filenames:
        <prefix>-HOMO_centers_s1-1_<ase_idx>.data
    """
    prefix = traj_path.stem
    cand = traj_path.parent / f"{prefix}-HOMO_centers_s1-1_{ase_idx}.data"
    return _find_exact(cand)


def find_dipole_file(traj_path: Path, ase_idx: int) -> Optional[Path]:
    """
    Examples CP2K tends to output:
        <prefix>-DIPOLE-TOTAL_DIPOLE-1_<ase_idx>.Dipole
    """
    prefix = traj_path.stem
    cand = traj_path.parent / f"{prefix}-DIPOLE-TOTAL_DIPOLE-1_{ase_idx}.Dipole"
    return _find_exact(cand)


def find_forces_file(traj_path: Path) -> Optional[Path]:
    """
    Forces are typically in:
        <prefix>-frc-1.xyz
    (But filenames vary. Adjust as needed.)
    """
    prefix = traj_path.stem
    cand = traj_path.parent / f"{prefix}-frc-1.xyz"
    return _find_exact(cand)


# ------------------------------------------------------------
# Parsers
# ------------------------------------------------------------

def _parse_mulliken(path: Path) -> np.ndarray:
    """
    Parse Mulliken charge file. Assumes CP2K standard table with:
        # Atom ... Net charge
    Returns array of charges in order of atoms in the file.
    """
    charges = []
    with open(path, "r") as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    q = float(parts[-1])
                    charges.append(q)
                except Exception:
                    continue
    return np.asarray(charges, dtype=float)


def _parse_homo_centers(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse HOMO centers file produced by CP2K.
    Format like:
        X   x y z   spread
    Returns:
      centers: (M,3)
      spreads: (M,)
    """
    centers = []
    spreads = []
    with open(path, "r") as f:
        for line in f:
            if line.strip().startswith("X"):
                parts = line.split()
                if len(parts) >= 5:
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    s = float(parts[4])
                    centers.append([x, y, z])
                    spreads.append(s)

    return np.asarray(centers, dtype=float), np.asarray(spreads, dtype=float)


def _parse_dipole(path: Path) -> np.ndarray:
    """
    Parse dipole file containing:
        x y z   (sometimes 9 components)
    We take the last 3 floats on the row.
    """
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    vals = [float(v) for v in parts[-3:]]
                    return np.asarray(vals, dtype=float)
                except Exception:
                    continue
    return np.zeros(3, dtype=float)


def _parse_forces(path: Path, n_atoms: int) -> np.ndarray:
    """
    Parse <prefix>-frc-1.xyz with possibly multiple frames.

    Typical CP2K layout (per frame):

        N
        i =    <step>, time = ...
        O  fx fy fz
        H  fx fy fz
        ...
        (N lines of forces)

    We:
      - ignore the "N" line and comment/time line
      - collect one (fx, fy, fz) triplet per atom line
      - group every n_atoms triplets into one frame

    Returns:
      - (n_frames, n_atoms, 3) if multiple frames
      - (n_atoms, 3) if only one frame detected
      - zeros((n_atoms, 3)) if nothing usable was found
    """
    frames: list[np.ndarray] = []
    current_forces: list[tuple[float, float, float]] = []

    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue

            # Header line with number of atoms, e.g. "193"
            if len(parts) == 1 and parts[0].isdigit():
                # If we already collected a full frame, finalize it
                if current_forces:
                    if len(current_forces) == n_atoms:
                        frames.append(np.asarray(current_forces, dtype=float))
                    current_forces = []
                continue

            # Comment / meta line, e.g. "i =    1, time = ..."
            if parts[0] in ("i", "#", "!", "Time", "Step"):
                continue

            # Atom force lines: symbol + fx fy fz
            if parts[0] in ("O", "H", "Na", "Cl", "X"):
                if len(parts) < 4:
                    continue
                try:
                    fx = float(parts[1])
                    fy = float(parts[2])
                    fz = float(parts[3])
                except ValueError:
                    continue
                current_forces.append((fx, fy, fz))

                # Hvis vi har n_atoms linjer, har vi én full frame
                if len(current_forces) == n_atoms:
                    frames.append(np.asarray(current_forces, dtype=float))
                    current_forces = []

    # Etter filslutt – ta med siste frame hvis den ser hel ut
    if current_forces and len(current_forces) == n_atoms:
        frames.append(np.asarray(current_forces, dtype=float))

    if not frames:
        # Fallback: ingenting forståelig → returner nuller (ikke crash)
        return np.zeros((n_atoms, 3), dtype=float)

    if len(frames) == 1:
        # Bare én frame
        return frames[0]

    # Flere frames: returner 3D-array (n_frames, n_atoms, 3)
    return np.stack(frames, axis=0)




# ------------------------------------------------------------
# Main loader function
# ------------------------------------------------------------
def load_cp2k_aux_data_for_frame(
    traj_path: Path,
    ase_idx: int,
    n_atoms: int,
) -> Tuple[
    Optional[np.ndarray],                     # mulliken charges
    Optional[Tuple[np.ndarray, np.ndarray]],  # homo centers + spreads
    Optional[np.ndarray],                     # dipole vector
    Optional[np.ndarray],                     # forces array (n_atoms,3)
]:
    ...
    # ---------- Mulliken ----------  (din nye kode her, som vi allerede fikset)
    mull = None
    mull_path = find_mulliken_file(traj_path)
    if mull_path is not None:
        all_charges = _parse_mulliken(mull_path)
        all_charges = np.asarray(all_charges, dtype=float).ravel()
        total = all_charges.size

        if total == 0:
            mull = None
        elif total == n_atoms:
            mull = all_charges
        elif total % n_atoms != 0:
            print(
                f"[MULLIKEN] Unexpected number of charges: total={total}, "
                f"n_atoms={n_atoms}. Disabling Mulliken."
            )
            mull = None
        else:
            n_frames = total // n_atoms
            frame_idx = ase_idx
            if 0 <= frame_idx < n_frames:
                mull = all_charges.reshape(n_frames, n_atoms)[frame_idx]
            else:
                print(
                    f"[MULLIKEN] ase_idx={ase_idx} out of range for "
                    f"n_frames={n_frames}. Disabling Mulliken."
                )
                mull = None

    # ---------- HOMO centers ----------
    homo_path = find_homo_centers_file(traj_path, ase_idx)
    homo = _parse_homo_centers(homo_path) if homo_path else None

    # ---------- Dipole ----------
    dipole_path = find_dipole_file(traj_path, ase_idx)
    dip = _parse_dipole(dipole_path) if dipole_path else None

    # ---------- Forces ----------
    frc = None
    frc_path = find_forces_file(traj_path)
    if frc_path is not None:
        all_forces = _parse_forces(frc_path, n_atoms)
        all_forces = np.asarray(all_forces, dtype=float)

        if all_forces.ndim == 2:
            # Enkeltframe: forvent (n_atoms, 3)
            if all_forces.shape == (n_atoms, 3):
                frc = all_forces
            else:
                print(
                    f"[FORCES] Unexpected shape for single frame: "
                    f"{all_forces.shape}, expected ({n_atoms}, 3). Disabling forces."
                )
                frc = None

        elif all_forces.ndim == 3:
            # Flere frames: (n_frames, n_atoms, 3)
            n_frames, n_a, n_c = all_forces.shape
            if n_a != n_atoms or n_c != 3:
                print(
                    f"[FORCES] Shape mismatch: all_forces.shape={all_forces.shape}, "
                    f"expected (*, {n_atoms}, 3). Disabling forces."
                )
                frc = None
            else:
                frame_idx = ase_idx
                if 0 <= frame_idx < n_frames:
                    frc = all_forces[frame_idx]
                else:
                    print(
                        f"[FORCES] ase_idx={ase_idx} out of range for "
                        f"n_frames={n_frames}. Disabling forces."
                    )
                    frc = None

    return mull, homo, dip, frc

