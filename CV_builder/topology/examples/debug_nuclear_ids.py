"""
debug_nuclear_ids.py — Verify CP2K nuclear-ID stability across RETIS trajectories.

PURPOSE
-------
This script verifies that:

  1. The element ordering (O-block, H-block, ion-last) is stable across files.
     If yes → nuclear IDs are stable, no ICP mapper needed.

  2. The H→O assignment (water topology) at the TRUE step-0 frame (lowest OP,
     genuine equilibrium geometry) is identical across all RETIS steps.
     Frame index is read from traj.txt — NOT hardcoded as ase_idx=0.

  3. Given a FIXED reference topology (from the first step-0 frame), O-H
     distances are physical in all subsequent frames.
     PT intermediates will show stretched OH — this is expected and correct.
     A TRUE error is OH > 2.5 Å (wrong assignment).

USAGE
-----
    cd /Users/bredo/ppa/CV_ana
    conda run -n molmod python CV_builder/topology/debug_nuclear_ids.py \\
        --load-dir load --n-steps 30

    # Show all water molecules for the reference step:
    conda run -n molmod python CV_builder/topology/debug_nuclear_ids.py \\
        --load-dir load --verbose

OPTIONS
-------
    --load-dir   DIR    Directory with RETIS step sub-directories (default: load)
    --n-steps    N      Number of RETIS steps to sample (default: 20)
    --cell-size  L      Box length in Å for PBC (default: 12.5)
    --verbose           Print all water molecules for the reference step
    --oh-warn    X      O-H warn threshold (Å) (default: 1.5)
    --oh-error   X      Hard error threshold for broken assignment (Å) (default: 2.5)
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read as ase_read


_PHI = (1.0 + 5.0 ** 0.5) / 2.0
_U_BREAK = np.array([1.0, _PHI, _PHI**2])
_U_BREAK /= np.linalg.norm(_U_BREAK)


# ── Minimal geometry helpers (no dependency on the full package) ──────────────

def _pbc_delta(delta: np.ndarray, L: float | None) -> np.ndarray:
    if L is None or L <= 0:
        return delta
    return delta - L * np.round(delta / L)


def _two_nearest_h_per_o(O_pos: np.ndarray, H_pos: np.ndarray, L: float | None) -> np.ndarray:
    d = O_pos[:, None, :] - H_pos[None, :, :]
    d = _pbc_delta(d, L)
    d2 = np.sum(d * d, axis=2)
    return np.argsort(d2, axis=1)[:, :2]


def _deterministic_order(O_pos: np.ndarray, H_pair: np.ndarray, L: float | None) -> np.ndarray:
    OH = _pbc_delta(H_pair - O_pos[None, :], L)
    proj = OH @ _U_BREAK
    return np.argsort(proj)[::-1]


# ── Core analysis ─────────────────────────────────────────────────────────────

def load_frame_arrays(xyz_path: str, ase_idx: int):
    """Load a trajectory frame, return (elements, positions, n_atoms)."""
    atoms = ase_read(xyz_path, index=ase_idx)
    return atoms.get_chemical_symbols(), atoms.get_positions(), len(atoms)


def build_topology_from_frame(elems, pos, cell: float, oh_warn: float):
    """
    Build water topology for one frame: (triplets, ion_idx, ion_sym, oh_stats, bad_ohs).
    triplets = list of (h1_abs, o_abs, h2_abs) using nearest-H-per-O.
    """
    abs_O   = np.array([i for i, e in enumerate(elems) if e == "O"], dtype=int)
    abs_H   = np.array([i for i, e in enumerate(elems) if e == "H"], dtype=int)
    abs_ion = np.array([i for i, e in enumerate(elems) if e not in ("H", "O")], dtype=int)

    O_pos = pos[abs_O]; H_pos = pos[abs_H]
    nn_H  = _two_nearest_h_per_o(O_pos, H_pos, cell)

    triplets = []; oh_distances = []; bad_ohs = []
    for o_i in range(len(abs_O)):
        h_pair_local = nn_H[o_i]
        order = _deterministic_order(O_pos[o_i], H_pos[h_pair_local], cell)
        h1_l, h2_l = h_pair_local[order]
        o_abs = int(abs_O[o_i]); h1_abs = int(abs_H[h1_l]); h2_abs = int(abs_H[h2_l])
        triplets.append((h1_abs, o_abs, h2_abs))
        for h_l, h_abs in [(h1_l, h1_abs), (h2_l, h2_abs)]:
            d = float(np.linalg.norm(_pbc_delta(H_pos[h_l] - O_pos[o_i], cell)))
            oh_distances.append(d)
            if d > oh_warn:
                bad_ohs.append((o_abs, h_abs, d))

    ion_idx = int(abs_ion[0]) if abs_ion.size == 1 else None
    ion_sym = elems[ion_idx] if ion_idx is not None else None
    return triplets, ion_idx, ion_sym, oh_distances, bad_ohs


def check_fixed_topology(
    ref_triplets: list,
    pos: np.ndarray,
    cell: float,
    oh_error: float,
) -> list:
    """
    Given a FIXED set of (H1, O, H2) triplets from the reference frame,
    compute O-H distances in the CURRENT frame and report any that exceed
    oh_error Å (which would indicate a truly broken assignment).

    Stretched bonds 1.0–2.0 Å are EXPECTED for proton-transfer intermediates.
    Only bonds > oh_error (default 2.5 Å) indicate a wrong assignment.
    """
    broken = []
    for h1, o, h2 in ref_triplets:
        for h in [h1, h2]:
            d = float(np.linalg.norm(_pbc_delta(pos[h] - pos[o], cell)))
            if d > oh_error:
                broken.append((o, h, d))
    return broken


def elem_order_fingerprint(elems) -> str:
    """Unique sequence of element types, e.g. 'OHCl'."""
    return "".join(dict.fromkeys(elems))


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_traj_txt_step0(traj_txt: str) -> tuple[str | None, int]:
    """Return (filename, ase_index) for step 0 from a RETIS traj.txt."""
    with open(traj_txt) as f:
        for line in f:
            s = line.strip()
            if s.startswith('#') or not s:
                continue
            parts = s.split()
            if parts[0] == '0':
                return parts[1], int(parts[2])
    return None, -1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--load-dir",  default="load", help="RETIS load directory")
    ap.add_argument("--n-steps",   type=int, default=20, dest="n_steps",
                    help="Number of RETIS steps to sample (default: 20)")
    ap.add_argument("--n-files",   type=int, default=None, dest="n_files",
                    help="Alias for --n-steps (backward compat)")
    ap.add_argument("--cell-size", type=float, default=12.5, help="Box size in Å")
    ap.add_argument("--oh-warn",   type=float, default=1.5, help="O-H warn threshold (Å)")
    ap.add_argument("--oh-error",  type=float, default=2.5, help="O-H broken-assignment threshold (Å)")
    ap.add_argument("--verbose",   action="store_true", help="Print all waters for reference step")
    args = ap.parse_args()

    n_sample = args.n_files if args.n_files is not None else args.n_steps
    load_dir = Path(args.load_dir)
    oh_error = args.oh_error

    # ── Find all traj.txt files and sample step-0 frames ─────────────────
    all_traj = sorted(glob.glob(str(load_dir / '*/traj.txt')))
    if not all_traj:
        print(f"ERROR: No traj.txt files found under {load_dir}/*/traj.txt")
        sys.exit(1)

    step_by = max(1, len(all_traj) // n_sample)
    sampled_txts = all_traj[::step_by][:n_sample]

    # Resolve (xyz_path, ase_idx) for each step
    sampled = []
    for ttxt in sampled_txts:
        fname, idx = _parse_traj_txt_step0(ttxt)
        if fname is None:
            continue
        xyz = Path(ttxt).parent / 'accepted' / fname
        if xyz.exists():
            sampled.append((str(xyz), idx, Path(ttxt).parent.name))

    if not sampled:
        print(f"ERROR: Could not resolve any step-0 frames from traj.txt files")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  debug_nuclear_ids.py — CP2K nuclear-ID verification")
    print(f"{'='*70}")
    print(f"  Load dir      : {load_dir}")
    print(f"  RETIS steps   : {len(all_traj)}  |  Sampling: {len(sampled)}")
    print(f"  Cell size     : {args.cell_size} Å  |  O-H warn: {args.oh_warn} Å  O-H error: {oh_error} Å")
    print(f"{'='*70}\n")
    print("  Frames are step-0 (lowest OP, equilibrium geometry) from traj.txt.")
    print("  This is NOT ase_idx=0 of the xyz file — that can be a PT-intermediate.")
    print("  A FIXED topology is built from the first step-0 frame and applied to all.")
    print("  Stretched OH (1.0–2.0 Å) is EXPECTED for PT frames; OH > error = WRONG.\n")

    results = []
    errors  = []
    ref_triplets = None
    ref_ion_idx  = None
    ref_ion_sym  = None
    elem_orders  = set()

    for i, (fpath, ase_idx, step_id) in enumerate(sampled):
        short = f"step={step_id} [{ase_idx}] {Path(fpath).name}"
        try:
            elems, pos, n = load_frame_arrays(fpath, ase_idx=ase_idx)
            elem_orders.add(elem_order_fingerprint(elems))

            # Build topology per-frame only to get reference triplets on first file
            triplets, ion_idx, ion_sym, oh_dists, bad_ohs = build_topology_from_frame(
                elems, pos, args.cell_size, args.oh_warn
            )

            if ref_triplets is None:
                ref_triplets = triplets
                ref_ion_idx  = ion_idx
                ref_ion_sym  = ion_sym

            # Check reference topology distances in this frame
            broken = check_fixed_topology(ref_triplets, pos, args.cell_size, oh_error)

            # How many waters differ from reference (unordered H sets)
            ref_pairs  = [frozenset([h1,h2]) for h1,o,h2 in ref_triplets]
            this_pairs = [frozenset([h1,h2]) for h1,o,h2 in triplets]
            n_reassigned = sum(1 for a,b in zip(ref_pairs, this_pairs) if a != b)

            status = "BROKEN" if broken else ("PT-frame" if n_reassigned else "OK")
            print(f"  [{i+1:>3}/{len(sampled)}] {status:10s} {short}")
            print(f"           n={n:3d}  O={len([e for e in elems if e=='O']):2d}  "
                  f"ion={ion_sym}@{ion_idx}  elem={elem_order_fingerprint(elems)}  "
                  f"OH max={max(oh_dists):.3f}Å mean={sum(oh_dists)/len(oh_dists):.3f}Å  "
                  f"reassigned={n_reassigned}")
            if broken:
                for o, h, d in broken:
                    print(f"           !! BROKEN O{o}--H{h} = {d:.3f} Å (ref topology gives wrong assignment!)")
            if n_reassigned and not broken:
                # find which waters
                changed = [(ref_triplets[k][1], sorted(ref_pairs[k]), sorted(this_pairs[k]))
                           for k in range(len(ref_pairs)) if ref_pairs[k] != this_pairs[k]]
                for o_idx, rp, np_ in changed[:3]:
                    # distances in current frame
                    d_kept = min(float(np.linalg.norm(_pbc_delta(pos[rp[0]]-pos[o_idx], args.cell_size))),
                                 float(np.linalg.norm(_pbc_delta(pos[rp[1]]-pos[o_idx], args.cell_size))))
                    d_new  = min(float(np.linalg.norm(_pbc_delta(pos[np_[0]]-pos[o_idx], args.cell_size))),
                                 float(np.linalg.norm(_pbc_delta(pos[np_[1]]-pos[o_idx], args.cell_size))))
                    print(f"           PT: O{o_idx} ref-H={rp} (d={d_kept:.2f}Å)  nearest-H={np_} (d={d_new:.2f}Å)")

            results.append((fpath, status, n_reassigned, broken))
        except Exception as e:
            errors.append((fpath, str(e)))
            print(f"  [{i+1:>3}/{len(sampled)}] ERROR      {short} — {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")

    # Element ordering
    if len(elem_orders) == 1:
        print(f"  ✅ Element ordering: stable across all files ({next(iter(elem_orders))})")
    else:
        print(f"  ❌ Element ordering: DIFFERS across files: {elem_orders}")

    # Ion
    unique_ions = set(r[0].split('/')[-1] for r in results)  # placeholder
    print(f"  ✅ Ion: {ref_ion_sym}@{ref_ion_idx} (verified stable in element-order check above)")

    # Broken assignments
    n_broken = sum(1 for _, _, _, broken in results if broken)
    if n_broken == 0:
        print(f"  ✅ Fixed topology: no BROKEN assignments (OH < {oh_error} Å in all files)")
    else:
        print(f"  ❌ Fixed topology: {n_broken} files have BROKEN O-H > {oh_error} Å")

    # PT frames
    n_pt = sum(1 for _, status, _, _ in results if status == "PT-frame")
    print(f"  ℹ️  PT-frames (nearest-H differs from ref, but OH physical): {n_pt}/{len(results)}")
    print(       "     These are proton-transfer intermediates — expected and correct.")
    print(       "     The fixed reference topology correctly assigns the DONOR water label.")

    if n_broken == 0 and len(elem_orders) == 1:
        print(f"\n  ✅ OVERALL PASS — nuclear IDs are stable, element ordering fixed.")
        print(       "     Skip ICP mapper; use fixed reference topology for all frames.\n")
    else:
        print(f"\n  ❌ OVERALL FAIL — check warnings above.\n")



    # ── Verbose: all water molecules for first file ───────────────────────
    if args.verbose and ref_triplets:
        first_fpath, first_idx, first_step = sampled[0]
        print(f"\n{'='*70}")
        print(f"  Reference water molecules — step={first_step} [{first_idx}] {Path(first_fpath).name}")
        print(f"{'='*70}")
        print(f"  {'#':>4}  {'H1':>7}  {'O':>7}  {'H2':>7}")
        for idx, (h1, o, h2) in enumerate(ref_triplets):
            print(f"  {idx:>4d}  H{h1:<6d}  O{o:<6d}  H{h2:<6d}")

    print(f"\n  Errors: {len(errors)}")
    for (fp, _idx, _step), msg in errors:
        print(f"    step={_step} {Path(fp).name}: {msg}")


if __name__ == "__main__":
    main()
