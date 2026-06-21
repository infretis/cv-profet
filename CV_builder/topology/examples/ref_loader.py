"""
ref_loader.py — Utility to load stable equilibrium frames for topology building.
This code auto detects the the topology file, instead of having to pass a seperate topology file
it just reads the correct frame from the accepted folder.

This topology is fed into the SystemTopology object, which is used to build the reference molecules
across all trajectories

In RETIS, time step 0 of a trajectory corresponds to the launching/ near equilibrium point
(lowest OP frame). This module provides the logic to locate these frames
across trajectories and prepare the reference DataFrame needed by SystemTopology.
"""
import logging
from glob import iglob
from pathlib import Path
import numpy as np
import pandas as pd
from ase.io import read as ase_read
from ase import Atoms

log = logging.getLogger(__name__)

def _parse_traj_txt_step0(traj_txt: str | Path) -> tuple[str | None, int]:
    """
    Read a RETIS traj.txt and return (filename, ase_index) for step 0.
    
    Step 0 corresponds to the near equilibrium geometry of the path.
    This is NOT necessarily ase_idx=0 because ∞RETIS generates backwards trajectories.
    """
    with open(traj_txt) as f:
        for line in f:
            s = line.strip()
            if s.startswith('#') or not s:
                continue
            parts = s.split()
            if parts[0] == '0':
                return parts[1], int(parts[2])
    return None, -1


class ReferenceTopologyLoader:
    def __init__(self, load_dir: str | Path):
        self.load_dir = Path(load_dir)
        self.df_reference: pd.DataFrame | None = None
        self.names_ref: np.ndarray | None = None
        
        # 1. Find the equilibrium frame
        xyz_path, ase_idx = self._find_step0_frame()
        log.debug("Using reference frame: %s [idx=%d]", xyz_path, ase_idx)
        
        # 2. Load and prepare
        df_raw = self._load_atoms(xyz_path, ase_idx=ase_idx)
        self.df_reference = df_raw.copy()
        self.names_ref = np.array(
            [f"{el}{i}" for el, i in zip(df_raw['element'], df_raw['orig_idx'])]
        )

    def _find_step0_frame(self) -> tuple[str, int]:
        """Search across RETIS steps to find a valid 'step 0' via traj.txt."""
        for ttxt in sorted(iglob(str(self.load_dir / "*/traj.txt"))):
            fname, idx = _parse_traj_txt_step0(ttxt)
            if fname is None:
                continue
            xyz = Path(ttxt).parent / 'accepted' / fname
            if xyz.exists():
                return str(xyz), idx
                
        raise FileNotFoundError(
            f"No step-0 frame found via traj.txt under {self.load_dir}. "
            f"Ensure traj.txt files exist in <load_dir>/*/traj.txt."
        )

    def _load_atoms(self, xyz_path: str, ase_idx: int = 0) -> pd.DataFrame:
        """Load atomic data into a raw dataframe preserving original indexing."""
        atoms = ase_read(xyz_path, index=ase_idx)
        assert isinstance(atoms, Atoms)

        elements = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        df = pd.DataFrame({
            'element': elements,
            'orig_idx': list(range(len(elements))),
            'x': pos[:, 0],
            'y': pos[:, 1],
            'z': pos[:, 2],
        })
        return df
