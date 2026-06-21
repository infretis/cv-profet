# CV_manager1/reaction_center.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from CV_frame_data.math_utils import mic_delta, mic_dist

def _mic_delta(a: np.ndarray, b: np.ndarray, box) -> np.ndarray:
    return mic_delta(b - a, box)

def _mic_dist(a: np.ndarray, b: np.ndarray, box) -> float:
    return mic_dist(a, b, box)

@dataclass(slots=True)
class ReactionCenter:
    O_d: int              # donor oxygen (invariant index)
    Hs: int               # stretched proton (invariant index)
    O_a: int              # nearest acceptor oxygen (invariant index)
    widx: int             # index into water_triplets for the donor water
    # handy geometry
    r_Od_Hs: float
    r_Oa_Hs: float
    R_Od_Oa: float
    delta: float          # PT coordinate: r_Od_Hs - r_Oa_Hs
    e_d: np.ndarray       # unit vector Od -> Hs   (3,)
    e_a: np.ndarray       # unit vector Oa -> Hs   (3,)

    def triplet(self) -> tuple[int,int,int]:
        return (self.O_d, self.Hs, self.O_a)

def select_reaction_center(coords: np.ndarray,
                            water_triplets: np.ndarray,  # (nW,3) as (H1,O,H2) in invariant indices
                            box) -> ReactionCenter:
    # 1) pick O_d and Hs as the GLOBAL longest O–H among all waters
    best = (-1, -1, -1.0, -1)  # (O_d, Hs, r, widx)
    for idx, (h1, o, h2) in enumerate(water_triplets):
        r1 = _mic_dist(coords[o], coords[h1], box)
        r2 = _mic_dist(coords[o], coords[h2], box)
        if r1 >= r2:
            if r1 > best[2]:
                best = (int(o), int(h1), r1, idx)
        else:
            if r2 > best[2]:
                best = (int(o), int(h2), r2, idx)

    O_d, Hs, rmax, widx = best
    # 2) pick O_a as closest oxygen (excluding O_d)
    O_list = np.unique(water_triplets[:, 1]).astype(int)
    dmin = 1e99; O_a = -1
    for o in O_list:
        if o == O_d: 
            continue
        d = _mic_dist(coords[Hs], coords[o], box)
        if d < dmin:
            dmin, O_a = d, o

    # 3) compile geometry
    v_d = _mic_delta(coords[O_d], coords[Hs], box)
    v_a = _mic_delta(coords[O_a], coords[Hs], box)
    nd = np.linalg.norm(v_d); na = np.linalg.norm(v_a)
    e_d = v_d / (nd if nd > 0 else 1.0)
    e_a = v_a / (na if na > 0 else 1.0)
    r_Od_Hs = nd
    r_Oa_Hs = na
    R_Od_Oa = _mic_dist(coords[O_d], coords[O_a], box)
    delta   = r_Od_Hs - r_Oa_Hs

    return ReactionCenter(
        O_d=O_d, Hs=Hs, O_a=O_a, widx=widx,
        r_Od_Hs=float(r_Od_Hs), r_Oa_Hs=float(r_Oa_Hs), R_Od_Oa=float(R_Od_Oa), delta=float(delta),
        e_d=e_d, e_a=e_a
    )
