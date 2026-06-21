import numpy as np

def has_pbc(box) -> bool:
    """Return True if box is usable for PBC MIC operations."""
    if box is None:
        return False
    b = np.asarray(box, dtype=float)
    if not np.all(np.isfinite(b)):
        return False
    if b.ndim == 0:
        return float(b) > 0.0
    if b.shape == (3,):
        return bool(np.all(b > 0.0))
    if b.shape == (3, 3):
        det = float(np.linalg.det(b))
        return abs(det) > 1e-12
    return False

def mic_delta(delta: np.ndarray, box) -> np.ndarray:
    """Apply minimum-image wrapping for scalar, length-3, or 3x3 cell boxes."""
    if not has_pbc(box):
        return delta

    b = np.asarray(box, dtype=float)
    d = np.asarray(delta, dtype=float)

    if b.ndim == 0:
        L = float(b)
        return d - L * np.rint(d / L)

    if b.shape == (3,):
        return d - b * np.rint(d / b)

    inv_b = np.linalg.inv(b)
    frac = np.matmul(d, inv_b)
    frac -= np.rint(frac)
    return np.matmul(frac, b)

def mic_dist(a: np.ndarray, b: np.ndarray, box) -> float:
    return float(np.linalg.norm(mic_delta(b - a, box)))

def box_lengths(box) -> np.ndarray | None:
    """Return (a,b,c) lengths for scalar/length-3/3x3 boxes, else None."""
    if not has_pbc(box):
        return None
    b = np.asarray(box, dtype=float)
    if b.ndim == 0:
        L = float(b)
        return np.array([L, L, L], dtype=float)
    if b.shape == (3,):
        return b.astype(float)
    return np.linalg.norm(b, axis=1)

def rational_switch(x: np.ndarray, x0: float, n: int = 16, m: int = 56) -> np.ndarray:
    """
    Computes the rational switching function:
        S(x) = (1 - (x/x0)^n) / (1 - (x/x0)^m)
    Handles x == x0 safely.
    """
    ratio = x / x0
    
    # Avoid overflow in power
    ratio = np.clip(ratio, 0.0, 1.5) 
    
    num = 1.0 - ratio**n
    den = 1.0 - ratio**m
    
    # Where x == x0, ratio == 1. L'Hopital's rule gives n/m
    out = np.empty_like(ratio)
    mask_one = np.isclose(ratio, 1.0, atol=1e-7)
    
    out[mask_one] = n / m
    out[~mask_one] = num[~mask_one] / den[~mask_one]
    
    # Where x is very large, ratio is > 1.5, the value is effectively 0
    # The clipping already helps, but ensuring it decays to zero:
    out[ratio >= 1.5] = 0.0
    
    return np.clip(out, 0.0, 1.0)
