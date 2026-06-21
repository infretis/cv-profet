"""
wham_helpers.py — Minimal WHAM utilities.
"""
import numpy as np


def get_k(interfaces, lambda_value):
    """
    Find integer k such that interfaces[k] < lambda_value <= interfaces[k+1].

    Returns 0 if lambda_value < interfaces[0], len(interfaces)-1 if above the last.
    """
    if lambda_value < interfaces[0]:
        return 0
    elif lambda_value > interfaces[-1]:
        return len(interfaces) - 1
    return int(np.searchsorted(interfaces, lambda_value, side="left"))