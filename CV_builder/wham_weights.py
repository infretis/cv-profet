import os
from typing import List, Tuple, Dict

import numpy as np

from typing import List
import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path


import numpy as np

def print_wham_weights(trajlabels: List[int], WFtot: Dict[int, float]) -> None:
    print("Retis trajectory \t WF-weight for each path")
    printed_count = 0
    zero_labels = []

    for label in trajlabels:
        weight = WFtot[label]
        if weight == 0.0:
            zero_labels.append(label)

        if printed_count < 50:
            print(f"{label} \t {weight}")
            printed_count += 1

    if zero_labels:
        print(f"\nPaths with 0.0 WHAM weight: {len(zero_labels)}\n -> A higher ensemble is completely disconnected from the ground state.")
        for z_label in zero_labels:
            print(f"{z_label} \t 0.0")
    else:
        print("\nNo paths with 0.0 WHAM weight.")

@dataclass(frozen=True)
class TrajWeights:
    weights: Dict[int, float]
    retis_steps: List[int]

    @classmethod
    def from_inputs(
        cls,
        raw_steps: List[str],
        load_dir: Path,
        interfaces: List[float],
        data_file: Path,
        approximate_threshold: float,
        approximate_factor: float,
        silent: bool = False,
        cache_filename: str = "wham_weights_cache.npz",
    ) -> "TrajWeights":
        """
        Compute (or load cached) WHAM weights.

        Cache key depends on:
          - interfaces
          - data_file path
          - approximate_threshold / approximate_factor
          - raw_steps (dirs used)

        The cache file is stored in load_dir / cache_filename.
        """
        cache_path = load_dir / cache_filename

        # ---- 1) Prøv å laste cache ----
        if cache_path.exists():
            try:
                data = np.load(cache_path, allow_pickle=True)
                meta = data["meta"].item()  # meta er et dict lagret via allow_pickle

                same_interfaces = np.allclose(
                    meta["interfaces"], np.asarray(interfaces, float)
                )
                same_datafile = Path(meta["data_file"]) == data_file
                same_thresh = meta["approximate_threshold"] == approximate_threshold
                same_factor = meta["approximate_factor"] == approximate_factor
                same_raw_steps = meta["raw_steps"] == list(raw_steps)

                if same_interfaces and same_datafile and same_thresh and same_factor and same_raw_steps:
                    if not silent:
                        print(f"[WHAM CACHE] Using cached weights from {cache_path}")
                    retis_steps = data["retis_steps"].astype(int).tolist()
                    w_keys = data["w_keys"].astype(int)
                    w_vals = data["w_vals"].astype(float)
                    weights = {int(k): float(v) for k, v in zip(w_keys, w_vals)}
                    if not silent:
                        print_wham_weights(retis_steps, weights)
                    return cls(weights=weights, retis_steps=retis_steps)
                else:
                    if not silent:
                        print("[WHAM CACHE] Cache exists but metadata mismatch → recomputing WHAM weights.")
            except Exception as e:
                if not silent:
                    print(f"[WHAM CACHE] Failed to read cache ({e}) → recomputing WHAM weights.")

        # ---- 2) Ingen gyldig cache: kjør full WHAM ----
        dirs = [str(load_dir / step) for step in raw_steps]

        retis_steps, weights = get_WHAMweights(
            dirs=dirs,
            lambda_interfaces=np.array(interfaces),
            ifile=str(data_file),
            approximate_threshold=approximate_threshold,
            approximate_factor=approximate_factor,
            silent=silent,
        )

        # ---- 3) Lagre cache for fremtidige runs ----
        try:
            meta = {
                "interfaces": np.asarray(interfaces, float),
                "data_file": str(data_file),
                "approximate_threshold": approximate_threshold,
                "approximate_factor": approximate_factor,
                "raw_steps": list(raw_steps),
            }
            np.savez(
                cache_path,
                retis_steps=np.asarray(retis_steps, int),
                w_keys=np.asarray(list(weights.keys()), int),
                w_vals=np.asarray(list(weights.values()), float),
                meta=meta,
            )
            if not silent:
                print(f"[WHAM CACHE] Saved WHAM weights to {cache_path}")
        except Exception as e:
            if not silent:
                print(f"[WHAM CACHE] Failed to write cache ({e}) → continuing without cache.")

        return cls(weights=weights, retis_steps=retis_steps)

    def valid_steps(self) -> List[int]:
        return self.retis_steps

    def get_weight(self, step: int) -> float:
        return self.weights[step]

def get_WHAMfactors(
    matrix: np.ndarray,
    lambda_interfaces: np.ndarray,
    i0_plus: int,
    Q: List[float]
) -> List[float]:
    """
    Compute WHAM factors (Chi_X) for each trajectory row in the matrix.

    For each row x in `matrix`:
        1. Extract λ_max from column index `i0_plus + 2`.
        2. Find the largest interface λ_i (from lambda_interfaces[:-1]) that is strictly less than λ_max.
            - If none is found but λ_max == the first interface, treat index as 0.
            - Otherwise, raise an error.
        3. Let Q_max = Q[indexQ].
        4. Sum raw Cxy values for that trajectory across all plus‐ensembles:
            sumC = sum(row[i0_plus : i0_plus + n_plus_ens])
        5. Compute Chi_X = Q_max * sumC.
        6. Append Chi_X to the output list.

    Parameters:
    -----------
    matrix: np.ndarray
        2D array where each row corresponds to one RETIS trajectory’s data.
        Column layout:
            - Column `i0_plus + 2` stores λ_max for that trajectory.
            - Columns `i0_plus` through `i0_plus + (n_interfaces - 1)` hold Cxy counts (unweighted).
    lambda_interfaces: np.ndarray
        Array of all interface λ values, including λ_B at the end.
    i0_plus: int
        Index of the first “C[0+]” column in each row.
        Then λ_max is at column index = i0_plus + 2.
    Q: List[float]
        WHAM Q‐factors computed previously (length = n_plus_ens + 1).

    Returns:
    --------
    List[float]
        A list of WHAM factors (Chi_X) for each row in `matrix`.
    """
    # Determine how many plus-ensembles there are:
    #   Total interfaces including λ_B = len(lambda_interfaces)
    #   Number of plus-ensembles = (len(lambda_interfaces) - 1)
    n_interfaces = len(lambda_interfaces)
    n_plus_ens = n_interfaces - 1

    # λ_max is stored in column index i0_plus + 2
    col_lambda_max = 2

    # Interfaces used to pick Q_index (exclude λ_B at the end)
    interfaces_without_B = lambda_interfaces[:-1]

    wham_factors: List[float] = []

    for row in matrix:
        # 1) Extract λ_max for this trajectory
        lambda_max = row[col_lambda_max]

        # 2) Find the largest interface λ_i < lambda_max
        #    Build a list of indices where interface < lambda_max
        valid_indices = [
            idx for idx, val in enumerate(interfaces_without_B) if val < lambda_max
        ]

        if valid_indices:
            indexQ = max(valid_indices)
        else:
            # If no interface is strictly less, but lambda_max == first interface,
            # we treat indexQ = 0
            if np.isclose(lambda_max, interfaces_without_B[0]):
                indexQ = 0
            else:
                raise ValueError(
                    f"lambda_max ({lambda_max}) is ≤ all TIS interfaces: {interfaces_without_B}"
                )

        # 3) Retrieve Q_max = Q[indexQ]
        Q_max = Q[indexQ]

        # 4) Sum raw Cxy values across all plus-ensembles:
        #    Those Cxy columns run from i0_plus to i0_plus + (n_plus_ens - 1)
        sumC = np.sum(row[i0_plus : i0_plus + n_plus_ens])

        # 5) Compute Chi_X and append
        Chi_X = Q_max * sumC
        wham_factors.append(Chi_X)

    return wham_factors

def gcd_of_floats(numbers: List[float]) -> float:
    """
    Compute the greatest common divisor (GCD) of a list of floats by
    converting each to a Fraction (via limit_denominator) and then
    combining their numerators/denominators.

    Returns a float ≃ GCD of the original list.
    """
    from math import gcd
    from fractions import Fraction
    from functools import reduce

    fractions = [Fraction(number).limit_denominator() for number in numbers]
    numerators = [f.numerator for f in fractions]
    denominators = [f.denominator for f in fractions]

    gcd_numerators = reduce(gcd, numerators)
    lcm_denominator = reduce(lambda a, b: a * b // gcd(a, b), denominators)

    result = Fraction(gcd_numerators, lcm_denominator)
    return float(result)


def load_transition_matrix(ifile: str ) -> Tuple[List[int], np.ndarray]:
    """
    Read a whitespace‐delimited file (skipping lines starting with '#'),
    replace any '----' with 0.0, and return a NumPy array containing only
    rows whose first column (as int) is in valid_steps.

    Each row becomes a list of floats; we filter by row[0] ∈ valid_steps.
    """
    matrix_rows = []
    with open(ifile) as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.strip().split()
            row = [float(x) if x != "----" else 0.0 for x in parts]
            matrix_rows.append(row)

    mat = np.array(matrix_rows)
    # Sort by first column (trajectory index), then keep only valid_steps
    mat = mat[mat[:, 0].argsort()]
    retis_steps = [int(x) for x in mat[:, 0]]
    return retis_steps, mat


def unweight_ha_weights(
    matrix: np.ndarray,
    n_interfaces: int,
    i0_plus: int
) -> Tuple[np.ndarray, List[float], List[float]]:
    """
    Given a matrix of shape (Nrows, Mcols), where:
        - The “Cxy” raw counts for [0+] appear at column index i0_plus,
        - The HA‐weights appear at column i0_plus + n_interfaces,
        - and so on for each interface y ∈ {0-, 0+, 1+, …},

    Perform the “unweighting” step:
        1) For each row x and each interface y, divide raw Cxy by its HA‐weight.
        2) Keep running sums of before‐weight (sumPxy) and after‐weight (sumPxy_afterw)
            per interface.

    We return:
        - The modified matrix (with Cxy / HA‐weight inserted back into x[y_index]),
        - A list sumPxy[y] (sum before‐weight for each interface y),
        - A list sumPxy_afterw[y] (sum after‐weight for each interface y).

    If HA‐weight is zero but raw Cxy>0, we raise a ZeroDivisionError.
    """
    i0_minus = i0_plus - 1
    sumPxy = [0.0] * n_interfaces
    sumPxy_afterw = [0.0] * n_interfaces

    for row in matrix:
        for y in range(n_interfaces):
            y1 = i0_minus + y         # index of raw Cxy
            y2 = y1 + n_interfaces    # index of HA‐weight for that same ensemble

            raw = row[y1]
            weight = row[y2]

            if weight > 0.0:
                sumPxy[y] += raw
                row[y1] = raw / weight
                sumPxy_afterw[y] += row[y1]
            elif raw > 0.0:
                raise ZeroDivisionError(f"HA‐weight is zero but raw Cxy={raw} for row {row}.")
            # else if raw=weight=0 ⇒ row[y1] stays 0.0

            # Store the running sum of raw Cxy back into the HA‐weight column:
            row[y2] = sumPxy[y]

    return matrix, sumPxy, sumPxy_afterw


def normalize_after_unweight(
    matrix: np.ndarray,
    n_interfaces: int,
    i0_plus: int,
    sumPxy: List[float],
    sumPxy_afterw: List[float]
) -> np.ndarray:
    """
    After un‐weighting, divide each column of Cxy by the average inverse HA‐weight for that interface y.

    For each interface y:
    AvInvW[y] = sumPxy_afterw[y] / sumPxy[y]  (if sumPxy[y]>0; else 1.0)
    Then for every row, row[y1] = row[y1] / AvInvW[y]

    Returns the modified matrix in place (same object).
    """
    i0_minus = i0_plus - 1

    for y in range(n_interfaces):
        if sumPxy[y] > 0.0:
            avg_invw = sumPxy_afterw[y] / sumPxy[y]
        else:
            avg_invw = 1.0

        y1 = i0_minus + y
        for row in matrix:
            row[y1] /= avg_invw

    return matrix


def wham_pq(
    n_plus_ens: int,
    lambda_interfaces: np.ndarray,
    lamres: float,
    eta: List[float],
    v_alpha: List[float]
) -> Tuple[List[float], List[float]]:
    """
    Compute P and Q (WHAM crossing probabilities and Q‐factors) using Lervik’s formulas (JCTC 2015).

    Inputs:
        - n_plus_ens   = number of “plus-ensembles” (n_interfaces - 1)
        - lambda_interfaces: array of interface values [λ0, λ1, …, λN]
        - lamres: bin‐width resolution
        - eta: list of length n_plus_ens containing Σ Cxy for each ensemble y
        - v_alpha: list of length len(lambda_values) (running v_α crossing histogram)

    Returns two lists:
        - P (length = n_plus_ens + 1)  # P_A(λi | λ0)
        - Q (length = n_plus_ens + 1)  # Q‐factors
    """
    P = [0.0] * (n_plus_ens + 1)
    Q = [0.0] * (n_plus_ens + 1)
    invQ = [0.0] * (n_plus_ens + 1)

    # Base case:
    P[0] = 1.0
    invQ[0] = eta[0]
    if invQ[0] == 0.0:
        return P, Q
    Q[0] = 1.0 / invQ[0]

    lambdaA = lambda_interfaces[0]
    for i in range(1, n_plus_ens):
        lambda_i = lambda_interfaces[i]
        alpha = round((lambda_i - lambdaA) / lamres)

        P[i] = v_alpha[alpha] * Q[i - 1]
        if P[i] == 0.0:
            return P, Q
        print(f"i={i}, lambda_i={lambda_i:.5f}, alpha={alpha}, v_alpha[alpha]={v_alpha[alpha]:.5f}, Q[i-1]={Q[i-1]:.5f}, P[i]={P[i]:.5f}")
        print(len(eta))
        invQ[i] = invQ[i - 1] + (eta[i] / P[i])
        Q[i] = 1.0 / invQ[i]

    # Finally, append the crossing at λ_B:
    P.append(v_alpha[-1] * Q[n_plus_ens - 1])
    return P, Q


def get_WHAMweights(
    dirs: List[str],
    lambda_interfaces: np.ndarray,
    ifile: str,
    approximate_threshold: float,
    approximate_factor: float,
    silent: bool = False
) -> Tuple[List[int], Dict[int, float]]:
    """
    Main driver to compute WHAM weights (WFtot) for RETIS trajectories.

    Steps:
    1) Compute differences between adjacent interfaces, round to 5 decimals.
    2) Compute lamres = GCD(differences).
    3) Optionally adjust lamres by approximate_factor if lamres ≤ approximate_threshold.
    4) Identify retis_steps = [basename(d) for d in dirs if basename(d).isdigit()].
      5) Build λ‐grid = [i * lamres for i in range(...)],
    initialize v_alpha, u_alpha, p_loc, eta, etc.
    6) Read the data file into a matrix, filter rows by retis_steps.
    7) Unweight HA‐weights and normalize.
    8) Loop over each row to fill eta, p_loc, v_alpha.
    9) Compute P, Q via wham_pq.
    10) Compute final WHAM factors (WFtot) using a helper (get_WHAMfactors).
    11) Return (trajlabels, WFtot).

    Returns:
    - List of trajectory labels (retis_steps).
    - A dict WFtot[label] = combined “semi-WHAM” weight for that trajectory.
    """
    # 1) Pairwise differences (rounded to 5 decimals).
    differences = [
        np.round(j - i, 5)
        for i, j in zip(lambda_interfaces[:-1], lambda_interfaces[1:])
    ]
    lamres = gcd_of_floats(differences)

    # 2) Print user‐friendly summary
    if not silent:
        print("=" * 50)
        print(f"Minimum raw difference between interfaces: {min(differences):.5f}")
        print(
            "Calculated GCD of differences: "
            f"{lamres:.5f}\n"
            "  → This 'lamres' is the bin‐width WHAM will use to index λ."
        )
        print(
            "If lamres is extremely small (e.g. ≈1e-05), you can speed up WHAM\n"
            "by multiplying lamres by a factor, provided that all true gaps\n"
            f"stay ≥ (factor × {lamres:.5f})."
        )
        print("Use --approximate_threshold and --approximate_factor to enable this.\n") #TODO: this is wrong. Should just ensure that all gaps stay >= lamres
        print("=" * 50, "\n")

    # 3) Apply approximate logic (skip if threshold = 0)
    if 0 < lamres <= approximate_threshold:
        min_gap = min(differences)
        # Maks faktor som fortsatt gir lamres_new <= min_gap
        max_safe_factor = min_gap / lamres if lamres > 0 else 1.0

        # Faktisk faktor vi bruker: begrenset både av brukerens ønske og fysikken
        effective_factor = min(approximate_factor, max_safe_factor)

        if effective_factor > 1.0:
            if not silent:
                print(
                    f"lamres ({lamres:.5f}) ≤ approximate_threshold ({approximate_threshold:.5f}) → "
                    f"multiplying by factor {effective_factor:.3f} (user factor={approximate_factor}, "
                    f"max safe={max_safe_factor:.3f})"
                )
            lamres *= effective_factor
            if not silent:
                print(f"Adjusted lamres: {lamres:.5f}\n")
        else:
            if not silent:
                print(
                    "Approximation requested, but lamres is already close to the minimum interface gap.\n"
                    "Keeping original lamres to avoid losing resolution.\n"
                )
    else:
        if not silent:
            print("No adjustment to lamres.\n")


    # 4) Determine retis_steps (trajectory indices) from directory names
    # retis_steps = sorted([int(os.path.basename(d)) for d in dirs if os.path.basename(d).isdigit()])

    # 5) Initialize several arrays/vectors for WHAM
    i0_plus = 4                         # “C[0+]” column index in data rows
    lambdaA = lambda_interfaces[0]
    lambdaB = lambda_interfaces[-1]
    n_interfaces = len(lambda_interfaces)
    n_plus_ens = n_interfaces - 1       # number of “plus-ensembles”

    # 5a) Create λ‐grid (from λA to λB in steps of lamres)
    alpha_start = round(lambdaA / lamres)
    alpha_end = round(lambdaB / lamres)
    lambda_values = [i * lamres for i in range(alpha_start, alpha_end + 1)]

    # 5b) Initialize crossing‐prob arrays
    v_alpha = [0.0] * len(lambda_values)
    u_alpha = [0.0] * len(lambda_values)
    v_alpha[0] = 1.0
    u_alpha[0] = 1.0

    # 5c) Initialize local‐crossing matrix p_loc[y][α]
    p_loc = [[0.0] * len(lambda_values) for _ in range(n_plus_ens)]
    # η[i] = Σ Cxy for each plus‐ensemble y = 0..n_plus_ens−1
    eta = [0.0] * n_plus_ens

    # 6) Load and filter transition matrix file
    print(f"Loading transition matrix from {ifile} " )
    retis_steps, matrix = load_transition_matrix(ifile)
    if not silent:
        print("Loaded matrix shape:", matrix.shape, "from file:", ifile)
        print("Requested retis_steps shape:", np.array(retis_steps).shape)

    # 7) Unweight HA‐weights and normalize
    matrix, sumPxy, sumPxy_afterw = unweight_ha_weights(
        matrix, n_interfaces, i0_plus
    )
    matrix = normalize_after_unweight(
        matrix, n_interfaces, i0_plus, sumPxy, sumPxy_afterw
    )

    # 8) Fill η, p_loc, v_alpha from each row
    for row in matrix:
        lambdamax = row[2]  # e.g. maximum λ for that trajectory
        for i in range(n_plus_ens):
            Cxy_index = i0_plus + i
            Cxy = row[Cxy_index]
            eta[i] += Cxy

            # Determine α_min/α_max for this row
            lambda_i = lambda_interfaces[i]
            alpha_min = round((lambda_i - lambdaA) / lamres)
            alpha_max = int(np.floor((lambdamax - lambdaA) / lamres))

            if alpha_max > len(v_alpha) - 1:
                alpha_max = len(v_alpha) - 1

            # Add to p_loc[i][α] for α ∈ [α_min, α_max]
            for α in range(alpha_min, alpha_max + 1):
                p_loc[i][α] += Cxy

            # Then increment v_alpha for α ∈ [α_min+1, α_max]
            for α in range(alpha_min + 1, alpha_max + 1):
                v_alpha[α] += Cxy

    # 9) Compute P, Q (WHAM crossing probabilities and Q‐factors)
    _, Q = wham_pq(n_plus_ens, lambda_interfaces, lamres, eta, v_alpha)
    if not silent:
        print("Computed Q‐factors:\n", Q,"\n")

    # 10) Compute final WHAM factors using an external helper
    #     (assumes get_WHAMfactors(matrix, lambda_interfaces, i0_plus, Q) exists)
    WHAMfactors = get_WHAMfactors(matrix, lambda_interfaces, i0_plus, Q)

    # 10a) “Semi” WHAM: just normalize [0-] counts (column i0_minus)
    i0_minus = i0_plus - 1
    WHAMfactorsMIN = [row[i0_minus] for row in matrix]
    sumWM = sum(WHAMfactorsMIN)
    if sumWM == 0.0:
        if not silent:
            print("sumWM is zero—skipping normalization (all zeros).")
    else:
        WHAMfactorsMIN = [w / sumWM for w in WHAMfactorsMIN]
        sumWM2 = sum(WHAMfactorsMIN)
        WHAMfactorsMIN = [w / sumWM2 for w in WHAMfactorsMIN]

    # 11) Build final WFtot per trajectory label
    trajlabels = [int(row[0]) for row in matrix]

    WFtot: Dict[int, float] = {}

    for idx, label in enumerate(trajlabels):
        WFtot[label] = WHAMfactorsMIN[idx] + WHAMfactors[idx]

    if not silent:
        # User requested 1) head of 50 in specific format, 2) all zeros
        print("Retis trajectory \t WF-weight for each path")
        printed_count = 0
        zero_labels = []

        for label in trajlabels:
            weight = WFtot[label]
            if weight == 0.0:
                zero_labels.append(label)

            if printed_count < 50:
                print(f"{label} \t {weight}")
                printed_count += 1

        if zero_labels:
            print(f"\nPaths with 0.0 WHAM weight: {len(zero_labels)}\n -> A higher ensmeble is completly disconnected from the ground state.")
            for z_label in zero_labels:
                print(f"{z_label} \t 0.0")
        else:
            print("\nNo paths with 0.0 WHAM weight.")

    return trajlabels, WFtot

