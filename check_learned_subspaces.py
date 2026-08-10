#!/usr/bin/env python3

import re
from pathlib import Path

import numpy as np


RUN = Path("results/observable_design_toy/confirmatory/R3")
CHECKPOINTS = RUN / "checkpoints"

OBJECTIVES = ("info", "cv", "fiber")
N_SEEDS = 2

RAW_NAMES = [
    "x1",
    "x2",
    "x1^2",
    "x1*x2",
    "x2^2",
]


# ---------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------

def projection_matrix(A):
    """
    A is assumed row-orthonormal: A @ A.T ~= I.
    P projects onto the row space of A.
    """
    return A.T @ A


def subspace_distance(A, B):
    """
    Projection-Frobenius distance used in the experiment brief.

    0 = identical subspaces.
    For equal R, larger = more different.
    """
    R = A.shape[0]
    PA = projection_matrix(A)
    PB = projection_matrix(B)

    return np.linalg.norm(PA - PB, ord="fro") / np.sqrt(2 * R)


def principal_angles(A, B):
    """
    Principal angles between the row spaces.

    Returns radians and degrees.
    """
    # Since rows are orthonormal, singular values of A B^T
    # are cosines of the principal angles.
    s = np.linalg.svd(A @ B.T, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)

    rad = np.arccos(s)
    deg = np.degrees(rad)

    return rad, deg


def procrustes_align(A_reference, A_other):
    """
    Rotate the ROW BASIS of A_other so it is as close as possible
    to A_reference.

    This does NOT change the subspace defined by A_other.
    It is useful only for looking at the matrices visually.
    """
    M = A_reference @ A_other.T
    U, _, Vt = np.linalg.svd(M)

    Q = U @ Vt
    return Q @ A_other


# ---------------------------------------------------------------------
# Load checkpoints
# ---------------------------------------------------------------------

pattern = re.compile(
    r"observable_(info|cv|fiber)_modelseed_(\d+)\.npz$"
)

found = {objective: {} for objective in OBJECTIVES}

for path in CHECKPOINTS.glob("observable_*_modelseed_*.npz"):
    match = pattern.match(path.name)

    if not match:
        continue

    objective = match.group(1)
    seed = int(match.group(2))

    data = np.load(path, allow_pickle=True)
    A = np.asarray(data["A"], dtype=float)

    found[objective][seed] = {
        "A": A,
        "path": path,
    }


# Require seeds completed for ALL three objectives.
common_seeds = sorted(
    set(found["info"])
    & set(found["cv"])
    & set(found["fiber"])
)

if len(common_seeds) < N_SEEDS:
    raise SystemExit(
        f"Need {N_SEEDS} common completed seeds, "
        f"but found only {common_seeds}"
    )

seeds = common_seeds[:N_SEEDS]

print("\nUsing model seeds:", seeds)


# ---------------------------------------------------------------------
# Shared standardization
# ---------------------------------------------------------------------

std_path = RUN / "design_standardization.npz"

if std_path.exists():
    std = np.load(std_path)

    center = np.asarray(std["center"], dtype=float)
    W = np.asarray(std["whitening"], dtype=float)

else:
    center = None
    W = None
    print(
        "\nWARNING: design_standardization.npz not found. "
        "Raw-basis coefficients will not be printed."
    )


# ---------------------------------------------------------------------
# 1. Print actual learned A matrices
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("LEARNED A MATRICES — standardized/whitened coordinates")
print("=" * 80)

for seed in seeds:
    print(f"\nMODEL SEED {seed}")

    for objective in OBJECTIVES:
        A = found[objective][seed]["A"]

        print(f"\n{objective.upper()}")
        print(A)

        gram = A @ A.T

        print(
            "max |A A^T - I| =",
            np.max(np.abs(gram - np.eye(A.shape[0])))
        )


# ---------------------------------------------------------------------
# 2. Convert to physically interpretable raw basis
#
# Phi = A W (b - center)
#
# therefore:
#
# Phi = C b + intercept
#
# C = A W
# intercept = -C center
# ---------------------------------------------------------------------

if W is not None:

    print("\n" + "=" * 80)
    print("COEFFICIENTS IN ORIGINAL PHYSICAL BASIS")
    print("basis =", RAW_NAMES)
    print("=" * 80)

    for seed in seeds:
        print(f"\nMODEL SEED {seed}")

        for objective in OBJECTIVES:
            A = found[objective][seed]["A"]

            C = A @ W
            intercept = -C @ center

            print(f"\n{objective.upper()}")

            for r, row in enumerate(C):
                terms = ", ".join(
                    f"{name}={coef:+.5f}"
                    for name, coef in zip(RAW_NAMES, row)
                )

                print(
                    f"  phi_{r + 1}: "
                    f"{terms}; "
                    f"intercept={intercept[r]:+.5f}"
                )


# ---------------------------------------------------------------------
# 3. Seed-to-seed stability WITHIN each objective
# ---------------------------------------------------------------------

seed1, seed2 = seeds

print("\n" + "=" * 80)
print(f"SEED STABILITY: seed {seed1} vs seed {seed2}")
print("=" * 80)

for objective in OBJECTIVES:
    A1 = found[objective][seed1]["A"]
    A2 = found[objective][seed2]["A"]

    distance = subspace_distance(A1, A2)
    _, angles_deg = principal_angles(A1, A2)

    print(f"\n{objective.upper()}")
    print(f"  subspace distance = {distance:.8f}")
    print(
        "  principal angles =",
        np.round(angles_deg, 4),
        "degrees",
    )
    print(
        f"  max angle         = "
        f"{np.max(angles_deg):.4f} degrees"
    )
    print(
        f"  mean angle        = "
        f"{np.mean(angles_deg):.4f} degrees"
    )


# ---------------------------------------------------------------------
# 4. Compare INFO / CV / FIBER WITHIN each seed
# ---------------------------------------------------------------------

pairs = [
    ("fiber", "info"),
    ("fiber", "cv"),
    ("info", "cv"),
]

print("\n" + "=" * 80)
print("OBJECTIVE-TO-OBJECTIVE SUBSPACE DIFFERENCES")
print("=" * 80)

for seed in seeds:

    print(f"\nMODEL SEED {seed}")

    for left, right in pairs:
        A = found[left][seed]["A"]
        B = found[right][seed]["A"]

        distance = subspace_distance(A, B)
        _, angles_deg = principal_angles(A, B)

        print(
            f"  {left:5s} vs {right:5s}: "
            f"distance={distance:.6f}  "
            f"angles(deg)="
            f"{np.round(angles_deg, 3)}"
        )


# ---------------------------------------------------------------------
# 5. Optional: align seed-2 row basis to seed-1
#
# Use this if you want to inspect whether the MATRICES themselves
# become similar after removing arbitrary row rotations.
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("PROCRUSTES-ALIGNED SEED-2 MATRICES")
print("=" * 80)

for objective in OBJECTIVES:
    A1 = found[objective][seed1]["A"]
    A2 = found[objective][seed2]["A"]

    A2_aligned = procrustes_align(A1, A2)

    print(f"\n{objective.upper()}")

    print("\nseed 1:")
    print(np.round(A1, 5))

    print("\nseed 2 aligned to seed 1:")
    print(np.round(A2_aligned, 5))

    print(
        "\nRMS aligned matrix difference =",
        np.sqrt(np.mean((A1 - A2_aligned) ** 2)),
    )


print("\nDone.")