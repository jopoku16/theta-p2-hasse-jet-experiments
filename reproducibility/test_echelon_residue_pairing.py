"""Test anti-diagonal pairings of complementary D=0 and D=2 obstructions.

In an echelon q-basis, the coefficient of a product is an anti-diagonal
bilinear form.  This script asks whether any such coefficient is nonzero on
every nonvanishing cross-wedge pair.  A successful coefficient would be a
candidate shadow of the geometric residue pairing.  Failure only rules out
this simple coordinate model; it does not rule out Hasse duality.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from d1_obstruction_audit import quotient_boundary

sys.path.insert(0, str(Path(__file__).parent / "code"))
import branchfree  # noqa: E402


def anti_diagonal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return coefficients of the product of two coordinate polynomials."""
    return np.convolve(left, right)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--coordinates", type=int, default=64)
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        source = [
            row for row in csv.DictReader(handle) if int(row["reflected_D"]) == 2
        ]

    families: defaultdict[tuple[int, int], list[tuple[int, np.ndarray]]] = defaultdict(list)
    environments: dict[tuple[int, int], dict[str, object]] = {}
    for row in source:
        p, k = int(row["p"]), int(row["k"])
        key = (p, k)
        environment = environments.get(key)
        if environment is None:
            environment = branchfree.build(p, k, p - 2, 3)
            environments[key] = environment

        i0 = int(row["i"])
        i2 = p * p + 1 - k - i0
        period = p * (p - 1)
        candidate_weight = k + 2 * i2 + period
        remainder = environment["reduce_Mw"](
            environment["theta"](environment["f"], i2),
            candidate_weight,
            p * p,
        )
        if remainder is None or np.any(remainder % p):
            raise RuntimeError(f"unexpected D2 remainder at {(p, k, i2)}")
        obstruction = (remainder // p) % p
        boundary = quotient_boundary(candidate_weight)
        right = np.zeros(args.coordinates, dtype=np.int64)
        available = obstruction[boundary : boundary + args.coordinates]
        right[: len(available)] = available
        left = np.array(
            [int(row[f"direct_{j}"]) for j in range(args.coordinates)],
            dtype=np.int64,
        )
        q1 = int(row["q1"]) % p
        families[key].append((q1, anti_diagonal(left, right) % p))

    total_nonzero = 0
    total_zero = 0
    globally_successful = 0
    for (p, k), rows in sorted(families.items()):
        nonzero_rows = [values for q1, values in rows if q1]
        zero_rows = [values for q1, values in rows if not q1]
        total_nonzero += len(nonzero_rows)
        total_zero += len(zero_rows)
        matrix = np.vstack(nonzero_rows)
        zero_counts = np.count_nonzero(matrix == 0, axis=0)
        best = np.flatnonzero(zero_counts == zero_counts.min())
        successful = np.flatnonzero(zero_counts == 0)
        globally_successful += bool(successful.size)
        zero_ok = all(not np.any(values % p) for values in zero_rows)
        print(
            f"p={p} k={k}: nonzero={len(nonzero_rows)} zero={len(zero_rows)} "
            f"best_failures={int(zero_counts.min())} "
            f"best_indices={best[:12].tolist()} "
            f"perfect_indices={successful[:12].tolist()} zero_pairs_zero={zero_ok}"
        )

    print(
        f"families with a perfect anti-diagonal coefficient: "
        f"{globally_successful}/{len(families)}"
    )
    print(f"cross-wedge nonzero pairs={total_nonzero}; zero pairs={total_zero}")


if __name__ == "__main__":
    main()
