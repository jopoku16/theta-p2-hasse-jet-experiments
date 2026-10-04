"""Fit quotient-coordinate functionals detecting the shifted conic.

For each (p,k) family in the reflected D=2 wedge, solve over F_p for a
linear functional on the first canonical D=0 obstruction coordinates whose
value is the shifted-conic polynomial q1. A successful fit is an exact
finite-data identity and a candidate residue functional, not a proof for
general p.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from identify_projection_corrections import solve_span


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--coordinates", type=int, default=12)
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    families: defaultdict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["reflected_D"]) == 2:
            families[(int(row["p"]), int(row["k"]))].append(row)

    failures: list[tuple[int, int]] = []
    for (p, k), family in sorted(families.items()):
        target = np.array([int(row["q1"]) for row in family], dtype=np.int64)
        solution = None
        used = 0
        rank = 0
        for coordinate_count in range(1, args.coordinates + 1):
            columns = [
                np.array(
                    [int(row[f"direct_{index}"]) for row in family],
                    dtype=np.int64,
                )
                for index in range(coordinate_count)
            ]
            answer = solve_span(columns, target, p)
            if answer is not None:
                solution, rank = answer
                used = coordinate_count
                break
        if solution is None:
            failures.append((p, k))
            print(f"p={p} k={k}: no functional in first {args.coordinates} coordinates")
            continue
        support = [
            (index, int(coefficient))
            for index, coefficient in enumerate(solution)
            if coefficient % p
        ]
        print(
            f"p={p} k={k}: rows={len(family)} first={used} "
            f"rank={rank} support={support}"
        )

    print(f"families={len(families)} failures={len(failures)}")
    if failures:
        print("failed families:", failures)


if __name__ == "__main__":
    main()
