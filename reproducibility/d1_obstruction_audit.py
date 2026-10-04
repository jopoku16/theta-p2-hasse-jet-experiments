"""Audit the three-curve, four-coordinate criterion on D=1 cells.

For a D=1 cell, reduce theta^i f modulo p^2 by the echelon basis of its
nominal-weight modular-form space.  The remainder is divisible by p.  This
script tests whether defects are confined to three adjacent quadratics and
how many leading quotient coordinates are needed to identify them.

The output is finite computational evidence, not a theorem.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "code"))
import branchfree  # noqa: E402


def load_rows(paths: list[Path]) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend({key: int(value) for key, value in row.items()} for row in csv.DictReader(handle))
    return rows


def curve_name(row: dict[str, int]) -> str | None:
    p, k, n, ip = (row[name] for name in ("p", "k", "n", "ip"))
    c = (ip * ip + (k - 1) * ip) % p
    targets = (
        ("q0", n * (n + 1) % p),
        ("qm", (n + 1) ** 2 % p),
        ("q1", (n + 1) * (n + 2) % p),
    )
    return next((name for name, target in targets if c == target), None)


def quotient_boundary(weight: int) -> int:
    return weight // 12 if weight % 12 == 2 else weight // 12 + 1


def confusion_key(predicted: bool, observed: bool) -> str:
    return "tp" if predicted and observed else "fp" if predicted else "fn" if observed else "tn"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_paths", type=Path, nargs="+")
    parser.add_argument("--max-prefix", type=int, default=6)
    args = parser.parse_args()

    rows = [row for row in load_rows(args.csv_paths) if row["ordy"] == 1 and row["k"] < row["p"] - 1]
    families: defaultdict[tuple[int, int], list[dict[str, int]]] = defaultdict(list)
    for row in rows:
        families[(row["p"], row["k"])].append(row)

    prefix_counts = {length: Counter() for length in range(1, args.max_prefix + 1)}
    curve_counts: Counter[str] = Counter()
    total_d1 = 0

    for (p, k), family in sorted(families.items()):
        environment = branchfree.build(p, k, p - 2, 3)
        for row in family:
            if row["D"] != 1:
                continue
            total_d1 += 1
            name = curve_name(row)
            observed = row["D"] - row["delta"] >= 1
            curve_counts[f"{'drop' if observed else 'no_drop'}_{name or 'off'}"] += 1

            i = row["n"] * p + row["ip"]
            weight = k + 2 * i
            remainder = environment["reduce_Mw"](
                environment["theta"](environment["f"], i), weight, p * p
            )
            if remainder is None or np.any(remainder % p):
                raise RuntimeError(f"unexpected non-p-divisible remainder at {(p, k, row['n'], row['ip'])}")
            obstruction = (remainder // p) % p
            boundary = quotient_boundary(weight)
            for length, counts in prefix_counts.items():
                predicted = name is not None and not np.any(obstruction[boundary : boundary + length])
                counts[confusion_key(predicted, observed)] += 1

    print(f"ordinary nondegenerate D=1 cells: {total_d1}")
    print("three-curve distribution:", dict(sorted(curve_counts.items())))
    print("prefix length: tp fp fn tn")
    for length, counts in prefix_counts.items():
        print(
            f"{length:13d}: {counts['tp']:2d} {counts['fp']:2d} "
            f"{counts['fn']:2d} {counts['tn']:4d}"
        )


if __name__ == "__main__":
    main()
