"""Probe the canonical lower-weight obstruction for D=1 band cells.

For a D=1 cell, an additional drop occurs exactly when theta^i f belongs
to its nominal-weight modular-form space modulo p^2.  The echelon reduction
used by branchfree.py gives a canonical remainder.  Because the reduction
vanishes modulo p, division by p produces an obstruction vector over F_p.

This script reports the rank and support of those vectors.  It is an
exploratory diagnostic, not a theorem or a fitted classification rule.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "code"))
import branchfree  # noqa: E402


def load_rows(path: Path) -> list[dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: int(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def probe(path: Path) -> None:
    rows = load_rows(path)
    p, k = rows[0]["p"], rows[0]["k"]
    d1 = [row for row in rows if row["D"] == 1]
    environment = branchfree.build(p, k, p - 2, 3)
    modulus = environment["M"]
    vectors: list[np.ndarray] = []
    labels: list[tuple[int, int, int]] = []
    leading_obstructions: list[int] = []
    support_offsets: Counter[int] = Counter()
    not_divisible = 0

    for row in d1:
        i = row["n"] * p + row["ip"]
        g = environment["theta"](environment["f"], i)
        remainder = environment["reduce_Mw"](g, k + 2 * i, modulus)
        if remainder is None or np.any(remainder % p):
            not_divisible += 1
            continue
        obstruction = (remainder // p) % p
        vectors.append(obstruction)
        labels.append((row["n"], row["ip"], row["delta"]))
        target_weight = k + 2 * i
        first_forbidden = target_weight // 12 if target_weight % 12 == 2 else target_weight // 12 + 1
        leading_obstructions.append(int(obstruction[first_forbidden]))
        support = np.flatnonzero(obstruction)
        if len(support):
            support_offsets[int(support[0]) - first_forbidden] += 1

    matrix = np.stack(vectors) if vectors else np.zeros((0, 0), dtype=np.int64)
    zero_indices = [index for index, vector in enumerate(vectors) if not np.any(vector)]
    leading_zero_indices = [index for index, value in enumerate(leading_obstructions) if value == 0]
    leading_false_zeros = [index for index in leading_zero_indices if np.any(vectors[index])]

    print(f"\n{path.name}: (p,k)=({p},{k})")
    print(f"  D=1 cells: {len(d1)}")
    print(f"  obstruction vectors divisible by p: {len(vectors)}/{len(d1)}")
    print(f"  divisibility failures: {not_divisible}")
    print(f"  zero obstruction vectors: {len(zero_indices)}")
    print(f"  leading-coordinate zeros: {len(leading_zero_indices)}")
    print(f"  leading-coordinate false zeros: {len(leading_false_zeros)}")
    print(f"  first-support offsets from quotient boundary: {dict(sorted(support_offsets.items()))}")
    print("  zeros (n,ip,delta):", [labels[index] for index in zero_indices])
    if leading_false_zeros:
        print("  false leading zeros (n,ip,delta):", [labels[index] for index in leading_false_zeros])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_paths", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.csv_paths:
        probe(path)


if __name__ == "__main__":
    main()
