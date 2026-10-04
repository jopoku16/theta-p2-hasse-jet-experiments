"""Test whether triangular boundary coefficients are prime-independent.

The restricted echelon coefficients are grouped by weight k, boundary line,
m=p-2-n, and relative position.  For groups represented at three or more
primes, the script searches for a small rational number whose reduction
modulo every prime equals the observed coefficient.  This distinguishes a
plausible universal combinatorial coefficient from a coordinate-dependent
projection value.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def small_rational(
    observations: list[tuple[int, int]], bound: int
) -> tuple[int, int] | None:
    for denominator in range(1, bound + 1):
        if any(denominator % p == 0 for p, _ in observations):
            continue
        for numerator in range(-bound, bound + 1):
            if math.gcd(abs(numerator), denominator) != 1:
                continue
            if all(
                numerator * pow(denominator, -1, p) % p == value % p
                for p, value in observations
            ):
                return numerator, denominator
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--bound", type=int, default=100)
    parser.add_argument("--minimum-primes", type=int, default=3)
    parser.add_argument("--examples", type=int, default=20)
    args = parser.parse_args()

    groups: defaultdict[
        tuple[int, str, int, int, int], list[tuple[int, int]]
    ] = defaultdict(list)
    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                int(row["k"]),
                row["line"],
                int(row["m"]),
                int(row["position"]),
                int(row["endpoint"]),
            )
            groups[key].append((int(row["p"]), int(row["coefficient"])))

    eligible = {
        key: sorted(values)
        for key, values in groups.items()
        if len({p for p, _ in values}) >= args.minimum_primes
    }
    stable: list[tuple[tuple[int, str, int, int, int], tuple[int, int]]] = []
    unstable: list[tuple[tuple[int, str, int, int, int], list[tuple[int, int]]]] = []
    for key, values in sorted(eligible.items()):
        rational = small_rational(values, args.bound)
        if rational is None:
            unstable.append((key, values))
        else:
            stable.append((key, rational))

    print(
        f"eligible_groups={len(eligible)} stable_small_rational={len(stable)} "
        f"unstable={len(unstable)} bound={args.bound}"
    )
    print("stable examples (key -> rational):")
    for key, rational in stable[: args.examples]:
        print(f"  {key} -> {rational[0]}/{rational[1]}")
    print("unstable examples (key -> (p,value)): ")
    for key, values in unstable[: args.examples]:
        print(f"  {key} -> {values}")


if __name__ == "__main__":
    main()
