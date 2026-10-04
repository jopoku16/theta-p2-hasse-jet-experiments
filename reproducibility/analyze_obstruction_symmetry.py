"""Compare exported D=1 obstruction prefixes under the two involutions.

This is a diagnostic for a possible transformation law.  Proportionality is
tested only on the exported coordinate prefix and is not asserted as a
theorem about the full obstruction vector.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def load(path: Path) -> list[dict[str, int | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, int | str]] = []
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    key: value if key == "curve" else int(value)
                    for key, value in raw.items()
                }
            )
        return rows


def proportional(left: tuple[int, ...], right: tuple[int, ...], p: int) -> bool:
    pivot = next((index for index, value in enumerate(left) if value % p), None)
    if pivot is None:
        return all(value % p == 0 for value in right)
    if right[pivot] % p == 0:
        return False
    scalar = right[pivot] * pow(left[pivot], -1, p) % p
    return all((scalar * x - y) % p == 0 for x, y in zip(left, right))


def classify(left: tuple[int, ...], right: tuple[int, ...], p: int) -> str:
    left_zero = all(value % p == 0 for value in left)
    right_zero = all(value % p == 0 for value in right)
    if left_zero and right_zero:
        return "both_zero_prefix"
    if left == right:
        return "equal_prefix"
    if proportional(left, right, p):
        return "proportional_prefix"
    if tuple(value % p == 0 for value in left) == tuple(value % p == 0 for value in right):
        return "same_zero_pattern"
    return "different"


def audit(rows: list[dict[str, int | str]], coordinate_names: list[str], transform: str) -> None:
    lookup = {
        (int(row["p"]), int(row["k"]), int(row["n"]), int(row["ip"])): row
        for row in rows
    }
    seen: set[tuple[int, int, int, int]] = set()
    classes: Counter[str] = Counter()
    defect_pairs: Counter[tuple[int, int]] = Counter()
    examples: list[str] = []

    for key, row in sorted(lookup.items()):
        p, k, n, ip = key
        if transform == "R":
            partner_key = (p, k, p - 2 - n, ip)
        elif transform == "S":
            partner_key = (p, k, n, (1 - k - ip) % p)
        else:
            partner_key = (p, k, p - 2 - n, (1 - k - ip) % p)
        if key in seen or partner_key not in lookup:
            continue
        seen.update((key, partner_key))
        partner = lookup[partner_key]
        left = tuple(int(row[name]) for name in coordinate_names)
        right = tuple(int(partner[name]) for name in coordinate_names)
        kind = classify(left, right, p)
        classes[kind] += 1
        defects = (int(row["defect"]), int(partner["defect"]))
        defect_pairs[defects] += 1
        if kind == "different" and len(examples) < 8:
            examples.append(f"{key} -> {partner_key}, defects={defects}")

    print(f"{transform}: {sum(classes.values())} complete pairs")
    print("  prefix relation:", dict(classes))
    print("  defect pairs:", dict(defect_pairs))
    for example in examples:
        print("   ", example)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--coordinates", type=int, default=12)
    args = parser.parse_args()
    rows = load(args.csv_path)
    coordinate_names = [f"o{index}" for index in range(args.coordinates)]
    for transform in ("R", "S", "RS"):
        audit(rows, coordinate_names, transform)


if __name__ == "__main__":
    main()
