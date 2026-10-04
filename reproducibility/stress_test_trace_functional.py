"""Falsification tests for a putative q-expansion trace functional.

The target pairing is geometric.  A finite linear fit to one (p, k) family
is therefore useful only if it has additional structure.  This script tests
three increasingly weak surrogates:

1. one canonical quotient coordinate, up to a scalar;
2. one coefficient vector shared by all tested weights for a fixed prime;
3. deterministic odd/even held-out prediction within each family.

Failure does not disprove geometric Hasse duality: quotient coordinates move
with the weight, and the true trace may require a weight-dependent basis.
Success with rank close to the number of training rows is also not evidence,
because it can be interpolation.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from identify_projection_corrections import solve_span


def load_families(path: Path) -> dict[tuple[int, int], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    families: defaultdict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["reflected_D"]) == 2:
            families[(int(row["p"]), int(row["k"]))].append(row)
    for family in families.values():
        family.sort(key=lambda row: (int(row["n"]), int(row["ip"])))
    return dict(families)


def matrix(rows: list[dict[str, str]], count: int) -> np.ndarray:
    return np.array(
        [[int(row[f"direct_{j}"]) for j in range(count)] for row in rows],
        dtype=np.int64,
    )


def target(rows: list[dict[str, str]]) -> np.ndarray:
    return np.array([int(row["q1"]) for row in rows], dtype=np.int64)


def one_coordinate_matches(
    rows: list[dict[str, str]], p: int, count: int
) -> list[tuple[int, int]]:
    y = target(rows) % p
    matches: list[tuple[int, int]] = []
    for j in range(count):
        x = matrix(rows, j + 1)[:, j] % p
        nz = np.flatnonzero(x)
        if not nz.size:
            continue
        c = int(y[nz[0]]) * pow(int(x[nz[0]]), -1, p) % p
        if np.array_equal((c * x) % p, y):
            matches.append((j, c))
    return matches


def held_out_result(
    rows: list[dict[str, str]], p: int, count: int
) -> tuple[bool, int, int] | None:
    train_rows = rows[::2]
    test_rows = rows[1::2]
    if not test_rows:
        return None
    x_train = matrix(train_rows, count)
    answer = solve_span(
        [x_train[:, j] for j in range(count)], target(train_rows), p
    )
    if answer is None:
        return None
    coefficients, rank = answer
    predicted = matrix(test_rows, count) @ coefficients % p
    observed = target(test_rows) % p
    correct = int(np.count_nonzero(predicted == observed))
    return correct == len(test_rows), correct, rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--coordinates", type=int, default=64)
    args = parser.parse_args()

    families = load_families(args.csv_path)
    by_prime: defaultdict[int, list[tuple[int, list[dict[str, str]]]]] = defaultdict(list)
    for (p, k), rows in sorted(families.items()):
        by_prime[p].append((k, rows))

    print("ONE-COORDINATE TEST")
    one_coordinate_successes = 0
    for (p, k), rows in sorted(families.items()):
        matches = one_coordinate_matches(rows, p, args.coordinates)
        one_coordinate_successes += bool(matches)
        print(f"p={p} k={k}: rows={len(rows)} matches={matches}")

    print("\nSHARED-WEIGHT TEST")
    shared_successes = 0
    shared_tests = 0
    for p, blocks in sorted(by_prime.items()):
        if len(blocks) < 2:
            continue
        shared_tests += 1
        rows = [row for _, family in blocks for row in family]
        x = matrix(rows, args.coordinates)
        answer = solve_span(
            [x[:, j] for j in range(args.coordinates)], target(rows), p
        )
        if answer is None:
            print(
                f"p={p}: weights={[k for k, _ in blocks]} rows={len(rows)} "
                f"no shared functional"
            )
            continue
        coefficients, rank = answer
        support = int(np.count_nonzero(coefficients % p))
        shared_successes += 1
        print(
            f"p={p}: weights={[k for k, _ in blocks]} rows={len(rows)} "
            f"rank={rank} support={support}"
        )

    print("\nODD/EVEN HELD-OUT TEST")
    prefix_sizes = [size for size in (4, 8, 12, 16, 24, 32, 48, 64) if size <= args.coordinates]
    held_out_successes = 0
    held_out_tests = 0
    for (p, k), rows in sorted(families.items()):
        outcomes: list[str] = []
        for count in prefix_sizes:
            result = held_out_result(rows, p, count)
            if result is None:
                outcomes.append(f"{count}:no-fit")
                continue
            exact, correct, rank = result
            held_out_tests += 1
            held_out_successes += exact
            outcomes.append(
                f"{count}:{correct}/{len(rows[1::2])},rank={rank}"
            )
        print(f"p={p} k={k}: " + "; ".join(outcomes))

    print("\nSUMMARY")
    print(
        f"one-coordinate families={one_coordinate_successes}/{len(families)}; "
        f"shared-weight primes={shared_successes}/{shared_tests}; "
        f"exact held-out fits={held_out_successes}/{held_out_tests}"
    )


if __name__ == "__main__":
    main()
