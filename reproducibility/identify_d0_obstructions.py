"""Search for simple modular representatives of the D=0 obstruction.

For each selected D=0 cell, compare its canonical lower-weight obstruction
with the canonical remainder of E_{p+1}^a theta^t f modulo p.  Since
E_{p+1} is congruent to E_2 modulo p, the search is performed directly on
q-expansions.  A reported match is an exact equality over the complete
truncation used by the filtration computation, up to a nonzero scalar.

This is a conjecture-finding diagnostic, not a proof of a uniform identity.
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


def load(path: Path) -> list[dict[str, int | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: value if key == "curve" else int(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def proportional(left: np.ndarray, right: np.ndarray, p: int) -> int | None:
    """Return right/left when the vectors are proportional and nonzero."""
    support = np.flatnonzero(left % p)
    if not support.size:
        return None
    pivot = int(support[0])
    if int(right[pivot]) % p == 0:
        return None
    scalar = int(right[pivot]) * pow(int(left[pivot]), -1, p) % p
    if np.all((scalar * left - right) % p == 0):
        return scalar
    return None


def e2_series(p: int, length: int) -> np.ndarray:
    sigma = np.zeros(length, dtype=np.int64)
    for divisor in range(1, length):
        sigma[divisor::divisor] = (sigma[divisor::divisor] + divisor) % p
    result = (-24 * sigma) % p
    result[0] = 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--p", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--max-a", type=int)
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    rows = [
        row
        for row in load(args.csv_path)
        if int(row["defect"]) == 0
        and (args.p is None or int(row["p"]) == args.p)
        and (args.k is None or int(row["k"]) == args.k)
    ]
    families: defaultdict[tuple[int, int], list[dict[str, int | str]]] = defaultdict(list)
    for row in rows:
        families[(int(row["p"]), int(row["k"]))].append(row)

    match_counts: Counter[tuple[int, int]] = Counter()
    unmatched: list[tuple[int, int, int, int]] = []
    examples: list[str] = []

    for (p, k), family in sorted(families.items()):
        environment = branchfree.build(p, k, p - 2, 3)
        length = environment["N"] + 1

        def conv(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return np.convolve(left, right)[:length] % p

        max_a = args.max_a if args.max_a is not None else p
        e2 = e2_series(p, length)
        e2_powers = [np.eye(1, length, 0, dtype=np.int64).ravel()]
        for _ in range(max_a):
            e2_powers.append(conv(e2_powers[-1], e2))
        theta_powers = {
            exponent: environment["theta"](environment["f"], exponent) % p
            for exponent in range(1, p)
        }
        candidates: dict[tuple[int, int], np.ndarray] = {}
        for a in range(max_a + 1):
            for exponent in range(1, p):
                candidates[(a, exponent)] = conv(e2_powers[a], theta_powers[exponent])

        for row in family:
            n, ip = int(row["n"]), int(row["ip"])
            i = n * p + ip
            lower_weight = int(row["lower_weight"])
            remainder = environment["reduce_Mw"](
                environment["theta"](environment["f"], i), lower_weight, p * p
            )
            if remainder is None or np.any(remainder % p):
                raise RuntimeError(f"unexpected D0 remainder at {(p, k, n, ip)}")
            obstruction = (remainder // p) % p
            matches: list[tuple[int, int, int]] = []
            for a in range(max_a + 1):
                exponent = (i - a) % (p - 1)
                if exponent == 0:
                    exponent = p - 1
                candidate_remainder = environment["reduce_Mw"](
                    candidates[(a, exponent)], lower_weight, p
                )
                if candidate_remainder is None:
                    continue
                scalar = proportional(candidate_remainder, obstruction, p)
                if scalar is not None:
                    matches.append((a, exponent, scalar))
                    match_counts[(a, exponent)] += 1
            if not matches:
                unmatched.append((p, k, n, ip))
            elif len(examples) < args.examples:
                examples.append(f"{(p, k, n, ip)}: {matches}")

    print(f"tested {len(rows)} nonzero D0 obstructions")
    print(f"matched {len(rows) - len(unmatched)}; unmatched {len(unmatched)}")
    print("most frequent (a,t) matches:", match_counts.most_common(20))
    for example in examples:
        print("  ", example)
    if unmatched:
        print("first unmatched:", unmatched[:20])


if __name__ == "__main__":
    main()
