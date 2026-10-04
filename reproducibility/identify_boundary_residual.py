"""Identify the residual on the three exceptional D=0 boundary lines.

Away from the lines the direct obstruction equals

    q1 * (E_2/12)^p * theta^(i-p) f.

On the lines this script searches the difference for a single modular block
or a sparse span of blocks (E_2/12)^a theta^t f in the same quotient.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from identify_d0_obstructions import proportional
from identify_projection_corrections import solve_span
from verify_d0_block_formula import e2_over_12

sys.path.insert(0, str(Path(__file__).parent / "code"))
import branchfree  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--max-a", type=int)
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["p"]) == args.p
            and int(row["k"]) == args.k
            and int(row["reflected_D"]) == 2
        ]
    p, k = args.p, args.k
    environment = branchfree.build(p, k, p - 2, 3)
    length = environment["N"] + 1

    def conv(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.convolve(left, right)[:length] % p

    max_a = p if args.max_a is None else args.max_a
    x = e2_over_12(p, length)
    x_powers = [np.zeros(length, dtype=np.int64)]
    x_powers[0][0] = 1
    for _ in range(max(p, max_a)):
        x_powers.append(conv(x_powers[-1], x))
    theta_powers = {
        exponent: environment["theta"](environment["f"], exponent) % p
        for exponent in range(1, p)
    }

    supports = Counter()
    single_matches = Counter()
    tested = 0
    for row in sorted(rows, key=lambda item: (int(item["n"]), int(item["ip"]))):
        n, ip = int(row["n"]), int(row["ip"])
        if ip not in {n + 2, 2 * p - k - 1 - n, 2 * p - k - n}:
            continue
        tested += 1
        i = int(row["i"])
        lower_weight = int(row["lower_weight"])
        direct_remainder = environment["reduce_Mw"](
            environment["theta"](environment["f"], i), lower_weight, p * p
        )
        if direct_remainder is None or np.any(direct_remainder % p):
            raise RuntimeError(f"unexpected direct remainder at {(p, k, n, ip)}")
        direct = (direct_remainder // p) % p
        q1 = int(row["q1"]) % p
        shifted = environment["reduce_Mw"](
            q1
            * conv(x_powers[p], environment["theta"](environment["f"], i - p))
            % p,
            lower_weight,
            p,
        )
        if shifted is None:
            raise RuntimeError(f"shifted reduction failed at {(p, k, n, ip)}")
        residual = (direct - shifted) % p

        labels: list[tuple[int, int]] = []
        candidates: list[np.ndarray] = []
        matches: list[tuple[int, int, int]] = []
        for a in range(max_a + 1):
            exponent = (i - a) % (p - 1)
            if exponent == 0:
                exponent = p - 1
            candidate = environment["reduce_Mw"](
                conv(x_powers[a], theta_powers[exponent]), lower_weight, p
            )
            if candidate is None:
                continue
            candidate %= p
            labels.append((a, exponent))
            candidates.append(candidate)
            scalar = proportional(candidate, residual, p)
            if scalar is not None:
                matches.append((a, exponent, scalar))
                single_matches[(a, exponent)] += 1

        span = solve_span(candidates, residual, p)
        if span is None:
            print(f"{(n, ip)} line residual outside candidate span; singles={matches}")
            continue
        coefficients, rank = span
        support = np.flatnonzero(coefficients % p)
        supports[len(support)] += 1
        terms = [
            (*labels[int(index)], int(coefficients[int(index)]))
            for index in support
        ]
        print(
            f"{(n, ip)} q1={q1} rank={rank} support={len(support)} "
            f"singles={matches} terms={terms}"
        )

    print(f"tested boundary rows={tested}")
    print("single matches:", dict(single_matches))
    print("span support sizes:", dict(sorted(supports.items())))


if __name__ == "__main__":
    main()
