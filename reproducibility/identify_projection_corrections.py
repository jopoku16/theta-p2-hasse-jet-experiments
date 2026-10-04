"""Search for simple modular representatives of the D=0 correction Pi_i.

Candidates have the form E_(p+1)^a theta^t f modulo p, with the exponent
t forced by the weight class. A match is checked on the complete
Sturm-safe q-expansion, not only on exported coordinate prefixes. This is
a conjecture-finding diagnostic and does not establish a uniform identity.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from identify_d0_obstructions import proportional
from verify_d0_block_formula import coefficient_unit, e2_over_12, load

sys.path.insert(0, str(Path(__file__).parent / "code"))
import branchfree  # noqa: E402


def solve_span(
    columns: list[np.ndarray], target: np.ndarray, p: int
) -> tuple[np.ndarray, int] | None:
    """Return one solution and the column rank for A x = target over F_p."""
    matrix = np.column_stack([*columns, target]) % p
    active = np.any(matrix != 0, axis=1)
    matrix = matrix[active].copy()
    column_count = len(columns)
    pivot_row = 0
    pivots: list[tuple[int, int]] = []
    for column in range(column_count):
        candidates = np.flatnonzero(matrix[pivot_row:, column] % p)
        if not candidates.size:
            continue
        row = pivot_row + int(candidates[0])
        matrix[[pivot_row, row]] = matrix[[row, pivot_row]]
        matrix[pivot_row] = (
            matrix[pivot_row] * pow(int(matrix[pivot_row, column]), -1, p)
        ) % p
        factors = matrix[:, column].copy()
        factors[pivot_row] = 0
        matrix = (matrix - factors[:, None] * matrix[pivot_row]) % p
        pivots.append((pivot_row, column))
        pivot_row += 1
        if pivot_row == matrix.shape[0]:
            break

    for row in range(matrix.shape[0]):
        if not np.any(matrix[row, :column_count]) and matrix[row, -1] % p:
            return None
    solution = np.zeros(column_count, dtype=np.int64)
    for row, column in pivots:
        solution[column] = matrix[row, -1] % p
    return solution, len(pivots)


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
        if (args.p is None or int(row["p"]) == args.p)
        and (args.k is None or int(row["k"]) == args.k)
    ]
    families: defaultdict[tuple[int, int], list[dict[str, int | str]]] = (
        defaultdict(list)
    )
    for row in rows:
        families[(int(row["p"]), int(row["k"]))].append(row)

    matches_by_a: Counter[int] = Counter()
    span_ranks: Counter[int] = Counter()
    span_supports: Counter[int] = Counter()
    unmatched: list[tuple[int, int, int, int]] = []
    outside_span: list[tuple[int, int, int, int]] = []
    examples: list[str] = []
    span_examples: list[str] = []

    for (p, k), family in sorted(families.items()):
        environment = branchfree.build(p, k, p - 2, 3)
        length = environment["N"] + 1

        def conv(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return np.convolve(left, right)[:length] % p

        x = e2_over_12(p, length)
        x_powers = [np.zeros(length, dtype=np.int64)]
        x_powers[0][0] = 1
        max_a = args.max_a if args.max_a is not None else p
        for _ in range(max(p, max_a)):
            x_powers.append(conv(x_powers[-1], x))
        theta_powers = {
            exponent: environment["theta"](environment["f"], exponent) % p
            for exponent in range(1, p)
        }

        for row in family:
            n, ip = int(row["n"]), int(row["ip"])
            i = n * p + ip
            lower_weight = int(row["lower_weight"])
            remainder = environment["reduce_Mw"](
                environment["theta"](environment["f"], i),
                lower_weight,
                p * p,
            )
            if remainder is None or np.any(remainder % p):
                raise RuntimeError(f"unexpected direct remainder at {(p, k, n, ip)}")
            direct = (remainder // p) % p

            jmax = i - p
            ka = coefficient_unit(i, jmax, k, p)
            raw = (
                (ka + i * (i + k - 1))
                * conv(
                    x_powers[p],
                    environment["theta"](environment["f"], jmax) % p,
                )
            ) % p
            for exponent_theta in range(n * p, n * p + p - k + 1):
                exponent_x = i - exponent_theta
                upper = n * p + p - k - exponent_theta
                coefficient = coefficient_unit(i, exponent_theta, k, p)
                coefficient *= (-1) ** upper * math.comb(
                    i - exponent_theta - 1, upper
                )
                raw = (
                    raw
                    + coefficient
                    * conv(
                        x_powers[exponent_x],
                        environment["theta"](
                            environment["f"], exponent_theta
                        )
                        % p,
                    )
                ) % p
            naive = environment["reduce_Mw"](raw, lower_weight, p)
            if naive is None:
                raise RuntimeError(f"naive reduction failed at {(p, k, n, ip)}")
            correction = (direct - naive) % p

            matches: list[tuple[int, int, int]] = []
            candidate_labels: list[tuple[int, int]] = []
            candidate_remainders: list[np.ndarray] = []
            for a in range(max_a + 1):
                exponent = (i - a) % (p - 1)
                if exponent == 0:
                    exponent = p - 1
                candidate = conv(x_powers[a], theta_powers[exponent])
                candidate_remainder = environment["reduce_Mw"](
                    candidate, lower_weight, p
                )
                if candidate_remainder is None:
                    continue
                candidate_labels.append((a, exponent))
                candidate_remainders.append(candidate_remainder % p)
                scalar = proportional(candidate_remainder, correction, p)
                if scalar is not None:
                    matches.append((a, exponent, scalar))
                    matches_by_a[a] += 1

            if not matches:
                unmatched.append((p, k, n, ip))
            elif len(examples) < args.examples:
                examples.append(f"{(p, k, n, ip)}: {matches}")

            span = solve_span(candidate_remainders, correction, p)
            if span is None:
                outside_span.append((p, k, n, ip))
            else:
                solution, rank = span
                support = np.flatnonzero(solution)
                span_ranks[rank] += 1
                span_supports[len(support)] += 1
                if len(span_examples) < args.examples:
                    terms = [
                        (*candidate_labels[int(index)], int(solution[int(index)]))
                        for index in support
                    ]
                    span_examples.append(
                        f"{(p, k, n, ip)} rank={rank}: {terms}"
                    )

    print(f"tested {len(rows)} projection corrections")
    print(f"matched {len(rows) - len(unmatched)}; unmatched {len(unmatched)}")
    print("most frequent Hasse exponents:", matches_by_a.most_common(20))
    for example in examples:
        print("  ", example)
    if unmatched:
        print("first unmatched:", unmatched[:20])
    print(
        f"span matches {len(rows) - len(outside_span)}; "
        f"outside span {len(outside_span)}"
    )
    print("span ranks:", dict(sorted(span_ranks.items())))
    print("solution support sizes:", dict(sorted(span_supports.items())))
    for example in span_examples:
        print("  ", example)
    if outside_span:
        print("first outside span:", outside_span[:20])


if __name__ == "__main__":
    main()
