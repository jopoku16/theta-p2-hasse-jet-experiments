"""Audit the two triangular D=0 boundary residuals on full remainders.

Put m=p-2-n and X=E_2/12 modulo p.  For the line

    i' + n = 2p-k-1

the observed residual is tested against the span of

    X^a theta^(2p-k-1-a) f,  p-k <= a <= p-k+2m+1,
    X^p theta^(p-k-1) f.

For the adjacent line i'+n=2p-k, it is tested against

    X^a theta^(2p-k-a) f,  p-k+1 <= a <= p-k+2m+3,
    X^p theta^(p-k) f.

All comparisons use the complete Sturm-safe canonical remainders.  This is
a finite exact audit, not a proof for arbitrary primes and eigenforms.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from d1_obstruction_audit import quotient_boundary
from identify_projection_corrections import solve_span
from verify_d0_block_formula import e2_over_12

sys.path.insert(0, str(Path(__file__).parent / "code"))
import branchfree  # noqa: E402


def matrix_rank(columns: list[np.ndarray], p: int) -> int:
    if not columns:
        return 0
    zero = np.zeros_like(columns[0])
    solved = solve_span(columns, zero, p)
    if solved is None:
        raise RuntimeError("rank computation failed")
    return solved[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["reflected_D"]) == 2
        ]
    families: defaultdict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        families[(int(row["p"]), int(row["k"]))].append(row)

    outcomes: Counter[str] = Counter()
    ranks: Counter[tuple[str, int, int]] = Counter()
    supports: Counter[tuple[str, int, int]] = Counter()
    separator_prefixes: Counter[tuple[str, int, int]] = Counter()
    examples: list[str] = []
    coefficient_rows: list[dict[str, int | str]] = []

    for (p, k), family in sorted(families.items()):
        environment = branchfree.build(p, k, p - 2, 3)
        length = environment["N"] + 1

        def conv(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return np.convolve(left, right)[:length] % p

        x = e2_over_12(p, length)
        x_powers = [np.zeros(length, dtype=np.int64)]
        x_powers[0][0] = 1
        for _ in range(p):
            x_powers.append(conv(x_powers[-1], x))

        for row in family:
            n, ip, i = int(row["n"]), int(row["ip"]), int(row["i"])
            if ip + n == 2 * p - k - 1:
                line = "L1"
            elif ip + n == 2 * p - k:
                line = "L2"
            else:
                continue

            lower_weight = int(row["lower_weight"])
            direct_mod_p2 = environment["reduce_Mw"](
                environment["theta"](environment["f"], i),
                lower_weight,
                p * p,
            )
            if direct_mod_p2 is None or np.any(direct_mod_p2 % p):
                raise RuntimeError(f"direct reduction failed at {(p, k, n, ip)}")
            direct = (direct_mod_p2 // p) % p

            q1 = int(row["q1"]) % p
            shifted = environment["reduce_Mw"](
                q1
                * conv(
                    x_powers[p],
                    environment["theta"](environment["f"], i - p),
                )
                % p,
                lower_weight,
                p,
            )
            if shifted is None:
                raise RuntimeError(f"shifted reduction failed at {(p, k, n, ip)}")
            residual = (direct - shifted) % p

            m = p - 2 - n
            if line == "L1":
                total = 2 * p - k - 1
                a_values = list(range(p - k, p - k + 2 * m + 2)) + [p]
            else:
                total = 2 * p - k
                a_values = list(range(p - k + 1, p - k + 2 * m + 4)) + [p]

            labels: list[tuple[int, int]] = []
            columns: list[np.ndarray] = []
            for exponent_x in a_values:
                exponent_theta = total - exponent_x
                if not 0 <= exponent_x <= p or not 1 <= exponent_theta < p:
                    raise RuntimeError(
                        f"bad triangular exponent at {(p, k, n, ip, exponent_x, exponent_theta)}"
                    )
                candidate = environment["reduce_Mw"](
                    conv(
                        x_powers[exponent_x],
                        environment["theta"](environment["f"], exponent_theta),
                    ),
                    lower_weight,
                    p,
                )
                if candidate is None:
                    raise RuntimeError(
                        f"candidate reduction failed at {(p, k, n, ip, exponent_x)}"
                    )
                labels.append((exponent_x, exponent_theta))
                columns.append(candidate % p)

            solution = solve_span(columns, residual, p)
            if solution is None:
                outcomes[f"{line}_outside_span"] += 1
                if len(examples) < args.examples:
                    examples.append(f"outside {(p, k, n, ip)} {line} labels={labels}")
                continue

            coefficients, rank = solution
            without_endpoint = solve_span(
                columns[:-1], np.zeros_like(residual), p
            )
            if without_endpoint is None:
                raise RuntimeError(f"rank audit failed at {(p, k, n, ip)}")
            rank_without_endpoint = without_endpoint[1]
            support = np.flatnonzero(coefficients % p)
            outcomes[f"{line}_in_span"] += 1
            if np.any(residual % p):
                outcomes[f"{line}_nonzero"] += 1
            else:
                outcomes[f"{line}_zero"] += 1
            if rank == rank_without_endpoint + 1:
                outcomes[f"{line}_endpoint_independent"] += 1
            else:
                outcomes[f"{line}_endpoint_dependent"] += 1
            expected_endpoint = (-(m + 1)) % p
            if int(coefficients[-1]) % p == expected_endpoint:
                outcomes[f"{line}_endpoint_coefficient_match"] += 1
            else:
                outcomes[f"{line}_endpoint_coefficient_failure"] += 1
            ranks[(line, m, rank)] += 1
            supports[(line, m, len(support))] += 1
            boundary = quotient_boundary(lower_weight)
            tails = [column[boundary:] % p for column in columns]
            separator_prefix = -1
            for prefix in range(1, len(tails[0]) + 1):
                rank_earlier = matrix_rank(
                    [column[:prefix] for column in tails[:-1]], p
                )
                rank_with_endpoint = matrix_rank(
                    [column[:prefix] for column in tails], p
                )
                if rank_with_endpoint == rank_earlier + 1:
                    separator_prefix = prefix
                    break
            if separator_prefix < 0:
                raise RuntimeError(
                    f"no endpoint separator at {(p, k, n, ip, line)}"
                )
            separator_prefixes[(line, m, separator_prefix)] += 1
            for position, ((exponent_x, exponent_theta), coefficient) in enumerate(
                zip(labels, coefficients, strict=True)
            ):
                coefficient_rows.append(
                    {
                        "p": p,
                        "k": k,
                        "n": n,
                        "ip": ip,
                        "m": m,
                        "line": line,
                        "position": position,
                        "endpoint": int(position == len(labels) - 1),
                        "x_exponent": exponent_x,
                        "theta_exponent": exponent_theta,
                        "coefficient": int(coefficient) % p,
                        "rank": rank,
                        "rank_without_endpoint": rank_without_endpoint,
                        "column_count": len(labels),
                        "separator_prefix": separator_prefix,
                    }
                )
            if len(examples) < args.examples:
                terms = [
                    (*labels[int(index)], int(coefficients[int(index)]))
                    for index in support
                ]
                examples.append(
                    f"{(p, k, n, ip)} {line} m={m} rank={rank} "
                    f"support={len(support)} terms={terms}"
                )

    print(f"outcomes={dict(sorted(outcomes.items()))}")
    print("rank counts by (line,m,rank):")
    for key, count in sorted(ranks.items()):
        print(f"  {key}: {count}")
    print("support counts by (line,m,support):")
    for key, count in sorted(supports.items()):
        print(f"  {key}: {count}")
    print("separator counts by (line,m,prefix length):")
    for key, count in sorted(separator_prefixes.items()):
        print(f"  {key}: {count}")
    print("examples:")
    for example in examples:
        print("  " + example)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(coefficient_rows[0]))
            writer.writeheader()
            writer.writerows(coefficient_rows)
        print(f"wrote {args.output} with {len(coefficient_rows)} coefficient rows")


if __name__ == "__main__":
    main()
