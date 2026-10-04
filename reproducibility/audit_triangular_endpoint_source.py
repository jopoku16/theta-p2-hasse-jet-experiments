"""Separate the triangular endpoint coefficient into unit and B blocks.

On the two exceptional triangular lines, Lucas and Wilson predict that the
unit-block coefficient after subtracting the shifted leading term is
-2(m+1), where m=p-2-n.  This script projects the remaining p-divisible
B-block sum and the minimal-weight projection correction onto the explicit
triangular basis.  The audit distinguishes the coefficient already present
in the naive A+B+C expression from the coefficient supplied by the missing
projection term.  All reductions use the complete Sturm-safe q-expansion.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from identify_projection_corrections import solve_span
from verify_d0_block_formula import coefficient_unit, e2_over_12

sys.path.insert(0, str(Path(__file__).parent / "code"))
import branchfree  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--examples", type=int, default=10)
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
    examples: list[str] = []
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
                total = 2 * p - k - 1
            elif ip + n == 2 * p - k:
                line = "L2"
                total = 2 * p - k
            else:
                continue

            m = p - 2 - n
            lower_weight = int(row["lower_weight"])
            if line == "L1":
                a_values = list(range(p - k, p - k + 2 * m + 2)) + [p]
            else:
                a_values = list(range(p - k + 1, p - k + 2 * m + 4)) + [p]

            columns: list[np.ndarray] = []
            for exponent_x in a_values:
                exponent_theta = total - exponent_x
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
                columns.append(candidate % p)

            raw_b = np.zeros(length, dtype=np.int64)
            for ell in range(n * p, n * p + p - k + 1):
                exponent_x = i - ell
                upper = n * p + p - k - ell
                coefficient = coefficient_unit(i, ell, k, p)
                coefficient = (
                    coefficient
                    * pow(-1, upper, p)
                    * math.comb(i - ell - 1, upper)
                ) % p
                raw_b = (
                    raw_b
                    + coefficient
                    * conv(
                        x_powers[exponent_x],
                        environment["theta"](environment["f"], ell) % p,
                    )
                ) % p

            reduced_b = environment["reduce_Mw"](raw_b, lower_weight, p)
            if reduced_b is None:
                raise RuntimeError(f"B-block reduction failed at {(p, k, n, ip)}")
            solution = solve_span(columns, reduced_b % p, p)
            if solution is None:
                outcomes[f"{line}_outside_span"] += 1
                continue
            coefficients, _ = solution
            endpoint_coefficient = int(coefficients[-1]) % p
            expected_b = 0
            outcomes[
                f"{line}_B_endpoint_zero_match"
                if endpoint_coefficient == expected_b
                else f"{line}_B_endpoint_failure"
            ] += 1

            ka = coefficient_unit(i, i - p, k, p)
            unit_minus_q1 = (
                ka + i * (i + k - 1) - int(row["q1"])
            ) % p
            expected_unit = (-2 * (m + 1)) % p
            outcomes[
                f"{line}_unit_match"
                if unit_minus_q1 == expected_unit
                else f"{line}_unit_failure"
            ] += 1

            unit_coefficient = (ka + i * (i + k - 1)) % p
            raw_naive = (
                raw_b
                + unit_coefficient
                * conv(
                    x_powers[p],
                    environment["theta"](environment["f"], i - p) % p,
                )
            ) % p
            naive = environment["reduce_Mw"](raw_naive, lower_weight, p)
            direct_mod_p2 = environment["reduce_Mw"](
                environment["theta"](environment["f"], i),
                lower_weight,
                p * p,
            )
            if (
                naive is None
                or direct_mod_p2 is None
                or np.any(direct_mod_p2 % p)
            ):
                raise RuntimeError(f"projection audit failed at {(p, k, n, ip)}")
            projection_correction = ((direct_mod_p2 // p) - naive) % p
            correction_solution = solve_span(columns, projection_correction, p)
            if correction_solution is None:
                outcomes[f"{line}_correction_outside_span"] += 1
            else:
                correction_endpoint = int(correction_solution[0][-1]) % p
                expected_correction = (m + 1) % p
                outcomes[
                    f"{line}_correction_endpoint_match"
                    if correction_endpoint == expected_correction
                    else f"{line}_correction_endpoint_failure"
                ] += 1

            if len(examples) < args.examples:
                examples.append(
                    f"{(p, k, n, ip)} {line} m={m} "
                    f"unit={unit_minus_q1} B={endpoint_coefficient} "
                    f"Pi={correction_endpoint if correction_solution is not None else 'NA'} "
                    f"direct_endpoint={(-(m + 1)) % p}"
                )

    print(f"outcomes={dict(sorted(outcomes.items()))}")
    print("examples:")
    for example in examples:
        print("  " + example)


if __name__ == "__main__":
    main()
