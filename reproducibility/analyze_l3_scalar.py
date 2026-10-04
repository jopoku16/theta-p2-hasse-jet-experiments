"""Diagnose the scalar on the boundary line i' = n + 2.

For each tested cell, this script recomputes the full Sturm-safe D=0
Bockstein remainder and the one-block representative

    X^(2n-2p+k+3) theta^(p-k) f,   X = E_2/12 (mod p).

It records the canonical quotient pivot and compares the proportionality
scalar with coefficient units already present in the A+B+C expansion.  The
purpose is diagnostic: a match may suggest a symbolic normalization, while
the pivot data reveal whether the better theorem is one-dimensional
nonvanishing.  No finite match reported here is a proof for arbitrary p.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from d1_obstruction_audit import quotient_boundary
from identify_d0_obstructions import proportional
from verify_d0_block_formula import coefficient_unit, e2_over_12

sys.path.insert(0, str(Path(__file__).parent / "code"))
import branchfree  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--show", type=int, default=12)
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["reflected_D"]) == 2
            and int(row["ip"]) == int(row["n"]) + 2
        ]

    families: defaultdict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        families[(int(row["p"]), int(row["k"]))].append(row)

    matches: Counter[str] = Counter()
    pivot_offsets: Counter[int] = Counter()
    records: list[dict[str, int]] = []

    for (p, k), family in sorted(families.items()):
        environment = branchfree.build(p, k, p - 2, 3)
        length = environment["N"] + 1

        def conv(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return np.convolve(left, right)[:length] % p

        x = e2_over_12(p, length)
        x_powers = [np.zeros(length, dtype=np.int64)]
        x_powers[0][0] = 1
        for _ in range(p + 1):
            x_powers.append(conv(x_powers[-1], x))

        ap = int(environment["f"][p]) % p
        for row in family:
            n = int(row["n"])
            i = int(row["i"])
            q1 = int(row["q1"]) % p
            lower_weight = int(row["lower_weight"])
            boundary = quotient_boundary(lower_weight)

            direct_mod_p2 = environment["reduce_Mw"](
                environment["theta"](environment["f"], i),
                lower_weight,
                p * p,
            )
            if direct_mod_p2 is None or np.any(direct_mod_p2 % p):
                raise RuntimeError(f"direct reduction failed at {(p, k, n)}")
            direct = (direct_mod_p2 // p) % p

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
                raise RuntimeError(f"shifted reduction failed at {(p, k, n)}")
            residual = (direct - shifted) % p

            exponent_x = 2 * n - 2 * p + k + 3
            candidate = environment["reduce_Mw"](
                conv(
                    x_powers[exponent_x],
                    environment["theta"](environment["f"], p - k),
                ),
                lower_weight,
                p,
            )
            if candidate is None:
                raise RuntimeError(f"candidate reduction failed at {(p, k, n)}")
            scalar = proportional(candidate % p, residual, p)
            if scalar is None:
                raise RuntimeError(f"one-block identity failed at {(p, k, n)}")

            support = np.flatnonzero(candidate % p)
            if not support.size:
                raise RuntimeError(f"zero candidate at {(p, k, n)}")
            pivot = int(support[0])
            pivot_offset = pivot - boundary
            pivot_offsets[pivot_offset] += 1

            jmax = i - p
            ka = coefficient_unit(i, jmax, k, p)
            unit = (ka + i * (i + k - 1)) % p
            block_coefficients: list[int] = []
            for ell in range(n * p, n * p + p - k + 1):
                upper = n * p + p - k - ell
                kb = coefficient_unit(i, ell, k, p)
                kb = kb * pow(-1, upper, p) * math.comb(i - ell - 1, upper) % p
                block_coefficients.append(kb)

            natural = {
                "q1": q1,
                "a_p": ap,
                "ka": ka,
                "unit": unit,
                "sum_b": sum(block_coefficients) % p,
                "first_b": block_coefficients[0],
                "last_b": block_coefficients[-1],
                "unit_plus_sum_b": (unit + sum(block_coefficients)) % p,
                "first_plus_last_b": (block_coefficients[0] + block_coefficients[-1])
                % p,
            }
            for label, value in natural.items():
                variants = {
                    label: value,
                    f"-{label}": (-value) % p,
                    f"a_p*{label}": ap * value % p,
                    f"-a_p*{label}": -ap * value % p,
                }
                for variant, candidate_scalar in variants.items():
                    if scalar == candidate_scalar % p:
                        matches[variant] += 1

            records.append(
                {
                    "p": p,
                    "k": k,
                    "n": n,
                    "m": p - 2 - n,
                    "scalar": scalar,
                    "scalar_over_ap": scalar * pow(ap, -1, p) % p,
                    "pivot": pivot,
                    "pivot_offset": pivot_offset,
                    "candidate_pivot": int(candidate[pivot]) % p,
                    "residual_pivot": int(residual[pivot]) % p,
                }
            )

    print(f"checked={len(records)} one-block boundary cells")
    print(f"pivot offsets from quotient boundary={dict(sorted(pivot_offsets.items()))}")
    print("natural scalar matches (nonzero counts only):")
    for label, count in matches.most_common():
        print(f"  {label}: {count}/{len(records)}")
    print("sample canonical records:")
    for record in records[: args.show]:
        print("  " + " ".join(f"{key}={value}" for key, value in record.items()))


if __name__ == "__main__":
    main()
