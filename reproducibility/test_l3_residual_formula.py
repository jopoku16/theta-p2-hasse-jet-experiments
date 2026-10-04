"""Test the one-block formula for the boundary line i' = n + 2.

The proposed residual after subtracting the shifted leading block is,
up to a nonzero scalar,

    (E_2/12)^(2n-2p+k+3) * theta^(p-k) f.

The audit compares the first exported quotient coordinates.  It is a finite
identity test and not a symbolic proof of the nonzero scalar.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from d1_obstruction_audit import quotient_boundary
from identify_d0_obstructions import proportional
from verify_d0_block_formula import e2_over_12

sys.path.insert(0, str(Path(__file__).parent / "code"))
import branchfree  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--coordinates", type=int, default=64)
    parser.add_argument("--full-remainder", action="store_true")
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

    outcomes = Counter()
    scalars: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
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
        for row in family:
            n, i = int(row["n"]), int(row["i"])
            lower_weight = int(row["lower_weight"])
            boundary = quotient_boundary(lower_weight)
            if args.full_remainder:
                direct_mod_p2 = environment["reduce_Mw"](
                    environment["theta"](environment["f"], i),
                    lower_weight,
                    p * p,
                )
                if direct_mod_p2 is None or np.any(direct_mod_p2 % p):
                    raise RuntimeError(f"direct reduction failed at {(p, k, n)}")
                direct = (direct_mod_p2 // p) % p
            else:
                direct = np.array(
                    [int(row[f"direct_{j}"]) for j in range(args.coordinates)],
                    dtype=np.int64,
                )
            q1 = int(row["q1"]) % p
            shifted_full = environment["reduce_Mw"](
                q1
                * conv(x_powers[p], environment["theta"](environment["f"], i - p))
                % p,
                lower_weight,
                p,
            )
            if shifted_full is None:
                raise RuntimeError(f"shifted reduction failed at {(p, k, n)}")
            if args.full_remainder:
                shifted = shifted_full % p
            else:
                shifted = np.zeros(args.coordinates, dtype=np.int64)
                available = shifted_full[boundary : boundary + args.coordinates]
                shifted[: len(available)] = available
            residual = (direct - shifted) % p

            exponent_x = 2 * n - 2 * p + k + 3
            exponent_theta = p - k
            if not 0 <= exponent_x < len(x_powers):
                raise RuntimeError(f"unexpected X exponent at {(p, k, n)}")
            candidate_full = environment["reduce_Mw"](
                conv(
                    x_powers[exponent_x],
                    environment["theta"](environment["f"], exponent_theta),
                ),
                lower_weight,
                p,
            )
            if candidate_full is None:
                raise RuntimeError(f"candidate reduction failed at {(p, k, n)}")
            if args.full_remainder:
                candidate = candidate_full % p
            else:
                candidate = np.zeros(args.coordinates, dtype=np.int64)
                available = candidate_full[boundary : boundary + args.coordinates]
                candidate[: len(available)] = available
            scalar = proportional(candidate, residual, p)
            if scalar is None:
                outcomes["failure"] += 1
                print(f"failure at {(p, k, n, n + 2)}")
            else:
                outcomes["proportional"] += 1
                scalars[(p, k)].append(scalar)

    print(f"outcomes={dict(outcomes)}")
    for key, values in sorted(scalars.items()):
        print(f"p={key[0]} k={key[1]} scalars={values}")


if __name__ == "__main__":
    main()
