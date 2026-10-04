"""Export the missing minimal-weight projection term in the D=0 formula.

The direct Bockstein obstruction and the naive one-step-lower A+B+C
expression are both reduced by the same canonical level-one basis.  Their
difference is the exact correction that a proof of cross-wedge duality must
explain.  This is a finite computational diagnostic, not a symbolic proof.
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
from verify_d0_block_formula import coefficient_unit, e2_over_12, load

sys.path.insert(0, str(Path(__file__).parent / "code"))
import branchfree  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_paths", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coordinates", type=int, default=12)
    args = parser.parse_args()

    rows = [row for path in args.csv_paths for row in load(path)]
    families: defaultdict[tuple[int, int], list[dict[str, int | str]]] = (
        defaultdict(list)
    )
    for row in rows:
        families[(int(row["p"]), int(row["k"]))].append(row)

    names = []
    for prefix in ("direct", "naive", "correction"):
        names.extend(f"{prefix}_{index}" for index in range(args.coordinates))
    fieldnames = [
        "p", "k", "n", "ip", "i", "lower_weight", "boundary",
        "reflected_D", "q1", "direct_zero", "naive_zero", "correction_zero",
        "correction_first_support", *names,
    ]
    output_rows: list[dict[str, int]] = []
    counts: Counter[tuple[bool, bool, bool]] = Counter()

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
            naive %= p
            correction = (direct - naive) % p
            boundary = quotient_boundary(lower_weight)
            correction_support = np.flatnonzero(correction[boundary:] % p)
            direct_zero = not np.any(direct % p)
            naive_zero = not np.any(naive % p)
            correction_zero = not np.any(correction % p)
            counts[(direct_zero, naive_zero, correction_zero)] += 1

            exported = {
                "p": p,
                "k": k,
                "n": n,
                "ip": ip,
                "i": i,
                "lower_weight": lower_weight,
                "boundary": boundary,
                "reflected_D": int(row["reflected_D"]),
                "q1": (ip * ip + (k - 1) * ip - (n + 1) * (n + 2)) % p,
                "direct_zero": int(direct_zero),
                "naive_zero": int(naive_zero),
                "correction_zero": int(correction_zero),
                "correction_first_support": (
                    -1 if not correction_support.size else int(correction_support[0])
                ),
            }
            for prefix, vector in (
                ("direct", direct),
                ("naive", naive),
                ("correction", correction),
            ):
                for index in range(args.coordinates):
                    position = boundary + index
                    exported[f"{prefix}_{index}"] = (
                        int(vector[position]) if position < len(vector) else 0
                    )
            output_rows.append(exported)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"wrote {args.output} with {len(output_rows)} D=0 cells")
    for key, count in sorted(counts.items()):
        print(
            f"direct_zero={key[0]} naive_zero={key[1]} "
            f"correction_zero={key[2]}: {count}"
        )


if __name__ == "__main__":
    main()
