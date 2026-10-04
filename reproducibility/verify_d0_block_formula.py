"""Verify the explicit A+B+C formula for the D=0 obstruction.

After the inversion identity, the p-divisible blocks give a finite sum of
terms X^(i-l) theta^l f, with X=E_2/12.  The p-divisible correction of the
unit block contributes i(i+k-1) X^p theta^(i-p) f.  This script checks that
the resulting class, reduced modulo the candidate lower-weight space, is
the same canonical obstruction obtained directly modulo p^2.

The calculation is exact in the Sturm-safe truncation used by branchfree.
It validates the formula in the tested range; it does not replace its
symbolic proof.
"""

from __future__ import annotations

import argparse
import csv
import math
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


def coefficient_unit(i: int, l: int, k: int, p: int) -> int:
    modulus = p * p
    product = 1
    for value in range(l + k, i + k):
        product = product * value % modulus
    coefficient = (math.comb(i, l) % modulus) * product % modulus
    if coefficient % p:
        raise RuntimeError(f"coefficient is not p-divisible at {(i, l, k, p)}")
    return coefficient // p % p


def e2_over_12(p: int, length: int) -> np.ndarray:
    sigma = np.zeros(length, dtype=np.int64)
    for divisor in range(1, length):
        sigma[divisor::divisor] = (sigma[divisor::divisor] + divisor) % p
    result = (-2 * sigma) % p
    result[0] = pow(12, -1, p)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--p", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()

    rows = [
        row
        for row in load(args.csv_path)
        if (args.p is None or int(row["p"]) == args.p)
        and (args.k is None or int(row["k"]) == args.k)
    ]
    families: defaultdict[tuple[int, int], list[dict[str, int | str]]] = defaultdict(list)
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
            n, ip = int(row["n"]), int(row["ip"])
            i = n * p + ip
            lower_weight = int(row["lower_weight"])
            direct = environment["reduce_Mw"](
                environment["theta"](environment["f"], i), lower_weight, p * p
            )
            if direct is None or np.any(direct % p):
                raise RuntimeError(f"unexpected direct remainder at {(p, k, n, ip)}")
            direct_obstruction = (direct // p) % p

            jmax = i - p
            ka = coefficient_unit(i, jmax, k, p)
            raw = (
                (ka + i * (i + k - 1))
                * conv(x_powers[p], environment["theta"](environment["f"], jmax) % p)
            ) % p

            for l in range(n * p, n * p + p - k + 1):
                exponent = i - l
                upper = n * p + p - k - l
                kb = coefficient_unit(i, l, k, p)
                kb = kb * ((-1) ** upper) * math.comb(i - l - 1, upper) % p
                raw = (
                    raw
                    + kb * conv(
                        x_powers[exponent],
                        environment["theta"](environment["f"], l) % p,
                    )
                ) % p

            predicted = environment["reduce_Mw"](raw, lower_weight, p)
            if predicted is None:
                outcome = "no_reduction"
            elif np.array_equal(predicted % p, direct_obstruction % p):
                outcome = "equal"
            elif np.array_equal((-predicted) % p, direct_obstruction % p):
                outcome = "opposite"
            else:
                support = np.flatnonzero(predicted % p)
                scalar = None
                if support.size and int(direct_obstruction[int(support[0])]) % p:
                    pivot = int(support[0])
                    scalar = (
                        int(direct_obstruction[pivot])
                        * pow(int(predicted[pivot]), -1, p)
                        % p
                    )
                if scalar is not None and np.all(
                    (scalar * predicted - direct_obstruction) % p == 0
                ):
                    outcome = f"proportional_{scalar}"
                else:
                    outcome = "different"
                if len(examples) < args.examples:
                    first = np.flatnonzero((predicted - direct_obstruction) % p)
                    examples.append(
                        f"{(p, k, n, ip)} first difference "
                        f"{int(first[0]) if first.size else -1}; scalar={scalar}"
                    )
            outcomes[outcome] += 1

    print(f"checked {len(rows)} D0 cells")
    print(dict(outcomes))
    for example in examples:
        print("  ", example)


if __name__ == "__main__":
    main()
