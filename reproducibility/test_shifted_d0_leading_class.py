"""Test the simplest shifted-conic representative for the D=0 obstruction.

The candidate is

    q1 * (E_2/12)^p * theta^(i-p) f,

reduced in the same lower-weight quotient as the direct Bockstein.  The
test is split by the D-value of the Tate-dual reflected cell.  Equality on
the D=0--D=2 wedge would turn the observed zero criterion into the same
leading-block mechanism as the proved D=2 formula.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from identify_d0_obstructions import proportional
from verify_d0_block_formula import e2_over_12

sys.path.insert(0, str(Path(__file__).parent / "code"))
import branchfree  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    families: defaultdict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        families[(int(row["p"]), int(row["k"]))].append(row)

    outcomes: Counter[tuple[int, str]] = Counter()
    scalars: Counter[tuple[int, int]] = Counter()
    examples: list[str] = []
    for (p, k), family in sorted(families.items()):
        environment = branchfree.build(p, k, p - 2, 3)
        length = environment["N"] + 1

        def conv(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return np.convolve(left, right)[:length] % p

        x = e2_over_12(p, length)
        xp = np.zeros(length, dtype=np.int64)
        xp[0] = 1
        for _ in range(p):
            xp = conv(xp, x)

        for row in family:
            n, ip = int(row["n"]), int(row["ip"])
            i = n * p + ip
            lower_weight = int(row["lower_weight"])
            direct_remainder = environment["reduce_Mw"](
                environment["theta"](environment["f"], i),
                lower_weight,
                p * p,
            )
            if direct_remainder is None or np.any(direct_remainder % p):
                raise RuntimeError(f"unexpected direct remainder at {(p, k, n, ip)}")
            direct = (direct_remainder // p) % p
            q1 = int(row["q1"]) % p
            raw = q1 * conv(
                xp, environment["theta"](environment["f"], i - p) % p
            ) % p
            candidate = environment["reduce_Mw"](raw, lower_weight, p)
            if candidate is None:
                outcome = "no_reduction"
            elif np.array_equal(candidate % p, direct):
                outcome = "equal"
            elif np.array_equal((-candidate) % p, direct):
                outcome = "opposite"
            elif not np.any(candidate) and not np.any(direct):
                outcome = "both_zero"
            else:
                scalar = proportional(candidate, direct, p)
                if scalar is None:
                    outcome = "different"
                else:
                    outcome = "proportional"
                    scalars[(p, scalar)] += 1
                if len(examples) < 20:
                    examples.append(
                        f"{(p, k, n, ip)} reflected_D={row['reflected_D']} "
                        f"q1={q1} outcome={outcome} scalar={scalar}"
                    )
            outcomes[(int(row["reflected_D"]), outcome)] += 1

    for key, value in sorted(outcomes.items()):
        print(f"reflected_D={key[0]} outcome={key[1]}: {value}")
    print("proportional scalars by prime:", dict(sorted(scalars.items())))
    for example in examples:
        print("  ", example)


if __name__ == "__main__":
    main()
