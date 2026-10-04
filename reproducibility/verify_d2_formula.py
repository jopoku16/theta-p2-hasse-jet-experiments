"""Recheck the proved D=2 conic obstruction in canonical coordinates."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from d1_obstruction_audit import load_rows

sys.path.insert(0, str(Path(__file__).parent / "code"))
import branchfree  # noqa: E402


def e2_over_12(p: int, length: int) -> np.ndarray:
    sigma = np.zeros(length, dtype=np.int64)
    for divisor in range(1, length):
        sigma[divisor::divisor] = (sigma[divisor::divisor] + divisor) % p
    result = (-2 * sigma) % p
    result[0] = pow(12, -1, p)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_paths", type=Path, nargs="+")
    parser.add_argument("--p", type=int)
    parser.add_argument("--k", type=int)
    args = parser.parse_args()

    rows = [
        row
        for row in load_rows(args.csv_paths)
        if row["ordy"] == 1
        and row["k"] < row["p"] - 1
        and row["D"] == 2
        and (args.p is None or row["p"] == args.p)
        and (args.k is None or row["k"] == args.k)
    ]
    families: defaultdict[tuple[int, int], list[dict[str, int]]] = defaultdict(list)
    for row in rows:
        families[(row["p"], row["k"])].append(row)

    outcomes: Counter[str] = Counter()
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
            n, ip = row["n"], row["ip"]
            i = n * p + ip
            candidate_weight = k + 2 * i + p * (p - 1)
            direct = environment["reduce_Mw"](
                environment["theta"](environment["f"], i),
                candidate_weight,
                p * p,
            )
            if direct is None or np.any(direct % p):
                raise RuntimeError(f"unexpected D2 remainder at {(p, k, n, ip)}")
            direct_obstruction = (direct // p) % p
            q0 = (ip * ip + (k - 1) * ip - n * (n + 1)) % p
            predicted_raw = q0 * conv(
                xp, environment["theta"](environment["f"], i - p) % p
            ) % p
            predicted = environment["reduce_Mw"](
                predicted_raw, candidate_weight, p
            )
            if predicted is None:
                outcomes["no_reduction"] += 1
            elif np.array_equal(predicted % p, direct_obstruction):
                outcomes["equal"] += 1
            elif np.array_equal((-predicted) % p, direct_obstruction):
                outcomes["opposite"] += 1
            elif not np.any(predicted) and not np.any(direct_obstruction):
                outcomes["both_zero"] += 1
            else:
                outcomes["different"] += 1

    print(f"checked {len(rows)} D=2 cells")
    print(dict(outcomes))


if __name__ == "__main__":
    main()
