"""Map where the shifted D=0 leading block already gives the obstruction.

This prefix audit reuses the exported direct Bockstein coordinates and only
recomputes the proposed mod-p representative.  It is intended to discover a
clean subwedge boundary after the full Sturm-safe equality test established
that 458 of 692 cross-wedge cells satisfy the formula exactly.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from d1_obstruction_audit import quotient_boundary
from verify_d0_block_formula import e2_over_12

sys.path.insert(0, str(Path(__file__).parent / "code"))
import branchfree  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--coordinates", type=int, default=64)
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle) if int(row["reflected_D"]) == 2
        ]
    families: defaultdict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        families[(int(row["p"]), int(row["k"]))].append(row)

    total = Counter()
    boundary_audit = Counter()
    boundary_vectors: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    first_supports: defaultdict[str, Counter[int]] = defaultdict(Counter)
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

        by_n: defaultdict[int, Counter[str]] = defaultdict(Counter)
        difference_rows: list[tuple[int, int]] = []
        equal_rows: list[tuple[int, int]] = []
        for row in sorted(family, key=lambda item: (int(item["n"]), int(item["ip"]))):
            n, ip = int(row["n"]), int(row["ip"])
            i = int(row["i"])
            lower_weight = int(row["lower_weight"])
            q1 = int(row["q1"]) % p
            raw = q1 * conv(
                xp, environment["theta"](environment["f"], i - p) % p
            ) % p
            candidate = environment["reduce_Mw"](raw, lower_weight, p)
            if candidate is None:
                raise RuntimeError(f"candidate reduction failed at {(p, k, n, ip)}")
            boundary = quotient_boundary(lower_weight)
            proposed = np.zeros(args.coordinates, dtype=np.int64)
            available = candidate[boundary : boundary + args.coordinates]
            proposed[: len(available)] = available
            direct = np.array(
                [int(row[f"direct_{j}"]) for j in range(args.coordinates)],
                dtype=np.int64,
            )
            outcome = "equal" if np.array_equal(proposed % p, direct % p) else "different"
            on_boundary = ip in {
                n + 2,
                2 * p - k - 1 - n,
                2 * p - k - n,
            }
            line = (
                "L3_ip=n+2"
                if ip == n + 2
                else "L1_sum=2p-k-1"
                if ip + n == 2 * p - k - 1
                else "L2_sum=2p-k"
                if ip + n == 2 * p - k
                else "interior"
            )
            if line != "interior":
                boundary_vectors[line].append(direct % p)
                support = np.flatnonzero(direct % p)
                first_supports[line][int(support[0]) if support.size else -1] += 1
            boundary_audit[(on_boundary, outcome == "different")] += 1
            by_n[n][outcome] += 1
            total[outcome] += 1
            (equal_rows if outcome == "equal" else difference_rows).append((n, ip))

        print(
            f"p={p} k={k}: equal={len(equal_rows)} different={len(difference_rows)}"
        )
        print(
            "  by n: "
            + ", ".join(
                f"{n}:E{counts['equal']}/D{counts['different']}"
                for n, counts in sorted(by_n.items())
            )
        )
        if difference_rows:
            print(f"  first different: {difference_rows[:12]}")

    print(f"total: {dict(total)}")
    print(
        "three-line audit "
        f"TP={boundary_audit[(True, True)]} "
        f"FP={boundary_audit[(True, False)]} "
        f"FN={boundary_audit[(False, True)]} "
        f"TN={boundary_audit[(False, False)]}"
    )
    for line, vectors in sorted(boundary_vectors.items()):
        matrix = np.vstack(vectors)
        universally_nonzero = np.flatnonzero(np.all(matrix != 0, axis=0))
        print(
            f"{line}: rows={len(vectors)} first_supports={dict(first_supports[line])} "
            f"universally_nonzero_coordinates={universally_nonzero[:16].tolist()}"
        )


if __name__ == "__main__":
    main()
