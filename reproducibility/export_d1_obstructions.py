"""Export canonical D=1 obstruction coordinates for structural analysis.

For each ordinary, nondegenerate D=1 cell, reduce theta^i f modulo p^2 by
the echelon basis of M_w at the nominal weight w=k+2i.  The remainder is
divisible by p.  This script records the first quotient coordinates of the
remainder divided by p, together with the three adjacent quadratic residues.

The resulting CSV is computational evidence.  It does not turn the observed
four-coordinate criterion into a theorem.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from d1_obstruction_audit import load_rows, quotient_boundary

sys.path.insert(0, str(Path(__file__).parent / "code"))
import branchfree  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_paths", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coordinates", type=int, default=12)
    args = parser.parse_args()

    rows = [
        row
        for row in load_rows(args.csv_paths)
        if row["ordy"] == 1 and row["k"] < row["p"] - 1 and row["D"] == 1
    ]
    families: defaultdict[tuple[int, int], list[dict[str, int]]] = defaultdict(list)
    for row in rows:
        families[(row["p"], row["k"])].append(row)

    coordinate_names = [f"o{index}" for index in range(args.coordinates)]
    fieldnames = [
        "p",
        "k",
        "n",
        "ip",
        "i",
        "delta",
        "defect",
        "weight",
        "boundary",
        "q0",
        "qm",
        "q1",
        "curve",
        "first_support_relative",
        *coordinate_names,
    ]

    output_rows: list[dict[str, int | str]] = []
    for (p, k), family in sorted(families.items()):
        environment = branchfree.build(p, k, p - 2, 3)
        for row in family:
            n, ip = row["n"], row["ip"]
            i = n * p + ip
            weight = k + 2 * i
            remainder = environment["reduce_Mw"](
                environment["theta"](environment["f"], i), weight, p * p
            )
            if remainder is None or np.any(remainder % p):
                raise RuntimeError(f"unexpected remainder at {(p, k, n, ip)}")

            obstruction = (remainder // p) % p
            boundary = quotient_boundary(weight)
            tail = obstruction[boundary:]
            support = np.flatnonzero(tail)
            c = (ip * ip + (k - 1) * ip) % p
            residues = {
                "q0": (c - n * (n + 1)) % p,
                "qm": (c - (n + 1) ** 2) % p,
                "q1": (c - (n + 1) * (n + 2)) % p,
            }
            curve = next((name for name, value in residues.items() if value == 0), "off")
            exported: dict[str, int | str] = {
                "p": p,
                "k": k,
                "n": n,
                "ip": ip,
                "i": i,
                "delta": row["delta"],
                "defect": row["D"] - row["delta"],
                "weight": weight,
                "boundary": boundary,
                **residues,
                "curve": curve,
                "first_support_relative": int(support[0]) if support.size else -1,
            }
            for index, name in enumerate(coordinate_names):
                position = boundary + index
                exported[name] = int(obstruction[position]) if position < len(obstruction) else 0
            output_rows.append(exported)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {args.output} with {len(output_rows)} D=1 cells")


if __name__ == "__main__":
    main()
