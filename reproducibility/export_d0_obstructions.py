"""Export the lower-weight obstruction for ordinary nondegenerate D=0 cells.

For i=np+i' with D=0, the candidate extra drop asks whether theta^i f is
represented modulo p^2 at weight k+2i-p(p-1).  Reduction by the canonical
level-one echelon basis at that weight leaves a p-divisible remainder in the
tested range.  Its quotient by p is the obstruction exported here.

The output is finite computational evidence, not a theorem.
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
        if row["ordy"] == 1 and row["k"] < row["p"] - 1
    ]
    families: defaultdict[tuple[int, int], list[dict[str, int]]] = defaultdict(list)
    for row in rows:
        families[(row["p"], row["k"])].append(row)

    coordinate_names = [f"o{index}" for index in range(args.coordinates)]
    fieldnames = [
        "p", "k", "n", "ip", "i", "delta", "defect", "lower_weight",
        "boundary", "obstruction_filtration", "obstruction_gap_units",
        "reflected_D", "q0", "qm", "q1", "curve",
        "first_support_relative", *coordinate_names,
    ]
    output_rows: list[dict[str, int | str]] = []

    for (p, k), family in sorted(families.items()):
        environment = branchfree.build(p, k, p - 2, 3)
        lookup = {(row["n"], row["ip"]): row for row in family}
        for row in (candidate for candidate in family if candidate["D"] == 0):
            n, ip = row["n"], row["ip"]
            i = n * p + ip
            lower_weight = k + 2 * i - p * (p - 1)
            remainder = environment["reduce_Mw"](
                environment["theta"](environment["f"], i), lower_weight, p * p
            )
            if remainder is None or np.any(remainder % p):
                raise RuntimeError(f"unexpected remainder at {(p, k, n, ip)}")

            obstruction = (remainder // p) % p
            boundary = quotient_boundary(lower_weight)
            tail = obstruction[boundary:]
            support = np.flatnonzero(tail)
            obstruction_filtration = (
                -1
                if not support.size
                else environment["om_p"](obstruction, lower_weight)
            )
            if obstruction_filtration is None:
                raise RuntimeError(f"obstruction filtration not found at {(p, k, n, ip)}")
            c = (ip * ip + (k - 1) * ip) % p
            residues = {
                "q0": (c - n * (n + 1)) % p,
                "qm": (c - (n + 1) ** 2) % p,
                "q1": (c - (n + 1) * (n + 2)) % p,
            }
            curve = next((name for name, value in residues.items() if value == 0), "off")
            reflected = lookup.get((p - 2 - n, ip))
            exported: dict[str, int | str] = {
                "p": p,
                "k": k,
                "n": n,
                "ip": ip,
                "i": i,
                "delta": row["delta"],
                "defect": row["D"] - row["delta"],
                "lower_weight": lower_weight,
                "boundary": boundary,
                "obstruction_filtration": obstruction_filtration,
                "obstruction_gap_units": (
                    -1
                    if obstruction_filtration < 0
                    else (obstruction_filtration - lower_weight) // (p - 1)
                ),
                "reflected_D": reflected["D"] if reflected is not None else -1,
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
    print(f"wrote {args.output} with {len(output_rows)} D=0 cells")


if __name__ == "__main__":
    main()
