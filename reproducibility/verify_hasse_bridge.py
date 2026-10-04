"""Audit the weight typing of the candidate and natural Hasse bridges.

This script verifies identities only.  It does not claim that the
divided-congruence Bockstein classes have already been identified with
Hasse-quotient classes, nor that the bridge is nonzero on an eigenspace.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from analyze_defects import dual_key, enrich, load_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_paths", type=Path, nargs="+")
    args = parser.parse_args()

    rows = [enrich(row) for path in args.csv_paths for row in load_rows(path)]
    rows = [row for row in rows if row["ordy"] == 1 and row["k"] < row["p"] - 1]
    lookup = {(row["p"], row["k"], row["n"], row["ip"]): row for row in rows}

    seen: set[tuple[int, int, int, int]] = set()
    pair_types: Counter[tuple[int, int]] = Counter()
    complete_pairs = 0
    cross_wedge_pairs = 0

    for row in rows:
        key = (row["p"], row["k"], row["n"], row["ip"])
        partner_key = dual_key(row)
        if key in seen or partner_key in seen or partner_key not in lookup:
            continue
        partner = lookup[partner_key]
        seen.update((key, partner_key))
        complete_pairs += 1
        pair_types[tuple(sorted((row["D"], partner["D"])))] += 1

        p, k = row["p"], row["k"]
        period = p * (p - 1)
        exponent = row["n"] * p + row["ip"]
        dual_exponent = partner["n"] * p + partner["ip"]
        assert exponent + dual_exponent == p * p + 1 - k

        if row["D"] == 0 and partner["D"] == 2:
            d0, d2 = row, partner
        elif row["D"] == 2 and partner["D"] == 0:
            d0, d2 = partner, row
        else:
            continue

        cross_wedge_pairs += 1
        i0 = d0["n"] * p + d0["ip"]
        i2 = d2["n"] * p + d2["ip"]
        w_minus = k + 2 * i0 - period
        w_plus = k + 2 * i2 + period
        dual_weight = period + 2 - w_minus
        bridge_shift = p * (p + 1)
        natural_minus = w_minus + period
        natural_plus = w_plus + period
        natural_bridge_shift = p * (3 * p - 1)

        assert w_minus + w_plus == period + 2 + bridge_shift
        assert dual_weight + bridge_shift == w_plus
        assert natural_minus == k + 2 * i0
        assert natural_plus == k + 2 * i2 + 2 * period
        assert natural_minus + natural_plus == (
            period + 2 + natural_bridge_shift
        )

    print(f"ordinary nondegenerate rows: {len(rows)}")
    print(f"complete Tate-dual pairs: {complete_pairs}")
    print(f"cross-wedge D0-D2 pairs: {cross_wedge_pairs}")
    print(f"pair types: {dict(sorted(pair_types.items()))}")
    print(
        "all exponent, candidate-bridge, and natural-Bockstein "
        "weight identities verified"
    )


if __name__ == "__main__":
    main()
