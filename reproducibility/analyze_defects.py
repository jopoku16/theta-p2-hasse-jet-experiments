"""Audit candidate laws for the second defect in theta cycles modulo p^2.

The input is the band_cells.csv file produced by branchfree.py.  This script
does not infer a theorem.  It reports exact confusion counts and lists the
false positives and false negatives of simple shifted-conic criteria.  It
also tests the reflection n -> p-2-n, which exchanges the original and
shifted conics modulo p.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    return [{key: int(value) for key, value in row.items()} for row in raw]


def residue(value: int, modulus: int) -> int:
    return value % modulus


def enrich(row: dict[str, int]) -> dict[str, int]:
    result = dict(row)
    p, k, n, ip = (result[name] for name in ("p", "k", "n", "ip"))
    i = n * p + ip
    result["i"] = i
    result["extra_drop"] = result["D"] - result["delta"]
    result["q0"] = residue(i * i + (k - 1) * i - n * (n + 1), p)
    result["q1"] = residue(i * i + (k - 1) * i - (n + 1) * (n + 2), p)
    result["q2"] = residue(i * i + (k - 1) * i - (n + 2) * (n + 3), p)
    result["n_reflect"] = p - 2 - n
    result["j_residue"] = residue((n - 1) + ip, p - 1)
    result["classical_low"] = p - k + 1
    result["at_classical_low"] = int(result["j_residue"] == result["classical_low"])
    result["at_period_boundary"] = int(result["j_residue"] == 0)
    result["degenerate_weight"] = int(k == p - 1)
    period = p * (p - 1)
    result["modp_lower"] = -(-(result["om_theta_i"] - result["k2i"]) // period)
    return result


def confusion(rows: list[dict[str, int]], predicate, target) -> Counter:
    counts: Counter[str] = Counter()
    for row in rows:
        predicted = bool(predicate(row))
        observed = bool(target(row))
        key = "tp" if predicted and observed else "fp" if predicted else "fn" if observed else "tn"
        counts[key] += 1
    return counts


def print_rows(label: str, rows: list[dict[str, int]]) -> None:
    print(f"\n{label}: {len(rows)}")
    fields = ("p", "k", "n", "ip", "D", "delta", "extra_drop", "q0", "q1", "j_residue", "classical_low")
    print(" ".join(f"{field:>13}" for field in fields))
    for row in rows:
        print(" ".join(f"{row[field]:13d}" for field in fields))


def add_reflection_data(rows: list[dict[str, int]]) -> None:
    """Attach data from the reflected cell when that cell is in the grid."""
    lookup = {(row["p"], row["k"], row["n"], row["ip"]): row for row in rows}
    for row in rows:
        key = (row["p"], row["k"], row["n_reflect"], row["ip"])
        reflected = lookup.get(key)
        row["reflection_present"] = int(reflected is not None)
        row["reflected_D"] = reflected["D"] if reflected is not None else -1
        row["reflected_delta"] = reflected["delta"] if reflected is not None else -1
        row["reflected_extra_drop"] = reflected["extra_drop"] if reflected is not None else -1


def root_partner(row: dict[str, int]) -> int:
    """Return the other root of x^2+(k-1)x-c for fixed c modulo p."""
    return residue(1 - row["k"] - row["ip"], row["p"])


def dual_key(row: dict[str, int]) -> tuple[int, int, int, int]:
    """Tate-dual index: i* = p^2+1-k-i."""
    return (row["p"], row["k"], row["n_reflect"], root_partner(row))


def print_reflection_pairs(rows: list[dict[str, int]]) -> None:
    drops = [row for row in rows if row["extra_drop"] >= 1]
    grouped: defaultdict[tuple[int, int, int], list[dict[str, int]]] = defaultdict(list)
    for row in drops:
        grouped[(row["p"], row["k"], row["ip"])].append(row)

    print("\nExtra-drop reflection groups")
    for key, group in sorted(grouped.items()):
        ns = sorted(row["n"] for row in group)
        reflected_pairs = sorted(
            (row["n"], row["n_reflect"])
            for row in group
            if row["n"] <= row["n_reflect"] and any(other["n"] == row["n_reflect"] for other in group)
        )
        print(f"p,k,ip={key}: n={ns}; paired={reflected_pairs}")


def print_reflection_law(rows: list[dict[str, int]]) -> None:
    lookup = {(row["p"], row["k"], row["n"], row["ip"]): row for row in rows}
    transitions: Counter[tuple[int, int, int, int]] = Counter()
    failures: list[tuple[dict[str, int], dict[str, int]]] = []
    for row in rows:
        if row["n"] >= row["n_reflect"]:
            continue
        reflected = lookup.get((row["p"], row["k"], row["n_reflect"], row["ip"]))
        if reflected is None:
            continue
        transitions[(
            row["D"],
            reflected["D"],
            reflected["D"] - row["D"],
            reflected["delta"] - row["delta"],
        )] += 1
        if row["extra_drop"] != reflected["extra_drop"]:
            failures.append((row, reflected))

    print("\nReflection transition audit")
    print(" D_left D_right  D_shift delta_shift  count")
    for transition, count in sorted(transitions.items()):
        print(" ".join(f"{value:8d}" for value in (*transition, count)))
    print(f"defect-invariance failures: {len(failures)}")
    for left, right in failures:
        print(
            f"  (p,k,ip)=({left['p']},{left['k']},{left['ip']}), "
            f"n={left['n']}->{right['n']}: "
            f"(D,delta)=({left['D']},{left['delta']})"
            f"->({right['D']},{right['delta']})"
        )


def print_involution_orbits(rows: list[dict[str, int]]) -> None:
    """Print residual-defect orbits under row reflection and root exchange."""
    lookup = {(row["p"], row["k"], row["n"], row["ip"]): row for row in rows}
    seen: set[tuple[int, int, int, int]] = set()
    print("\nD<=1 defect orbits under row reflection R and root exchange S")
    for row in sorted(rows, key=lambda item: (item["p"], item["k"], item["n"], item["ip"])):
        if row["D"] > 1 or row["extra_drop"] < 1:
            continue
        orbit_keys = {
            (row["p"], row["k"], n, ip)
            for n in (row["n"], row["n_reflect"])
            for ip in (row["ip"], root_partner(row))
        }
        canonical = min(orbit_keys)
        if canonical in seen:
            continue
        seen.add(canonical)
        cells = [lookup[key] for key in sorted(orbit_keys) if key in lookup]
        pattern = [
            f"(n={cell['n']},ip={cell['ip']},D={cell['D']},delta={cell['delta']})"
            for cell in cells
        ]
        drops = sum(cell["extra_drop"] >= 1 for cell in cells)
        print(f"  (p,k)=({row['p']},{row['k']}), drops={drops}/{len(cells)}: " + " ".join(pattern))


def print_duality_audit(rows: list[dict[str, int]]) -> None:
    lookup = {(row["p"], row["k"], row["n"], row["ip"]): row for row in rows}
    seen: set[tuple[int, int, int, int]] = set()
    defect_pairs: Counter[tuple[int, int]] = Counter()
    failures: list[tuple[dict[str, int], dict[str, int]]] = []
    pairs = 0
    for row in rows:
        key = (row["p"], row["k"], row["n"], row["ip"])
        partner_key = dual_key(row)
        if key in seen or partner_key in seen or partner_key not in lookup:
            continue
        partner = lookup[partner_key]
        seen.update((key, partner_key))
        pairs += 1
        defects = (row["extra_drop"], partner["extra_drop"])
        defect_pairs[defects] += 1
        if defects[0] != defects[1]:
            failures.append((row, partner))

    print("\nTate-dual-complement audit")
    print(f"complete pairs: {pairs}")
    print(f"ordered defect pairs: {dict(sorted(defect_pairs.items()))}")
    print(f"defect-invariance failures: {len(failures)}")
    for left, right in failures:
        p, k = left["p"], left["k"]
        left_i = left["n"] * p + left["ip"]
        right_i = right["n"] * p + right["ip"]
        midpoint = residue(left["ip"] ** 2 + (k - 1) * left["ip"] - (left["n"] + 1) ** 2, p)
        print(
            f"  (p,k)=({p},{k}), (n,ip)=({left['n']},{left['ip']})"
            f"->({right['n']},{right['ip']}), i+i*={left_i + right_i}, "
            f"p^2+1-k={p * p + 1 - k}, midpoint_residue={midpoint}, "
            f"defects={left['extra_drop']}->{right['extra_drop']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_paths", type=Path, nargs="+")
    args = parser.parse_args()

    rows = [enrich(row) for path in args.csv_paths for row in load_rows(path)]
    add_reflection_data(rows)
    ordinary_nondegenerate = [row for row in rows if row["ordy"] == 1 and row["degenerate_weight"] == 0]
    d0 = [row for row in ordinary_nondegenerate if row["D"] == 0]

    print(f"rows={len(rows)} ordinary_nondegenerate={len(ordinary_nondegenerate)} D0={len(d0)}")
    print("extra-drop distribution:", dict(sorted(Counter(row["extra_drop"] for row in rows).items())))

    target = lambda row: row["extra_drop"] >= 1
    candidates = {
        "original conic q0=0": lambda row: row["q0"] == 0,
        "shifted conic q1=0": lambda row: row["q1"] == 0,
        "second shift q2=0": lambda row: row["q2"] == 0,
        "q1=0 and classical low": lambda row: row["q1"] == 0 and row["at_classical_low"] == 1,
        "q1=0 and reflected D=2": lambda row: row["q1"] == 0 and row["reflected_D"] == 2,
        "q1=0 and reflected D2 drop": lambda row: (
            row["q1"] == 0 and row["reflected_D"] == 2 and row["reflected_extra_drop"] >= 1
        ),
    }
    print("\nD=0 ordinary, nondegenerate candidate audit")
    for name, predicate in candidates.items():
        counts = confusion(d0, predicate, target)
        print(f"{name:31s} tp={counts['tp']:2d} fp={counts['fp']:2d} fn={counts['fn']:2d} tn={counts['tn']:3d}")

    lower_bound_types = Counter(
        (row["reflected_D"], row["q1"] == 0, row["extra_drop"], row["modp_lower"])
        for row in d0
    )
    print("\nD=0 types: (reflected_D, q1_zero, extra_drop, modp_lower) -> count")
    for cell_type, count in sorted(lower_bound_types.items()):
        print(f"  {cell_type} -> {count}")

    q1_d0 = [row for row in d0 if row["q1"] == 0]
    print_rows("True shifted-conic drops", [row for row in q1_d0 if target(row)])
    print_rows("Shifted-conic false positives", [row for row in q1_d0 if not target(row)])

    by_family: defaultdict[tuple[int, int], Counter] = defaultdict(Counter)
    for row in q1_d0:
        by_family[(row["p"], row["k"])]["drop" if target(row) else "no_drop"] += 1
    print("\nShifted-conic D=0 cells by family")
    for family, counts in sorted(by_family.items()):
        print(f"{family}: {dict(counts)}")

    reflected_d0 = [
        row
        for row in d0
        if row["q1"] == 0 and row["reflected_D"] == 2 and row["reflected_extra_drop"] >= 1
    ]
    print_rows("D=0 drops predicted by reflected D=2 drops", reflected_d0)
    print_reflection_pairs(ordinary_nondegenerate)
    print_reflection_law(ordinary_nondegenerate)
    print_involution_orbits(ordinary_nondegenerate)
    print_duality_audit(ordinary_nondegenerate)


if __name__ == "__main__":
    main()
