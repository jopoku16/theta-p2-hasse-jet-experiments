"""Rational-reconstruction diagnostic for third-line scalars.

For each fixed (k,m), combine the exact finite-field scalars across tested
primes by CRT and ask whether they come from a small rational number.  A
stable reconstruction can expose a characteristic-zero structural constant;
failure is evidence that the scalar retains genuinely p-adic information.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sympy import ZZ
from sympy.ntheory.modular import solve_congruence
from sympy.polys.modulargcd import _integer_rational_reconstruction

from d1_obstruction_audit import quotient_boundary
from identify_d0_obstructions import proportional
from verify_d0_block_formula import e2_over_12

sys.path.insert(0, str(Path(__file__).parent / "code"))
import branchfree  # noqa: E402


def scalar_for(row: dict[str, str]) -> int:
    p, k, n = int(row["p"]), int(row["k"]), int(row["n"])
    i = int(row["i"])
    m = p - 2 - n
    lower_weight = int(row["lower_weight"])
    env = branchfree.build(p, k, p - 2, 3)
    length = env["N"] + 1

    def conv(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.convolve(left, right)[:length] % p

    x = e2_over_12(p, length)
    powers = [np.eye(1, length, 0, dtype=np.int64).ravel()]
    for _ in range(p + 1):
        powers.append(conv(powers[-1], x))

    direct_p2 = env["reduce_Mw"](env["theta"](env["f"], i), lower_weight, p * p)
    if direct_p2 is None or np.any(direct_p2 % p):
        raise RuntimeError((p, k, n, "direct"))
    direct = (direct_p2 // p) % p
    q1 = int(row["q1"]) % p
    shifted = env["reduce_Mw"](
        q1 * conv(powers[p], env["theta"](env["f"], i - p)) % p,
        lower_weight,
        p,
    )
    block = env["reduce_Mw"](
        conv(powers[k - 2 * m - 1], env["theta"](env["f"], p - k)),
        lower_weight,
        p,
    )
    if shifted is None or block is None:
        raise RuntimeError((p, k, n, "reduction"))
    scalar = proportional(block, (direct - shifted) % p, p)
    if scalar is None:
        raise RuntimeError((p, k, n, "shape"))
    return int(scalar)


def main() -> None:
    path = Path(sys.argv[1])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["reflected_D"]) == 2
            and int(row["ip"]) == int(row["n"]) + 2
        ]

    groups: defaultdict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        p, k, n = int(row["p"]), int(row["k"]), int(row["n"])
        scalar = scalar_for(row)
        groups[(k, p - 2 - n)].append((scalar, p))

    for key, values in sorted(groups.items()):
        if len(values) < 3:
            continue
        crt = solve_congruence(*values)
        if crt is None:
            raise RuntimeError((key, values))
        residue, modulus = map(int, crt)
        rational = _integer_rational_reconstruction(residue, modulus, ZZ)
        checks = " ".join(f"{p}:{value}" for value, p in values)
        print(f"k={key[0]:2d} m={key[1]:2d} N={len(values):2d} rational={rational} {checks}")


if __name__ == "__main__":
    main()
