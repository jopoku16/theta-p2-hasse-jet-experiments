"""Compare the L3 residual scalar with intrinsic p-adic constants.

This is a diagnostic only.  It recomputes the exact scalar on i'=n+2 and
prints normalizations by the Wilson quotient and the first divided
coefficient of E_{p-1}.  The goal is to detect whether the scalar can be a
uniform elementary unit or necessarily remembers the integral Hasse lift.
"""

from __future__ import annotations

import csv
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from sympy import bernoulli

from d1_obstruction_audit import quotient_boundary
from identify_d0_obstructions import proportional
from verify_d0_block_formula import e2_over_12

sys.path.insert(0, str(Path(__file__).parent / "code"))
import branchfree  # noqa: E402


def divided_eisenstein_q_coefficient(p: int) -> int:
    """Return ([q]E_{p-1})/p modulo p."""
    b = Fraction(bernoulli(p - 1))
    coefficient = Fraction(-2 * (p - 1), 1) / b
    modulus = p * p
    residue = coefficient.numerator * pow(coefficient.denominator, -1, modulus)
    residue %= modulus
    if residue % p:
        raise RuntimeError(f"E_{{p-1}}-1 is not p-divisible at p={p}")
    return residue // p % p


def main() -> None:
    path = Path(sys.argv[1])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["reflected_D"]) == 2
            and int(row["ip"]) == int(row["n"]) + 2
        ]

    for row in rows:
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
        exponent = k - 2 * m - 1
        block = env["reduce_Mw"](
            conv(powers[exponent], env["theta"](env["f"], p - k)),
            lower_weight,
            p,
        )
        if shifted is None or block is None:
            raise RuntimeError((p, k, n, "reduction"))
        scalar = proportional(block, (direct - shifted) % p, p)
        if scalar is None:
            raise RuntimeError((p, k, n, "scalar"))

        wilson = (math.factorial(p - 1) + 1) // p % p
        hasse_q1 = divided_eisenstein_q_coefficient(p)
        ap = int(env["f"][p]) % p
        boundary = quotient_boundary(lower_weight)
        pivot = int(np.flatnonzero(block[boundary:] % p)[0]) + boundary
        normalizers = {
            "c": scalar,
            "c/W": scalar * pow(wilson, -1, p) % p if wilson else -1,
            "c/H": scalar * pow(hasse_q1, -1, p) % p if hasse_q1 else -1,
            "c/ap": scalar * pow(ap, -1, p) % p if ap else -1,
        }
        print(
            f"p={p:2d} k={k:2d} m={m:2d} a={exponent:2d} "
            f"W={wilson:2d} H={hasse_q1:2d} ap={ap:2d} pivot={pivot:3d} "
            + " ".join(f"{name}={value:2d}" for name, value in normalizers.items())
        )


if __name__ == "__main__":
    main()
