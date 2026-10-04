"""Blindly test the i'=n+2 residual beyond the discovery table.

The supported level-one forms are the same forms used by branchfree.py.
For each ordinary family and each admissible m, this script checks whether
the L3 residual is zero or is proportional to the predicted one-block
class.  It is a falsification test, not a proof.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from sympy import primerange

from identify_d0_obstructions import proportional
from verify_d0_block_formula import e2_over_12

sys.path.insert(0, str(Path(__file__).parent / "code"))
import branchfree  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-p", type=int, default=61)
    args = parser.parse_args()
    checked = zero = failed_shape = wrong_scalar = 0

    for p in primerange(13, args.max_p + 1):
        for k in (8, 12, 16, 18, 20, 22, 26):
            if k >= p - 1:
                continue
            env = branchfree.build(p, k, p - 2, 3)
            if not env["ordy"]:
                continue
            length = env["N"] + 1

            def conv(left: np.ndarray, right: np.ndarray) -> np.ndarray:
                return np.convolve(left, right)[:length] % p

            x = e2_over_12(p, length)
            powers = [np.eye(1, length, 0, dtype=np.int64).ravel()]
            for _ in range(p + 1):
                powers.append(conv(powers[-1], x))

            family_scalars: list[tuple[int, int]] = []
            for m in range(1, (k - 2) // 2):
                n = p - 2 - m
                ip = p - m
                i = n * p + ip
                lower_weight = k + 2 * i - p * (p - 1)
                direct_p2 = env["reduce_Mw"](
                    env["theta"](env["f"], i), lower_weight, p * p
                )
                if direct_p2 is None or np.any(direct_p2 % p):
                    raise RuntimeError((p, k, m, "direct"))
                direct = (direct_p2 // p) % p
                q1 = (-k * m) % p
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
                    raise RuntimeError((p, k, m, "reduction"))
                residual = (direct - shifted) % p
                checked += 1
                if not np.any(residual):
                    zero += 1
                    family_scalars.append((m, 0))
                    continue
                scalar = proportional(block, residual, p)
                if scalar is None:
                    failed_shape += 1
                    family_scalars.append((m, -1))
                else:
                    expected = -m * math.factorial(k - 2 * m - 2) % p
                    if scalar != expected:
                        wrong_scalar += 1
                    family_scalars.append((m, scalar))
            if family_scalars:
                print(f"p={p} k={k} scalars={family_scalars}")

    print(
        f"checked={checked} zero={zero} failed_shape={failed_shape} "
        f"wrong_scalar={wrong_scalar}"
    )


if __name__ == "__main__":
    main()
