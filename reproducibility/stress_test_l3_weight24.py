"""Test the L3 residual in the first two-dimensional cusp-form weight.

The discovery audit used weights whose relevant level-one cusp space is
one-dimensional.  This script tests weight 24 with the independent forms
Delta*E4^3 and Delta*E6^2 and with projective linear combinations.  It
checks ordinarity, the predicted one-block shape, and the resulting scalar.
The computation is a falsification test, not a proof.
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


def weight_24_forms(p: int, length: int) -> dict[str, np.ndarray]:
    modulus = p * p

    def conv(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.convolve(left, right)[:length] % modulus

    def sigma(power: int) -> np.ndarray:
        values = np.zeros(length, dtype=np.int64)
        for divisor in range(1, length):
            values[divisor::divisor] = (
                values[divisor::divisor] + pow(divisor, power, modulus)
            ) % modulus
        return values

    one = np.zeros(length, dtype=np.int64)
    one[0] = 1
    e4 = 240 * sigma(3) % modulus
    e4[0] = 1
    e6 = -504 * sigma(5) % modulus
    e6[0] = 1
    product = one.copy()
    for n in range(1, length):
        updated = product.copy()
        updated[n:] = (updated[n:] - product[: length - n]) % modulus
        product = updated
    eta24 = one.copy()
    for _ in range(24):
        eta24 = conv(eta24, product)
    delta = np.zeros(length, dtype=np.int64)
    delta[1:] = eta24[:-1]
    f4 = conv(delta, conv(conv(e4, e4), e4))
    f6 = conv(delta, conv(e6, e6))
    return {"DeltaE4^3": f4, "DeltaE6^2": f6}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-p", type=int, default=61)
    args = parser.parse_args()
    k = 24
    checked = ordinary = failed_shape = zero = wrong_scalar = 0

    for p in primerange(29, args.max_p + 1):
        env = branchfree.build(p, 26, p - 2, 3)
        length = env["N"] + 1
        generators = weight_24_forms(p, length)

        def conv(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return np.convolve(left, right)[:length] % p

        x = e2_over_12(p, length)
        powers = [np.eye(1, length, 0, dtype=np.int64).ravel()]
        for _ in range(p + 1):
            powers.append(conv(powers[-1], x))

        forms: dict[str, np.ndarray] = dict(generators)
        for lam in range(p):
            forms[f"f4+{lam}f6"] = (generators["DeltaE4^3"] + lam * generators["DeltaE6^2"]) % (p * p)

        family_scalars: dict[str, list[tuple[int, int]]] = {}
        for label, form in forms.items():
            if int(form[p]) % p == 0:
                continue
            ordinary += 1
            scalars: list[tuple[int, int]] = []
            for m in range(1, (k - 2) // 2):
                n = p - 2 - m
                ip = p - m
                i = n * p + ip
                lower_weight = k + 2 * i - p * (p - 1)
                direct_p2 = env["reduce_Mw"](
                    env["theta"](form, i), lower_weight, p * p
                )
                if direct_p2 is None or np.any(direct_p2 % p):
                    raise RuntimeError((p, label, m, "direct"))
                direct = (direct_p2 // p) % p
                q1 = (-k * m) % p
                shifted = env["reduce_Mw"](
                    q1 * conv(powers[p], env["theta"](form, i - p)) % p,
                    lower_weight,
                    p,
                )
                exponent = k - 2 * m - 1
                block = env["reduce_Mw"](
                    conv(powers[exponent], env["theta"](form, p - k)),
                    lower_weight,
                    p,
                )
                if shifted is None or block is None:
                    raise RuntimeError((p, label, m, "reduction"))
                residual = (direct - shifted) % p
                checked += 1
                if not np.any(residual):
                    zero += 1
                    scalars.append((m, 0))
                    continue
                scalar = proportional(block, residual, p)
                if scalar is None:
                    failed_shape += 1
                    scalars.append((m, -1))
                else:
                    expected = -m * math.factorial(k - 2 * m - 2) % p
                    if scalar != expected:
                        wrong_scalar += 1
                    scalars.append((m, scalar))
            family_scalars[label] = scalars

        distinct = {tuple(values) for values in family_scalars.values()}
        print(
            f"p={p} ordinary_forms={len(family_scalars)} "
            f"distinct_scalar_rows={len(distinct)} examples={list(distinct)[:4]}"
        )

    print(
        f"ordinary_forms={ordinary} checked={checked} zero={zero} "
        f"failed_shape={failed_shape} wrong_scalar={wrong_scalar}"
    )


if __name__ == "__main__":
    main()
