"""Audit the two uniform low-point jet formulas.

This checks the modified-Serre triangular identities, the modularized
first-jet expansion, the closed reset-jet formula, and the two cutoff
containments used in Proposition 5.3 of the manuscript.  The checks use
complete Sturm-safe q-expansions for every (p, k) family in the released
cross-wedge record.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

from verify_periodic_first_jet import eisenstein

sys.path.insert(0, str(Path(__file__).parent / "code"))
import branchfree  # noqa: E402


ROOT = Path(__file__).resolve().parent


def main() -> None:
    with (ROOT / "data" / "d0_obstruction_coordinates.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["reflected_D"]) == 2
        ]

    families = sorted({(int(row["p"]), int(row["k"])) for row in rows})
    triangular_identities = 0
    modularized_jets = 0
    first_cutoffs = 0
    second_cutoffs = 0

    for p, k in families:
        env = branchfree.build(p, k, p - 2, 3)
        length = env["N"] + 1
        modulus = p * p
        one = np.zeros(length, dtype=np.int64)
        one[0] = 1

        def conv(left: np.ndarray, right: np.ndarray, mod: int) -> np.ndarray:
            return np.convolve(left, right)[:length] % mod

        def theta(series: np.ndarray, power: int, mod: int) -> np.ndarray:
            return np.array(
                [
                    pow(index, power, mod) * int(series[index]) % mod
                    for index in range(length)
                ],
                dtype=np.int64,
            )

        def power(series: np.ndarray, exponent: int, mod: int) -> np.ndarray:
            result = one % mod
            base = series % mod
            value = exponent
            while value:
                if value & 1:
                    result = conv(result, base, mod)
                base = conv(base, base, mod)
                value >>= 1
            return result

        def rising(start: int, count: int) -> int:
            result = 1
            for value in range(start, start + count):
                result *= value
            return result

        e2 = eisenstein(2, length, modulus)
        e4 = eisenstein(4, length, modulus)
        hasse = eisenstein(p - 1, length, modulus)
        ep1 = eisenstein(p + 1, length, modulus)
        f = env["f"].copy() % modulus
        x = e2 * pow(12, -1, modulus) % modulus
        b12 = ep1 * pow(12, -1, modulus) % modulus

        alpha_difference = (hasse - one) % modulus
        assert not np.any(alpha_difference % p)
        alpha = (alpha_difference // p) % p
        c_difference = (ep1 - conv(hasse, e2, modulus)) % modulus
        assert not np.any(c_difference % p)
        xi = (c_difference // p) * pow(12, -1, p) % p
        x_mod_p = x % p
        x_to_p = power(x_mod_p, p, p)

        def serred(series: np.ndarray, weight: int) -> np.ndarray:
            return (
                theta(series, 1, modulus)
                - weight
                * pow(12, -1, modulus)
                * conv(e2, series, modulus)
            ) % modulus

        low = p - k + 1
        modified = [f.copy()]
        if low:
            modified.append(serred(f, k))
        for index in range(1, low):
            modified.append(
                (
                    serred(modified[index], k + 2 * index)
                    - index
                    * (index + k - 1)
                    * pow(144, -1, modulus)
                    * conv(e4, modified[index - 1], modulus)
                )
                % modulus
            )

        a_powers = [power(hasse, index, modulus) for index in range(low + 1)]
        b_powers = [power(b12, index, modulus) for index in range(low + 1)]
        x_powers = [power(x, index, modulus) for index in range(low + 1)]

        for t in range(low + 1):
            forward = np.zeros(length, dtype=np.int64)
            inverse = np.zeros(length, dtype=np.int64)
            modularized = np.zeros(length, dtype=np.int64)
            for j in range(t + 1):
                coefficient = (
                    math.comb(t, j) * rising(k + j, t - j)
                ) % modulus
                forward = (
                    forward
                    + coefficient
                    * conv(x_powers[t - j], modified[j], modulus)
                ) % modulus
                inverse = (
                    inverse
                    + ((-1) ** (t - j) * coefficient) % modulus
                    * conv(x_powers[t - j], theta(f, j, modulus), modulus)
                ) % modulus
                modularized = (
                    modularized
                    + coefficient
                    * conv(
                        a_powers[j],
                        conv(b_powers[t - j], modified[j], modulus),
                        modulus,
                    )
                ) % modulus

            u_t = theta(f, t, modulus)
            if not np.array_equal(forward, u_t):
                raise AssertionError(f"forward identity failed at {(p, k, t)}")
            if not np.array_equal(inverse, modified[t]):
                raise AssertionError(f"inverse identity failed at {(p, k, t)}")
            difference = (modularized - u_t) % modulus
            if np.any(difference % p):
                raise AssertionError(f"modularized lift failed at {(p, k, t)}")
            actual_jet = (difference // p) % p
            if t == 0:
                expected_jet = np.zeros(length, dtype=np.int64)
            else:
                expected_jet = (
                    t * conv(alpha, theta(f, t, p), p)
                    + t
                    * (k + t - 1)
                    * conv(xi, theta(f, t - 1, p), p)
                ) % p
            if not np.array_equal(actual_jet, expected_jet):
                raise AssertionError(
                    f"modularized first jet failed at {(p, k, t)}"
                )
            triangular_identities += 1
            modularized_jets += 1

        s = p - k
        lift = f.copy()
        lift_weight = k
        for _ in range(s):
            lift = (
                conv(hasse, theta(lift, 1, modulus), modulus)
                + lift_weight
                * pow(12, -1, modulus)
                * conv(c_difference, lift, modulus)
            ) % modulus
            lift_weight += p + 1
        pre_difference = (lift - theta(f, s, modulus)) % modulus
        assert not np.any(pre_difference % p)
        eta_s = (pre_difference // p) % p
        s_term = (
            conv(alpha, theta(f, s, p), p)
            + conv(x_to_p, theta(f, s - 1, p), p)
        ) % p
        pre_low_correction = (eta_s + k * s_term) % p

        reset_difference = (modified[low] - theta(f, low, modulus)) % modulus
        assert not np.any(reset_difference % p)
        reset_jet = (reset_difference // p) % p
        reset_closed = np.zeros(length, dtype=np.int64)
        low_factorial = math.factorial(low) % p
        x_mod_p_powers = [power(x_mod_p, index, p) for index in range(low + 1)]
        for j in range(low):
            coefficient = (
                -low_factorial
                * pow((math.factorial(j) * (low - j)) % p, -1, p)
            ) % p
            reset_closed = (
                reset_closed
                + coefficient
                * conv(x_mod_p_powers[low - j], theta(f, j, p), p)
            ) % p
        if not np.array_equal(reset_jet, reset_closed):
            raise AssertionError(f"reset formula failed at {(p, k)}")

        for row in rows:
            if int(row["p"]) != p or int(row["k"]) != k:
                continue
            n = int(row["n"])
            ip = int(row["ip"])
            lower_weight = int(row["lower_weight"])
            if ip + n == 2 * p - k - 1:
                if not env["in_Mw"](pre_low_correction, lower_weight, p):
                    raise AssertionError(
                        f"first cutoff failed at {(p, k, n, ip)}"
                    )
                first_cutoffs += 1
            elif ip + n == 2 * p - k:
                if not env["in_Mw"](reset_jet, lower_weight, p):
                    raise AssertionError(
                        f"second cutoff failed at {(p, k, n, ip)}"
                    )
                second_cutoffs += 1

    print(
        f"families={len(families)} "
        f"triangular_identities={triangular_identities} "
        f"modularized_jets={modularized_jets} "
        f"first_cutoffs={first_cutoffs} "
        f"second_cutoffs={second_cutoffs} failures=0"
    )


if __name__ == "__main__":
    main()
