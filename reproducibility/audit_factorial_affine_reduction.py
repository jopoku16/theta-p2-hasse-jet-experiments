"""Audit the exact affine reduction and its two q-series recurrences.

The manuscript proves these identities symbolically.  This script checks
all coefficients in the Sturm-safe truncation for every cross-wedge row in
the released data.  It is a sign and branch audit, not a substitute for the
proof.
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


def main() -> None:
    data = Path(__file__).parent / "data" / "d0_obstruction_coordinates.csv"
    with data.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if int(row["reflected_D"]) == 2
        ]

    cache: dict[tuple[int, int], dict[str, object]] = {}
    affine_checked = 0
    recurrence_checked = 0

    for row in rows:
        p, k = int(row["p"]), int(row["k"])
        key = (p, k)
        if key not in cache:
            env = branchfree.build(p, k, p - 2, 3)
            length = env["N"] + 1
            modulus = p * p
            one = np.zeros(length, dtype=np.int64)
            one[0] = 1

            def conv_mod(left: np.ndarray, right: np.ndarray, mod: int) -> np.ndarray:
                return np.convolve(left, right)[:length] % mod

            def theta(series: np.ndarray, power: int, mod: int) -> np.ndarray:
                return np.array(
                    [pow(index, power, mod) * int(series[index]) % mod
                     for index in range(length)],
                    dtype=np.int64,
                )

            e2 = eisenstein(2, length, modulus)
            e4 = eisenstein(4, length, modulus)
            hasse = eisenstein(p - 1, length, modulus)
            ep1 = eisenstein(p + 1, length, modulus)
            f = env["f"].copy() % modulus

            def serred(series: np.ndarray, weight: int) -> np.ndarray:
                return (
                    theta(series, 1, modulus)
                    - weight * pow(12, -1, modulus)
                    * conv_mod(e2, series, modulus)
                ) % modulus

            def modified_serre(index: int) -> np.ndarray:
                if index == 0:
                    return f.copy()
                previous = f.copy()
                current = serred(f, k)
                if index == 1:
                    return current
                for j in range(1, index):
                    following = (
                        serred(current, k + 2 * j)
                        - j * (j + k - 1) * pow(144, -1, modulus)
                        * conv_mod(e4, previous, modulus)
                    ) % modulus
                    previous, current = current, following
                return current

            def theta_lift(series: np.ndarray, weight: int) -> np.ndarray:
                return (
                    conv_mod(hasse, theta(series, 1, modulus), modulus)
                    + weight * pow(12, -1, modulus)
                    * conv_mod(
                        (ep1 - conv_mod(hasse, e2, modulus)) % modulus,
                        series,
                        modulus,
                    )
                ) % modulus

            low = p - k + 1
            lift = modified_serre(low)
            lift_weight = k + 2 * low
            jets: dict[int, np.ndarray] = {}
            phis: dict[int, np.ndarray] = {}
            us: dict[int, np.ndarray] = {}
            for r in range(low, p):
                if r > low:
                    lift = theta_lift(lift, lift_weight)
                    lift_weight += p + 1
                u = theta(f, r, modulus)
                difference = (lift - u) % modulus
                assert not np.any(difference % p)
                jets[r] = (difference // p) % p
                periodic = (theta(f, r + p - 1, modulus) - u) % modulus
                assert not np.any(periodic % p)
                phis[r] = (periodic // p) % p
                us[r] = u % p

            alpha_difference = (hasse - one) % modulus
            assert not np.any(alpha_difference % p)
            alpha = (alpha_difference // p) % p
            theta_alpha = theta(alpha, 1, p)
            c_difference = (ep1 - conv_mod(hasse, e2, modulus)) % modulus
            assert not np.any(c_difference % p)
            xi = (c_difference // p) * pow(12, -1, p) % p
            x = e2 * pow(12, -1, p) % p
            x_powers = [one % p]
            for _ in range(p):
                x_powers.append(conv_mod(x_powers[-1], x, p))
            assert np.array_equal((theta_alpha + x_powers[p]) % p, x % p)
            y_series = (xi + x_powers[p]) % p
            assert env["in_Mw"](y_series, p + 1, p)

            s = p - k
            u_s = theta(f, s, p)
            families: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for d in range(1, k - 2):
                r = s + d
                u_r = us[r]
                u_previous = theta(f, r - 1, p)
                z_d = conv_mod(x_powers[d], u_s, p)
                cal_c = (
                    -phis[r] - jets[r] + (d + 1) * conv_mod(alpha, u_r, p)
                    - d * (d + 1 - k)
                    * conv_mod(x_powers[p], u_previous, p)
                ) % p
                cal_m = (
                    -phis[r] + 2 * conv_mod(alpha, u_r, p)
                    - (2 * d - k) * conv_mod(x_powers[p], u_previous, p)
                    + math.factorial(d - 1) * z_d
                ) % p
                families[d] = (cal_c, cal_m)

            for d in range(1, k - 3):
                r = s + d
                cal_c, cal_m = families[d]
                next_c, next_m = families[d + 1]
                expected_eta = (
                    theta(jets[r], 1, p)
                    + conv_mod(alpha, us[r + 1], p)
                    + (r + 1) * conv_mod(xi, us[r], p)
                ) % p
                if not np.array_equal(jets[r + 1], expected_eta):
                    first = int(np.flatnonzero((jets[r + 1] - expected_eta) % p)[0])
                    raise AssertionError(
                        f"eta recurrence failed at {(p, k, d, first)}"
                    )
                expected_c = (
                    theta(cal_c, 1, p)
                    + (k - d - 1)
                    * conv_mod((xi + x_powers[p]) % p, us[r], p)
                    - (d + 1) * conv_mod(x, us[r], p)
                ) % p
                extra = (
                    -2 * conv_mod(x, us[r], p)
                    + math.factorial(d) * pow(144, -1, p)
                    * conv_mod(
                        e4 % p,
                        conv_mod(x_powers[d - 1], u_s, p),
                        p,
                    )
                    - math.factorial(d - 1)
                    * conv_mod(x_powers[d], us[s + 1], p)
                ) % p
                expected_m = (theta(cal_m, 1, p) + extra) % p
                if not np.array_equal(next_c, expected_c):
                    first = int(np.flatnonzero((next_c - expected_c) % p)[0])
                    raise AssertionError(
                        f"C recurrence failed at {(p, k, d, first)}"
                    )
                if not np.array_equal(next_m, expected_m):
                    first = int(np.flatnonzero((next_m - expected_m) % p)[0])
                    raise AssertionError(
                        f"M recurrence failed at {(p, k, d, first)}"
                    )
                recurrence_checked += 1

            cache[key] = {
                "f": f,
                "theta": theta,
                "conv": conv_mod,
                "alpha": alpha,
                "jets": jets,
                "phis": phis,
                "us": us,
                "x_powers": x_powers,
                "families": families,
                "u_s": u_s,
            }

        m = p - 2 - int(row["n"])
        d = int(row["ip"]) - (p - k + m + 1)
        if d < 1:
            continue
        item = cache[key]
        theta = item["theta"]
        conv_mod = item["conv"]
        alpha = item["alpha"]
        jets = item["jets"]
        phis = item["phis"]
        us = item["us"]
        x_powers = item["x_powers"]
        cal_c, cal_m = item["families"][d]
        s = p - k
        r = s + d
        period_count = int(row["n"]) + 1
        a = p - 2 * m - d - 1
        q1 = (m * (2 * d - k) + d * (d + 1 - k)) % p
        direct_jet = (
            period_count * phis[r]
            - jets[r]
            - a * conv_mod(alpha, us[r], p)
        ) % p
        shifted = q1 * conv_mod(
            x_powers[p], theta(item["f"], r - 1, p), p
        ) % p
        z_d = conv_mod(x_powers[d], item["u_s"], p)
        left = (
            direct_jet - shifted + m * math.factorial(d - 1) * z_d
        ) % p
        right = (cal_c + m * cal_m) % p
        assert np.array_equal(left, right)
        affine_checked += 1

    print(
        f"affine_checked={affine_checked} "
        f"recurrence_checked={recurrence_checked} failures=0"
    )


if __name__ == "__main__":
    main()
