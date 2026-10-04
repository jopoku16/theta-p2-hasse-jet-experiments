"""Verify an explicit lower-weight lift and periodic first-jet formula.

For a D=0 cross-wedge cell, write i=r+L(p-1), 1<=r<p.  The script
constructs a modular lift G_r of theta^r f modulo p:

* before the ordinary low point, iterate the modular theta lift T_w;
* at and after the low point, start with the modified Serre derivative
  f_(p-k+1) and then iterate T_w.

It multiplies G_r by the required Hasse power to obtain a form h of the
candidate lower weight.  It then checks, on complete Sturm-safe q-series,

    beta_w(theta^i f) = [(theta^i f-h)/p] modulo M_w.

This is a verification of the explicit formula in the tested range, not a
replacement for the symbolic proof recorded in the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from sympy import bernoulli

sys.path.insert(0, str(Path(__file__).parent / "code"))
import branchfree  # noqa: E402


def eisenstein(weight: int, length: int, modulus: int) -> np.ndarray:
    b = Fraction(bernoulli(weight))
    scale = Fraction(-2 * weight, 1) / b
    scale_mod = scale.numerator * pow(scale.denominator, -1, modulus) % modulus
    sigma = np.zeros(length, dtype=np.int64)
    for divisor in range(1, length):
        sigma[divisor::divisor] = (
            sigma[divisor::divisor]
            + pow(divisor, weight - 1, modulus)
        ) % modulus
    result = scale_mod * sigma % modulus
    result[0] = 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--p", type=int)
    parser.add_argument("--k", type=int)
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["reflected_D"]) == 2
            and (args.p is None or int(row["p"]) == args.p)
            and (args.k is None or int(row["k"]) == args.k)
        ]
    if args.limit is not None:
        rows = rows[: args.limit]

    checked = bad_weight = bad_mod_p = bad_beta = 0
    cache: dict[tuple[int, int], dict[str, object]] = {}

    for row in rows:
        p, k = int(row["p"]), int(row["k"])
        key = (p, k)
        if key not in cache:
            env = branchfree.build(p, k, p - 2, 3)
            length = env["N"] + 1
            modulus = p * p

            def conv(left: np.ndarray, right: np.ndarray) -> np.ndarray:
                return np.convolve(left, right)[:length] % modulus

            cache[key] = {
                "env": env,
                "length": length,
                "modulus": modulus,
                "e2": eisenstein(2, length, modulus),
                "e4": eisenstein(4, length, modulus),
                "A": eisenstein(p - 1, length, modulus),
                "B": eisenstein(p + 1, length, modulus),
                "conv": conv,
            }
        item = cache[key]
        env = item["env"]
        length = int(item["length"])
        modulus = int(item["modulus"])
        e2 = item["e2"]
        e4 = item["e4"]
        hasse = item["A"]
        ep1 = item["B"]
        conv = item["conv"]
        f = env["f"].copy() % modulus

        def theta(series: np.ndarray, power: int = 1) -> np.ndarray:
            return np.array(
                [pow(index, power, modulus) * int(series[index]) % modulus
                 for index in range(length)],
                dtype=np.int64,
            )

        def serred(series: np.ndarray, weight: int) -> np.ndarray:
            return (
                theta(series)
                - weight * pow(12, -1, modulus) * conv(e2, series)
            ) % modulus

        def theta_lift(series: np.ndarray, weight: int) -> np.ndarray:
            return (
                conv(hasse, theta(series))
                + weight
                * pow(12, -1, modulus)
                * conv((ep1 - conv(hasse, e2)) % modulus, series)
            ) % modulus

        def modified_serre(index: int) -> np.ndarray:
            if index == 0:
                return f.copy()
            previous = f.copy()
            current = serred(f, k)
            if index == 1:
                return current
            inv144 = pow(144, -1, modulus)
            for j in range(1, index):
                following = (
                    serred(current, k + 2 * j)
                    - j * (j + k - 1) * inv144 * conv(e4, previous)
                ) % modulus
                previous, current = current, following
            return current

        i = int(row["i"])
        lower_weight = int(row["lower_weight"])
        r = i % (p - 1)
        if r == 0:
            r = p - 1
        period_count = (i - r) // (p - 1)
        if period_count != int(row["n"]) + 1:
            raise RuntimeError((p, k, row["n"], row["ip"], r, period_count))

        low = p - k + 1
        if r < low:
            lift = f.copy()
            lift_weight = k
            for _ in range(r):
                lift = theta_lift(lift, lift_weight)
                lift_weight += p + 1
        else:
            lift = modified_serre(low)
            lift_weight = k + 2 * low
            for _ in range(r - low):
                lift = theta_lift(lift, lift_weight)
                lift_weight += p + 1

        difference = lower_weight - lift_weight
        if difference < 0 or difference % (p - 1):
            bad_weight += 1
            continue
        hasse_power = difference // (p - 1)
        h = lift.copy()
        for _ in range(hasse_power):
            h = conv(h, hasse)

        direct_series = theta(f, i)
        if np.any((direct_series - h) % p):
            bad_mod_p += 1
            continue
        first_jet = ((direct_series - h) % modulus) // p % p
        explicit = env["reduce_Mw"](first_jet, lower_weight, p)

        direct_remainder = env["reduce_Mw"](
            direct_series, lower_weight, modulus
        )
        if direct_remainder is None or np.any(direct_remainder % p):
            raise RuntimeError((p, k, row["n"], row["ip"], "direct"))
        canonical = (direct_remainder // p) % p
        if explicit is None or not np.array_equal(explicit % p, canonical):
            bad_beta += 1
            continue
        checked += 1

    print(
        f"rows={len(rows)} checked={checked} bad_weight={bad_weight} "
        f"bad_mod_p={bad_mod_p} bad_beta={bad_beta}"
    )


if __name__ == "__main__":
    main()
