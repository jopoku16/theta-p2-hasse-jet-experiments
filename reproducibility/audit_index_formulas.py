"""Audit all algebraic indices and filtration gaps used in the manuscript."""

from __future__ import annotations

from sympy import primerange


def factorial_valuation(value: int, prime: int) -> int:
    total = 0
    while value:
        value //= prime
        total += value
    return total


def binomial_valuation(top: int, bottom: int, prime: int) -> int:
    return (
        factorial_valuation(top, prime)
        - factorial_valuation(bottom, prime)
        - factorial_valuation(top - bottom, prime)
    )


def main() -> None:
    cells = 0
    for p in primerange(7, 102):
        P = p * (p - 1)
        for k in range(4, p - 1, 2):
            ell0 = p - k + 1
            for m in range(1, (k - 2) // 2):
                n = p - 2 - m
                for d in range(0, k - 2 * m):
                    ip = p - k + m + 1 + d
                    i = n * p + ip
                    r = p - k + d
                    L = n + 1
                    w = k + 2 * i - P
                    assert i == r + L * (p - 1)
                    assert 2 <= r <= p - 1
                    assert w == p * p - (2 * m + 1) * p - k + 2 * m + 2 + 2 * d

                    if r < ell0:
                        v = k + r * (p + 1)
                        a = k - 2 * m - 2
                    else:
                        v = k + r * (p + 1) - ell0 * (p - 1)
                        a = p - 2 * m - d - 1
                    assert w - v == a * (p - 1)
                    assert a >= 0

                    q1 = (ip * (ip + k - 1) - (n + 1) * (n + 2)) % p
                    assert q1 == (m * (2 * d - k) + d * (d + 1 - k)) % p

                    if d in (0, 1):
                        total = 2 * p - k - 1 + d
                        if p <= 43:
                            for ell in range(n * p):
                                quotient, residue = divmod(ell, p)
                                actual = (
                                    binomial_valuation(i, ell, p)
                                    + factorial_valuation(i + k - 1, p)
                                    - factorial_valuation(ell + k - 1, p)
                                )
                                predicted = (
                                    n
                                    + 1
                                    - quotient
                                    - int(residue >= p - k + 1)
                                    + int(residue > ip)
                                )
                                assert actual == predicted
                                in_leading_interval = (
                                    n * p - k + 1 <= ell <= i - p
                                )
                                assert (actual == 1) == in_leading_interval

                        for t in range(0, p - k + 1):
                            exponent_x = ip - t
                            if t <= m + 1:
                                exponent_theta = p - 2 - m + t
                                assert exponent_x + exponent_theta == total
                                if d == 0:
                                    assert p - k <= exponent_x <= p - k + 2 * m + 1
                                else:
                                    assert p - k + 1 <= exponent_x <= p - k + 2 * m + 3
                            else:
                                exponent_theta = t - m - 1
                                assert exponent_x + exponent_theta == p - k + d
                                finite_weight = k + (p - k + d) * (p + 1)
                                expected_gap = (2 * m + 2 + d - k) * (p - 1)
                                assert finite_weight - w == expected_gap
                                assert expected_gap <= 0

                    z_filtration = k + (p - k + d) * (p + 1)
                    assert z_filtration - w == (d - k + 2 * m + 2) * (p - 1)

                    if 1 <= d <= k - 2 * m - 2:
                        next_w = w + 2
                        recurrence_low = k + 2 * (p - k) + 2 + d * (p + 1)
                        recurrence_high = k + (p - k + d - 1) * (p + 1) + 4
                        assert recurrence_low - next_w == (
                            -(p - 2 - d - 2 * m) * (p - 1)
                        )
                        assert recurrence_high - next_w == (
                            (d - k + 2 * m + 1) * (p - 1)
                        )
                        assert recurrence_low < next_w
                        assert recurrence_high < next_w

                    i_dagger = p * p + 1 - k - i
                    r_minus = k + 2 * i
                    r_plus = k + 2 * i_dagger + 2 * P
                    assert r_minus + r_plus == P + 2 + p * (3 * p - 1)
                    cells += 1

    print(f"audited_cells={cells} failures=0 max_prime=101")


if __name__ == "__main__":
    main()
