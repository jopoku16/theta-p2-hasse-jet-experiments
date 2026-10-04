"""Audit all algebraic indices and filtration gaps used in the manuscript."""

from __future__ import annotations

from sympy import primerange


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

                    z_filtration = k + (p - k + d) * (p + 1)
                    assert z_filtration - w == (d - k + 2 * m + 2) * (p - 1)

                    i_dagger = p * p + 1 - k - i
                    r_minus = k + 2 * i
                    r_plus = k + 2 * i_dagger + 2 * P
                    assert r_minus + r_plus == P + 2 + p * (3 * p - 1)
                    cells += 1

    print(f"audited_cells={cells} failures=0 max_prime=101")


if __name__ == "__main__":
    main()
