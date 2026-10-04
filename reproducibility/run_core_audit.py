"""Run the four exact audits reported in the paper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

CHECKS = (
    (
        ("verify_periodic_first_jet.py", "data/d0_obstruction_coordinates.csv"),
        "rows=692 checked=692 bad_weight=0 bad_mod_p=0 bad_beta=0",
    ),
    (
        ("stress_test_l3_uniformity.py", "--max-p", "61"),
        "checked=464 zero=0 failed_shape=0 wrong_scalar=0",
    ),
    (
        ("stress_test_l3_weight24.py", "--max-p", "43"),
        "ordinary_forms=186 checked=1860 zero=0 failed_shape=0 wrong_scalar=0",
    ),
    (
        ("audit_index_formulas.py",),
        "audited_cells=210176 failures=0 max_prime=101",
    ),
)


def main() -> int:
    for arguments, expected in CHECKS:
        command = [sys.executable, *arguments]
        print(f"\n$ {' '.join(command)}", flush=True)
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(result.stdout, end="")
        if result.returncode != 0:
            print(f"FAILED: command returned {result.returncode}", file=sys.stderr)
            return result.returncode
        if expected not in result.stdout:
            print(f"FAILED: expected summary not found: {expected}", file=sys.stderr)
            return 1

    print("\nAll four core audits passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
