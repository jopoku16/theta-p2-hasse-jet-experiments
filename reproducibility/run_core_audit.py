"""Run the seven exact audits reported in the paper."""

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
        ("stress_test_l3_uniformity.py", "--max-p", "79"),
        "checked=652 zero=0 failed_shape=0 wrong_scalar=0",
    ),
    (
        ("stress_test_l3_weight24.py", "--max-p", "43"),
        "ordinary_forms=186 checked=1860 zero=0 failed_shape=0 wrong_scalar=0",
    ),
    (
        ("audit_index_formulas.py",),
        "audited_cells=210176 failures=0 max_prime=101",
    ),
    (
        ("audit_factorial_affine_reduction.py",),
        "affine_checked=614 recurrence_checked=156 failures=0",
    ),
    (
        (
            "audit_triangular_endpoint_source.py",
            "data/d0_obstruction_coordinates.csv",
            "--examples",
            "0",
        ),
        (
            "outcomes={'L1_B_endpoint_zero_match': 78, "
            "'L1_correction_endpoint_match': 78, 'L1_unit_match': 78, "
            "'L2_B_endpoint_zero_match': 78, "
            "'L2_correction_endpoint_match': 78, 'L2_unit_match': 78}"
        ),
    ),
    (
        ("audit_low_point_jets.py",),
        (
            "families=15 triangular_identities=273 modularized_jets=273 "
            "first_cutoffs=78 second_cutoffs=78 failures=0"
        ),
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

    print("\nAll seven core audits passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
