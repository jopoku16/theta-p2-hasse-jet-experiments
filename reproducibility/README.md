# Reproducibility package

This archive supports the manuscript **Experiments with Theta Cycles Modulo
p^2: Hasse-Jet Duality and a Shifted-Conic Conjecture**.

## Environment

- Python 3.13.7
- NumPy 2.4.6
- SymPy 1.14.0

Install the pinned dependencies with:

    python -m pip install -r requirements.txt

Run all commands from the root of the extracted archive.

Run the seven main checks together with:

    python run_core_audit.py

The audit stops if a command fails or an expected exact summary is missing.

## Primary checks

Explicit first-Hasse-jet lift:

    python verify_periodic_first_jet.py data/d0_obstruction_coordinates.csv

Expected final line:

    rows=692 checked=692 bad_weight=0 bad_mod_p=0 bad_beta=0

Blind third-line extension:

    python stress_test_l3_uniformity.py --max-p 79

Expected final line:

    checked=652 zero=0 failed_shape=0 wrong_scalar=0

Independent weight-24 forms:

    python stress_test_l3_weight24.py --max-p 43

Expected final line:

    ordinary_forms=186 checked=1860 zero=0 failed_shape=0 wrong_scalar=0

Symbolic index and filtration audit:

    python audit_index_formulas.py

Expected final line:

    audited_cells=210176 failures=0 max_prime=101

Affine reduction and recurrence audit:

    python audit_factorial_affine_reduction.py

Expected final line:

    affine_checked=614 recurrence_checked=156 failures=0

Triangular endpoint source decomposition:

    python audit_triangular_endpoint_source.py data/d0_obstruction_coordinates.csv --examples 0

The output reports 78 matches on each line for the unit block, zero
finite-block endpoint, and first-Hasse-jet correction.  Every failure
category must be absent.

Low-point jet identities and cutoff reductions:

    python audit_low_point_jets.py

Expected final line:

    families=15 triangular_identities=273 modularized_jets=273 first_cutoffs=78 second_cutoffs=78 failures=0

## Interpretation

These commands reproduce the exact finite calculations reported in the
manuscript. They also audit the new symbolic low-point formulas. They do not
prove the factorial block reduction or the two triangular core-series
containments for arbitrary primes. The manuscript labels those statements
as conjectural.

## Contents

The folder contains only code, data, dependencies, and instructions needed
to check the reported calculations.
