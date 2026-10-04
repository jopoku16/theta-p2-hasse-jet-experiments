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

## Primary checks

Explicit first-Hasse-jet lift:

    python verify_periodic_first_jet.py data/d0_obstruction_coordinates.csv

Expected final line:

    rows=692 checked=692 bad_weight=0 bad_mod_p=0 bad_beta=0

Blind third-line extension:

    python stress_test_l3_uniformity.py --max-p 61

Expected final line:

    checked=464 zero=0 failed_shape=0 wrong_scalar=0

Independent weight-24 forms:

    python stress_test_l3_weight24.py --max-p 43

Expected final line:

    ordinary_forms=186 checked=1860 zero=0 failed_shape=0 wrong_scalar=0

Symbolic index and filtration audit:

    python audit_index_formulas.py

Expected final line:

    audited_cells=210176 failures=0 max_prime=101

## Interpretation

These commands reproduce the exact finite calculations reported in the
manuscript. They do not prove the factorial block reduction or the two
triangular endpoint coefficient identities for arbitrary primes. The
manuscript labels those statements as conjectural.

## Contents

The archive contains only code, data, dependencies, and instructions needed
to check the reported calculations.
