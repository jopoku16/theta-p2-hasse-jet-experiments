# Theta cycles modulo p^2

This repository supports the paper:

> **Experiments with Theta Cycles Modulo p^2: Hasse-Jet Duality and a
> Shifted-Conic Conjecture**  
> Hurcheson Asante Ofori and Jeffery Opoku, University of Texas Rio Grande Valley

The paper studies an unresolved part of the weight filtration of theta
iterates modulo `p^2`. The calculations use exact integer and finite-field
arithmetic. No floating-point approximation or random sampling is used.

## Main finite results

- The main data set has 692 ordinary cross-wedge cells.
- The proposed quadratic matches all 458 interior cells.
- All 234 failures lie on three stated boundary lines.
- The data contain 20 interior zeros, exactly where the shifted quadratic
  vanishes.
- Two later tests give 2,324 exact matches with the proposed factorial
  boundary scalar.
- An index and filtration audit checks 210,176 parameter cells through
  prime 101 with no recorded failure.

These are exact finite results. They do not prove the conjectures for every
prime. The paper states the remaining factorial block identity and two
endpoint identities as conjectures.

## Repository layout

- `manuscript/`: the paper PDF and all LaTeX source files.
- `reproducibility/`: scripts, exact data tables, pinned dependencies, and
  detailed commands.

## Quick checks

```text
cd reproducibility
python -m pip install -r requirements.txt
python run_core_audit.py
```

The audit stops if a command fails or an expected exact summary is missing.
The four commands run by the audit are:

```text
python verify_periodic_first_jet.py data/d0_obstruction_coordinates.csv
python stress_test_l3_uniformity.py --max-p 61
python stress_test_l3_weight24.py --max-p 43
python audit_index_formulas.py
```

Expected final lines:

```text
rows=692 checked=692 bad_weight=0 bad_mod_p=0 bad_beta=0
checked=464 zero=0 failed_shape=0 wrong_scalar=0
ordinary_forms=186 checked=1860 zero=0 failed_shape=0 wrong_scalar=0
audited_cells=210176 failures=0 max_prime=101
```

Release `v1.1.0` contains the corrected author record, the current manuscript,
the exact audit runner, and the full source needed to reproduce these checks.

## Compile the paper

Run the following command from `manuscript/`:

```text
pdflatex experimental_manuscript.tex
pdflatex experimental_manuscript.tex
```

## Citation

Citation information is provided in `CITATION.cff`. Please cite the paper
and the archived release when a DOI becomes available.
