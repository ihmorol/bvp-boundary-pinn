# Step 16 · Export

[Index](README.md) · page 2, the classical reference · group 6, checks at L = 20 · [← step 15](15-slope-check.md) · [step 17 →](17-reference-a.md)

**In one line:** save the verified reference on a fine grid, with a record of how it was made and a fingerprint of each file.

## What it does
- The solution is evaluated on 4 001 points, first step 5 × 10⁻⁴, dense at the wall. This grid is also where every PINN run is tested.
- Saved per branch in `reference/locked-branchA-L20-tol1e-08/`: `reference.npz` (η, the 12 states z, the 5 highest derivatives H, the solver mesh), `meta.json` (settings, versions, every check, the case fingerprint), `SHA256SUMS`.
- Also saved: `reference/study.json` and `study.md` (table TB1), `l_study.npz`, and the reference figures.

## In the code
- `export` ([reference/run.py:79](../../../src/bvp_pinn/reference/run.py#L79)) builds the arrays and the record.
- `write_reference` ([reference/io.py:18](../../../src/bvp_pinn/reference/io.py#L18)) writes the files and their fingerprints, at [reference/run.py:183](../../../src/bvp_pinn/reference/run.py#L183).
- Figures: `plots.make_all` at [reference/run.py:195](../../../src/bvp_pinn/reference/run.py#L195).

## Ready?
Yes. The rebuild of 2026-11-26 matched the saved reference exactly (largest difference 0.0).
