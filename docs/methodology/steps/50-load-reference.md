# Step 50 · Load the reference

[Index](README.md) · page 5, acceptance and evaluation · group 15, test · [← step 49](49-acceptance-rule.md) · [step 51 →](51-same-grid.md)

**In one line:** after all runs are judged, load reference A, and refuse it unless it is the right, verified solution.

## What it does
- Finds the reference folders by name from the settings (`locked-branchA-L20-tol1e-08`, same for B).
- Checks every file's fingerprint and that the reference was built from the locked case file.
- Branch A must sit at L = 20, pass gate G3 and decay (m > 0). Otherwise evaluation stops: no PINN is compared with an unverified target.
- Each run must have been trained at the same L as the reference.

## In the code
1. `reference_folders` ([evaluation/evaluate.py:36](../../../src/bvp_pinn/evaluation/evaluate.py#L36)).
2. `load_references` ([line 55](../../../src/bvp_pinn/evaluation/evaluate.py#L55)) calls `load_reference` ([reference/io.py:26](../../../src/bvp_pinn/reference/io.py#L26)), which checks `SHA256SUMS` and the case fingerprint, then checks L, `G3_fixed_L_pass` and `far_field_valid` of branch A.
3. `evaluate_run` ([evaluation/evaluate.py:96](../../../src/bvp_pinn/evaluation/evaluate.py#L96)) refuses a run with another L.

## Checked by
`test_evaluation_refuses_a_target_that_failed_its_checks`, `test_saved_target_reference_passes_the_guard` in [tests/test_evaluation.py](../../../tests/test_evaluation.py); `test_loader_round_trip_and_tamper` in [tests/test_reference_verify.py](../../../tests/test_reference_verify.py).

## Ready?
Yes.
