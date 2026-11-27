# Step 22 · Reference for checking only

[Index](README.md) · page 2, the classical reference · group 7, cross-checks · [← step 21](21-l-study.md) · [step 23 →](23-points.md)

**In one line:** the reference is only for testing. Training never sees it.

## What it does
- Branch A is compared with each PINN run after training and after the acceptance rule ([steps 50 to 57](50-load-reference.md)).
- Branch B is used only to say which branch a run landed nearest to.
- Training and acceptance code must not read reference data (gate G5). Otherwise a PINN result could be tuned toward the answer.

## In the code
- The training package `bvp_pinn.pinn` never imports `reference.io` or `bvp_pinn.evaluation`; only `bvp_pinn.evaluation` loads reference data (`load_references`, [evaluation/evaluate.py:55](../../../src/bvp_pinn/evaluation/evaluate.py#L55)).

## Checked by
`test_training_code_never_reads_the_reference` in [tests/test_pinn.py](../../../tests/test_pinn.py) reads every import of every training file.

## Ready?
Yes.
