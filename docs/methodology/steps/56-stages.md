# Step 56 · Stages

[Index](README.md) · page 5, acceptance and evaluation · group 15, test · [← step 55](55-nearest-branch.md) · [step 57 →](57-solve-bvp-timing.md)

**In one line:** how accurate each run was before the refine chain, and how much each window kept its accuracy while later windows were trained.

## What it does
- **Per window, per stage:** for every saved stage file (`stage<k>.pt`), the largest error of each field on each window [0, 0.5], [0.5, 2], [2, 6], [6, 20]. For B2 and B3 this shows whether training a new window spoiled the ones already learned (retention, gate G4).
- **After Adam:** the largest wall error and the largest field relative L2 of the last stage file, before L-BFGS and Levenberg-Marquardt. All arms share the same refine chain, which may hide arm differences, so both "after Adam" and "final" are reported.

## In the code
- `window_errors` ([evaluation/evaluate.py:129](../../../src/bvp_pinn/evaluation/evaluate.py#L129)) and `after_adam` ([line 144](../../../src/bvp_pinn/evaluation/evaluate.py#L144)).

## Ready?
Yes.
