# Step 54 · Equations and conditions

[Index](README.md) · page 5, acceptance and evaluation · group 15, test · [← step 53](53-wall-values.md) · [step 55 →](55-nearest-branch.md)

**In one line:** how well the PINN meets the equations and conditions on the test grid, and how good its own error estimate was.

## What it records
| Item | Meaning |
|---|---|
| `residual_mse` | mean square of each of R14 to R18 on the 4 001 points (the plain rows of [step 34](34-residual-rows.md), no scale) |
| `boundary_abs` | the size of each of B1 to B12 (about 10⁻¹⁵ when exact) |
| `error_estimate` | for each wall value: the estimated error of [step 47](47-newton-step-estimate.md) next to the true error PINN − reference |

The last row tests the estimator itself: if estimated and true errors agree, the acceptance rule was judging the right thing.

## In the code
- In `evaluate_run` ([evaluation/evaluate.py:96](../../../src/bvp_pinn/evaluation/evaluate.py#L96)), using `physics.residuals` and `losses.boundary_errors`; written to `metrics.json`.

## Ready?
Yes.
