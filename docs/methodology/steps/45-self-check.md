# Step 45 · Self check

[Index](README.md) · page 4, training one run · group 13, refine · [← step 44](44-damping.md) · [step 46 →](46-run-folder.md)

**In one line:** right after training, the run measures itself, without the reference: wall values, decay powers, condition errors, where the residual is still large, and whether the θ tail truly decays.

## What it records
| Item | How | Why |
|---|---|---|
| wall values | f″(0), θ(0), θ′(0), φ′(0), χ′(0), f(20) | the numbers the study reports |
| m, m_p | from f(20), θ(20) ([step 11](11-far-decay.md)) | m > 0 and m_p > 0, or the tail cannot decay |
| max \|B_i\| | all 12 conditions | about 10⁻¹⁵ in exact mode |
| failure share | share of monitor points where some row is above 10⁻³ | where the equations are still missed |
| tail share | fit θ ≈ C₁ + C₂ q^(−m) on [10, 20]; tail share = \|C₁\|/max θ | C₁ is a constant part that never decays |
| monitor mean squares | each R, plain and on the final scale | compare arms |

The tail fit matters because a constant added to θ far out has almost no residual in Eqs. 16 and 17, so a small loss can hide it (run E8: tail share 1.8, against 0.003 for the reference).

## In the code
- `self_check` ([pinn/selfcheck.py:26](../../../src/bvp_pinn/pinn/selfcheck.py#L26)), `failure_share` ([line 44](../../../src/bvp_pinn/pinn/selfcheck.py#L44)), `tail_fit` ([line 51](../../../src/bvp_pinn/pinn/selfcheck.py#L51)). Called at [pinn/train.py:112](../../../src/bvp_pinn/pinn/train.py#L112).

## Checked by
`test_tail_fit_finds_a_constant_tail` in [tests/test_pinn.py](../../../tests/test_pinn.py).

## Ready?
Yes.
