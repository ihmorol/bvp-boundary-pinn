# Step 48 · Score

[Index](README.md) · page 5, acceptance and evaluation · group 14, acceptance · [← step 47](47-newton-step-estimate.md) · [step 49 →](49-acceptance-rule.md)

**In one line:** one number per run: the largest estimated relative error of four wall values.

## The equation
```text
score = max over k in {f″(0), θ(0), θ′(0), φ′(0)} of |δz_k(0)| / |z_k(0)|
```
δz from [step 47](47-newton-step-estimate.md), z from the PINN itself. Lower is better.

- On the pilot runs E3 to E8 the score ranked the runs in the same order as their true error.
- φ(0) is not in the score. Adding it is an open decision: it would have rejected the seed 1 pilot.

## In the code
- `SCORE_KEYS` ([pinn/estimate.py:42](../../../src/bvp_pinn/pinn/estimate.py#L42)); computed in `estimate` ([line 75](../../../src/bvp_pinn/pinn/estimate.py#L75)). Saved to `error_estimate.json` with the estimated error of every wall value.

## Ready?
Yes.
