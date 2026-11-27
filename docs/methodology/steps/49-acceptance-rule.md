# Step 49 · Acceptance rule

[Index](README.md) · page 5, acceptance and evaluation · group 14, acceptance · [← step 48](48-score.md) · [step 50 →](50-load-reference.md)

**In one line:** a fixed rule, set before any campaign run, accepts or rejects each finished run without the reference.

## The rule
A run is **accepted** only if all of these hold:

| Test | Bar | Why |
|---|---|---|
| training finished | no non-finite loss | |
| far field decays | m > 0 and m_p > 0 | otherwise θ, φ cannot go to zero ([step 11](11-far-decay.md)) |
| tail share | at most 0.05 | no constant, non-decaying part in θ ([step 45](45-self-check.md)) |
| largest \|B_i\| | at most 10⁻⁸ | all 12 conditions met |
| score | at most 10⁻³ | estimated wall error ([step 48](48-score.md)) |

- If only the estimate is missing (for example the engine could not run), the verdict is **undecided**, not rejected.
- **Rejected runs stay.** They appear in every table and get one line in `failures.md`. A seed is never re-run to get a better result.
- The bars come from the pilots: the good runs had scores 4 × 10⁻⁵ to 3 × 10⁻⁴, the failed ones 0.04 and above; the reference's tail share is 0.003, failed runs 0.6 and 1.8.

## In the code
1. `assess_run` ([pinn/estimate.py:112](../../../src/bvp_pinn/pinn/estimate.py#L112)) loads the run, runs the estimate, writes `error_estimate.json`, then the verdict `selection.json` (both written whole, through a temporary file).
2. `judge` ([pinn/estimate.py:87](../../../src/bvp_pinn/pinn/estimate.py#L87)) applies the rule above.
3. The bars: `selection` block of [configs/pinn.json](../../../configs/pinn.json) (`score_max`, `tail_share_max`, `boundary_max`).

## Checked by
`test_acceptance_rule` in [tests/test_pinn.py](../../../tests/test_pinn.py), including that a NaN score never passes.

## Ready?
Yes. Open decision: whether φ(0) joins the score.
