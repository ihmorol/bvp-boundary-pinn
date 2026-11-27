# Step 55 · Nearest branch

[Index](README.md) · page 5, acceptance and evaluation · group 15, test · [← step 54](54-equations-and-conditions.md) · [step 56 →](56-stages.md)

**In one line:** which solution branch the run landed on, A (the target) or B.

## What it does
- The relative L2 distance of the run's θ curve from θ of branch A and from θ of branch B; the closer one is named.
- If the run's curve is not finite, the answer is "none".
- A run that is not nearest to A gets a line in `failures.md`, even if accepted.

## In the code
- `nearest_branch` ([evaluation/evaluate.py:161](../../../src/bvp_pinn/evaluation/evaluate.py#L161)).

## Ready?
Yes.
