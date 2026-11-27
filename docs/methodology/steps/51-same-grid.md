# Step 51 · Same grid

[Index](README.md) · page 5, acceptance and evaluation · group 15, test · [← step 50](50-load-reference.md) · [step 52 →](52-field-errors.md)

**In one line:** the PINN is evaluated on exactly the reference's 4 001 points, so every comparison is point by point.

## What it does
- The trained network gives all 12 states z and the 5 highest derivatives H at the reference grid: 17 curves.
- B1's window networks: each point is sent to the network that owns its window.
- The predictions are saved next to the run as `pred_eval.npz`, so figures never need to re-run the network.

## In the code
- `predict` ([pinn/model.py:98](../../../src/bvp_pinn/pinn/model.py#L98)) with `window_of` for B1.
- Called in `evaluate_run` ([evaluation/evaluate.py:96](../../../src/bvp_pinn/evaluation/evaluate.py#L96)).

## Ready?
Yes.
