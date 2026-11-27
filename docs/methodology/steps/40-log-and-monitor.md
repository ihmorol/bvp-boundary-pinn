# Step 40 · Log and monitor

[Index](README.md) · page 4, training one run · group 12, Adam · [← step 39](39-loss-weights.md) · [step 41 →](41-stage-checkpoint.md)

**In one line:** every 100 steps, record every loss term separately, and the loss on the monitor points.

## What it records (one history row)
- total loss, each of R14 to R18 (mean square), each of B1 to B12 (about 10⁻¹⁵ in exact mode, logged only), each weighted term, each weight, the learning rate, seconds.
- **monitor loss:** the full-problem loss on the 1 000 monitor points, without a gradient. If it rises while the training loss falls, the network is fitting the points, not the equations.
- A short progress line is printed every 2 000 Adam steps, 1 000 L-BFGS evaluations and 20 LM steps.

## In the code
- `row` ([pinn/train.py:230](../../../src/bvp_pinn/pinn/train.py#L230)), `monitor_loss` ([line 148](../../../src/bvp_pinn/pinn/train.py#L148)); progress printer `progress` ([pinn/run.py:42](../../../src/bvp_pinn/pinn/run.py#L42)). Rows are written to `history.csv` at the end ([step 46](46-run-folder.md)).

## Ready?
Yes.
