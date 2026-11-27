# Step 69 · Timing

[Index](README.md) · page 6, the main run on Colab · group 18, main run · [← step 68](68-dry-run.md) · [step 70 →](70-campaign.md)

**In one line:** measure how long one Adam step and one Levenberg-Marquardt step take on this machine, and print the campaign estimate.

## What it does
- Times B0 and B1 at the real settings (2 000 points, 40 × 4 network): one full Adam step (loss and backward) and one LM step on the balanced scale.
- Estimates each run and the whole campaign.

## The equation
```text
run time ≈ adam_steps × step × 1.05 + 1.25 × lbfgs_iters × step + lm_iters × lm_step
campaign ≈ Σ over arms of run time × number of seeds, divided by runs at once
```
The 1.05 covers logging and monitor loss; 1.25 is L-BFGS's limit of loss evaluations per iteration, so this is an upper bound.

**Numbers from the development laptop (one CPU thread):** 0.19 s per Adam step; 21 s per LM step for B0, 32 s for B1. That gives about 3 hours per B0 run on that CPU. On a T4 GPU the estimate is 13 to 30 hours for the campaign (worked out, not measured); this step replaces it with a measurement.

## In the code
- `step_timing` ([tools/main_run.py:270](../../../tools/main_run.py#L270)); the timer [tools/time_pinn_step.py](../../../tools/time_pinn_step.py) with `--device`, `--lm`, `--json`.

## Ready?
Yes.
