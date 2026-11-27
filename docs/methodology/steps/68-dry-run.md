# Step 68 · Dry run

[Index](README.md) · page 6, the main run on Colab · group 18, main run · [← step 67](67-reference-step.md) · [step 69 →](69-timing.md)

**In one line:** run every arm once with tiny settings on the real device, then evaluate it, so a problem shows in a minute, not after hours.

## What it does
- A dry campaign: arms B0 to B3, seed 0, with the dry settings (a 16 × 2 network, 120 points, 40 Adam steps, 5 L-BFGS iterations, 3 LM steps), on the chosen device.
- Then the evaluation and the figures of that dry campaign.
- Everything goes to `<Drive folder>/smoke`, apart from the real runs. Its numbers mean nothing; only that the chain runs.
- Runs once per device, job count and commit, so new scheduling code is tried on the device for a minute before hours of training. With 2 runs at once it also checks the parallel path.

## In the code
- `step_smoke` ([tools/main_run.py:323](../../../tools/main_run.py#L323)); the dry settings `DRY_RUN` ([pinn/runs.py:31](../../../src/bvp_pinn/pinn/runs.py#L31)).

## Ready?
Yes. On a GPU this is the first real test of the GPU path.
