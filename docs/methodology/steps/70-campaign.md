# Step 70 · Campaign

[Index](README.md) · page 6, the main run on Colab · group 18, main run · [← step 69](69-timing.md) · [step 71 →](71-evaluate.md)

**In one line:** train every arm and seed (4 × 5 = 20 runs), one run after another, with every run folder in Drive.

## What it does
1. Lists the planned runs. Each has a fixed folder name from its settings ([step 46](46-run-folder.md)).
2. Runs that have `meta.json` are finished and skipped, whatever their result.
3. Runs with a model but no verdict (the session stopped during the acceptance rule) are judged now, never retrained.
4. The rest are trained ([steps 23 to 46](23-points.md)) and judged ([steps 47 to 49](47-newton-step-estimate.md)), one or more at a time.
5. After the campaign command, every planned run must have `meta.json`; otherwise the step stops, and the next start resumes.

This step always starts again on a restart (unlike the earlier steps), so seeds or arms added later still run. A run that was cut off starts again from its beginning; at roughly 0.5 to 4 hours per run, depending on the device, that is the most a disconnect loses.

## In the code
- `step_campaign` ([tools/main_run.py:300](../../../tools/main_run.py#L300)) runs `python -m bvp_pinn.pinn.campaign` with `BVP_RUNS_DIR` set to the Drive folder.
- [pinn/campaign.py](../../../src/bvp_pinn/pinn/campaign.py): `main` ([line 57](../../../src/bvp_pinn/pinn/campaign.py#L57)), `planned` ([line 31](../../../src/bvp_pinn/pinn/campaign.py#L31)), `unjudged` ([line 43](../../../src/bvp_pinn/pinn/campaign.py#L43)), `_one` ([line 48](../../../src/bvp_pinn/pinn/campaign.py#L48)).

## Checked by
`test_resumed_campaign_judges_a_run_that_lost_its_verdict` in [tests/test_evaluation.py](../../../tests/test_evaluation.py); the local restart test of 2026-11-26.

## Ready?
The code is ready. The campaign has not run. Its settings are pilot values (open decision).
