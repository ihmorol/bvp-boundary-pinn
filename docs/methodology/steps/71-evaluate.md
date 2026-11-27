# Step 71 · Evaluate

[Index](README.md) · page 6, the main run on Colab · group 18, main run · [← step 70](70-campaign.md) · [step 72 →](72-package.md)

**In one line:** turn the finished run folders into metrics, solve_bvp timings, tables and figures, all kept in Drive.

## What it does
1. `python -m bvp_pinn.evaluation.evaluate`: [steps 50 to 56](50-load-reference.md) for every run of the campaign; `summary.csv`, `failures.md`, `settings.json`.
2. `python -m bvp_pinn.evaluation.classical`: [step 57](57-solve-bvp-timing.md), 5 repeats.
3. `python -m bvp_pinn.evaluation.plots`: [steps 58 and 59](58-tables.md).

Results go to `<Drive folder>/results/<campaign>/` (`BVP_RESULTS_DIR`). The step runs on every start, because it is cheap (a few minutes) and rebuilds everything from the run folders.

## In the code
- `step_evaluate` ([tools/main_run.py:312](../../../tools/main_run.py#L312)).

## Ready?
Yes, tested on dry runs.
