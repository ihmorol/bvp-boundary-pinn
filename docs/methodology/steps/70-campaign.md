# Step 70 · Campaign

[Index](README.md) · page 6, the main run on Colab · group 18, main run · [← step 69](69-timing.md) · [step 71 →](71-evaluate.md)

**In one line:** train every arm and seed (4 × 5 = 20 runs), one or more at a time, each in its own process, with every run folder in Drive.

## What it does
1. Lists the planned runs. Each has a fixed folder name from its settings ([step 46](46-run-folder.md)).
2. Runs that have `meta.json` are finished and skipped, whatever their result.
3. Runs with a model but no verdict (the session stopped during the acceptance rule) are judged now, never retrained.
4. The rest are trained ([steps 23 to 46](23-points.md)) and judged ([steps 47 to 49](47-newton-step-estimate.md)), in the order of arms and seeds, up to `JOBS` at a time. Each run is a new process with one thread and its own seed, so it trains exactly as it would alone (checked bit for bit by `test_a_run_beside_another_trains_exactly_as_alone`).
5. A run that runs out of GPU memory while others share the GPU has no result: its failure record is removed, `needs_alone.json` is written, and it trains again from its start with no other run beside it. Out of memory with no other run is a result like any other failure.
6. With `SESSION_HOURS`, a run starts only if it is expected to end before the session limit (10 minutes kept free). Expected time of one run alone: the median of this folder's finished runs of that arm that ran alone, else the timing estimate ([step 69](69-timing.md)) scaled by how far it was from the finished runs; with others at once, 1.6 times that until runs of this session measure the real factor. When nothing more fits, the main run says how many runs are left and stops cleanly.
7. Each finished run gets `schedule.json`: runs at once (the most that trained beside it), device, start, end. Runs from before this file existed get one written later if every earlier session ran one at a time. The cost table and FX1, FX2 show it, because times measured with others at once are longer than alone.
8. Every ten minutes a line shows how busy the GPU is (the share of time a GPU operation ran), its memory, and the CPU load (`logs/usage.log`).
9. After the campaign command, every planned run must have `meta.json`; otherwise the step stops (or, at the session limit, ends cleanly), and the next start resumes.

This step always starts again on a restart (unlike the earlier steps), so seeds or arms added later still run. A run that was cut off starts again from its beginning; at roughly 0.5 to 4 hours per run, depending on the device, that is the most a disconnect loses per run in progress.

## In the code
- `step_campaign` ([tools/main_run.py:418](../../../tools/main_run.py#L418)) runs `python -m bvp_pinn.pinn.campaign` with `BVP_RUNS_DIR` set to the Drive folder; `run_time_priors` ([line 374](../../../tools/main_run.py#L374)) and `deadline` ([line 369](../../../tools/main_run.py#L369)) for the session limit; `monitor` ([line 410](../../../tools/main_run.py#L410)) for the usage lines.
- [pinn/campaign.py](../../../src/bvp_pinn/pinn/campaign.py): `main` ([line 207](../../../src/bvp_pinn/pinn/campaign.py#L207)), `planned` ([line 67](../../../src/bvp_pinn/pinn/campaign.py#L67)), `unjudged` ([line 79](../../../src/bvp_pinn/pinn/campaign.py#L79)), `schedule` ([line 142](../../../src/bvp_pinn/pinn/campaign.py#L142)), `worker` ([line 90](../../../src/bvp_pinn/pinn/campaign.py#L90)).

## Checked by
`test_resumed_campaign_judges_a_run_that_lost_its_verdict` in [tests/test_evaluation.py](../../../tests/test_evaluation.py); the scheduling tests in [tests/test_campaign.py](../../../tests/test_campaign.py) (runs at once, the session limit, out of memory beside others); the local restart test of 2026-11-26. The new scheduling has not run on a GPU yet; the dry run of [step 68](68-dry-run.md) tries it first in each new Colab session.

## Ready?
The code is ready. The campaign has started on a free Colab T4 GPU: in the first session B0 seeds 0 to 4 finished (about 43 minutes each), then Colab's GPU allowance ran out during B1 seed 0 (seen 2026-11-28). Its settings are pilot values (open decision).
