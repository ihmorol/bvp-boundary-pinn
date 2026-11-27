# Step 46 · Run folder

[Index](README.md) · page 4, training one run · group 13, refine · [← step 45](45-self-check.md) · [step 47 →](47-newton-step-estimate.md)

**In one line:** everything one run made goes into one folder, named after its settings, and meta.json is written last.

## What it does
- **Name:** `bvp-<arm>-s<seed>-<hash>`, where hash is the first 8 characters of the SHA-256 of all the run's settings. Same settings, same folder; changed settings, new folder.
- **At the start:** `config.json` (every setting and the seed).
- **At the end:** `model.pt` (final weights), `stage<k>.pt`, `points.npz` (training and monitor points), `history.csv` (every row of [step 40](40-log-and-monitor.md)), then `meta.json` (result, times, versions, device, GPU, parameter count, refine stages, self check), written whole through a temporary file.
- A crashed run writes `meta.json` with its error. It counts as a result and is never re-run.
- `meta.json` existing means "finished", which is how a restarted campaign skips it.

## In the code
- `run_id` ([pinn/runs.py:102](../../../src/bvp_pinn/pinn/runs.py#L102)), `write_json` ([line 107](../../../src/bvp_pinn/pinn/runs.py#L107)), `RunFolder` ([line 122](../../../src/bvp_pinn/pinn/runs.py#L122)), `save` ([line 134](../../../src/bvp_pinn/pinn/runs.py#L134)), `save_failure` ([line 150](../../../src/bvp_pinn/pinn/runs.py#L150)).
- Folder location: `runs/`, or `BVP_RUNS_DIR` (Google Drive on Colab).

## Checked by
`test_every_arm_trains_end_to_end`, `test_a_crashed_run_is_recorded_and_not_repeated` in [tests/test_pinn.py](../../../tests/test_pinn.py).

## Ready?
Yes.
