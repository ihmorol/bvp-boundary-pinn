# Step 64 · Pins

[Index](README.md) · page 6, the main run on Colab · group 18, main run · [← step 63](63-packages.md) · [step 65 →](65-device.md)

**In one line:** the first session records the commit and the training settings; every later session must match them, so one campaign never mixes code or settings.

## What it does
- On the first start it writes `steps/start.json` in the Drive folder: the commit, a fingerprint of configs/pinn.json, and any uncommitted changes in src, tools, tests and engine/bvp_engine.cpp.
- On every later start it compares. Any difference stops the run with a message: use the pinned commit and settings, or a new Drive folder.
- Data files the reference step rewrites (engine/out, reference/) are not counted as code changes.

## Failures and restarts (page 6, group 19)
| Event | What happens |
|---|---|
| A step fails | The run stops with the step name and the last 25 lines of its output; the full output is in `logs/<step>.log` |
| Colab disconnects | Run all again: finished steps are skipped, finished runs are skipped, the run that was in progress starts again from its beginning |
| Half-written files | `meta.json`, `selection.json` and `metrics.json` are written through a temporary file and a rename, so they are whole or absent; `meta.json` marks a run finished, `selection.json` marks it judged |
| The cell is stopped | The notebook stops the whole process group, so no training keeps running in the background |

## In the code
- `check_same_code` ([tools/main_run.py:141](../../../tools/main_run.py#L141)); `StepFailed` and `sh` ([line 99](../../../tools/main_run.py#L99)) for the failure message and logs; `write_json` ([pinn/runs.py:107](../../../src/bvp_pinn/pinn/runs.py#L107)); the process-group stop in cell "Step D" of the notebook.

## Checked by
A local restart test on 2026-11-26: a lost verdict was judged late, a cut-off run was trained again, a deleted reference was restored, and changed settings stopped the run. That test also found and fixed a bug where the check counted data files as code.

## Ready?
Yes.
