# Step 64 · Pins

[Index](README.md) · page 6, the main run on Colab · group 18, main run · [← step 63](63-packages.md) · [step 65 →](65-device.md)

**In one line:** the first session records the commit and the training settings; every later session must have the same settings and the same training code, so one campaign never mixes how runs train.

## What it does
- On the first start it writes `steps/start.json` in the Drive folder: the commit, a fingerprint of configs/pinn.json, and any uncommitted changes in src, tools, tests and engine/bvp_engine.cpp.
- On every later start it compares. Other settings, or uncommitted code changes, stop the run. Another commit is allowed only if `git diff` against the first commit shows no change in the files that decide how a run trains and is judged (`METHOD_PATHS`): src/bvp_pinn apart from pinn/campaign.py (scheduling) and evaluation/ (rebuilt for every run at the end), configs/, engine/bvp_engine.cpp, uv.lock and pyproject.toml. Otherwise it stops and names the files; set `REPO_COMMIT` to the first commit, or use a new Drive folder.
- Commit 2895060 was the last to change training code; everything committed after it changed none (`test_later_commits_changed_no_training_code`).
- Data files the reference step rewrites (engine/out, reference/) are not counted as code changes.

## Failures and restarts (page 6, group 19)
| Event | What happens |
|---|---|
| A step fails | The run stops with the step name and the last 25 lines of its output; the full output is in `logs/<step>.log` |
| Colab disconnects | Run all again: finished steps are skipped, finished runs are skipped, the runs that were in progress start again from their beginning |
| A run runs out of GPU memory beside other runs | No result: its failure record is removed and it trains again from its start with the GPU to itself ([step 70](70-campaign.md)). Out of memory alone is a result, like any failure |
| Half-written files | `meta.json`, `selection.json` and `metrics.json` are written through a temporary file and a rename, so they are whole or absent; `meta.json` marks a run finished, `selection.json` marks it judged |
| The cell is stopped | The notebook stops the whole process group, so no training keeps running in the background |

## In the code
- `check_same_code` ([tools/main_run.py:170](../../../tools/main_run.py#L170)) and `method_changes` ([line 161](../../../tools/main_run.py#L161)); `StepFailed` and `sh` ([line 119](../../../tools/main_run.py#L119)) for the failure message and logs; `write_json` ([pinn/runs.py:107](../../../src/bvp_pinn/pinn/runs.py#L107)); the process-group stop in cell "Step D" of the notebook.

## Checked by
A local restart test on 2026-11-26: a lost verdict was judged late, a cut-off run was trained again, a deleted reference was restored, and changed settings stopped the run. That test also found and fixed a bug where the check counted data files as code. The check that allows a newer commit with the same training code (2026-11-28) is tested on the git history (`test_later_commits_changed_no_training_code`); its first real use is the next Colab session ([PARALLEL_RUNS_2026-11-28.md](../../reviews/PARALLEL_RUNS_2026-11-28.md)).

## Ready?
Yes.
