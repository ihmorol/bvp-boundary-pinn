# Step 72 · Package

[Index](README.md) · page 6, the main run on Colab · group 18, main run · [← step 71](71-evaluate.md)

**In one line:** copy the settings into the Drive folder and write a README that says where everything is.

## What the Drive folder holds at the end
| Folder or file | Content |
|---|---|
| `steps/` | one marker per finished step; `start.json` with the pinned commit and settings fingerprint |
| `logs/` | the full output of every step |
| `environment-*.json` | machine, packages, GPU and settings of every session |
| `configs/` | the three config files the run used |
| `reference/`, `engine_out/` | the reference built in this run, its study and error bars, the engine's mesh study and branch search |
| `runs/` | one folder per run: settings, history, models, verdict, metrics, predictions |
| `results/<campaign>/` | summary, failures, settings, tables TB2, TB3, TX1, timing, figures |
| `smoke/` | the dry run; its numbers mean nothing |
| `README.txt` | this list and the commit |

Runs on every start.

## In the code
- `step_package` ([tools/main_run.py:319](../../../tools/main_run.py#L319)).

## Ready?
Yes.
