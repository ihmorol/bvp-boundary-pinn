# Step 60 · Settings

[Index](README.md) · page 6, the main run on Colab · group 17, notebook · [← step 59](59-figures.md) · [step 61 →](61-gpu-and-drive.md)

**In one line:** one notebook cell holds every choice of the run, so nothing else needs editing.

## What it holds
| Setting | Default | Meaning |
|---|---|---|
| `DRIVE_FOLDER` | bvp-boundary-pinn/main-run-P4 | where everything is kept, inside My Drive; reuse it to resume |
| `REPO_URL`, `REPO_BRANCH` | the GitHub repository, impl/campaign-readiness | the code |
| `REPO_COMMIT` | empty | empty: the commit of the first session in this Drive folder |
| `DEVICE`, `JOBS` | auto | GPU if present; runs at once |
| `ARMS`, `SEEDS`, `CAMPAIGN_NAME` | None | None: what configs/pinn.json says (B0 to B3, seeds 0 to 4, "P4") |
| `PINN_SETTINGS` | the committed values | points, network size, Adam, L-BFGS and LM budgets, window edges |
| `RUN_TESTS`, `SMOKE_TEST`, `MEASURE_TIMING` | True | the checks of steps 66, 68, 69 |
| `CLASSICAL_REPEATS` | 5 | [step 57](57-solve-bvp-timing.md) |
| `PYTHON_VERSION`, `UV_VERSION`, `TORCH_VERSION` | 3.12.13, 0.10.9, 2.15.0 | exact software versions |

The physical values cannot be set here: a setting that names one is refused ([step 5](05-guards.md)).

## In the code
- The settings cell of [tools/colab_main_run.ipynb](../../../tools/colab_main_run.ipynb).
- `PINN_SETTINGS` is written into the "common" block of configs/pinn.json by `apply_pinn_settings` ([tools/main_run.py:118](../../../tools/main_run.py#L118)), which then checks every arm. If the values equal the committed ones, nothing changes.

## Ready?
Yes. The training values are still pilot values (open decision).
