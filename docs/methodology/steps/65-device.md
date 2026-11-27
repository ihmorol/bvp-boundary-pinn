# Step 65 · Device

[Index](README.md) · page 6, the main run on Colab · group 18, main run · [← step 64](64-pins.md) · [step 66 →](66-tests.md)

**In one line:** choose GPU or CPU and how many runs at once, and record the machine of every session.

## What it does
- `auto`: the GPU if PyTorch sees one, else the CPU. Before anything long, a small float64 matrix product runs on the GPU, so a broken driver fails here and not hours later.
- Runs at once (`JOBS` auto): 1 on a GPU; on CPU the number of cores, at most 4. Each run uses one CPU thread.
- Writes `environment-<time>.json` in the Drive folder: Python, every installed package version, GPU name, CUDA, commit, settings.
- If a later session trains on a different device, it prints a warning. Accuracy does not depend on the device beyond rounding, but run times do, and the cost table lists the device per arm.

## In the code
- `pick_device` ([tools/main_run.py:160](../../../tools/main_run.py#L160)), `check_same_device` ([line 178](../../../tools/main_run.py#L178)), `environment` ([line 189](../../../tools/main_run.py#L189)).

## Ready?
Yes. The GPU path is read and reasoned, not run (there is no GPU on the development machine); the dry run of [step 68](68-dry-run.md) checks it on Colab.
