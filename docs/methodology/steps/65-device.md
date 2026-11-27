# Step 65 · Device

[Index](README.md) · page 6, the main run on Colab · group 18, main run · [← step 64](64-pins.md) · [step 66 →](66-tests.md)

**In one line:** choose GPU or CPU and how many runs at once, and record the machine of every session.

## What it does
- `auto`: the GPU if PyTorch sees one, else the CPU. Before anything long, a small float64 matrix product runs on the GPU, so a broken driver fails here and not hours later.
- Runs at once (`JOBS` auto): 2 on a GPU (1 if the machine has one CPU); on CPU the number of cores, at most 4. Each run uses one CPU thread and its own process. On a GPU it prints the GPU memory and warns if the runs may not fit (about 3 GB per run). Why 2 and not more: on a GPU each run still needs a CPU to send it its many small operations, and free Colab has 2 CPUs ([PARALLEL_RUNS_2026-11-28.md](../../reviews/PARALLEL_RUNS_2026-11-28.md)).
- Writes `environment-<time>.json` in the Drive folder: Python, every installed package version, GPU name, CUDA, commit, settings.
- If a later session trains on a different device, it prints a warning. Accuracy does not depend on the device beyond rounding, but run times do, and the cost table lists the device per arm.

## In the code
- `pick_device` ([tools/main_run.py:201](../../../tools/main_run.py#L201)), `check_same_device` ([line 229](../../../tools/main_run.py#L229)), `environment` ([line 257](../../../tools/main_run.py#L257)).

## Ready?
Yes. The GPU path is read and reasoned, not run (there is no GPU on the development machine); the dry run of [step 68](68-dry-run.md) checks it on Colab.
