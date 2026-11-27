# Step 61 · GPU and Drive

[Index](README.md) · page 6, the main run on Colab · group 17, notebook · [← step 60](60-settings.md) · [step 62 →](62-code.md)

**In one line:** check for a GPU, connect Google Drive and make the results folder.

## What it does
- Runs `nvidia-smi` only if it exists (CPU runtimes have none) and prints the GPU name, driver and memory.
- Stops with a clear message if `DEVICE = "cuda"` but there is no GPU.
- Mounts Google Drive at /content/drive and creates the results folder under My Drive.

## In the code
- Cell "Step A" of [tools/colab_main_run.ipynb](../../../tools/colab_main_run.ipynb).

## Checked by
The notebook cells ran end to end in a simulated Colab on 2026-11-26 (fake Drive, no GPU). That run found and fixed a crash when `nvidia-smi` is missing.

## Ready?
Yes. Not yet run on a real Colab machine.
