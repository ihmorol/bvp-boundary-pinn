# Step 61 · GPU and Drive

[Index](README.md) · page 6, the main run on Colab · group 17, notebook · [← step 60](60-settings.md) · [step 62 →](62-code.md)

**In one line:** check for a GPU, connect Google Drive and make the results folder.

## What it does
- Runs `nvidia-smi` only if it exists (CPU runtimes have none) and prints the GPU name, driver and memory.
- Stops with a clear message if `DEVICE = "cuda"` but there is no GPU.
- Mounts Google Drive at /content/drive. If the folder holds an earlier session, it prints the first session's commit, every finished run (these are never trained again) and the finished steps. If it holds none and `MUST_RESUME` is True, it stops: a missing shortcut must not start the campaign over. Otherwise it creates the folder.
- Notes the time the runtime started in /content/session_start.txt, for the session limit ([step 70](70-campaign.md)).

## Continuing in another Google account
The results are in the Drive of the account that started the run. Share the folder `bvp-boundary-pinn` with the second account as Editor; in the second account add it to My Drive as a shortcut (*Shared with me*, right-click, *Organize > Add shortcut*). Colab then sees it at the same path, so `DRIVE_FOLDER` stays the same. If the shortcut does not show up, download the folder (a zip file), unzip it, and upload the folder to the second account's My Drive (*New > Folder upload*) under the same name.

## In the code
- Cell "Step A" of [tools/colab_main_run.ipynb](../../../tools/colab_main_run.ipynb).

## Checked by
The notebook cells ran end to end in a simulated Colab on 2026-11-26 (fake Drive, no GPU). That run found and fixed a crash when `nvidia-smi` is missing.

## Ready?
Yes. Not yet run on a real Colab machine.
