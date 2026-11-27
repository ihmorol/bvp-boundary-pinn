# Step 62 · Code

[Index](README.md) · page 6, the main run on Colab · group 17, notebook · [← step 61](61-gpu-and-drive.md) · [step 63 →](63-packages.md)

**In one line:** get the code (the branch tip, or `REPO_COMMIT`); the main run then checks that its training code is the same as in the first session of this Drive folder.

## What it does
- Clones the repository (once per machine) and fetches the branch.
- Picks the commit: `REPO_COMMIT` if set, otherwise the branch tip. Until 2026-11-28 it took the commit of the first session; now a newer commit may be used if it changes only scheduling, evaluation, tools, tests or docs, which [step 64](64-pins.md) checks.
- Checks it out and removes stray untracked files from the code folders.
- If the repository is private, it uses a GitHub token from Colab's Secrets (`GITHUB_TOKEN`), hides it in every printed line and does not keep it in the clone's settings. The repository is public today, so no token is needed.

## In the code
- Cell "Step B" of [tools/colab_main_run.ipynb](../../../tools/colab_main_run.ipynb).

## Ready?
Yes. The code is on GitHub (commit 64c2e48, pushed 2026-11-27).
