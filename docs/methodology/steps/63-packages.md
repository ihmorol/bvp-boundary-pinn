# Step 63 · Packages

[Index](README.md) · page 6, the main run on Colab · group 17, notebook · [← step 62](62-code.md) · [step 64 →](64-pins.md)

**In one line:** install exactly the package versions the code was tested with, in a private Python environment.

## What it does
1. Installs uv 0.10.9, then creates /content/venv with Python 3.12.13 (uv downloads it if Colab's Python differs).
2. Exports the exact versions from `uv.lock` (`uv export --frozen`) and installs them: numpy 2.5.5, scipy 1.18.1, matplotlib 3.11.2, sympy 1.14.0, pytest 9.2.0 and their dependencies.
3. torch 2.15.0: on a GPU the PyPI build (with CUDA 13.2); on CPU the `+cpu` build of the lock file.
4. CUDA 13.2 needs a recent NVIDIA driver. If this torch cannot use the GPU, it installs the PyTorch build that Colab itself ships for that GPU and records it. If no build can use the GPU, it stops and says to set `DEVICE = "cpu"`.
5. Writes `installed.json`, so a rerun on the same machine skips all of this.
6. Runs every command of this environment, and later the main run, with Colab's `PYTHONPATH` and `MPLBACKEND` removed and `MPLBACKEND=Agg`. Colab sets them for its own Python 3.13 kernel; inherited, its inline matplotlib backend stops `import matplotlib` here (seen on Colab on 2026-11-27), and its `PYTHONPATH` could pull in Colab's own packages. The output of each command is shown in the notebook, because Colab hides what subprocesses print.

## Words used here
| Term | Meaning |
|---|---|
| CUDA 13.2 | The GPU software version the torch build was made for; it needs a driver at least as new |

## In the code
- Cell "Step C" of [tools/colab_main_run.ipynb](../../../tools/colab_main_run.ipynb).

## Checked by
The CPU path was installed and checked here; the GPU build was checked to resolve against the same versions (dry run of the installer).

## Ready?
Yes. The GPU install has not run on a real Colab machine yet.
