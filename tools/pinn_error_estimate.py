"""Reference-free error estimate for a trained PINN (gap R6), with the C++ engine.

    g++ -O2 -std=c++17 -o engine/bvp_engine engine/bvp_engine.cpp
    uv run python tools/pinn_error_estimate.py runs/<run_id> [runs/<run_id> ...]
    uv run python tools/pinn_error_estimate.py --self-test

For each run: the PINN states on a wall-clustered mesh go to `engine/bvp_engine --estimate`,
which takes one Newton step of the box scheme from that state. The step is the estimated
error (PINN minus solution). Writes runs/<run_id>/error_estimate.json, with a selection score:
the largest estimated relative error of f''(0), theta(0), theta'(0) and phi'(0). It reads no
reference solution, so it can be reported for any run, and runs are chosen by this score, not
by their loss (a small loss can hide a wrong tail, run E8). evaluate.py compares it with the
true error. On E3 to E8 its ranking equals the ranking by true error.

--self-test checks the estimator on the reference itself plus a known smooth change, and
prints how much of that change it recovers. That uses the reference only to test the tool.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bvp_pinn.reference.solver import wall_mesh  # noqa: E402

ENGINE = ROOT / "engine" / "bvp_engine"
WALL = {"f2_0": 2, "theta_0": 6, "th1_0": 7, "phi_0": 8, "ph1_0": 9, "g1_0": 5, "ch1_0": 11}


def engine_step(eta, z):
    """One Newton step from the 12 states z (12, N) on eta (N,). Returns the (12, N) step."""
    if not ENGINE.exists():
        raise SystemExit("build the engine first: g++ -O2 -std=c++17 -o engine/bvp_engine engine/bvp_engine.cpp")
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "in.csv", Path(tmp) / "out.csv"
        np.savetxt(src, np.column_stack([eta, z.T]), delimiter=",", header="eta,12 states", comments="", fmt="%.17g")
        run = subprocess.run([str(ENGINE), "--estimate", str(src), str(out)], cwd=ROOT, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(run.stderr)
        rows = list(csv.reader(open(out)))[1:]
    return np.array(rows, dtype=float)[:, 1:].T, run.stdout.strip()


def estimate_run(folder, n=3201):
    from bvp_pinn.pinn import model as nets
    from bvp_pinn.pinn.evaluate import load_run
    torch.set_default_dtype(torch.float64)
    cfg, model = load_run(folder)
    eta = wall_mesh(cfg["L"], n, 2e-3)
    z = nets.predict(model, torch.tensor(eta).reshape(-1, 1))[0].numpy()
    step, log = engine_step(eta, z)
    # Selection score: the largest estimated relative error of the reported wall values,
    # relative to the PINN's own values. Lower is better. Used to pick runs after training
    # (change P-C); it never enters a gradient.
    score = max(abs(step[i, 0]) / max(abs(z[i, 0]), 1e-12) for i in (2, 6, 7, 9))
    result = {"nodes": n, "engine": log, "score": float(score),
              "wall_estimated_error": {k: float(step[i, 0]) for k, i in WALL.items()},
              "max_estimated_error": {k: float(np.abs(step[i]).max()) for k, i in
                                      {"f": 0, "g": 4, "theta": 6, "phi": 8, "chi": 10}.items()}}
    (folder / "error_estimate.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def self_test():
    from bvp_pinn.reference.io import load_reference
    ref = load_reference(ROOT / "reference" / "locked-branchA-L20-tol1e-08")[0]
    eta, z = ref["eta"], ref["z"]
    base, _ = engine_step(eta, z)
    bump = 1e-3 * np.exp(-eta)  # a smooth change of theta and its true slope
    zp = z.copy()
    zp[6] += bump
    zp[7] += -bump
    step, _ = engine_step(eta, zp)
    print(f"on the reference: max |estimate| {np.abs(base).max():.2e} (box-scheme error level)")
    print(f"with a 1e-3 theta change: estimated theta(0) error {step[6, 0]:.3e}, "
          f"recovered share of the change at the wall {step[6, 0] / 1e-3:.3f}")


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    for arg in argv:
        r = estimate_run(Path(arg))
        print(Path(arg).name, f"score {r['score']:.3f}", json.dumps(r["wall_estimated_error"]))


if __name__ == "__main__":
    main(sys.argv[1:])
