"""Error estimate and acceptance rule for finished PINN runs, from the command line.

    uv run python tools/pinn_error_estimate.py runs/<run_id> [runs/<run_id> ...]
    uv run python tools/pinn_error_estimate.py --self-test

The work is done by bvp_pinn.pinn.estimate (no reference data; it builds engine/bvp_engine with
g++ if needed). New runs get this automatically at the end of bvp_pinn.pinn.run; this command is
for older run folders. Writes runs/<run_id>/error_estimate.json and selection.json.

--self-test checks the estimator on the reference itself plus a known smooth change, and prints
how much of that change it recovers. That uses the reference only to test the tool.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bvp_pinn.pinn.estimate import assess_run, engine_step  # noqa: E402


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
        folder = Path(arg)
        verdict = assess_run(folder)
        est = json.loads((folder / "error_estimate.json").read_text())
        score = "n/a" if est.get("score") is None else f"{est['score']:.3g}"
        print(folder.name, f"score {score}", f"accepted {verdict['accepted']}", "; ".join(verdict["reasons"]))


if __name__ == "__main__":
    main(sys.argv[1:])
