"""Error estimate without the reference, and the acceptance rule (docs/pinn/PINN_PLAN.md section 3).

Estimate: the PINN states on a wall-clustered mesh go to `engine/bvp_engine --estimate`, which
takes one Newton step of the box scheme from that state. For a state near the solution the step
is the error (PINN minus solution) up to second-order terms and the box scheme's own error
(about 1e-7). The engine reads only the locked case file, never a reference solution.

Score: the largest estimated relative error of f''(0), theta(0), theta'(0) and phi'(0), relative
to the PINN's own values. On runs E3 to E8 it ranks the runs as the true error does
(docs/pinn/IMPLEMENTATION_NOTES.md section 7.2).

Rule (configs/pinn.json, "selection"; declared 2026-11-24, before the campaign). A run is
accepted if all of these hold:
  - training finished (no non-finite loss)
  - the far field decays: m > 0 and mp > 0 (selfcheck.py)
  - tail_share <= tail_share_max: theta has no constant part on [L/2, L]
  - every boundary error <= boundary_max
  - score <= score_max
Acceptance never removes a run. Rejected runs stay in every table and get a line in failures.md.
One run per arm and seed; a seed is never re-run to get a better result.
"""
from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from ..config import ROOT, load_case
from ..reference.solver import wall_mesh
from .model import predict
from .runs import CONFIG_PATH, load_run, read_config

ENGINE_SOURCE = ROOT / "engine" / "bvp_engine.cpp"
ENGINE = ROOT / "engine" / "bvp_engine"
WALL = {"f2_0": 2, "theta_0": 6, "th1_0": 7, "phi_0": 8, "ph1_0": 9, "g1_0": 5, "ch1_0": 11}
SCORE_KEYS = ("f2_0", "theta_0", "th1_0", "ph1_0")
FIELDS = {"f": 0, "g": 4, "theta": 6, "phi": 8, "chi": 10}


def ensure_engine():
    """Build engine/bvp_engine with g++ if it is missing or older than its source."""
    if ENGINE.exists() and ENGINE.stat().st_mtime >= ENGINE_SOURCE.stat().st_mtime:
        return ENGINE
    build = subprocess.run(["g++", "-O2", "-std=c++17", "-o", str(ENGINE), str(ENGINE_SOURCE)],
                           capture_output=True, text=True)
    if build.returncode:
        raise RuntimeError(f"engine build failed: {build.stderr.strip()}")
    return ENGINE


def engine_step(eta, z):
    """One box-scheme Newton step from the 12 states z (12, N) on eta (N,). Returns (step, engine log).

    The engine reads configs/physical-case-pending.json itself; load_case() first checks that the
    file is the locked one."""
    load_case()
    ensure_engine()
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "in.csv", Path(tmp) / "out.csv"
        np.savetxt(src, np.column_stack([eta, z.T]), delimiter=",", header="eta,12 states", comments="", fmt="%.17g")
        run = subprocess.run([str(ENGINE), "--estimate", str(src), str(out)], cwd=ROOT, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(run.stderr.strip())
        with open(out) as fh:
            rows = list(csv.reader(fh))[1:]
    return np.array(rows, dtype=float)[:, 1:].T, run.stdout.strip()


def estimate(model, L, n=3201):
    """Estimated errors of a trained model and its score. Lower score is better."""
    eta = wall_mesh(L, n, 2e-3)
    with torch.no_grad():
        z = predict(model, torch.tensor(eta).reshape(-1, 1))[0].numpy()
    step, log = engine_step(eta, z)
    score = max(abs(step[WALL[k], 0]) / max(abs(z[WALL[k], 0]), 1e-12) for k in SCORE_KEYS)
    return {"nodes": n, "engine": log, "score": float(score),
            "wall_estimated_error": {k: float(step[i, 0]) for k, i in WALL.items()},
            "max_estimated_error": {k: float(np.abs(step[i]).max()) for k, i in FIELDS.items()}}


def judge(meta, est, rule):
    """Apply the acceptance rule. accepted is None when the estimate could not be made."""
    reasons = []
    if meta.get("result") != "finished":
        reasons.append(f"result: {meta.get('result')}")
    check = meta.get("self_check") or {}
    if check and not check.get("decaying_far_field"):
        reasons.append(f"far field does not decay (m {check.get('m'):.3g}, mp {check.get('mp'):.3g})")
    share = check.get("tail_share")
    if share is not None and share > rule["tail_share_max"]:
        reasons.append(f"tail share {share:.3g} > {rule['tail_share_max']}")
    bmax = check.get("max_boundary_abs")
    if bmax is not None and bmax > rule["boundary_max"]:
        reasons.append(f"boundary error {bmax:.3g} > {rule['boundary_max']}")
    if est.get("score") is None:
        reasons.append(f"no estimate: {est.get('error')}")
    elif est["score"] > rule["score_max"]:
        reasons.append(f"estimate score {est['score']:.3g} > {rule['score_max']}")
    if est.get("score") is None and len(reasons) == 1:
        accepted = None  # only the estimate is missing: undecided, not rejected
    else:
        accepted = not reasons
    return {"accepted": accepted, "reasons": reasons, "rule": rule}


def assess_run(folder, config_path=CONFIG_PATH):
    """Estimate and judge one finished run folder. Writes error_estimate.json and selection.json."""
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    rule = read_config(config_path)["selection"]
    try:
        cfg, model = load_run(folder)
        est = estimate(model, cfg["L"])
    except (RuntimeError, ValueError, OSError) as err:
        est = {"score": None, "error": str(err)}
    (folder / "error_estimate.json").write_text(json.dumps(est, indent=2) + "\n")
    verdict = judge(meta, est, rule)
    (folder / "selection.json").write_text(json.dumps(verdict, indent=2) + "\n")
    return verdict
