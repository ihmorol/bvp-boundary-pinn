"""Configs and run folders (docs/CODING_RULES.md section 7)."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess

import numpy as np
import torch

from ..config import ROOT

CONFIG_PATH = ROOT / "configs" / "pinn.json"
RUNS = ROOT / "runs"

# A dry run checks the whole pipeline in about a minute. Its results mean nothing.
DRY_RUN = {"adam_steps": 40, "lbfgs_iters": 5, "n_points": 120, "n_monitor": 60, "log_every": 10,
           "width": 16, "depth": 2}


def resolve(arm, config_path=CONFIG_PATH, dry_run=False, **overrides):
    """Shared settings, then the arm's settings, then any overrides."""
    raw = json.loads(config_path.read_text())
    cfg = {**raw["common"], **raw["arms"][arm], "arm": arm, **overrides}
    if dry_run:
        cfg.update(DRY_RUN, dry_run=True)
    return cfg


def run_id(cfg, seed):
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    return f"bvp-{cfg['arm']}-s{seed}-{digest}"


def git_commit():
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    return {"commit": out.stdout.strip(), "dirty": bool(dirty.stdout.strip())}


class RunFolder:
    """runs/<run_id>/ with config.json, history.csv, meta.json and the model checkpoints."""

    def __init__(self, cfg, seed, root=RUNS):
        self.path = root / run_id(cfg, seed)
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "config.json").write_text(json.dumps({**cfg, "seed": seed}, indent=2) + "\n")
        self.rows = []

    def log(self, row):
        self.rows.append(row)

    def save(self, model, status, points, meta):
        torch.save(model.state_dict(), self.path / "model.pt")
        for k, state in enumerate(status.pop("stage_states")):
            torch.save(state, self.path / f"stage{k}.pt")
        np.savez(self.path / "points.npz", **{k: v.cpu().numpy() for k, v in points.items()})
        keys = list(dict.fromkeys(k for r in self.rows for k in r))
        with open(self.path / "history.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, keys)
            writer.writeheader()
            writer.writerows(self.rows)
        meta = {**meta, **status, "git": git_commit(), "python": platform.python_version(),
                "torch": torch.__version__, "numpy": np.__version__, "machine": platform.machine(),
                "processor": platform.processor(), "threads": torch.get_num_threads()}
        (self.path / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
