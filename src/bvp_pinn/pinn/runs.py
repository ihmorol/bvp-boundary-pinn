"""Configs and run folders (docs/team/CODING_RULES.md section 7).

configs/pinn.json has four blocks:
  common     training settings shared by every arm
  arms       what each arm changes (B0 to B3)
  selection  the acceptance rule applied after training (estimate.py); not part of a run's config
  campaign   the arms and seeds of the final campaign (campaign.py)
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from ..config import PARAMETER_NAMES, ROOT, load_case, load_reference_settings
from .model import build

CONFIG_PATH = ROOT / "configs" / "pinn.json"
RUNS = ROOT / "runs"

# A dry run checks the whole pipeline in under a minute per arm. Its results mean nothing.
DRY_RUN = {"adam_steps": 40, "lbfgs_iters": 5, "lm_iters": 3, "n_points": 120, "n_monitor": 60,
           "log_every": 10, "width": 16, "depth": 2}

# Keys a run config may hold besides those in configs/pinn.json (common and arm blocks).
RUN_KEYS = {"arm", "dry_run", "campaign", "init_from"}
REFINE_METHODS = ("lbfgs", "bfgs", "lm")
SCALES = (None, "none", "log", "balanced")
STRUCTURES = ("single", "windows", "staged")


def read_config(config_path=CONFIG_PATH):
    return json.loads(Path(config_path).read_text())


def resolve(arm, config_path=CONFIG_PATH, dry_run=False, **overrides):
    """Shared settings, then the arm's settings, then any overrides. Checked before use."""
    raw = read_config(config_path)
    cfg = {**raw["common"], **raw["arms"][arm], "arm": arm, **overrides}
    if dry_run:
        cfg.update(DRY_RUN, dry_run=True)
    validate(cfg, raw)
    return cfg


def validate(cfg, raw=None):
    """Fail before training, not hours into it, on settings the code cannot run or must not change.

    The physical values are not training settings: a key that names one is refused, and so is
    any key configs/pinn.json does not define (a typo would otherwise be ignored). L must be the
    reference's L, so the PINN and the reference solve the same problem."""
    raw = raw or read_config()
    physical = sorted(set(cfg) & set(PARAMETER_NAMES))
    if physical:
        raise ValueError(f"{physical} are locked supervisor values, not settings; they cannot be set (AGENTS.md section 3)")
    known = set(raw["common"]).union(*(set(a) for a in raw["arms"].values()), RUN_KEYS)
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise ValueError(f"unknown settings {unknown}; configs/pinn.json does not define them")
    if cfg["L"] != load_reference_settings()["L_reference"]:
        raise ValueError(f"L = {cfg['L']} differs from the reference L; the PINN must solve the same problem")
    edges = cfg["windows"]
    if edges[0] != 0.0 or edges[-1] != cfg["L"] or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f"windows {edges} must rise from 0 to L = {cfg['L']}")
    if cfg["structure"] not in STRUCTURES:
        raise ValueError(f"unknown structure {cfg['structure']!r}")
    if cfg["bc_mode"] not in ("soft", "linear", "exact"):
        raise ValueError(f"unknown bc_mode {cfg['bc_mode']!r}")
    if cfg.get("residual_scale") not in SCALES:
        raise ValueError(f"unknown residual_scale {cfg.get('residual_scale')!r}")
    for item in cfg.get("refine", "lbfgs").split("+"):
        method, _, scale = item.partition(":")
        if method not in REFINE_METHODS or (scale and scale not in SCALES):
            raise ValueError(f"unknown refine stage {item!r}")
    if cfg["structure"] == "staged" and cfg.get("replay") not in ("physics", "self"):
        raise ValueError("staged arms need replay = physics or self")


def parse_set(items):
    """KEY=VALUE strings from the command line, numbers turned into numbers."""
    return {k: _number(v) for k, v in (s.split("=", 1) for s in items)}


def _number(text):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


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
        self.path = Path(root) / run_id(cfg, seed)
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
                "processor": platform.processor(), "threads": torch.get_num_threads(),
                "parameters": sum(x.numel() for x in model.parameters())}
        (self.path / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


def is_finished(folder):
    """A run folder is finished once meta.json exists, whatever its result."""
    return (Path(folder) / "meta.json").exists()


def load_run(folder, checkpoint="model.pt"):
    """The config and the trained networks of a run folder, on the CPU.

    Sets float64 as the default dtype first: the constraint layer builds its fixed tensors in
    the default dtype, and every run is trained in float64 (CODING_RULES section 5)."""
    torch.set_default_dtype(torch.float64)
    folder = Path(folder)
    cfg = json.loads((folder / "config.json").read_text())
    model = build(cfg, load_case()[0])
    model.load_state_dict(torch.load(folder / checkpoint, map_location="cpu"))
    return cfg, model
