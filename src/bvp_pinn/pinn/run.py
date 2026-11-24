"""Train one arm with one seed.

    uv run python -m bvp_pinn.pinn.run --arm B0 --seed 0
    uv run python -m bvp_pinn.pinn.run --arm B2 --seed 0 --dry-run
    uv run python -m bvp_pinn.pinn.run --arm B0 --set init_from=runs/<run_id> adam_steps=0 refine=bfgs

Writes runs/<run_id>/. Evaluation against the reference is a separate step
(bvp_pinn.pinn.evaluate), so nothing here can read the reference (gate G5).
"""
from __future__ import annotations

import argparse
import time

import torch

from ..config import ROOT, load_case
from . import model as nets
from .points import make_points
from .runs import RunFolder, resolve
from .train import train


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, choices=["B0", "B1", "B2", "B3"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="tiny settings to check the pipeline")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override a config value, for example --set adam_steps=2000 replay_weight=10")
    args = ap.parse_args(argv)
    overrides = {k: _number(v) for k, v in (s.split("=", 1) for s in args.set)}
    return run(args.arm, args.seed, args.dry_run, args.device, **overrides)


def run(arm, seed, dry_run=False, device="cpu", **overrides):
    cfg = resolve(arm, dry_run=dry_run, **overrides)
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(device)
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(seed)
    p, case_sha = load_case()
    model = nets.build(cfg, p)
    if cfg.get("init_from"):  # continue from a finished run of the same network and problem
        model.load_state_dict(torch.load(ROOT / cfg["init_from"] / "model.pt", map_location=device))
    train_pts, monitor_pts = make_points(cfg, seed)
    folder = RunFolder(cfg, seed)
    print(f"{folder.path.name}: {sum(x.numel() for x in model.parameters())} parameters, "
          f"{len(train_pts)} points, device {device}")
    start = time.perf_counter()
    status = train(model, p, cfg, train_pts, monitor_pts, folder.log, seed)
    status["wall_seconds"] = time.perf_counter() - start
    folder.save(model, status, {"train": train_pts, "monitor": monitor_pts},
                {"seed": seed, "case_sha256": case_sha, "device": device})
    print(f"{status['result']} in {status['wall_seconds']:.1f} s; "
          f"refine ({status.get('refine')}) accepted: {status.get('refine_accepted')}; wrote {folder.path}")
    return folder.path


def _number(text):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


if __name__ == "__main__":
    main()
