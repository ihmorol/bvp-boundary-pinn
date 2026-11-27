"""Train one arm with one seed.

    uv run python -m bvp_pinn.pinn.run --arm B0 --seed 0
    uv run python -m bvp_pinn.pinn.run --arm B2 --seed 0 --dry-run
    uv run python -m bvp_pinn.pinn.run --arm B0 --set init_from=runs/<run_id> adam_steps=0 refine=bfgs

Writes runs/<run_id>/, then applies the acceptance rule (estimate.py): error_estimate.json
and selection.json. Evaluation against the reference is a separate step
(bvp_pinn.evaluation.evaluate), so nothing here can read the reference (gate G5).
"""
from __future__ import annotations

import argparse
import time

import torch

from ..config import ROOT, load_case
from . import model as nets
from .estimate import assess_run
from .points import make_points
from .runs import RUNS, RunFolder, parse_set, resolve
from .train import train


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, choices=["B0", "B1", "B2", "B3"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="tiny settings to check the pipeline")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--no-estimate", action="store_true", help="skip the error estimate and the acceptance rule")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override a config value, for example --set adam_steps=2000 replay_weight=10")
    args = ap.parse_args(argv)
    return run(args.arm, args.seed, args.dry_run, args.device, assess=not args.no_estimate, **parse_set(args.set))


PROGRESS_EVERY = {"adam": 2000, "lbfgs": 1000, "bfgs": 500, "lm": 20}


def progress(name):
    """A history-row callback that prints a short line now and then, so a run of hours shows it is alive."""
    def show(row):
        phase, step = row.get("phase"), row.get("step", 0)
        if phase in PROGRESS_EVERY and step % PROGRESS_EVERY[phase] == 0:
            print(f"  {name}: {phase} step {step}, loss {row['loss']:.3e}, {row.get('seconds', 0):.0f} s", flush=True)
    return show


def run(arm, seed, dry_run=False, device="cpu", runs_root=RUNS, assess=True, **overrides):
    """Train, save the run folder, then judge it. Returns the run folder path.

    A run always solves Eqs. 14 to 18 with B1 to B12 at the locked values: resolve() refuses any
    setting that names a physical value or another L, and train() checks the values."""
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
    folder = RunFolder(cfg, seed, runs_root)
    gpu = torch.cuda.get_device_name(torch.device(device)) if torch.device(device).type == "cuda" else None
    print(f"{folder.path.name}: {sum(x.numel() for x in model.parameters())} parameters, "
          f"{len(train_pts)} points, device {device}{f' ({gpu})' if gpu else ''}", flush=True)
    show = progress(folder.path.name)
    start = time.perf_counter()
    try:
        status = train(model, p, cfg, train_pts, monitor_pts, lambda row: (folder.log(row), show(row)), seed)
    except Exception as err:  # a crash is a result too: record it, so a resumed campaign does not repeat it
        folder.save_failure(f"error: {type(err).__name__}: {err}", time.perf_counter() - start,
                            {"seed": seed, "case_sha256": case_sha, "device": device, "gpu": gpu})
        raise
    status["wall_seconds"] = time.perf_counter() - start
    folder.save(model, status, {"train": train_pts, "monitor": monitor_pts},
                {"seed": seed, "case_sha256": case_sha, "device": device, "gpu": gpu})
    print(f"{status['result']} in {status['wall_seconds']:.1f} s; "
          f"refine ({status.get('refine')}) accepted: {status.get('refine_accepted')}; wrote {folder.path}", flush=True)
    if assess:
        torch.set_default_device("cpu")
        verdict = assess_run(folder.path)
        print(f"acceptance: {verdict['accepted']} {'; '.join(verdict['reasons'])}", flush=True)
    return folder.path


if __name__ == "__main__":
    main()
