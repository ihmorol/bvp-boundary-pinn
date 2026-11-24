"""Run the campaign: every arm and seed of the "campaign" block in configs/pinn.json (TODO phase P4).

    uv run python -m bvp_pinn.pinn.campaign                        # every run, one at a time
    uv run python -m bvp_pinn.pinn.campaign --jobs 3               # three runs at once, one thread each
    uv run python -m bvp_pinn.pinn.campaign --arms B1 --seeds 2 3  # a part, for example one Colab session
    uv run python -m bvp_pinn.pinn.campaign --list                 # what is done and what is left
    uv run python -m bvp_pinn.pinn.campaign --dry-run --jobs 4     # the whole loop with tiny settings

Every run gets campaign=<name> in its config, so evaluation reads campaign runs only, never the
pilots. A run whose folder already has meta.json is skipped, so a campaign split over several
sessions resumes where it stopped. A failed run is a result too: it is kept, not re-run.
"""
from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from .estimate import ensure_engine
from .run import run
from .runs import RUNS, is_finished, parse_set, read_config, resolve, run_id


def campaign_name(dry_run=False):
    name = read_config()["campaign"]["name"]
    return f"{name}-dry" if dry_run else name


def planned(arms, seeds, dry_run=False, runs_root=RUNS, **overrides):
    """(arm, seed, run folder, finished) for every run of the campaign."""
    tag = campaign_name(dry_run)
    out = []
    for arm in arms:
        for seed in seeds:
            cfg = resolve(arm, dry_run=dry_run, campaign=tag, **overrides)
            folder = runs_root / run_id(cfg, seed)
            out.append((arm, seed, folder, is_finished(folder)))
    return out


def _one(arm, seed, dry_run, device, overrides):
    """One run in a worker process. Returns (arm, seed, message)."""
    try:
        path = run(arm, seed, dry_run=dry_run, device=device, campaign=campaign_name(dry_run), **overrides)
        return arm, seed, f"done: {path.name}"
    except Exception as err:  # the other runs go on; the missing folder shows up in failures.md
        return arm, seed, f"failed: {type(err).__name__}: {err}"


def main(argv=None):
    block = read_config()["campaign"]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", nargs="*", default=block["arms"])
    ap.add_argument("--seeds", nargs="*", type=int, default=block["seeds"])
    ap.add_argument("--jobs", type=int, default=1, help="runs at once, each with its own process")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true", help="only show what is done and what is left")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="pilot use only, never in the campaign")
    args = ap.parse_args(argv)
    overrides = parse_set(args.set)
    plan = planned(args.arms, args.seeds, args.dry_run, **overrides)
    todo = [(arm, seed) for arm, seed, _, done in plan if not done]
    for arm, seed, folder, done in plan:
        print(f"{arm} seed {seed}: {'finished' if done else 'to run'} ({folder.name})")
    if args.list or not todo:
        return plan
    ensure_engine()  # the acceptance rule needs it after every run
    print(f"campaign {campaign_name(args.dry_run)}: {len(todo)} runs, {args.jobs} at a time", flush=True)
    if args.jobs == 1:
        for arm, seed in todo:
            print(_one(arm, seed, args.dry_run, args.device, overrides)[2], flush=True)
        return plan
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
        jobs = [pool.submit(_one, arm, seed, args.dry_run, args.device, overrides) for arm, seed in todo]
        for job in as_completed(jobs):
            arm, seed, message = job.result()
            print(f"{arm} seed {seed}: {message}", flush=True)
    return plan


if __name__ == "__main__":
    main()
