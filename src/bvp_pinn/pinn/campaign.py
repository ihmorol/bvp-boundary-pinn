"""Run the campaign: every arm and seed of the "campaign" block in configs/pinn.json (TODO phase P4).

    uv run python -m bvp_pinn.pinn.campaign                        # every run, one at a time
    uv run python -m bvp_pinn.pinn.campaign --jobs 2               # two runs at once, one thread each
    uv run python -m bvp_pinn.pinn.campaign --arms B1 --seeds 2 3  # a part, for example one Colab session
    uv run python -m bvp_pinn.pinn.campaign --list                 # what is done and what is left
    uv run python -m bvp_pinn.pinn.campaign --dry-run --jobs 4     # the whole loop with tiny settings
    uv run python -m bvp_pinn.pinn.campaign --jobs 2 --deadline 1764300000 --expect B1=8000 B2=2600

Every run gets campaign=<name> in its config, so evaluation reads campaign runs only, never the
pilots. A run whose folder already has meta.json is skipped, so a campaign split over several
sessions resumes where it stopped. A failed run is a result too: it is kept, not re-run.
A run that finished training but lost its verdict (the session stopped during the acceptance
rule) is judged on the next start; it is never trained again.

How runs are scheduled (this changes no training setting; a run trains the same way alone or with
others, in its own process with one thread and its own seed):

- Each run trains in a new process (python -m bvp_pinn.pinn.campaign --worker), so runs at once
  share no memory or state, and a finished run frees its GPU memory.
- Runs start in the order of --arms and --seeds while fewer than --jobs are running.
- A run that runs out of GPU memory while other runs share the GPU has no result: the machine was
  full, the method did not fail. Its failure record is removed, needs_alone.json is written, and it
  trains again from its start with no other run on the GPU. Out of memory with no other run is a
  result like any other failure.
- With --deadline (a Unix time), a run starts only if its expected time ends before the deadline,
  so a Colab session does not spend its last hours on runs it cannot finish. --expect gives the
  time of one run of an arm alone, in seconds; runs that finish in this session replace it.
- schedule.json in each run folder records how many runs shared the machine while it trained, so
  run times measured with others at once are never compared with times measured alone unmarked.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from ..config import ROOT
from .estimate import assess_run, ensure_engine, is_judged
from .run import run
from .runs import RUNS, is_finished, parse_set, read_config, resolve, run_id, write_json

# Until a run with others at once finishes in this session, it is assumed to take this many times
# its time alone. Runs that finish in the session replace the guess with what they measured.
SLOWDOWN_GUESS = 1.6
ALONE = "needs_alone.json"    # written when a run ran out of GPU memory while sharing the GPU
SCHEDULE = "schedule.json"    # runs at once, device and times of a finished run
OUT_OF_MEMORY_EXIT = 3        # worker exit code: no result, train again alone


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


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


def unjudged(plan):
    """Finished runs with a model but no verdict yet."""
    return [folder for _, _, folder, done in plan if done and (folder / "model.pt").exists() and not is_judged(folder)]


def out_of_memory(text):
    """True for the errors PyTorch, cuBLAS and cuSOLVER raise when the GPU memory is full."""
    t = text.lower()
    return "out of memory" in t or "outofmemoryerror" in t or "alloc_failed" in t


def worker(spec):
    """One run, in its own process. Exit code 0: finished (accepted or not), 1: failed (recorded in
    meta.json, a result), OUT_OF_MEMORY_EXIT: out of GPU memory while sharing it (no result)."""
    s = json.loads(spec)
    try:
        run(s["arm"], s["seed"], dry_run=s["dry_run"], device=s["device"], campaign=campaign_name(s["dry_run"]),
            **s["overrides"])
        return 0
    except Exception as err:
        text = f"{type(err).__name__}: {err}"
        if s["shared"] and out_of_memory(text):
            forget_out_of_memory(Path(s["folder"]), text)
            print(f"{Path(s['folder']).name}: out of GPU memory with other runs at once. This is no result; "
                  "it trains again from its start, alone.", flush=True)
            return OUT_OF_MEMORY_EXIT
        traceback.print_exc()
        print(f"{Path(s['folder']).name}: failed: {text}", flush=True)
        return 1


def forget_out_of_memory(folder, text):
    """Remove the failure record run() wrote, so the run counts as not done, and mark it to run alone."""
    (folder / "meta.json").unlink(missing_ok=True)
    write_json(folder / ALONE, {"time": now(), "error": text[:1000]})


class Job:
    """A run in a child process."""

    def __init__(self, item, shared, dry_run, device, overrides):
        arm, seed, folder, _ = item
        spec = json.dumps({"arm": arm, "seed": seed, "folder": str(folder), "dry_run": dry_run, "device": device,
                           "overrides": overrides, "shared": shared})
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")]))}
        self.proc = subprocess.Popen([sys.executable, "-m", "bvp_pinn.pinn.campaign", "--worker", spec], env=env)

    def poll(self):
        """None while running, else the exit code."""
        return self.proc.poll()


def expected(arm, prior, measured, slowdown, shared):
    """Expected seconds of one run of this arm: what runs of the arm took in this session under the same
    sharing, else its time alone (prior) times the slowdown when shared. None if nothing is known."""
    same = measured.get((arm, shared))
    if same:
        return statistics.median(same)
    if arm not in prior:
        return None
    return prior[arm] * (slowdown if shared else 1.0)


def schedule(todo, jobs, submit, deadline=None, prior=None, device=None, clock=time.time, sleep=time.sleep,
             poll_seconds=5.0, say=print):
    """Start the runs of todo, in order, while fewer than `jobs` run. A run marked alone (its last try ran
    out of memory next to others) starts only when nothing else runs, and nothing starts beside it.
    With a deadline, a run starts only if it is expected to end before it. Returns the items not started.

    todo: (arm, seed, run folder, alone) tuples. submit(item, shared) starts one and returns an object
    whose poll() is None while it runs and its exit code after."""
    prior, queue, running = dict(prior or {}), list(todo), []
    measured, ratios = {}, []
    while queue or running:
        while len(running) < jobs and not any(r["alone"] for r in running):
            slowdown = statistics.median(ratios) if ratios else SLOWDOWN_GUESS
            pick = None
            for item in queue:
                alone = item[3] or jobs == 1
                if item[3] and running:
                    continue
                need = expected(item[0], prior, measured, slowdown, shared=not alone)
                if deadline is None or need is None or clock() + need <= deadline:
                    pick, shared = item, not alone
                    break
            if pick is None:
                break
            queue.remove(pick)
            running.append({"item": pick, "job": submit(pick, shared), "shared": shared, "alone": pick[3],
                            "started": clock(), "start_text": now(), "peak": 1})
            for r in running:
                r["peak"] = max(r["peak"], len(running))
            note = " alone, after running out of memory beside others" if pick[3] else ""
            say(f"{pick[0]} seed {pick[1]}: started{note} ({len(running)} running)")
        if not running:
            break  # nothing left fits before the deadline
        sleep(poll_seconds)
        for r in list(running):
            code = r["job"].poll()
            if code is None:
                continue
            running.remove(r)
            arm, seed, folder, _ = r["item"]
            seconds = clock() - r["started"]
            if code == OUT_OF_MEMORY_EXIT or (code < 0 and r["shared"]):
                # no result (out of memory, or the process was killed while sharing the machine): again, alone
                if code < 0:
                    forget_out_of_memory(Path(folder), f"process ended by signal {-code}")
                queue.append((arm, seed, folder, True))
                say(f"{arm} seed {seed}: no result (exit {code}); it will train again alone")
                continue
            if code < 0:
                say(f"{arm} seed {seed}: stopped by signal {-code} without a result; the next start trains it again")
                continue
            if Path(folder).exists():
                write_json(Path(folder) / SCHEDULE, {"jobs": jobs, "max_runs_at_once": r["peak"], "device": device,
                                                     "started": r["start_text"], "finished": now(),
                                                     "seconds": round(seconds, 1), "alone_after_out_of_memory": r["alone"]})
            if code == 0:
                key = (arm, r["peak"] > 1)
                measured.setdefault(key, []).append(seconds)
                if r["peak"] > 1 and prior.get(arm):
                    ratios.append(seconds / prior[arm])
            verdict = Path(folder) / "selection.json"
            accepted = f", accepted {json.loads(verdict.read_text())['accepted']}" if verdict.exists() else ""
            say(f"{arm} seed {seed}: {'done' if code == 0 else 'failed (recorded)'} in {seconds / 3600:.2f} h{accepted} "
                f"({Path(folder).name}; at most {r['peak']} runs at once)")
    return queue


def main(argv=None):
    if argv is None and len(sys.argv) == 3 and sys.argv[1] == "--worker":
        raise SystemExit(worker(sys.argv[2]))
    block = read_config()["campaign"]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", nargs="*", default=block["arms"])
    ap.add_argument("--seeds", nargs="*", type=int, default=block["seeds"])
    ap.add_argument("--jobs", type=int, default=1, help="runs at once, each in its own process")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true", help="only show what is done and what is left")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="only with --dry-run")
    ap.add_argument("--deadline", type=float, default=None, help="Unix time: start no run expected to end after it")
    ap.add_argument("--expect", nargs="*", default=[], metavar="ARM=SECONDS", help="time of one run alone, per arm")
    args = ap.parse_args(argv)
    overrides = parse_set(args.set)
    if overrides and not args.dry_run:
        raise SystemExit("--set changes the run settings, so those runs would not be campaign runs; "
                         "use bvp_pinn.pinn.run for pilots, or --set together with --dry-run")
    plan = planned(args.arms, args.seeds, args.dry_run, **overrides)
    todo = [(arm, seed, folder, (folder / ALONE).exists()) for arm, seed, folder, done in plan if not done]
    for arm, seed, folder, done in plan:
        print(f"{arm} seed {seed}: {'finished' if done else 'to run'} ({folder.name})")
    if args.list:
        return plan
    late = unjudged(plan)
    if late or todo:
        ensure_engine()  # the acceptance rule needs it after every run
    for folder in late:
        verdict = assess_run(folder)
        print(f"{folder.name}: judged late, accepted {verdict['accepted']} {'; '.join(verdict['reasons'])}", flush=True)
    if not todo:
        return plan
    prior = {k: float(v) for k, v in (e.split("=", 1) for e in args.expect)}
    limit = f", none starting after it is expected to end past {datetime.datetime.fromtimestamp(args.deadline):%H:%M}" \
        if args.deadline else ""
    print(f"campaign {campaign_name(args.dry_run)}: {len(todo)} runs, {args.jobs} at a time{limit}", flush=True)
    submit = lambda item, shared: Job(item, shared, args.dry_run, args.device, overrides)
    left = schedule(todo, args.jobs, submit, deadline=args.deadline, prior=prior, device=args.device,
                    say=lambda text: print(text, flush=True))
    if left:
        print(f"Not started, expected to end after the session limit: "
              f"{', '.join(f'{a} seed {s}' for a, s, *_ in left)}. They run in the next session.", flush=True)
    waiting = {(a, s) for a, s, *_ in left}
    broken = [f"{a} seed {s}" for a, s, folder, _ in todo if (a, s) not in waiting and not is_finished(folder)]
    if broken:
        raise SystemExit(f"These runs ended without a record (see their output above): {', '.join(broken)}")
    return plan


if __name__ == "__main__":
    main()
