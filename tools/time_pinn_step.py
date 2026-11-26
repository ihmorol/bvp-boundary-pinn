"""Time one PINN training step, so speed claims can be checked on any machine.

    uv run python tools/time_pinn_step.py                 # B0, configs/pinn.json, 1 thread
    uv run python tools/time_pinn_step.py --threads 4 --points 1000 --mode soft
    uv run python tools/time_pinn_step.py --profile       # torch.profiler table of the slowest ops
    uv run python tools/time_pinn_step.py --arm B1 --device cuda --jets-only --lm --json t.json

Prints the median time of: the equation loss forward, the boundary errors, and a full step
(loss and backward), for the forward-jet engine and the old nested-autograd engine. With --lm it
also times one Levenberg-Marquardt step on the balanced scale (the last refine stage), after one
warm-up step. The first calls are warm-up and not counted. Close other heavy programs first: the
timings are noisy. --json writes the numbers, for the run-time estimate of tools/main_run.py.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bvp_pinn.config import load_case  # noqa: E402
from bvp_pinn.pinn import jets, lm, losses  # noqa: E402
from bvp_pinn.pinn import model as nets  # noqa: E402
from bvp_pinn.pinn import train as tr  # noqa: E402
from bvp_pinn.pinn.points import make_points  # noqa: E402
from bvp_pinn.pinn.runs import resolve  # noqa: E402

JETS = jets.supports


def sync():
    if torch.get_default_device().type == "cuda":
        torch.cuda.synchronize()


def median_seconds(fn, reps, warm=2):
    for _ in range(warm):
        fn()
    sync()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync()
        out.append(time.perf_counter() - t)
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", default="B0")
    ap.add_argument("--mode", default=None, help="bc_mode override: soft, linear or exact")
    ap.add_argument("--points", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--float32", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--jets-only", action="store_true", help="skip the slow nested-autograd engine")
    ap.add_argument("--lm", action="store_true", help="also time one Levenberg-Marquardt step")
    ap.add_argument("--json", default=None, help="write the timings to this file")
    a = ap.parse_args()
    torch.set_default_dtype(torch.float32 if a.float32 else torch.float64)
    torch.set_default_device(a.device)
    torch.set_num_threads(a.threads)
    p = load_case()[0]
    cfg = resolve(a.arm, **({"bc_mode": a.mode} if a.mode else {}), **({"n_points": a.points} if a.points else {}))
    torch.manual_seed(0)
    model = nets.build(cfg, p)
    stage = tr.plan_stages(cfg, make_points(cfg, 0)[0])[0]

    def full_step():
        terms, *_ = tr.loss_parts(model, p, cfg, stage)
        model.zero_grad()
        sum(terms).backward()

    print(f"{a.arm}, bc_mode {cfg['bc_mode']}, {cfg['n_points']} points, width {cfg['width']}, depth {cfg['depth']}, "
          f"{a.threads} thread(s), {torch.get_default_dtype()}, device {a.device}")
    out = {"arm": a.arm, "device": a.device, "threads": a.threads, "n_points": cfg["n_points"],
           "width": cfg["width"], "depth": cfg["depth"], "torch": torch.__version__}
    for name in ("jets",) if a.jets_only else ("jets", "autograd"):
        jets.supports = JETS if name == "jets" else (lambda net: False)
        eq = median_seconds(lambda: losses.equation_mse(model, stage["groups"], p), a.reps)
        bc = median_seconds(lambda: losses.boundary_errors(model, p, cfg["L"]), a.reps)
        step = median_seconds(full_step, a.reps)
        print(f"  {name:8s} equation loss {eq * 1e3:7.1f} ms   boundary errors {bc * 1e3:6.1f} ms   "
              f"full step {step * 1e3:7.1f} ms")
        out[f"{name}_full_step_s"] = step
        if a.profile:
            with torch.profiler.profile() as prof:
                full_step()
            print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12))
    jets.supports = JETS
    if a.lm:
        full = tr.full_problem(cfg, make_points(cfg, 0)[0])
        w = torch.ones(len(tr.loss_parts(model, p, cfg, full)[0]))
        c = {**cfg, "residual_scale": tr.final_scale(cfg)}
        out["lm_step_s"] = median_seconds(lambda: lm.minimize(model, p, c, full, w, 1), reps=1, warm=1)
        print(f"  Levenberg-Marquardt step ({c['residual_scale']} scale): {out['lm_step_s']:.2f} s")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
