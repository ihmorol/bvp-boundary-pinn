"""Manufactured-solution check of the PINN training code (no reference data).

    uv run python tools/pinn_manufactured.py                       # B0 settings of configs/pinn.json
    uv run python tools/pinn_manufactured.py --set adam_steps=4000 refine=bfgs

A known answer u* is built first: the smooth start profile with a different rate and wall
temperature than the student's start, passed through the exact constraint layer. So u* meets
all 12 boundary conditions B1 to B12 exactly. The source S = R(u*) is added to Eqs. 14 to 18,
so the forced equations R(u) - S = 0 have the solution u = u*. The same training code then
trains a fresh network on the forced equations, and we measure how close it gets to u*.

What it tests: the jets, the constraint layer, the losses and the optimizers on this
operator and these boundary conditions, from a start that is not the answer. What it does not
test: the thin wall layer of the real solution, because u* is smooth. A pass says the code
can solve this kind of problem; it says nothing about the accuracy on the real case.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bvp_pinn.config import load_case  # noqa: E402
from bvp_pinn.pinn import losses, physics  # noqa: E402
from bvp_pinn.pinn import model as nets  # noqa: E402
from bvp_pinn.pinn.points import make_points  # noqa: E402
from bvp_pinn.pinn.run import _number  # noqa: E402
from bvp_pinn.pinn.runs import resolve  # noqa: E402
from bvp_pinn.pinn.train import train  # noqa: E402

TEACHER = {"start": "smooth", "start_rate": 1.5, "start_theta0": 0.5, "zero_last": True}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", type=Path, default=None, help="optional JSON file for the result")
    args = ap.parse_args(argv)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    p = load_case()[0]
    cfg = resolve("B0", **{k: _number(v) for k, v in (s.split("=", 1) for s in args.set)})
    if cfg["bc_mode"] != "exact":
        raise SystemExit("this check needs bc_mode=exact, so u* meets B1 to B12 by construction")
    teacher = nets.build({**cfg, **TEACHER}, p)
    for x in teacher.parameters():
        x.requires_grad_(False)

    plain = physics.residuals

    def forced(eta, z, H, p_):
        zt, Ht = nets.states(teacher[0], eta.detach())
        return plain(eta, z, H, p_) - plain(eta.detach(), zt, Ht, p_).detach()

    losses.residuals = forced  # the training code now solves R(u) - R(u*) = 0
    student = nets.build(cfg, p)
    train_pts, monitor_pts = make_points(cfg, args.seed)
    status = train(student, p, cfg, train_pts, monitor_pts, lambda row: None, args.seed)
    losses.residuals = plain

    eta = torch.linspace(0, cfg["L"], 2001).reshape(-1, 1)
    zs, zt = nets.predict(student, eta)[0].numpy(), nets.predict(teacher, eta)[0].numpy()
    e = eta.numpy().ravel()
    rel = {name: float(np.sqrt(np.trapezoid((zs[i] - zt[i]) ** 2, e) / np.trapezoid(zt[i] ** 2, e)))
           for name, i in {"f": 0, "g": 4, "theta": 6, "phi": 8, "chi": 10}.items()}
    wall = {k: [float(zs[i, 0]), float(zt[i, 0])] for k, i in {"f2_0": 2, "theta_0": 6, "th1_0": 7, "ph1_0": 9}.items()}
    result = {"rel_l2": rel, "wall_student_vs_known": wall, "max_rel_l2": max(rel.values()),
              "forced_monitor_mse": status.get("monitor_equation_mse"), "refine_accepted": status.get("refine_accepted"),
              "seconds": status.get("adam_seconds", 0) + status.get("refine_seconds", 0), "cfg": cfg}
    print(json.dumps({k: v for k, v in result.items() if k != "cfg"}, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    main()
