"""Training for the four arms (docs/pinn/PINN_PLAN.md section 2).

  B0  one network, the whole problem in one stage
  B1  one network per window, trained together, with continuity at the window edges
  B2  one network, windows added from the wall outward; earlier windows keep their residual (physics replay)
  B3  as B2, but earlier windows are held to the network's own outputs from the stage before (self-replay)

All 12 boundary conditions hold in every stage: built into the output, or as loss terms
(cfg["bc_mode"], see constraints.py). Adam runs with gradient-norm weights; then a refine stage
(cfg["refine"]: L-BFGS, full BFGS or Levenberg-Marquardt) runs on the full problem with the weights frozen, and is
kept only if the monitor loss improves. The reference solution is never used here.

Options added on 2026-11-23 (docs/plans/LITERATURE_GAP_STRATEGY.md section 6). Each defaults to
the old behaviour, so old configs give the same runs:
  residual_scale   "none" or "log": fixed factor (eta + b)^order on each residual (losses.py)
  weight_cap       upper bound on the gradient-norm weights
  best_checkpoint  restore the Adam state with the lowest monitor loss (one-stage arms)
  resample_every   new stratified training points every this many Adam steps (one network only)
  refine           "lbfgs", "bfgs" (dense inverse Hessian, bfgs.py) or "lm" (Levenberg-Marquardt, lm.py),
                   or several in order joined by "+", such as "lbfgs+lm". lm_iters sets the LM steps
  residual_scale   also "balanced" (losses.py), chosen on 2026-11-23 after runs X1 and X2
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from . import bfgs, lm
from .constraints import SOFT_BCS
from .losses import GradNormWeights, boundary_errors, equation_mse, interface_mse, self_replay_mse
from .points import column, in_range, wall_points
from .selfcheck import self_check


def plan_stages(cfg, train):
    """The training stages of an arm. Each stage names its residual points per network."""
    edges = cfg["windows"]
    windows = list(zip(edges[:-1], edges[1:]))
    if cfg["structure"] == "single":
        return [{"groups": [(0, train)]}]
    if cfg["structure"] == "windows":
        return [{"groups": [(k, in_range(train, a, b)) for k, (a, b) in enumerate(windows)], "interface": True}]
    return [{"groups": [(0, in_range(train, a, b))], "old": in_range(train, 0.0, a)} for a, b in windows]


def full_problem(cfg, eta):
    """The whole problem on the given points: what the refine stage and the monitor use."""
    if cfg["structure"] == "windows":
        return plan_stages(cfg, eta)[0]
    return {"groups": [(0, eta)]}


def loss_parts(model, p, cfg, stage, with_bc=False):
    """The weighted terms (equation loss first, then soft boundary terms, then interface),
    plus the five equation losses, the 12 boundary errors and the replay term, for logging.

    The boundary errors are only computed when the loss needs them or with_bc is set: in exact
    mode they are zero by construction and are only logged.
    """
    eq = equation_mse(model, stage["groups"], p, cfg.get("residual_scale"))
    soft = SOFT_BCS[cfg["bc_mode"]]
    bc = boundary_errors(model, p, cfg["L"]) if soft or with_bc else None
    terms = [eq.sum()] + [bc[i] ** 2 for i in soft]
    if stage.get("interface"):
        terms.append(interface_mse(model))
    replay = torch.zeros(())
    if stage.get("old") is not None and len(stage["old"]):
        if cfg["replay"] == "physics":
            replay = equation_mse(model, [(0, stage["old"])], p, cfg.get("residual_scale")).sum()
        else:
            replay = self_replay_mse(model, stage["old"], stage["frozen"])
    return terms, eq, bc, replay


def weighted(terms, replay, w, cfg):
    return (w * torch.stack(terms)).sum() + cfg.get("replay_weight", 0.0) * replay


def train(model, p, cfg, train_pts, monitor_pts, log, seed=0):
    """Run all stages, then the refine stage. Returns a status dict; log(row) receives history rows."""
    stages = plan_stages(cfg, train_pts)
    weights = GradNormWeights(len(loss_parts(model, p, cfg, stages[0])[0]), cfg["weight_alpha"],
                              cfg.get("weight_cap"))
    steps = cfg["adam_steps"] // len(stages)
    start = time.perf_counter()
    status = {"stages": len(stages), "adam_steps_per_stage": steps, "stage_states": []}
    for k, stage in enumerate(stages):
        if cfg.get("replay") == "self" and len(stage["old"]):
            stage["frozen"] = model[0](stage["old"]).detach()  # no_grad would break the constraint layer
        ok = adam_stage(model, p, cfg, stage, steps, weights, monitor_pts, lambda r: log({"stage": k, **r}),
                        start, seed, best=len(stages) == 1 and cfg.get("best_checkpoint", False))
        status["stage_states"].append(copy.deepcopy(model.state_dict()))
        if not ok:
            status["result"] = f"non-finite loss in Adam stage {k}"
            return status
    status["weights"] = weights.w.tolist()
    status["adam_seconds"] = time.perf_counter() - start
    status.update(refine_phase(model, p, cfg, weights.w, train_pts, monitor_pts, log, start))
    # Training-side numbers that compare across arms and modes (no weights, no reference).
    monitor = full_problem(cfg, monitor_pts)["groups"]
    status["monitor_equation_mse"] = equation_mse(model, monitor, p).detach().tolist()
    final_scale = cfg.get("refine", "lbfgs").split("+")[-1].partition(":")[2] or cfg.get("residual_scale")
    status["final_residual_scale"] = final_scale
    status["monitor_scaled_equation_mse"] = equation_mse(model, monitor, p, final_scale).detach().tolist()
    status["final_boundary_abs"] = boundary_errors(model, p, cfg["L"]).detach().abs().tolist()
    status["self_check"] = self_check(model, p, cfg["L"], monitor_pts)
    status["result"] = "finished"
    return status


def adam_stage(model, p, cfg, stage, steps, weights, monitor_pts, log, start, seed=0, best=False):
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, (cfg["lr_end"] / cfg["lr"]) ** (1 / max(steps, 1)))
    every = cfg.get("resample_every", 0) if cfg["structure"] == "single" else 0
    rng = np.random.default_rng([seed, 1])  # its own stream: the fixed points stay as they were
    best_loss, best_state = float("inf"), None
    for i in range(steps):
        if every and i > 0 and i % every == 0:
            stage = {"groups": [(0, column(wall_points(cfg["n_points"], cfg["L"], cfg["stretch"], rng)))]}
        logging = i % cfg["log_every"] == 0 or i == steps - 1
        terms, eq, bc, replay = loss_parts(model, p, cfg, stage, with_bc=logging)
        if i > 0 and i % cfg["weight_every"] == 0:
            weights.update(params, terms)
        loss = weighted(terms, replay, weights.w, cfg)
        if not torch.isfinite(loss):
            return False
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if logging:
            mon = monitor_loss(model, p, cfg, weights.w, monitor_pts)
            log(row("adam", i, loss, eq, bc, terms, replay, weights.w, start, lr=sched.get_last_lr()[0], monitor=mon))
            if best and mon < best_loss:
                best_loss, best_state = mon, copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return True


def monitor_loss(model, p, cfg, w, monitor_pts):
    """Full-problem loss on the monitor points, with the given weights. Detached."""
    terms = loss_parts(model, p, cfg, full_problem(cfg, monitor_pts))[0]
    return weighted(terms, torch.zeros(()), w, cfg).item()


def refine_phase(model, p, cfg, w, train_pts, monitor_pts, log, start):
    """The refine stages after Adam, in order: cfg["refine"] is one method or several joined
    by "+", for example "lbfgs+lm". A stage may name its own residual scale after a colon,
    for example "lbfgs:log+lm:balanced"; otherwise it uses cfg["residual_scale"]. Each stage
    is rolled back on its own if it makes its monitor loss worse.

    Why per-stage scales (runs X3 and X4, IMPLEMENTATION_NOTES 7.2): the log scale brings the
    far tail into the decaying state, and the balanced scale then lets Levenberg-Marquardt
    finish. The balanced scale from the start left the tail almost flat (X4, m = 0.013)."""
    methods = cfg.get("refine", "lbfgs").split("+")
    stages = []
    for item in methods:
        method, _, scale = item.partition(":")
        c = {**cfg, "residual_scale": scale} if scale else cfg
        stages.append(refine_once(model, p, c, w, train_pts, monitor_pts, log, start, method))
    last = methods[-1].partition(":")[2]
    out = {k: v for k, v in stages[-1].items()}
    before = stages[0]["monitor_before_refine"]
    after = monitor_loss(model, p, {**cfg, "residual_scale": last} if last else cfg, w, monitor_pts)
    accepted = any(s["refine_accepted"] for s in stages)
    seconds = sum(s["refine_seconds"] for s in stages)
    calls = sum(s["refine_evaluations"] for s in stages)
    out.update({"refine": "+".join(methods), "refine_stages": stages, "refine_seconds": seconds,
                "refine_evaluations": calls, "monitor_before_refine": before, "monitor_after_refine": after,
                "refine_accepted": accepted})
    # Old names, so evaluate.py and earlier run folders keep working.
    out.update({"lbfgs_evaluations": calls, "lbfgs_seconds": seconds, "monitor_before_lbfgs": before,
                "monitor_after_lbfgs": after, "lbfgs_accepted": accepted})
    return out


def refine_once(model, p, cfg, w, train_pts, monitor_pts, log, start, method):
    """One refine stage on the full problem: L-BFGS, full BFGS or Levenberg-Marquardt.
    Rolled back if the monitor loss gets worse or is not finite."""
    stage = full_problem(cfg, train_pts)
    before = monitor_loss(model, p, cfg, w, monitor_pts)
    saved = copy.deepcopy(model.state_dict())
    calls = [0]

    def objective():
        terms, eq, bc, replay = loss_parts(model, p, cfg, stage, with_bc=calls[0] % cfg["log_every"] == 0)
        loss = weighted(terms, replay, w, cfg)
        if calls[0] % cfg["log_every"] == 0:
            log({"stage": "refine", **row(method, calls[0], loss, eq, bc, terms, replay, w, start)})
        calls[0] += 1
        return loss

    t0 = time.perf_counter()
    if method == "bfgs":
        info = bfgs.minimize(model, objective, cfg["lbfgs_iters"])
    elif method == "lbfgs":
        opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=cfg["lbfgs_iters"],
                                history_size=cfg["lbfgs_history"], tolerance_grad=1e-12,
                                tolerance_change=1e-16, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = objective()
            loss.backward()
            return loss

        opt.step(closure)
        info = {}
    elif method == "lm":
        info = lm.minimize(model, p, cfg, train_pts, cfg.get("lm_iters", cfg["lbfgs_iters"]),
                           log=lambda r: log({"stage": "refine", **r, "seconds": round(time.perf_counter() - start, 3)}))
    else:
        raise ValueError(f"unknown refine {method!r}")
    after = monitor_loss(model, p, cfg, w, monitor_pts)
    accepted = after == after and after <= before  # after == after is False for NaN
    if not accepted:
        model.load_state_dict(saved)
    return {"refine": method, "refine_evaluations": calls[0], "refine_seconds": time.perf_counter() - t0,
            "monitor_before_refine": before, "monitor_after_refine": after, "refine_accepted": accepted, **info}


def row(phase, step, loss, eq, bc, terms, replay, w, start, lr=None, monitor=None):
    """One history row: every loss term separately, never only the sum."""
    out = {"phase": phase, "step": step, "seconds": round(time.perf_counter() - start, 3), "lr": lr,
           "loss": loss.item(), "monitor": monitor, "replay": replay.item()}
    out.update({f"R{14 + i}": v.item() for i, v in enumerate(eq)})
    if bc is not None:
        out.update({f"B{i + 1}": v.item() for i, v in enumerate(bc)})
    out.update({f"term{i}": t.item() for i, t in enumerate(terms)})
    out.update({f"w{i}": float(v) for i, v in enumerate(w)})
    return out
