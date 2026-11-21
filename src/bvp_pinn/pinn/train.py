"""Training for the four arms (docs/PINN_PLAN.md section 2).

  B0  one network, the whole problem in one stage
  B1  one network per window, trained together, with continuity at the window edges
  B2  one network, windows added from the wall outward; earlier windows keep their residual (physics replay)
  B3  as B2, but earlier windows are held to the network's own outputs from the stage before (self-replay)

All 12 boundary conditions hold in every stage: built into the output, or as loss terms
(cfg["bc_mode"], see constraints.py). Adam runs with gradient-norm weights; then L-BFGS runs on
the full problem with the weights frozen, and is kept only if the monitor loss improves.
The reference solution is never used here.
"""
from __future__ import annotations

import copy
import time

import torch

from .constraints import SOFT_BCS
from .losses import GradNormWeights, boundary_errors, equation_mse, interface_mse, self_replay_mse
from .points import in_range


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
    """The whole problem on the given points: what L-BFGS and the monitor use."""
    if cfg["structure"] == "windows":
        return plan_stages(cfg, eta)[0]
    return {"groups": [(0, eta)]}


def loss_parts(model, p, cfg, stage):
    """The weighted terms (equation loss first, then soft boundary terms, then interface),
    plus the five equation losses, all 12 boundary errors and the replay term, for logging."""
    eq = equation_mse(model, stage["groups"], p)
    bc = boundary_errors(model, p, cfg["L"])
    terms = [eq.sum()] + [bc[i] ** 2 for i in SOFT_BCS[cfg["bc_mode"]]]
    if stage.get("interface"):
        terms.append(interface_mse(model))
    replay = torch.zeros(())
    if stage.get("old") is not None and len(stage["old"]):
        if cfg["replay"] == "physics":
            replay = equation_mse(model, [(0, stage["old"])], p).sum()
        else:
            replay = self_replay_mse(model, stage["old"], stage["frozen"])
    return terms, eq, bc, replay


def weighted(terms, replay, w, cfg):
    return (w * torch.stack(terms)).sum() + cfg.get("replay_weight", 0.0) * replay


def train(model, p, cfg, train_pts, monitor_pts, log):
    """Run all stages, then L-BFGS. Returns a status dict; log(row) receives history rows."""
    stages = plan_stages(cfg, train_pts)
    weights = GradNormWeights(len(loss_parts(model, p, cfg, stages[0])[0]), cfg["weight_alpha"])
    steps = cfg["adam_steps"] // len(stages)
    start = time.perf_counter()
    status = {"stages": len(stages), "adam_steps_per_stage": steps, "stage_states": []}
    for k, stage in enumerate(stages):
        if cfg.get("replay") == "self" and len(stage["old"]):
            stage["frozen"] = model[0](stage["old"]).detach()  # no_grad would break the constraint layer
        ok = adam_stage(model, p, cfg, stage, steps, weights, monitor_pts, lambda r: log({"stage": k, **r}), start)
        status["stage_states"].append(copy.deepcopy(model.state_dict()))
        if not ok:
            status["result"] = f"non-finite loss in Adam stage {k}"
            return status
    status["weights"] = weights.w.tolist()
    status["adam_seconds"] = time.perf_counter() - start
    status.update(lbfgs_phase(model, p, cfg, weights.w, train_pts, monitor_pts, log, start))
    # Training-side numbers that compare across arms and modes (no weights, no reference).
    monitor = full_problem(cfg, monitor_pts)["groups"]
    status["monitor_equation_mse"] = equation_mse(model, monitor, p).detach().tolist()
    status["final_boundary_abs"] = boundary_errors(model, p, cfg["L"]).detach().abs().tolist()
    status["result"] = "finished"
    return status


def adam_stage(model, p, cfg, stage, steps, weights, monitor_pts, log, start):
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, (cfg["lr_end"] / cfg["lr"]) ** (1 / max(steps, 1)))
    for i in range(steps):
        terms, eq, bc, replay = loss_parts(model, p, cfg, stage)
        if i > 0 and i % cfg["weight_every"] == 0:
            weights.update(params, terms)
        loss = weighted(terms, replay, weights.w, cfg)
        if not torch.isfinite(loss):
            return False
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if i % cfg["log_every"] == 0 or i == steps - 1:
            log(row("adam", i, loss, eq, bc, terms, replay, weights.w, start, lr=sched.get_last_lr()[0],
                    monitor=monitor_loss(model, p, cfg, weights.w, monitor_pts)))
    return True


def monitor_loss(model, p, cfg, w, monitor_pts):
    """Full-problem loss on the monitor points, with the given weights. Detached."""
    terms = loss_parts(model, p, cfg, full_problem(cfg, monitor_pts))[0]
    return float(weighted(terms, torch.zeros(()), w, cfg))


def lbfgs_phase(model, p, cfg, w, train_pts, monitor_pts, log, start):
    """L-BFGS on the full problem. Rolled back if the monitor loss gets worse or is not finite."""
    stage = full_problem(cfg, train_pts)
    before = monitor_loss(model, p, cfg, w, monitor_pts)
    saved = copy.deepcopy(model.state_dict())
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=cfg["lbfgs_iters"],
                            history_size=cfg["lbfgs_history"], tolerance_grad=1e-12,
                            tolerance_change=1e-16, line_search_fn="strong_wolfe")
    calls = [0]

    def closure():
        opt.zero_grad()
        terms, eq, bc, replay = loss_parts(model, p, cfg, stage)
        loss = weighted(terms, replay, w, cfg)
        loss.backward()
        if calls[0] % cfg["log_every"] == 0:
            log({"stage": "lbfgs", **row("lbfgs", calls[0], loss, eq, bc, terms, replay, w, start)})
        calls[0] += 1
        return loss

    t0 = time.perf_counter()
    opt.step(closure)
    after = monitor_loss(model, p, cfg, w, monitor_pts)
    accepted = after == after and after <= before  # after == after is False for NaN
    if not accepted:
        model.load_state_dict(saved)
    return {"lbfgs_evaluations": calls[0], "lbfgs_seconds": time.perf_counter() - t0,
            "monitor_before_lbfgs": before, "monitor_after_lbfgs": after, "lbfgs_accepted": accepted}


def row(phase, step, loss, eq, bc, terms, replay, w, start, lr=None, monitor=None):
    """One history row: every loss term separately, never only the sum."""
    out = {"phase": phase, "step": step, "seconds": round(time.perf_counter() - start, 3), "lr": lr,
           "loss": float(loss), "monitor": monitor, "replay": float(replay)}
    out.update({f"R{14 + i}": float(v) for i, v in enumerate(eq)})
    out.update({f"B{i + 1}": float(v) for i, v in enumerate(bc)})
    out.update({f"term{i}": float(t) for i, t in enumerate(terms)})
    out.update({f"w{i}": float(v) for i, v in enumerate(w)})
    return out
