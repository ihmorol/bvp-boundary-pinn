"""Full BFGS on the network weights, with a dense inverse Hessian (cfg["refine"] = "bfgs").

Nocedal and Wright, Numerical Optimization, 2nd ed., Algorithm 6.1, with the strong Wolfe
line search that torch.optim.LBFGS also uses. The first inverse Hessian is (s'y / y'y) I
(their eq. 6.20). Why: Kiyani et al. (2025) report errors near 1e-8 in float64 with full
quasi-Newton where L-BFGS stalled near 1e-3, and our loss is badly conditioned (Acta
Numerica 2024, section 7). Our network has about 5 200 weights, so the dense matrix is about
220 MB in float64. The update is rank two and in place, so no second n x n matrix is made.
"""
from __future__ import annotations

import torch
from torch.optim.lbfgs import _strong_wolfe


def _flat(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


def _write(params, x):
    i = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(x[i:i + n].view_as(p))
            i += n


def minimize(model, objective, max_iter, gtol=1e-14, stall_limit=20):
    """Minimize objective() (a scalar tensor that depends on model's weights) in place.

    Stops after max_iter iterations, when the largest gradient entry is below gtol, or after
    stall_limit iterations in a row whose loss drop is below 1e-15 of the loss.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    def evaluate(x):
        _write(params, x)
        loss = objective()
        grads = torch.autograd.grad(loss, params)
        return loss.item(), torch.cat([g.reshape(-1) for g in grads])

    def along(x, t, d):
        return evaluate(x + t * d)

    x = _flat(params)
    f, g = evaluate(x)
    H, stalls, resets, k = None, 0, 0, 0
    stop = "max_iter"
    for k in range(1, max_iter + 1):
        d = -g if H is None else -(H @ g)
        gtd = float(g @ d)
        if gtd >= 0:  # not a descent direction: start the Hessian again
            H, d = None, -g
            gtd = float(g @ d)
            resets += 1
        t = 1.0 if H is not None else min(1.0, 1.0 / float(g.abs().sum()))
        f_new, g_new, t, _ = _strong_wolfe(along, x, t, d, f, g, gtd)
        s = t * d
        y = g_new - g
        x = x + s
        sy, yy = float(s @ y), float(y @ y)
        if sy > 1e-12 * yy and yy > 0:  # curvature condition; skip the update otherwise
            if H is None:
                H = torch.eye(x.numel(), dtype=x.dtype, device=x.device) * (sy / yy)
            rho = 1.0 / sy
            Hy = H @ y
            H.addr_(Hy, s, alpha=-rho)
            H.addr_(s, Hy, alpha=-rho)
            H.addr_(s, s, alpha=rho * rho * float(y @ Hy) + rho)
        stalls = stalls + 1 if abs(f - f_new) <= 1e-15 * abs(f) else 0
        f, g = f_new, g_new
        if not torch.isfinite(g).all():
            stop = "non-finite gradient"
            break
        if float(g.abs().max()) < gtol:
            stop = "gradient below gtol"
            break
        if stalls >= stall_limit:
            stop = "loss stalled"
            break
    _write(params, x)
    return {"bfgs_iterations": k, "bfgs_stop": stop, "bfgs_resets": resets, "bfgs_final_loss": f}
