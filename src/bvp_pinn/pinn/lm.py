"""Levenberg-Marquardt on the network weights (cfg["refine"] = "lm").

In exact mode all 12 boundary conditions are built in, so the loss is a plain sum of squares,
loss = r . r, with r the (scaled) residuals of Eqs. 14 to 18 at the training points divided by
sqrt(N). Levenberg-Marquardt uses that structure: with J = dr/dw, each step solves

    (J'J + mu I) dw = -J'r

and mu is lowered after a good step and raised after a bad one (Nielsen's rule: Madsen,
Nielsen and Tingleff, "Methods for non-linear least squares problems", 2004, section 3.2).
J'J is the Gauss-Newton matrix, which for a linear model is exactly the Hessian that sets the
convergence rate (Acta Numerica 2024, section 7.5.3). The squeezing-flow paper (2026, local
copy) uses the same polish after its global search.

Cost on our network (about 5 200 weights, 1 000 points, 5 equations): J is about 200 MB and
J'J about 220 MB in float64; one Cholesky factorization per trial step.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.func import functional_call, jacrev, vmap

from .losses import residual_scale
from .model import states
from .physics import residuals


class _Residuals(nn.Module):
    """Scaled residuals of Eqs. 14 to 18 at the given points, as an (N, 5) tensor."""

    def __init__(self, net, p, scale):
        super().__init__()
        self.net, self.p, self.scale = net, p, scale

    def forward(self, eta):
        z, H = states(self.net, eta)
        r = residuals(eta, z, H, self.p)
        s = residual_scale(eta, self.p, self.scale)
        return r if s is None else r * s


def minimize(model, p, cfg, eta, max_iter, log=None, chunk=100):
    """LM on model[0] (one network, exact mode). Returns a status dict.

    The Jacobian is built per point: each residual row depends on its own point and on the
    two ends (through the constraint layer), so vmap over points of a per-point jacrev gives
    the full matrix in one batched pass. chunk is the number of points per batch (memory).
    """
    if cfg["bc_mode"] != "exact" or len(model) != 1:
        raise ValueError("refine=lm needs one network in exact mode (loss = sum of squares)")
    net = model[0]
    res = _Residuals(net, p, cfg.get("residual_scale"))
    eta = eta.detach().clone()
    root_n = math.sqrt(eta.shape[0])
    names = [k for k, v in net.named_parameters() if v.requires_grad]
    w = {k: v.detach().clone() for k, v in net.named_parameters() if v.requires_grad}
    sizes = [w[k].numel() for k in names]

    def call(params, e):
        return functional_call(res, {f"net.{k}": v for k, v in params.items()}, (e,))

    def r_of(params):
        return call(params, eta).reshape(-1) / root_n

    def r_point(params, e):  # e has shape (1,): one point
        return call(params, e.reshape(1, 1)).reshape(5)

    point_jac = vmap(jacrev(r_point), in_dims=(None, 0), chunk_size=chunk)

    def jac(params):
        J = point_jac(params, eta)  # name -> (N, 5, *shape)
        return torch.cat([J[k].reshape(eta.shape[0] * 5, -1) for k in names], 1) / root_n

    def flat(d):
        return torch.cat([d[k].reshape(-1) for k in names])

    def unflat(x):
        out, i = {}, 0
        for k, n in zip(names, sizes):
            out[k] = x[i:i + n].view_as(w[k])
            i += n
        return out

    with torch.no_grad():
        r = r_of(w)
    f = float(r @ r)
    mu, nu, accepted, stop = None, 2.0, 0, "max_iter"
    k = 0
    for k in range(1, max_iter + 1):
        J = jac(w)
        g = J.T @ r
        A = J.T @ J
        del J
        if mu is None:
            mu = 1e-3 * float(A.diagonal().max())
        x = flat(w)
        while True:
            A.diagonal().add_(mu)
            L, info = torch.linalg.cholesky_ex(A)
            A.diagonal().sub_(mu)
            if info == 0:
                step = -torch.cholesky_solve(g.reshape(-1, 1), L).reshape(-1)
                trial = unflat(x + step)
                with torch.no_grad():
                    r_new = r_of(trial)
                f_new = float(r_new @ r_new)
                predicted = float(step @ (mu * step - g))  # f - model value, > 0 for a descent step
                gain = (f - f_new) / predicted if predicted > 0 else -1.0
            else:
                gain = -1.0
            if gain > 0 and math.isfinite(f_new):
                w, r, f = {n: t.clone() for n, t in trial.items()}, r_new, f_new
                mu *= max(1 / 3, 1 - (2 * gain - 1) ** 3)
                nu = 2.0
                accepted += 1
                break
            mu *= nu
            nu *= 2
            if mu > 1e20:
                stop = "mu too large"
                break
        if log is not None:
            log({"phase": "lm", "step": k, "loss": f, "mu": mu})
        if stop != "max_iter":
            break
        if float(g.abs().max()) < 1e-14 or f < 1e-28:
            stop = "converged"
            break
    with torch.no_grad():
        for n, v in net.named_parameters():
            if n in w:
                v.copy_(w[n])
    return {"lm_iterations": k, "lm_accepted_steps": accepted, "lm_stop": stop, "lm_final_loss": f, "lm_mu": mu}
