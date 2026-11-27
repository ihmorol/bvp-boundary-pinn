"""Levenberg-Marquardt on the network weights (a refine stage: "lm" in cfg["refine"]).

The refine stage minimizes the training objective with the loss weights w frozen
(train.weighted, for the full problem):

    loss = w0 * (1/N) * sum over points of sum_k R_k^2      equation term (scaled residuals)
         + sum_i w_i * B_i^2                                 soft boundary terms (soft, linear modes)
         + w_I * mean of the squared interface jumps         B1 only

That is a plain sum of squares, loss = r . r, with the rows

    point rows    sqrt(w0 / N) * R_k(eta_j)                  5 per point
    global rows   sqrt(w_i) * B_i  and  sqrt(w_I / n_J) * jump_j

Levenberg-Marquardt uses that structure: with J = dr/dw, each step solves

    (J'J + mu I) dw = -J'r

and mu is lowered after a good step and raised after a bad one (Nielsen's rule: Madsen,
Nielsen and Tingleff, "Methods for non-linear least squares problems", 2004, section 3.2).
J'J is the Gauss-Newton matrix (Acta Numerica 2024, section 7.5.3). The squeezing-flow paper
(2026, local copy) uses the same polish after its global search.

Structure. A point row depends only on the network that owns the point, because the
constraint layer reads only that network's own ends. So J'J = D + C'C, where D has one block
J_k'J_k per network and C holds the few global rows (36 interface jumps for B1 with four
windows; the soft boundary errors in soft and linear modes; none for one network in exact
mode). Each trial step factors the blocks of D + mu I one at a time and adds C'C through the
Woodbury identity. B1 therefore costs four block factorizations, not one of four times the size.

Cost on the 40 x 4 network (about 5 200 weights per network): each block J_k'J_k is about
220 MB in float64; the point Jacobian is built per point (vmap of jacrev) in chunks.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.func import functional_call, jacrev, vmap

from .constraints import SOFT_BCS
from .losses import boundary_errors, interface_jumps, residual_scale
from .model import states
from .physics import residuals


class _PointResiduals(nn.Module):
    """Scaled residuals of Eqs. 14 to 18 of one network at the given points, as (N, 5)."""

    def __init__(self, net, p, scale):
        super().__init__()
        self.net, self.p, self.scale = net, p, scale

    def forward(self, eta):
        z, H = states(self.net, eta)
        r = residuals(eta, z, H, self.p)
        s = residual_scale(eta, self.p, self.scale)
        return r if s is None else r * s


class _GlobalRows(nn.Module):
    """Weighted soft boundary errors and B1 interface jumps, as one flat tensor (may be empty)."""

    def __init__(self, model, p, cfg, interface, w):
        super().__init__()
        self.model, self.p, self.L = model, p, cfg["L"]
        self.soft = SOFT_BCS[cfg["bc_mode"]]
        self.interface = interface
        self.root_soft = w[1:1 + len(self.soft)].sqrt()
        self.root_if = w[1 + len(self.soft)].sqrt() if interface else None

    def forward(self):
        rows = []
        if self.soft:
            rows.append(self.root_soft * boundary_errors(self.model, self.p, self.L)[self.soft])
        if self.interface:
            jumps = interface_jumps(self.model)
            rows.append(self.root_if * jumps / math.sqrt(jumps.numel()))
        return torch.cat(rows) if rows else torch.zeros(0)


def minimize(model, p, cfg, stage, w, max_iter, log=None, chunk=100):
    """LM on all networks of model, for the stage's points, with the weights w frozen.

    stage is train.full_problem(cfg, points): {"groups": [(network index, points)], "interface": bool}.
    Returns a status dict. chunk is the number of points per Jacobian batch (memory).
    """
    w = torch.as_tensor(w).detach()
    nets = list(model)
    groups = {k: eta.detach().clone() for k, eta in stage["groups"] if eta.shape[0]}
    root = math.sqrt(float(w[0]) / sum(e.shape[0] for e in groups.values()))
    names = [[n for n, v in net.named_parameters() if v.requires_grad] for net in nets]
    params = [{n: v.detach().clone() for n, v in net.named_parameters() if v.requires_grad} for net in nets]
    sizes = [sum(params[k][n].numel() for n in names[k]) for k in range(len(nets))]
    point_res = [_PointResiduals(net, p, cfg.get("residual_scale")) for net in nets]
    glob = _GlobalRows(model, p, cfg, bool(stage.get("interface")), w)

    def point_rows(ps, k, eta):
        out = functional_call(point_res[k], {f"net.{n}": v for n, v in ps[k].items()}, (eta,))
        return out.reshape(-1) * root

    def global_rows(ps):
        flat = {f"model.{k}.{n}": v for k, d in enumerate(ps) for n, v in d.items()}
        return functional_call(glob, flat, ())

    def r_of(ps):
        with torch.no_grad():
            return [point_rows(ps, k, e) for k, e in groups.items()], global_rows(ps)

    def block_jacobian(ps, k, eta):
        def one(pk, e):  # e has shape (1,): one point
            out = functional_call(point_res[k], {f"net.{n}": v for n, v in pk.items()}, (e.reshape(1, 1),))
            return out.reshape(5)
        J = vmap(jacrev(one), in_dims=(None, 0), chunk_size=chunk)(ps[k], eta)
        return torch.cat([J[n].reshape(eta.shape[0] * 5, -1) for n in names[k]], 1) * root

    def global_jacobian(ps, n_rows):
        J = jacrev(global_rows)(ps)
        return torch.cat([J[k][n].reshape(n_rows, -1) for k in range(len(nets)) for n in names[k]], 1)

    def normal_equations(ps, r_points, r_global):
        """Blocks A_k = J_k'J_k, the gradient g = J'r, and the global rows C (or None)."""
        A, g = [], []
        r_by_net = dict(zip(groups, r_points))
        for k in range(len(nets)):
            if k in groups:
                Jk = block_jacobian(ps, k, groups[k])
                A.append(Jk.T @ Jk)
                g.append(Jk.T @ r_by_net[k])
                del Jk
            else:
                A.append(torch.zeros(sizes[k], sizes[k]))
                g.append(torch.zeros(sizes[k]))
        g = torch.cat(g)
        C = None
        if r_global.numel():
            C = global_jacobian(ps, r_global.numel())
            g = g + C.T @ r_global
        return A, g, C

    def solve(A, g, C, mu):
        """h with (blockdiag(A) + mu I + C'C) h = -g, or None if a block is not positive definite."""
        ys, Zs, off = [], [], 0
        for Ak, n in zip(A, sizes):
            Ak.diagonal().add_(mu)
            Lk, info = torch.linalg.cholesky_ex(Ak)
            Ak.diagonal().sub_(mu)
            if info != 0:
                return None
            ys.append(-torch.cholesky_solve(g[off:off + n].reshape(-1, 1), Lk).reshape(-1))
            if C is not None:
                Zs.append(torch.cholesky_solve(C[:, off:off + n].T.contiguous(), Lk))
            off += n
            del Lk
        y = torch.cat(ys)
        if C is None:
            return y
        Z = torch.cat(Zs, 0)  # (D + mu I)^-1 C'
        S = torch.eye(C.shape[0]) + C @ Z
        return y - Z @ torch.linalg.solve(S, C @ y)  # Woodbury

    def flat(ps):
        return torch.cat([ps[k][n].reshape(-1) for k in range(len(nets)) for n in names[k]])

    def unflat(x):
        out, i = [], 0
        for k in range(len(nets)):
            d = {}
            for n in names[k]:
                m = params[k][n].numel()
                d[n] = x[i:i + m].view_as(params[k][n])
                i += m
            out.append(d)
        return out

    def value(r_points, r_global):
        return float(sum(r @ r for r in r_points) + r_global @ r_global)

    r_points, r_global = r_of(params)
    f = value(r_points, r_global)
    mu, nu, accepted, stop, k = None, 2.0, 0, "max_iter", 0
    for k in range(1, max_iter + 1):
        A, g, C = normal_equations(params, r_points, r_global)
        if mu is None:
            diag = torch.cat([Ak.diagonal() for Ak in A])
            if C is not None:
                diag = diag + C.pow(2).sum(0)
            mu = 1e-3 * float(diag.max())
        x = flat(params)
        while True:
            h = solve(A, g, C, mu)
            gain, f_new = -1.0, float("nan")
            if h is not None:
                trial = unflat(x + h)
                rp_new, rg_new = r_of(trial)
                f_new = value(rp_new, rg_new)
                predicted = float(h @ (mu * h - g))  # f - model value, > 0 for a descent step
                gain = (f - f_new) / predicted if predicted > 0 else -1.0
            if gain > 0 and math.isfinite(f_new):
                params = [{n: t.clone() for n, t in d.items()} for d in trial]
                r_points, r_global, f = rp_new, rg_new, f_new
                mu *= max(1 / 3, 1 - (2 * gain - 1) ** 3)
                nu = 2.0
                accepted += 1
                break
            mu *= nu
            nu *= 2
            if mu > 1e20:
                stop = "mu too large"
                break
        del A, C
        if log is not None:
            log({"phase": "lm", "step": k, "loss": f, "mu": mu})
        if stop != "max_iter":
            break
        if float(g.abs().max()) < 1e-14 or f < 1e-28:
            stop = "converged"
            break
    with torch.no_grad():
        for net, d in zip(nets, params):
            for n, v in net.named_parameters():
                if n in d:
                    v.copy_(d[n])
    return {"lm_iterations": k, "lm_accepted_steps": accepted, "lm_stop": stop, "lm_final_loss": f, "lm_mu": mu}
