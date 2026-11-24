"""Checks of a trained network that need no reference solution.

They tell from the PINN alone whether it can be the decaying solution that B10 and B11 ask
for, and where its residual is still large. evaluate.py compares with the reference later.

  wall values   f''(0), theta(0), theta'(0), phi'(0), chi'(0), f(L): the numbers the paper reports
  m, mp         far-field decay powers of theta and phi (reference/equations.py, far_conditions).
                theta ~ q^-m and phi ~ q^-mp tend to zero only if m > 0 and mp > 0.
                Branch B of the reference fails this (m < 0), so the check is branch-aware.
  failure share share of points where the largest |R14..R18| is above eps (FI-PINN style)
  tail          theta on [L/2, L] fitted to C1 + C2 q^-m, and phi to D1 + E q^-mp + P q^-m.
                C1 and D1 are the parts that do not decay. Both have almost no residual in
                Eqs. 16 and 17, so a small loss can hide them (run E8); only B10 and B11 hold
                them at zero. tail_share = |C1| / max |theta| on [L/2, L]: near 0 for a
                decaying tail (docs/reviews/PINN_METHOD_SYNTHESIS_2026-11-23.md, change P-C)
"""
from __future__ import annotations

import torch

from .losses import boundary_errors
from .model import predict
from .physics import residuals


def self_check(model, p, L, points=None, eps=1e-3):
    e = torch.tensor([[0.0], [float(L)]])
    z = predict(model, e)[0]
    f_L, th_L = float(z[0, 1]), float(z[6, 1])
    A = 1 + (p["theta_w"] - 1) * th_L
    m = p["Pr"] * p["kappa"] * f_L / (1 + p["Nr"] * A**3)
    mp = p["Le"] * p["Pr"] * p["kappa"] * f_L
    out = {"f2_0": float(z[2, 0]), "theta_0": float(z[6, 0]), "th1_0": float(z[7, 0]),
           "ph1_0": float(z[9, 0]), "ch1_0": float(z[11, 0]), "f_L": f_L,
           "m": m, "mp": mp, "decaying_far_field": bool(m > 0 and mp > 0),
           "max_boundary_abs": float(boundary_errors(model, p, L).detach().abs().max())}
    out.update(tail_fit(model, L, p, m, mp))
    if points is not None:
        out["failure_eps"] = eps
        out["failure_share"] = failure_share(model, p, points, eps)
    return out


def failure_share(model, p, points, eps):
    z, H = predict(model, points)
    cols = [r.reshape(-1, 1) for r in z]
    R = residuals(points, cols, [r.reshape(-1, 1) for r in H], p)
    return float((R.abs().max(1).values > eps).double().mean())


def tail_fit(model, L, p, m, mp, n=200):
    """Constant (non-decaying) parts of theta and phi on [L/2, L], by linear least squares."""
    if not (m > 0 and mp > 0):
        return {"tail_C1_theta": None, "tail_D1_phi": None, "tail_share": None, "theta_L": None}
    eta = torch.linspace(L / 2, L, n).reshape(-1, 1)
    z = predict(model, eta)[0]
    q = eta.reshape(-1) + p["kappa"]
    one = torch.ones_like(q)
    th, ph = z[6], z[8]
    c = torch.linalg.lstsq(torch.stack([one, q ** -m], 1), th.reshape(-1, 1)).solution.reshape(-1)
    basis = [one, q ** -mp] + ([q ** -m] if abs(mp - m) > 1e-6 else [])
    d = torch.linalg.lstsq(torch.stack(basis, 1), ph.reshape(-1, 1)).solution.reshape(-1)
    return {"tail_C1_theta": float(c[0]), "tail_D1_phi": float(d[0]),
            "tail_share": float(c[0].abs() / th.abs().max()), "theta_L": float(th[-1])}
