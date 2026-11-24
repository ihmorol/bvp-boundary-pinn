"""Loss terms, and the gradient-norm weights of Wang, Sankaran, Wang and Perdikaris (2023)."""
from __future__ import annotations

import torch

from .model import states
from .physics import boundary, residuals


ORDERS = torch.tensor([4.0, 2.0, 2.0, 2.0, 2.0])  # highest derivative order of R14 to R18


def residual_scale(eta, p, kind):
    """Fixed positive factor for each residual at each point, as an (N, 5) tensor, or None.

    "log": R_k times (eta + b)^order_k. With s = log(eta + b), d/deta = (1/(eta + b)) d/ds, so
    this is each equation written in the variable s, where the thin wall layer and the slow
    power-law tails both have size of order one. It uses only eta, b and the derivative orders,
    never the reference or the current residuals. The zero set of every equation is unchanged.

    "balanced": like "log" near the wall, but each momentum row stops growing where its own
    far-field balance takes over, so far-out f and g rows no longer dominate the loss
    (docs/reviews/PINN_METHOD_SYNTHESIS_2026-11-23.md, change P-A). With q = eta + kappa:
      R14 times q^4/(1 + c14 q^2), c14 = (1/Da)/(1 + Delta): the Darcy term against f4
      R15 times q^2/(1 + c15 q^2), c15 = 2 Delta B/(1 + Delta/2): the g term against g2
      R16, R17, R18 times q^2
    Built from eta and the locked coefficients only.
    """
    if kind in (None, "none"):
        return None
    if kind == "log":
        return (eta + p["b"]) ** ORDERS.to(eta)
    if kind == "balanced":
        q2 = (eta + p["kappa"]) ** 2
        c14 = (1 / p["Da"]) / (1 + p["Delta"])
        c15 = 2 * p["Delta"] * p["B"] / (1 + p["Delta"] / 2)
        return torch.cat([q2 * q2 / (1 + c14 * q2), q2 / (1 + c15 * q2), q2, q2, q2], 1)
    raise ValueError(f"unknown residual_scale {kind!r}")


def equation_mse(model, groups, p, scale=None):
    """Mean square of each of the 5 residuals over all points, as a (5,) tensor.

    groups is a list of (network index, points), so window networks each see their own points.
    scale names a fixed residual scale (residual_scale); None or "none" means plain residuals.
    """
    total, n = 0.0, 0
    for k, eta in groups:
        eta = eta.detach()
        z, H = states(model[k], eta)
        r = residuals(eta, z, H, p)
        s = residual_scale(eta, p, scale)
        if s is not None:
            r = r * s
        total = total + r.pow(2).sum(0)
        n += eta.shape[0]
    return total / n


def boundary_errors(model, p, L):
    """B1 to B12 as a (12,) tensor: the wall from the first network, eta = L from the last."""
    e0 = torch.zeros(1, 1)
    eL = torch.full((1, 1), float(L))
    return boundary(states(model[0], e0)[0], states(model[-1], eL)[0], p, L)


def interface_jumps(model):
    """B1 arm: jump of each of the 12 states at each inner window edge, as one flat tensor."""
    jumps = []
    for left, right in zip(model[:-1], model[1:]):
        e = torch.full((1, 1), float(right.lo))
        zl, zr = states(left, e)[0], states(right, e)[0]
        jumps += [a - b for a, b in zip(zl, zr)]
    return torch.cat(jumps).reshape(-1)


def interface_mse(model):
    """B1 arm: mean square jump of the 12 states between neighbouring window networks."""
    return interface_jumps(model).pow(2).mean()


def self_replay_mse(model, eta, frozen):
    """B3 arm: distance of the five outputs from the network's own outputs frozen at the last stage."""
    return (model[0](eta) - frozen).pow(2).mean()


class GradNormWeights:
    """Global weights from Algorithm 1, steps (c) and (d), of Wang, Sankaran, Wang and Perdikaris,
    "An expert's guide to training physics-informed neural networks", arXiv:2308.08468 (2023):

    lambda_hat_i = (sum over j of ||grad L_j||) / ||grad L_i||      (their eqs. 2.12 to 2.14)
    lambda_i    <- alpha lambda_i + (1 - alpha) lambda_hat_i        (eq. 2.15), alpha = 0.9

    So every weighted term pulls on the network with the same gradient norm. Term 0 is the
    equation loss; the rest are the soft boundary terms and, for B1, the interface term.

    cap bounds every weight (review H2: weights reached 3e9 to 7e10 in the pilots). Bounded
    weights are also what the convergence theory of adaptive weights needs.
    """

    def __init__(self, n, alpha, cap=None):
        self.w = torch.ones(n)
        self.alpha = alpha
        self.cap = cap

    def update(self, params, terms):
        norms = torch.stack([_grad_norm(t, params) for t in terms])
        hat = norms.sum() / (norms + 1e-30)
        self.w = self.alpha * self.w + (1 - self.alpha) * hat.detach()
        if self.cap is not None:
            self.w = self.w.clamp(max=self.cap)


def _grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.sqrt(sum(g.pow(2).sum() for g in grads if g is not None))
