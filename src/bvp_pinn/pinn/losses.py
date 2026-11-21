"""Loss terms, and the gradient-norm weights of Wang, Sankaran, Wang and Perdikaris (2023)."""
from __future__ import annotations

import torch

from .model import states
from .physics import boundary, residuals


def equation_mse(model, groups, p):
    """Mean square of each of the 5 residuals over all points, as a (5,) tensor.

    groups is a list of (network index, points), so window networks each see their own points.
    """
    total, n = 0.0, 0
    for k, eta in groups:
        eta = eta.detach().clone().requires_grad_(True)
        z, H = states(model[k], eta)
        total = total + residuals(eta, z, H, p).pow(2).sum(0)
        n += eta.shape[0]
    return total / n


def boundary_errors(model, p, L):
    """B1 to B12 as a (12,) tensor: the wall from the first network, eta = L from the last."""
    e0 = torch.zeros(1, 1).requires_grad_(True)
    eL = torch.full((1, 1), float(L)).requires_grad_(True)
    return boundary(states(model[0], e0)[0], states(model[-1], eL)[0], p, L)


def interface_mse(model):
    """B1 arm: mean square jump of the 12 states between neighbouring window networks."""
    jumps = []
    for left, right in zip(model[:-1], model[1:]):
        e = torch.full((1, 1), float(right.lo)).requires_grad_(True)
        zl, zr = states(left, e)[0], states(right, e)[0]
        jumps += [a - b for a, b in zip(zl, zr)]
    return torch.cat(jumps).pow(2).mean()


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
    """

    def __init__(self, n, alpha):
        self.w = torch.ones(n)
        self.alpha = alpha

    def update(self, params, terms):
        norms = torch.stack([_grad_norm(t, params) for t in terms])
        hat = norms.sum() / (norms + 1e-30)
        self.w = self.alpha * self.w + (1 - self.alpha) * hat.detach()


def _grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.sqrt(sum(g.pow(2).sum() for g in grads if g is not None))
