"""The networks (eta in, five fields out) and their derivatives by autograd."""
from __future__ import annotations

import math

import torch
from torch import nn

from .constraints import Constrained

ORDERS = (4, 2, 2, 2, 2)  # f is fourth order; g, theta, phi, chi are second order


class MLP(nn.Module):
    """tanh network from eta to (f, g, theta, phi, chi) on one interval [lo, hi].

    The input is mapped to [-1, 1] inside the network. With input_map="log" the map is
    log(1 + (eta - lo)/scale), which spreads the thin wall layer over more of the input range.
    """

    def __init__(self, width, depth, lo, hi, input_map="linear", scale=0.1):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 5))
        self.net = nn.Sequential(*layers)
        self.lo, self.hi, self.input_map, self.scale = lo, hi, input_map, scale

    def forward(self, eta):
        s = eta - self.lo
        if self.input_map == "log":
            s = torch.log1p(s / self.scale) / math.log1p((self.hi - self.lo) / self.scale)
        else:
            s = s / (self.hi - self.lo)
        return self.net(2 * s - 1)


def build(cfg, p):
    """One MLP on [0, L], or one MLP per window for the B1 arm, with the boundary layer of cfg["bc_mode"]."""
    edges = cfg["windows"] if cfg["structure"] == "windows" else [0.0, cfg["L"]]
    nets = [MLP(cfg["width"], cfg["depth"], a, b, cfg["input_map"], cfg["map_scale"])
            for a, b in zip(edges[:-1], edges[1:])]
    if cfg["bc_mode"] != "soft":
        nets = [Constrained(n, cfg["bc_mode"], p, cfg["L"], cfg["tfc_decay"]) for n in nets]
    return nn.ModuleList(nets).to(torch.get_default_dtype())


def states(net, eta):
    """All 12 states z and the 5 highest derivatives H at eta, each an (N, 1) column.

    z = [f, f1, f2, f3, g, g1, theta, th1, phi, ph1, chi, ch1] and H = [f4, g2, th2, ph2, ch2],
    the same order as reference/equations.py. eta must have requires_grad=True.
    """
    u = net(eta)
    z, H = [], []
    for k, order in enumerate(ORDERS):
        d = [u[:, k:k + 1]]
        for _ in range(order):
            d.append(torch.autograd.grad(d[-1].sum(), eta, create_graph=True)[0])
        z += d[:-1]
        H.append(d[-1])
    return z, H


def window_of(model, eta):
    """Index of the network that owns each point (always 0 for a single network)."""
    inner = torch.tensor([m.lo for m in model[1:]], dtype=eta.dtype, device=eta.device)
    return torch.bucketize(eta.reshape(-1), inner, right=True)


def predict(model, eta):
    """z and H on any grid, routing points to their window network. For evaluation only."""
    idx = window_of(model, eta)
    z = torch.zeros(12, eta.shape[0], dtype=eta.dtype, device=eta.device)
    H = torch.zeros(5, eta.shape[0], dtype=eta.dtype, device=eta.device)
    for k, net in enumerate(model):
        m = idx == k
        if m.any():
            e = eta[m].detach().clone().requires_grad_(True)
            zk, Hk = states(net, e)
            z[:, m] = torch.cat(zk, 1).T.detach()
            H[:, m] = torch.cat(Hk, 1).T.detach()
    return z, H
