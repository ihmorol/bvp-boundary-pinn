"""A fixed start profile added to the raw network output (cfg["start"] = "smooth").

It is the same analytic guess both reference solvers start from (reference/solver.py,
smooth_guess): every field is c + d*exp(-rate*eta). It is not reference data: it uses only the
locked parameters, a decay rate and a start wall temperature theta0.

Why: from a random start the network falls into a "cold wall" state (theta near 0,
docs/reviews/PINN_REVIEW_2026-11-21.md, H1). With this profile under the network, and the
network's last layer set to zero, training starts from a warm, decaying profile instead.
The constraint layer still makes all 12 boundary conditions exact on top of it.
"""
from __future__ import annotations

import torch
from torch import nn


class SmoothStart(nn.Module):
    """Fields f, g, theta, phi, chi = c + d*exp(-rate*eta), matching smooth_guess."""

    def __init__(self, p, rate=1.0, theta0=0.3):
        super().__init__()
        a = 1.0 / (1.0 + p["N1"] * rate + p["N1"] / p["kappa"])
        ratio = p["Nt"] / p["Nb"]
        f0 = p["fw"] / (p["Pr"] * p["Le"]) * ratio * theta0 * rate
        self.rate = rate
        self.register_buffer("c", torch.tensor([f0 + a / rate, 0.0, 0.0, 0.0, 0.0]))
        self.register_buffer("d", torch.tensor([-a / rate, p["ell"] * a * rate, theta0, -ratio * theta0, 1.0]))

    def jets(self, eta, K):
        """Values and the first K eta-derivatives, each an (N, 5) tensor."""
        e = torch.exp(-self.rate * eta)
        return [self.c + self.d * e] + [(-self.rate) ** k * self.d * e for k in range(1, K + 1)]

    def forward(self, eta):
        return self.jets(eta, 0)[0]
