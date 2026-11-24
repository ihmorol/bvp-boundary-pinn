"""Boundary conditions built into the network output (theory of functional connections).

The output is the raw network N plus fixed smooth shapes, each times how far N misses one
condition. So those conditions hold for any network weights. Design and sources:
docs/pinn/BC_ENFORCEMENT_REVIEW.md section 7.

Modes:
  soft    no change; all 12 conditions are loss terms
  linear  the 9 linear conditions exact (B1 B2 B3 B5 B6 B7 B8 B9 B12); B4, B10, B11 stay in the loss
  exact   all 12 exact. B4, B10, B11 each fix one slope from values the layer leaves alone (our own step)
"""
from __future__ import annotations

import torch
from torch import nn

from ..reference.equations import far_conditions

SOFT_BCS = {"soft": list(range(12)), "linear": [3, 9, 10], "exact": []}  # indices into B1..B12
F, G, TH, PH, CH = range(5)


class Constrained(nn.Module):
    """Wraps one MLP. Each window network owns the wall end, the far end, both or neither."""

    def __init__(self, net, mode, p, L, decay=0.5):
        super().__init__()
        self.net, self.mode, self.p, self.L = net, mode, p, L
        self.lo, self.hi = net.lo, net.hi
        self.wall, self.far = net.lo == 0.0, net.hi == L
        self.fields = [Field(rows, net.lo, net.hi, decay) for rows in self.functionals()]

    def functionals(self):
        """For each field, the conditions it carries: (end, {derivative order: coefficient})."""
        p, rows = self.p, [[] for _ in range(5)]
        if self.mode == "soft":
            return rows
        if self.wall:
            rows[F] += [("lo", {0: 1.0}), ("lo", {1: 1 + p["N1"] / p["kappa"], 2: -p["N1"]})]  # B2, B1
            rows[G] += [("lo", {0: 1.0})]                                                   # B3
            rows[PH] += [("lo", {1: 1.0})]                                                  # B5
            rows[CH] += [("lo", {0: 1.0})]                                                  # B6
            if self.mode == "exact":
                rows[TH] += [("lo", {0: 1.0}), ("lo", {1: 1.0})]                            # keep, B4
                rows[PH] = [("lo", {0: 1.0})] + rows[PH]                                     # keep
        if self.far:
            rows[F] += [("hi", {1: 1.0}), ("hi", {2: 1.0})]                                 # B7, B8
            rows[G] += [("hi", {0: 1.0})]                                                   # B9
            rows[CH] += [("hi", {0: 1.0})]                                                  # B12
            if self.mode == "exact":
                rows[TH] += [("hi", {0: 1.0}), ("hi", {1: 1.0})]                            # keep, B10
                rows[PH] += [("hi", {0: 1.0}), ("hi", {1: 1.0})]                            # keep, B11
        return rows

    def forward(self, eta):
        N = self.net(eta)
        lo = end_values(self.net, self.net.lo) if self.wall else None
        hi = end_values(self.net, self.net.hi) if self.far else None
        miss = self.misses(lo, hi)
        cols = [N[:, k:k + 1] + self.fields[k].shape(eta, miss[k]) for k in range(5)]
        return torch.cat(cols, 1)

    def misses(self, lo, hi):
        """Target minus the raw network's value, for every carried condition, in build order."""
        p, fl = self.p, self.fields
        miss = [[] for _ in range(5)]
        if self.mode == "soft":
            return miss
        th1_0 = None
        if self.wall:
            th0 = lo[TH, 0]
            if self.mode == "exact":  # B4 solved for theta'(0); stable form of equations.b4_stable
                Aw = 1 + (p["theta_w"] - 1) * th0
                Qw = (1 - th0) * (p["theta_w"] + Aw) * (p["theta_w"] ** 2 + Aw**2)
                th1_0 = -p["Bi"] * (1 - th0) - p["Rd"] * Qw
                miss[TH] += [torch.zeros(()), th1_0 - lo[TH, 1]]
                miss[PH] += [torch.zeros(())]
            else:
                th1_0 = lo[TH, 1]
            ph1_0 = -p["Nt"] / p["Nb"] * th1_0                                              # B5
            miss[F] += [p["fw"] / (p["Pr"] * p["Le"]) * ph1_0 - lo[F, 0],                    # B2
                        1 - ((1 + p["N1"] / p["kappa"]) * lo[F, 1] - p["N1"] * lo[F, 2])]   # B1
            miss[PH] += [ph1_0 - lo[PH, 1]]
            miss[CH] += [1 - lo[CH, 0]]                                                     # B6
        if self.far:
            miss[F] += [-hi[F, 1], -hi[F, 2]]                                               # B7, B8
        f_built = fl[F].value_at  # f is complete now; g and the far theta, phi need its end values
        if self.wall:
            f2_0 = lo[F, 2] + f_built("lo", 2, miss[F])
            miss[G] += [-p["ell"] * f2_0 - lo[G, 0]]                                         # B3
        if self.far:
            miss[G] += [-hi[G, 0]]                                                          # B9
            miss[CH] += [-hi[CH, 0]]                                                        # B12
            if self.mode == "exact":  # B10, B11: zero when theta'(L), phi'(L) take these values
                fL = hi[F, 0] + f_built("hi", 0, miss[F])
                zb = [fL, None, None, None, None, None, hi[TH, 0], 0.0, hi[PH, 0], 0.0]
                b10, b11 = far_conditions(zb, self.L, p)  # with zero slopes these are -(target slopes)
                miss[TH] += [torch.zeros(()), -b10 - hi[TH, 1]]
                miss[PH] += [torch.zeros(()), -b11 - hi[PH, 1]]
        return [torch.stack(m) if m else None for m in miss]


class Field:
    """Correction shapes for one field: phi_j with C_i[phi_j] = 1 if i = j, else 0."""

    def __init__(self, rows, lo, hi, decay):
        self.rows, self.lo, self.hi, self.decay = rows, lo, hi, decay
        if not rows:
            return
        count = {"lo": 0, "hi": 0}
        self.supports = []
        for end, _ in rows:  # powers 0, 1, 2, ... of the distance to the row's end
            self.supports.append((end, count[end]))
            count[end] += 1
        M = torch.stack([torch.stack([self._apply(row, s) for s in self.supports]) for row in rows])
        self.inv = torch.linalg.inv(M).detach()
        # Values and first two derivatives of every phi_j at both ends, for building end values.
        self.ends = {end: torch.stack([torch.stack([self._phi_derivative(j, end, d) for j in range(len(rows))])
                                       for d in range(3)]).detach() for end in ("lo", "hi")}

    def support(self, eta, end, n):
        d = eta - self.lo if end == "lo" else self.hi - eta
        return d**n * torch.exp(-d / self.decay)

    def _derivative(self, end, n, at, order):
        x = torch.full((1, 1), float(getattr(self, at))).requires_grad_(True)
        y = self.support(x, end, n)
        for _ in range(order):
            y = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        return y.reshape(())

    def _apply(self, row, s):
        end, coef = row
        return sum(c * self._derivative(s[0], s[1], end, d) for d, c in coef.items())

    def _phi_derivative(self, j, at, order):
        return sum(self.inv[i, j] * self._derivative(e, n, at, order) for i, (e, n) in enumerate(self.supports))

    def shape(self, eta, miss):
        if miss is None:
            return 0.0
        S = torch.cat([self.support(eta, e, n) for e, n in self.supports], 1)
        return S @ (self.inv @ miss).reshape(-1, 1)

    def value_at(self, end, order, miss):
        """Correction to the order-th derivative at one end. miss must hold all of this field's misses."""
        return self.ends[end][order] @ torch.stack(miss) if miss else 0.0


def end_values(net, x):
    """Raw network value and first two derivatives at one point, as a (5, 3) tensor."""
    e = torch.full((1, 1), float(x)).requires_grad_(True)
    u = net(e)
    cols = []
    for k in range(5):
        d = [u[:, k:k + 1]]
        for _ in range(2):
            d.append(torch.autograd.grad(d[-1].sum(), e, create_graph=True)[0])
        cols.append(torch.cat(d, 1).reshape(3))
    return torch.stack(cols)
