"""Derivatives of the network in eta by forward propagation (Taylor mode).

The network has one input, eta. So the first K derivatives of every layer can be pushed
forward through the layers with the chain rule, in plain tensor arithmetic:

  linear layer   y_0 = W a_0 + b,  y_k = W a_k for k >= 1
  tanh layer     Faa di Bruno formula up to order 4 (tanh_jet below)

One pass gives all five outputs and all orders 0..K. Training then needs one ordinary
backward pass for the weights. model.states_autograd gets the same numbers with 12 nested
autograd.grad calls (one per output and order), each over a larger graph than the one before.
tests/test_pinn.py checks that both give the same values and the same weight gradients.
"""
from __future__ import annotations

import math

import torch

from .constraints import Constrained

K_MAX = 4  # f needs f''''; the other four fields need second derivatives


def tanh_jet(a):
    """Derivatives of tanh(a(eta)) from the derivatives a = [a, a', a'', ...] (up to order 4).

    With t = tanh(a) and D = 1 - t^2, the derivatives of tanh are
      tanh' = D, tanh'' = -2 t D, tanh''' = D (6 t^2 - 2), tanh'''' = 8 t D (2 - 3 t^2).
    """
    K = len(a) - 1
    t = torch.tanh(a[0])
    out = [t]
    if K == 0:
        return out
    D = 1 - t * t
    s1 = D
    out.append(s1 * a[1])
    if K == 1:
        return out
    s2 = -2 * t * D
    a1sq = a[1] * a[1]
    out.append(s2 * a1sq + s1 * a[2])
    if K == 2:
        return out
    s3 = D * (6 * t * t - 2)
    out.append(s3 * a1sq * a[1] + 3 * s2 * a[1] * a[2] + s1 * a[3])
    if K == 3:
        return out
    s4 = 8 * t * D * (2 - 3 * t * t)
    out.append(s4 * a1sq * a1sq + 6 * s3 * a1sq * a[2] + s2 * (4 * a[1] * a[3] + 3 * a[2] * a[2]) + s1 * a[4])
    return out


def input_jet(mlp, eta, K):
    """The network's mapped input x(eta) in [-1, 1] and its first K eta-derivatives."""
    s = eta - mlp.lo
    width = mlp.hi - mlp.lo
    if mlp.input_map == "log":
        a, C = mlp.scale, math.log1p(width / mlp.scale)
        x = [2 * torch.log1p(s / a) / C - 1]
        # d^k/deta^k log(a + s) = (-1)^(k-1) (k-1)! / (a + s)^k
        x += [2 / C * (-1) ** (k - 1) * math.factorial(k - 1) / (a + s) ** k for k in range(1, K + 1)]
        return x
    zero = torch.zeros_like(eta)
    return [2 * s / width - 1] + [torch.full_like(eta, 2 / width)] + [zero] * (K - 1)


def mlp_jets(mlp, eta, K=K_MAX):
    """Raw MLP outputs and their eta-derivatives: a list of K+1 tensors of shape (N, 5)."""
    a = input_jet(mlp, eta, K)
    n = eta.shape[0]
    for layer in mlp.net:
        if isinstance(layer, torch.nn.Linear):
            y = torch.cat(a, 0) @ layer.weight.T  # all orders in one matrix product
            a = list(y.split(n, 0))
            a[0] = a[0] + layer.bias              # the bias only moves the value
        else:
            a = tanh_jet(a)
    if getattr(mlp, "start", None) is not None:
        a = [x + y for x, y in zip(a, mlp.start.jets(eta, K))]
    return a


def _support_jet(d, n, tau, sign, K):
    """phi(d) = d^n exp(-d/tau) with d = sign*(eta - end), and its first K eta-derivatives.

    By the Leibniz rule, d^j phi/dd^j = sum_i C(j, i) n!/(n-i)! d^(n-i) (-1/tau)^(j-i) exp(-d/tau).
    """
    e = torch.exp(-d / tau)
    out = []
    for j in range(K + 1):
        term = 0.0
        for i in range(min(j, n) + 1):
            term = term + math.comb(j, i) * math.perm(n, i) * d ** (n - i) * (-1 / tau) ** (j - i)
        out.append(sign**j * term * e)
    return out


def constrained_jets(cnet, eta, K=K_MAX):
    """Same as mlp_jets for a Constrained network: raw output plus the boundary shapes."""
    u = mlp_jets(cnet.net, eta, K)
    if cnet.mode == "soft":
        return u
    ends = torch.tensor([[float(cnet.net.lo)], [float(cnet.net.hi)]], dtype=eta.dtype, device=eta.device)
    e = torch.stack(mlp_jets(cnet.net, ends, 2), -1)  # (2 ends, 5 fields, orders 0..2)
    miss = cnet.misses(e[0] if cnet.wall else None, e[1] if cnet.far else None)
    cols = [[u[j][:, k:k + 1] for k in range(5)] for j in range(K + 1)]
    for k, field in enumerate(cnet.fields):
        if miss[k] is None:
            continue
        coef = field.inv @ miss[k]
        for col, (end, n) in enumerate(field.supports):
            d, sign = (eta - field.lo, 1.0) if end == "lo" else (field.hi - eta, -1.0)
            for j, phi in enumerate(_support_jet(d, n, field.decay, sign, K)):
                cols[j][k] = cols[j][k] + phi * coef[col]
    return [torch.cat(c, 1) for c in cols]


def supports(net):
    """True if jets can differentiate this network (an MLP, or an MLP inside the constraint layer)."""
    inner = net.net if isinstance(net, Constrained) else net
    return hasattr(inner, "input_map") and isinstance(getattr(inner, "net", None), torch.nn.Sequential)


def states(net, eta):
    """z and H as in model.states_autograd, from one forward pass. eta needs no grad."""
    u = constrained_jets(net, eta) if isinstance(net, Constrained) else mlp_jets(net, eta)
    col = lambda j, k: u[j][:, k:k + 1]
    z = [col(0, 0), col(1, 0), col(2, 0), col(3, 0)]   # f, f1, f2, f3
    for k in range(1, 5):                              # g, theta, phi, chi and their slopes
        z += [col(0, k), col(1, k)]
    H = [col(4, 0)] + [col(2, k) for k in range(1, 5)]  # f4, g2, th2, ph2, ch2
    return z, H
