"""Eqs. 14 to 19 on PyTorch tensors.

No second copy of the equations: this calls the same functions as the classical reference
(reference/equations.py), which tests/test_equations_sympy.py checks against SymPy. There is no
way to add a term or a source here: every loss, optimizer and check solves Eqs. 14 to 18 with
B1 to B12 as given (AGENTS.md section 3).
"""
from __future__ import annotations

import torch

from ..reference import equations as eq


def residuals(eta, z, H, p):
    """R14 to R18 as an (N, 5) tensor.

    Each equation is divided by the coefficient of its highest derivative, so all five are in
    the units of f'''', g'', theta'', phi'', chi''. This only rescales; the zero set is unchanged.
    """
    R = eq.residual_terms(eta, z, H, p)
    coef = (1 + p["Delta"], 1 + p["Delta"] / 2, eq.thermal_coefficient(z, p),
            1 / (p["Le"] * p["Pr"]), 1 / p["Lb"])
    return torch.cat([r / c for r, c in zip(R, coef)], 1)


def boundary(z0, zL, p, L):
    """B1 to B12 as a (12,) tensor. z0 and zL are the 12 states at eta = 0 and eta = L.

    Same conditions as the reference: asymptotic far conditions for theta and phi at L.
    """
    return torch.cat([b.reshape(1) for b in eq.boundary_terms(z0, zL, p, L=L)])
