"""solve_bvp driver: wall-clustered meshes, consistent guesses, extension in L."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import brentq

from . import equations as eq


def wall_mesh(L, n, first_step):
    """n points on [0, L], exponentially stretched so the first step is first_step."""
    if first_step * (n - 1) >= L:
        return np.linspace(0.0, L, n)
    gap = lambda beta: L * np.expm1(beta / (n - 1)) / np.expm1(beta) - first_step
    beta = brentq(gap, 1e-9, 500.0)
    x = L * np.expm1(beta * np.linspace(0.0, 1.0, n)) / np.expm1(beta)
    x[0], x[-1] = 0.0, L
    return x


def smooth_guess(eta, p, rate, theta0):
    """Decaying profiles that satisfy B1, B2, B3, B5, B6 exactly at eta=0.

    Derivative states are the true derivatives of the value states, so the guess
    is consistent. B4 is left to the solver because it depends on the unknown wall
    temperature.
    """
    e = np.exp(-rate * eta)
    a = 1.0 / (1.0 + p["N1"] * rate + p["N1"] / p["kappa"])
    ratio = p["Nt"] / p["Nb"]
    f0 = p["fw"] / (p["Pr"] * p["Le"]) * ratio * theta0 * rate
    th = theta0 * e
    g = p["ell"] * a * rate * e
    return np.vstack([
        f0 + a / rate * (1 - e), a * e, -a * rate * e, a * rate**2 * e,
        g, -rate * g,
        th, -rate * th,
        -ratio * th, ratio * rate * th,
        e, -rate * e,
    ])


WALL_KEYS = ("f_0", "f2_0", "g_0", "g1_0", "theta_0", "th1_0", "phi_0", "ph1_0", "ch1_0", "f_L")
CORE_KEYS = tuple(k for k in WALL_KEYS if k != "ch1_0")  # chi has no L-independent limit


def far_L(L, settings):
    """L for eq.boundary when asymptotic far conditions are on, else None.

    settings["far_form"] ("local", the default, or "unit") picks the form of those conditions
    (equations.far_conditions); only the far-field error bar uses "unit"."""
    return L if settings.get("far_field") == "asymptotic" else None


@dataclass
class Result:
    L: float
    tol: float
    label: str
    status: int
    message: str
    x: np.ndarray
    y: np.ndarray
    sol: object
    seconds: float
    max_rms: float
    bc: np.ndarray
    wall: dict = field(default_factory=dict)
    rms: np.ndarray = None

    @property
    def ok(self):
        return self.status == 0

    def summary(self):
        return {
            "label": self.label, "L": self.L, "tol": self.tol, "status": self.status,
            "message": self.message, "nodes": int(self.x.size), "seconds": round(self.seconds, 3),
            "max_rms_residual": self.max_rms, "max_abs_bc": float(np.max(np.abs(self.bc))),
            "wall": self.wall,
        }


def wall_values(sol, L):
    za, zb = sol(0.0), sol(L)
    return {
        "f_0": za[0], "f2_0": za[2], "g_0": za[4], "g1_0": za[5], "theta_0": za[6],
        "th1_0": za[7], "phi_0": za[8], "ph1_0": za[9], "ch1_0": za[11], "f_L": zb[0],
    }


def solve_case(p, L, tol, settings, *, label, x=None, y=None, guess=None):
    """One solve_bvp call. Either (x, y) or a guess spec must be given."""
    if x is None:
        x = wall_mesh(L, settings["initial_nodes"], settings["first_step"])
        y = smooth_guess(x, p, guess["rate"], guess["theta0"])
    start = time.perf_counter()
    form = settings.get("far_form", "local")
    res = solve_bvp(lambda e, z: eq.rhs(e, z, p), lambda a, b: eq.boundary(a, b, p, L=far_L(L, settings), far_form=form),
                    x, y, tol=tol, bc_tol=tol, max_nodes=settings["max_nodes"])
    seconds = time.perf_counter() - start
    bc = eq.boundary(res.y[:, 0], res.y[:, -1], p, L=far_L(L, settings), far_form=form)
    wall = {k: float(v) for k, v in wall_values(res.sol, L).items()} if res.status == 0 else {}
    return Result(L=float(L), tol=float(tol), label=label, status=int(res.status), message=str(res.message),
                  x=res.x, y=res.y, sol=res.sol, seconds=seconds,
                  max_rms=float(np.max(res.rms_residuals)), bc=bc, wall=wall,
                  rms=res.rms_residuals)


def extend(result, L_new, settings):
    """Mesh and guess on [0, L_new] from a solution on [0, L_old].

    Beyond the old end each state keeps its far-end value. There f1, f2, g,
    theta, phi, chi are already near zero and f is nearly constant.
    """
    x = wall_mesh(L_new, settings["initial_nodes"], settings["first_step"])
    return x, result.sol(np.minimum(x, result.L))


def rel_diff(a, b, keys=WALL_KEYS, floor=1e-8):
    """Largest relative difference between two wall-value dicts over keys."""
    return max(abs(a[k] - b[k]) / max(abs(a[k]), abs(b[k]), floor) for k in keys)
