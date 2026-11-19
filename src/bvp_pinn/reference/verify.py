"""Solver-level checks from docs/SPEC_reference_solutions.md section 5 (V3, V8 and V12)."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import make_interp_spline

from . import equations as eq
from . import solver as sv

# Manufactured states: smooth, keep theta in (0, 0.5) so A = 1 + c*theta stays positive.
_A = np.array([0.2, 0.5, -0.3, 0.4, 0.1, -0.2, 0.3, -0.4, 0.2, 0.3, 0.6, -0.5])
_R = np.array([1.0, 1.2, 0.8, 1.5, 0.9, 1.1, 1.3, 0.7, 1.0, 1.4, 0.6, 1.2])
_C = np.array([0.1, 0, 0, 0, 0, 0, 0.1, 0, -0.05, 0, 0, 0])
_D = np.full(12, 0.05)


def manufactured(eta):
    e = np.exp(-_R[:, None] * eta)
    z = _A[:, None] * e + _C[:, None] + _D[:, None] * np.cos(eta)
    dz = -_R[:, None] * _A[:, None] * e - _D[:, None] * np.sin(eta)
    return z, dz


def mms_order(p, L, meshes):
    """V3: nodal error of solve_bvp on fixed uniform meshes against a manufactured solution.

    The forcing is dz_m - F(eta, z_m), and the boundary function is shifted by its
    value at z_m, so z_m solves the forced problem exactly. Meshes must double.
    """
    def fun(e, z):
        zm, dzm = manufactured(e)
        return eq.rhs(e, z, p) + dzm - eq.rhs(e, zm, p)

    end = manufactured(np.array([0.0, L]))[0]
    shift = eq.boundary(end[:, 0], end[:, 1], p)
    bc = lambda za, zb: eq.boundary(za, zb, p) - shift

    errors, kept = [], []
    for n in meshes:
        x = np.linspace(0.0, L, n)
        res = solve_bvp(fun, bc, x, manufactured(x)[0] + 0.02 * np.exp(-x),
                        tol=1e-12, bc_tol=1e-12, max_nodes=n)
        kept.append(bool(res.x.size == n))
        errors.append(float(np.max(np.abs(res.y - manufactured(res.x)[0]))))
    orders = [float(np.log2(errors[i] / errors[i + 1])) for i in range(len(errors) - 1)]
    return {"L": L, "meshes": list(meshes), "nodal_max_errors": errors, "observed_orders": orders,
            "mesh_kept_fixed": all(kept)}


def core_then_chi(p, L, tol, settings, guess, reference):
    """V8: solve the ten-state core (no chi), then chi as a linear BVP on the fixed core.

    R14 to R17 and B1 to B5, B7 to B11 do not contain chi, so the split is exact.
    Returns the largest relative difference from the monolithic reference.
    """
    core_bc_rows = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]

    def pad(z10):
        return np.vstack([z10, np.zeros((2,) + z10.shape[1:])])

    x = sv.wall_mesh(L, settings["initial_nodes"], settings["first_step"])
    y = sv.smooth_guess(x, p, guess["rate"], guess["theta0"])
    core = solve_bvp(lambda e, z: eq.rhs(e, pad(z), p)[:10],
                     lambda a, b: eq.boundary(pad(a[:, None])[:, 0], pad(b[:, None])[:, 0], p,
                                                 L=sv.far_L(L, settings))[core_bc_rows],
                     x, y[:10], tol=tol, bc_tol=tol, max_nodes=settings["max_nodes"])
    chi = solve_bvp(lambda e, w: eq.rhs(e, np.vstack([core.sol(e), w]), p)[10:],
                    lambda a, b: np.array([a[0] - 1.0, b[0]]),
                    core.x, np.vstack([np.exp(-guess["rate"] * core.x), -guess["rate"] * np.exp(-guess["rate"] * core.x)]),
                    tol=tol, bc_tol=tol, max_nodes=settings["max_nodes"])
    grid = sv.wall_mesh(L, 2001, settings["first_step"])
    full = reference.sol(grid)
    split = np.vstack([core.sol(grid), chi.sol(grid)])
    scale = np.maximum(np.max(np.abs(full), axis=1), 1e-8)
    profile_rel = float(np.max(np.max(np.abs(split - full), axis=1) / scale))
    split_wall = sv.wall_values(lambda s: np.concatenate([core.sol(s), chi.sol(s)]), L)
    return {"core_status": int(core.status), "chi_status": int(chi.status),
            "max_profile_rel_diff": profile_rel,
            "max_wall_rel_diff": float(sv.rel_diff(reference.wall, split_wall))}


SLOPE_LINKS = (("f", "f1"), ("f1", "f2"), ("f2", "f3"), ("f3", "f4"), ("g", "g1"), ("g1", "g2"),
               ("theta", "th1"), ("th1", "th2"), ("phi", "ph1"), ("ph1", "ph2"), ("chi", "ch1"), ("ch1", "ch2"))


def slope_check(eta, z, H):
    """Slope of each saved curve (quintic spline of the saved points) against the saved next state.

    For the last link of each field the target is the highest derivative that Eqs. 14 to 18 demand
    (H), so a small mismatch there means the saved curves satisfy the equations. Uses only the
    saved arrays, not the solver. Returns max |slope - target| / max |target| per link.
    """
    states = dict(zip(eq.STATE_NAMES, z)) | dict(zip(eq.HIGH_NAMES, H))
    out = {}
    for a, b in SLOPE_LINKS:
        slope = make_interp_spline(eta, states[a], k=5).derivative()(eta)
        out[f"{a}->{b}"] = float(np.max(np.abs(slope - states[b])) / np.max(np.abs(states[b])))
    return out
