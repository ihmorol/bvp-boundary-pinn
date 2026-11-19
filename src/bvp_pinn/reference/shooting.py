"""Second, independent reference: multiple shooting with a Runge-Kutta integrator.

solve_bvp uses 4th-order collocation on one adaptive mesh. Here [0, L] is cut into segments;
each segment is an initial value problem integrated by SciPy's adaptive 8th-order
Dormand-Prince method (DOP853), and Newton makes the segments join and the boundary
conditions hold. Plain single shooting fails here: some modes grow like exp(1.8*eta).
Only the equations are shared with solve_bvp, and those are checked against SymPy and the source.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.integrate import solve_ivp

from . import equations as eq
from . import solver as sv


def _flow(fun, a, b, y0, rtol, dense=False):
    r = solve_ivp(fun, (a, b), y0, method="DOP853", rtol=rtol, atol=rtol * 1e-2, dense_output=dense)
    if not r.success:
        raise RuntimeError(r.message)
    return r


def multiple_shooting(fun, bc, nodes, S, rtol=1e-11, tol=1e-11, max_iter=40):
    """Solve y' = fun(x, y), bc(y(a), y(b)) = 0 on segments given by nodes.

    S is (n, M), the guessed state at the left end of each segment. Returns (sol, iterations,
    largest mismatch); sol(x) evaluates the DOP853 dense output of the segment holding x.
    """
    n, M = S.shape
    S = np.array(S, dtype=float)

    def mismatch(S):
        E = np.column_stack([_flow(fun, nodes[i], nodes[i + 1], S[:, i], rtol).y[:, -1] for i in range(M)])
        return np.concatenate([(E[:, :-1] - S[:, 1:]).T.ravel(), bc(S[:, 0], E[:, -1])]), E

    F, E = mismatch(S)
    for it in range(max_iter):
        if np.max(np.abs(F)) < tol:
            break
        # Jacobian by finite differences, one 12x12 block per segment.
        J = np.zeros((n * M, n * M))
        for i in range(M):
            Phi = np.empty((n, n))
            for j in range(n):
                d = np.zeros(n)
                d[j] = 1e-7 * max(1.0, abs(S[j, i]))
                Phi[:, j] = (_flow(fun, nodes[i], nodes[i + 1], S[:, i] + d, rtol).y[:, -1] - E[:, i]) / d[j]
            if i < M - 1:
                J[n * i:n * i + n, n * i:n * i + n] = Phi
                J[n * i:n * i + n, n * i + n:n * i + 2 * n] = -np.eye(n)
        b0 = bc(S[:, 0], E[:, -1])
        Ba, Bb = np.empty((n, n)), np.empty((n, n))
        for j in range(n):
            d = np.zeros(n)
            d[j] = 1e-7
            Ba[:, j] = (bc(S[:, 0] + d, E[:, -1]) - b0) / 1e-7
            Bb[:, j] = (bc(S[:, 0], E[:, -1] + d) - b0) / 1e-7
        J[n * (M - 1):, :n] += Ba
        J[n * (M - 1):, n * (M - 1):] += Bb @ Phi
        step = np.linalg.solve(J, F).reshape(M, n).T
        for _ in range(20):  # halve the Newton step until the mismatch falls
            try:
                F_new, E_new = mismatch(S - step)
            except RuntimeError:
                F_new = None
            if F_new is not None and np.max(np.abs(F_new)) < np.max(np.abs(F)):
                break
            step /= 2
        else:
            raise RuntimeError("Newton line search failed")
        S, F, E = S - step, F_new, E_new
    else:
        raise RuntimeError("Newton did not converge")

    dense = [_flow(fun, nodes[i], nodes[i + 1], S[:, i], rtol, dense=True).sol for i in range(M)]

    def sol(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        seg = np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, M - 1)
        out = np.empty((n, x.size))
        for i in np.unique(seg):
            out[:, seg == i] = dense[i](x[seg == i])
        return out

    return sol, it, float(np.max(np.abs(F)))


def rk_reference(p, L, settings, guess):
    """RK multiple-shooting solution of the locked case, started from the smooth guess only.

    Also solves at a looser rtol to estimate the integrator's own error.
    """
    r = settings["rk"]
    nodes = sv.wall_mesh(L, r["segments"] + 1, r["first_segment"])
    fun = lambda x, y: eq.rhs(x, y, p)
    bc = lambda a, b: eq.boundary(a, b, p, L=sv.far_L(L, settings))
    S0 = sv.smooth_guess(nodes[:-1], p, guess["rate"], guess["theta0"])
    start = time.perf_counter()
    sol, iterations, mismatch = multiple_shooting(fun, bc, nodes, S0, rtol=r["rtol"])
    seconds = time.perf_counter() - start
    loose = multiple_shooting(fun, bc, nodes, S0, rtol=r["rtol_check"])[0]
    grid = sv.wall_mesh(L, 2001, settings["first_step"])
    tight = sol(grid)
    return sol, {
        "method": "multiple shooting with DOP853", "segments": r["segments"], "rtol": r["rtol"],
        "start": "smooth guess, independent of solve_bvp", "newton_iterations": iterations,
        "max_mismatch": mismatch, "seconds": round(seconds, 1),
        "self_diff_vs_rtol_check": float(np.max(np.abs(tight - loose(grid))) / np.max(np.abs(tight))),
    }
