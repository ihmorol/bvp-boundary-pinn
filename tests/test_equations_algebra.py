"""SPEC V2 and equations.md integrity tests 3 to 5."""
import numpy as np

from bvp_pinn.config import load_case
from bvp_pinn.reference import equations as eq
from bvp_pinn.reference import solver as sv

P = load_case()[0]


def test_residual_is_linear_in_highest_derivatives():
    rng = np.random.default_rng(2)
    e = rng.uniform(0, 8, 40)
    z = rng.uniform(-1, 1, (12, 40))
    z[6] = rng.uniform(0, 1, 40)
    H = rng.uniform(-3, 3, (5, 40))
    lhs = eq.residuals(e, z, H, P) - eq.residuals(e, z, np.zeros_like(H), P)
    MH = np.einsum("nij,jn->in", eq.highest_matrix(z, P), H)
    np.testing.assert_allclose(lhs, MH, rtol=1e-11, atol=1e-11)


def test_highest_derivatives_zero_the_residual():
    rng = np.random.default_rng(3)
    e = rng.uniform(0, 8, 40)
    z = rng.uniform(-1, 1, (12, 40))
    z[6] = rng.uniform(0, 1, 40)
    H = eq.highest_derivatives(e, z, P)
    np.testing.assert_allclose(eq.residuals(e, z, H, P), 0, atol=1e-10)
    dz = eq.rhs(e, z, P)
    assert dz.shape == (12, 40)
    np.testing.assert_array_equal(dz[[0, 1, 2, 4, 6, 8, 10]], z[[1, 2, 3, 5, 7, 9, 11]])


def test_b4_stable_equals_original():
    th = np.linspace(-0.5, 1.2, 30)
    th1 = np.linspace(-2, 2, 30)
    np.testing.assert_allclose(eq.b4_stable(th, th1, P), eq.b4_original(th, th1, P), rtol=1e-12, atol=1e-12)



def test_smooth_guess_satisfies_wall_conditions_except_b4():
    x = sv.wall_mesh(10.0, 400, 1e-3)
    assert np.isclose(x[1] - x[0], 1e-3) and x[0] == 0 and x[-1] == 10
    y = sv.smooth_guess(x, P, 1.0, 0.5)
    bc = eq.boundary(y[:, 0], y[:, -1], P)
    np.testing.assert_allclose(bc[[0, 1, 2, 4, 5]], 0, atol=1e-14)
    np.testing.assert_allclose(np.gradient(y[0], x), y[1], atol=1e-3)


def test_far_conditions_vanish_on_decaying_modes():
    """theta = C q^-m and phi = E q^-mp + P q^-m (the far-field solutions) satisfy both conditions."""
    L, fL, C, E = 20.0, 0.19, 0.3, -0.2
    q = L + P["kappa"]
    th = C * q ** -1.0  # placeholder to get A; m then fixed below and theta rebuilt with it
    for _ in range(50):  # m depends on A(theta_L); iterate to the fixed point
        A = 1 + (P["theta_w"] - 1) * th
        m = P["Pr"] * P["kappa"] * fL / (1 + P["Nr"] * A**3)
        th = C * q**-m
    mp = P["Le"] * P["Pr"] * P["kappa"] * fL
    Pc = -(P["Nt"] / P["Nb"]) * C * m / (m - mp)
    zb = np.zeros(12)
    zb[0], zb[6], zb[7] = fL, th, -m * th / q
    zb[8] = E * q**-mp + Pc * q**-m
    zb[9] = -mp * E * q ** (-mp - 1) - m * Pc * q ** (-m - 1)
    np.testing.assert_allclose(eq.far_conditions(zb, L, P), 0, atol=1e-14)
