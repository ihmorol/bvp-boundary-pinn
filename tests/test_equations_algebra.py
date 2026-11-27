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


def test_far_conditions_follow_from_eqs_16_and_17():
    """Reference audit D3: the decay powers in far_conditions come from Eqs. 16 and 17 themselves.

    Far out (f = f_L, f' = f'' = 0, g = chi = 0), the linear part of residual_terms is applied to a
    power q^-k: the source part (theta = phi = 0) is subtracted and the amplitude is tiny, so the
    quadratic terms drop out. k is then found by root finding on R16 (phi = 0) and on R17
    (theta = 0), and compared with the m and mp that far_conditions uses. Finally the particular
    phi mode P q^-m forced by theta is solved from R17, and far_conditions must vanish on it.
    None of this uses the m formula, so the test is not circular.
    """
    from scipy.optimize import brentq
    L, fL, amp, eta = 20.0, 0.19, 1e-6, 1e4
    q = eta + P["kappa"]

    def lin(th, ph):  # (theta, th1, th2) and (phi, ph1, ph2) -> linear part of (R16, R17) at eta
        def R(t, f_):
            z = np.zeros(12)
            z[0], z[6], z[7], z[8], z[9] = fL, t[0], t[1], f_[0], f_[1]
            out = eq.residual_terms(eta, z, [0.0, 0.0, t[2], f_[2], 0.0], P)
            return np.array([out[2], out[3]])
        return R(th, ph) - R((0, 0, 0), (0, 0, 0))

    def mode(k, c):  # c q^-k and its first two derivatives
        return c * q**-k, -k * c * q ** (-k - 1), k * (k + 1) * c * q ** (-k - 2)

    m = brentq(lambda k: lin(mode(k, amp), (0, 0, 0))[0] / (amp * q ** (-k - 2)), 0.01, 2.0)
    mp = brentq(lambda k: lin((0, 0, 0), mode(k, amp))[1] / (amp * q ** (-k - 2)), 0.01, 2.0)

    qL, th_L = L + P["kappa"], 1e-9  # tiny theta at L, so A(theta_L) = 1 as in the expansion above
    zb = np.zeros(12)
    zb[0], zb[6] = fL, th_L
    m_code = eq.far_conditions(zb, L, P)[0] * qL / th_L       # B10 = theta' + m theta / q with theta' = 0
    zb[6], zb[8] = 0.0, 1e-9
    mp_code = eq.far_conditions(zb, L, P)[1] * qL / 1e-9      # B11 with theta = phi' = 0
    np.testing.assert_allclose([m_code, mp_code], [m, mp], rtol=1e-5)

    th = mode(m, amp)  # theta forces a phi part P q^-m; R17 is linear in P
    r0, r1 = lin(th, (0, 0, 0))[1], lin(th, mode(m, 1.0))[1]
    Pc = -r0 / (r1 - r0)
    E = 0.3 * amp
    phi = [a + b for a, b in zip(mode(mp, E), mode(m, Pc))]
    assert abs(lin(th, phi)[1]) < 1e-6 * abs(r0)  # theta and phi together solve Eq. 17
    zb = np.zeros(12)
    zb[0] = fL
    zb[6], zb[7] = amp * qL**-m, -m * amp * qL ** (-m - 1)
    zb[8] = E * qL**-mp + Pc * qL**-m
    zb[9] = -mp * E * qL ** (-mp - 1) - m * Pc * qL ** (-m - 1)
    b10, b11 = eq.far_conditions(zb, L, P)
    assert abs(b10) < 1e-6 * abs(zb[7]) and abs(b11) < 1e-6 * abs(zb[9])
