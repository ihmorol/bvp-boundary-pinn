"""SPEC V1: compare equations.py with a separate SymPy transcription of docs/problem/equations.md."""
import numpy as np
import sympy as sp

from bvp_pinn.config import load_case
from bvp_pinn.reference import equations as eq

names = "eta f f1 f2 f3 f4 g g1 g2 th th1 th2 ph ph1 ph2 ch ch1 ch2"
syms = sp.symbols(names)
eta, f, f1, f2, f3, f4, g, g1, g2, th, th1, th2, ph, ph1, ph2, ch, ch1, ch2 = syms
P = {k: sp.Symbol(k) for k in load_case()[0]}
Delta, B, Da, kappa, ds, tw = P["Delta"], P["B"], P["Da"], P["kappa"], P["delta_star"], P["theta_w"]
lam, bm, eps, b, Pr, Le = P["lambda_m"], P["beta_m"], P["epsilon"], P["b"], P["Pr"], P["Le"]
Nr, Nt, Nb, Lb, Pe = P["Nr"], P["Nt"], P["Nb"], P["Lb"], P["Pe"]
q, h, c = eta + kappa, eta + b, tw - 1
A = 1 + c * th

R = [
    (1 + Delta) * (f4 + 2 * f3 / q - f2 / q**2 + f1 / q**3) + kappa / q * (f * f3 - f1 * f2)
    + kappa / q**2 * (f * f2 - f1**2) - kappa / q**3 * f * f1
    - sp.Rational(1, 2) * ds * ((eta * f3 + 3 * f2) + (sp.Rational(1, 2) * eta * f2 + f1) / q)
    - (f2 + f1 / q) / Da - Delta * (g2 + g1 / q) - 2 * bm / h**4 * ((1 / q - 4 / h) * th + th1),
    (1 + Delta / 2) * (g2 + g1 / q) - Delta * B * (f2 + f1 / q + 2 * g) - kappa / q * (f1 * g - f * g1)
    - ds / 2 * (eta * g1 + 3 * g),
    (th2 + th1 / q) / Pr + Nr / Pr * (A**3 * th1 / q + 3 * c * A**2 * th1**2 + A**3 * th2)
    + kappa / q * f * th1 + Nt / Pr * th1**2 + Nb / Pr * th1 * ph1
    + 2 * bm * lam * (th - eps) / (Pr * h**3) * (kappa * f / q * (1 - 2 / h**2) - 2 * f1 / h) - ds * eta / 2 * th1,
    (ph2 + ph1 / q) / (Le * Pr) + Nt / (Le * Pr * Nb) * (th2 + th1 / q) + kappa / q * f * ph1 - ds * eta / 2 * ph1,
    (ch2 + ch1 / q) / Lb + kappa / q * f * ph + Pe / Pr * (ch * ph2 + ch1 * ph1) - ds * eta / 2 * ph1,
]


def random_states(rng, n=50):
    e = rng.uniform(0.0, 8.0, n)
    z = rng.uniform(-1.0, 1.0, (12, n))
    z[6] = rng.uniform(0.0, 1.0, n)  # theta inside the range where A > 0
    H = rng.uniform(-3.0, 3.0, (5, n))
    return e, z, H


def test_residuals_match_sympy_locked_and_random_parameters():
    rng = np.random.default_rng(0)
    locked = load_case()[0]
    other = dict(locked, delta_star=0.37, kappa=0.6, b=0.4, theta_w=1.7)  # exercise every term, incl. unsteady ones
    for p in (locked, other):
        funcs = sp.lambdify(syms, [r.subs({P[k]: v for k, v in p.items()}) for r in R], "numpy")
        e, z, H = random_states(rng)
        f_, f1_, f2_, f3_, g_, g1_, th_, th1_, ph_, ph1_, ch_, ch1_ = z
        f4_, g2_, th2_, ph2_, ch2_ = H
        expected = np.array(funcs(e, f_, f1_, f2_, f3_, f4_, g_, g1_, g2_, th_, th1_, th2_,
                                  ph_, ph1_, ph2_, ch_, ch1_, ch2_))
        got = eq.residuals(e, z, H, p)
        np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-11)


def test_boundary_matches_hand_formulas():
    p = load_case()[0]
    rng = np.random.default_rng(1)
    za, zb = rng.uniform(-1, 1, 12), rng.uniform(-1, 1, 12)
    za[6] = 0.4
    tw_ = p["theta_w"]
    expected = [
        za[1] - 1 - p["N1"] * (za[2] - za[1] / p["kappa"]),
        za[0] - p["fw"] / (p["Pr"] * p["Le"]) * za[9],
        za[4] + p["ell"] * za[2],
        -za[7] - p["Bi"] * (1 - za[6]) - p["Rd"] / (tw_ - 1) * (tw_**4 - (1 + (tw_ - 1) * za[6])**4),
        za[9] + p["Nt"] / p["Nb"] * za[7],
        za[10] - 1,
        zb[1], zb[2], zb[4], zb[6], zb[8], zb[10],
    ]
    np.testing.assert_allclose(eq.boundary(za, zb, p), expected, rtol=1e-12, atol=1e-13)


def test_reduced_momentum_is_an_exact_rewrite_of_eq14():
    """q*R14 = dPhi/deta - 2*kappa*f1**2/q for any functions f, g, theta (symbolic, all parameters free)."""
    x = sp.Symbol("x")
    F, G, T = (sp.Function(n)(x) for n in ("F", "G", "T"))
    d = lambda u, k=1: sp.diff(u, x, k)
    sub = {eta: x, f: F, f1: d(F), f2: d(F, 2), f3: d(F, 3), f4: d(F, 4), g: G, g1: d(G), g2: d(G, 2),
           th: T, th1: d(T)}
    R14 = R[0].subs(sub, simultaneous=True)
    z = [F, d(F), d(F, 2), d(F, 3), G, d(G), T]
    Phi = eq.momentum_flux(x, z, P)
    assert sp.simplify(sp.expand((x + kappa) * R14 - (d(Phi) - 2 * kappa * d(F) ** 2 / (x + kappa)))) == 0


def test_reduced_momentum_holds_on_the_reference():
    """Independent of R14: the finite-difference slope of Phi on the branch-A reference equals 2*kappa*f1**2/q.

    Phi uses only f..f3, g1 and theta, so this checks the rewrite on real data. The 2e-5 gap at the
    wall is the error of second-order differences on the 4001-point grid, not of the identity.
    """
    from bvp_pinn.config import ROOT
    p = load_case()[0]
    data = np.load(ROOT / "reference" / "locked-branchA-L20-tol1e-08" / "reference.npz")
    e, z = data["eta"], data["z"]
    slope = np.gradient(eq.momentum_flux(e, z, p), e, edge_order=2)
    np.testing.assert_allclose(slope, 2 * p["kappa"] * z[1] ** 2 / (e + p["kappa"]), atol=1e-4)
