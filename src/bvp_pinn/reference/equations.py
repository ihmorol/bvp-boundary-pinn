"""Eqs. 14 to 19 in NumPy, in the first-order form used by solve_bvp.

Transcribed from docs/problem/equations.md. tests/test_equations_sympy.py compares
this file with a separate SymPy transcription, so a copying error here has to
be made twice in the same way to pass. Both transcriptions start from equations.md,
so that test checks the code against equations.md, not against the source page. The
check against the source page is the visual one recorded in equations.md (by
eye, on the enlarged page; AGENTS.md section 3).

State order (12 rows): f, f1, f2, f3, g, g1, theta, th1, phi, ph1, chi, ch1.
Highest derivatives (5 rows): f4, g2, th2, ph2, ch2.
Arrays may have a trailing point axis: z is (12, N), H is (5, N).
"""
from __future__ import annotations

import numpy as np

STATE_NAMES = ("f", "f1", "f2", "f3", "g", "g1", "theta", "th1", "phi", "ph1", "chi", "ch1")
HIGH_NAMES = ("f4", "g2", "th2", "ph2", "ch2")
BC_NAMES = tuple(f"B{i}" for i in range(1, 13))


def residuals(eta, z, H, p):
    """Return R14 to R18 stacked as (5, ...). Each target is zero."""
    return np.stack(residual_terms(eta, z, H, p))


# What each term is, and whether it can go (docs/pinn/RUNTIME_STRATEGY.md section 2 has the sizes).
#
# Every term below is in the supervisor-confirmed Eqs. 14 to 18, so none is removed.
# - delta_star terms (one per equation): exactly zero at the locked case (delta_star = 0). They
#   need no extra derivative (f3, f2, g1, th1, ph1 exist anyway), so keeping them costs almost
#   nothing. They stay so the code still matches the printed equation for any delta_star.
# - No other term is negligible. On the branch-A reference the smallest ones (kappa/q*f*phi in
#   Eq. 18, the f*f1 term in Eq. 14) are 4e-4 to 1e-3 of the largest. That is still far above
#   the residual we need (about 1e-6 or lower), and far from the wall they shape the slow tails.
# - The derivative orders are what the physics needs: f'''' comes from removing the pressure
#   from the two momentum equations of a curved sheet. reduced_momentum below puts it back as
#   one first-order unknown, so f needs only f'''. g, theta, phi, chi are second order (diffusion).
# - In Eq. 14, the dipole force 2*beta_m/h**4*(...) is about 1250 at the wall and balances
#   (1+Delta)*f''''. That thin, strong wall layer is a training difficulty, not a code artifact.


def residual_terms(eta, z, H, p):
    """R14 to R18 as a tuple. Plain arithmetic only, so PyTorch tensors work too (the PINN uses this)."""
    f, f1, f2, f3, g, g1, th, th1, ph, ph1, ch, ch1 = z
    f4, g2, th2, ph2, ch2 = H
    kappa, Delta, ds = p["kappa"], p["Delta"], p["delta_star"]
    Pr, Le, Nt, Nb, Nr, bm = p["Pr"], p["Le"], p["Nt"], p["Nb"], p["Nr"], p["beta_m"]
    q = eta + kappa
    h = eta + p["b"]
    c = p["theta_w"] - 1.0
    A = 1.0 + c * th

    R14 = ((1 + Delta) * (f4 + 2 * f3 / q - f2 / q**2 + f1 / q**3)
           + kappa / q * (f * f3 - f1 * f2)
           + kappa / q**2 * (f * f2 - f1**2)
           - kappa / q**3 * f * f1
           - 0.5 * ds * ((eta * f3 + 3 * f2) + (0.5 * eta * f2 + f1) / q)
           - (f2 + f1 / q) / p["Da"]
           - Delta * (g2 + g1 / q)
           - 2 * bm / h**4 * ((1 / q - 4 / h) * th + th1))
    R15 = ((1 + Delta / 2) * (g2 + g1 / q)
           - Delta * p["B"] * (f2 + f1 / q + 2 * g)
           - kappa / q * (f1 * g - f * g1)
           - ds / 2 * (eta * g1 + 3 * g))
    R16 = ((th2 + th1 / q) / Pr
           + Nr / Pr * (A**3 * th1 / q + 3 * c * A**2 * th1**2 + A**3 * th2)
           + kappa / q * f * th1
           + Nt / Pr * th1**2
           + Nb / Pr * th1 * ph1
           + 2 * bm * p["lambda_m"] * (th - p["epsilon"]) / (Pr * h**3)
           * (kappa * f / q * (1 - 2 / h**2) - 2 * f1 / h)
           - ds * eta / 2 * th1)
    R17 = ((ph2 + ph1 / q) / (Le * Pr)
           + Nt / (Le * Pr * Nb) * (th2 + th1 / q)
           + kappa / q * f * ph1
           - ds * eta / 2 * ph1)
    # Printed as confirmed by the supervisor, including kappa/q*f*phi and the ph1 term.
    R18 = ((ch2 + ch1 / q) / p["Lb"]
           + kappa / q * f * ph
           + p["Pe"] / Pr * (ch * ph2 + ch1 * ph1)
           - ds * eta / 2 * ph1)
    return R14, R15, R16, R17, R18


def momentum_flux(eta, z, p):
    """Phi in q*R14 = dPhi/deta - 2*kappa*f1**2/q. Uses f up to f3 only, plus g1 and theta.

    Each part of R14 times q is an exact derivative except the kappa*f1**2 part:
      (1+Delta)*q*(f4 + 2f3/q - f2/q^2 + f1/q^3) = d/deta[(1+Delta)*(q f3 + f2 - f1/q)]
      q*(convective terms)                      = d/deta[kappa*(f f2 - f1^2 + f f1/q)] - 2 kappa f1^2/q
      -q*(f2 + f1/q)/Da                         = d/deta[-q f1/Da]
      -Delta*q*(g2 + g1/q)                      = d/deta[-Delta q g1]
      -2 beta_m q/h^4 ((1/q - 4/h) theta + th1) = d/deta[-2 beta_m q theta/h^4]
      delta_star part                           = d/deta[-delta_star/2 (q eta f2 + 2 q f1 - eta f1/2 - f/2)]
    tests/test_equations_sympy.py proves the identity symbolically.
    """
    f, f1, f2, f3, g, g1, th = z[:7]
    kappa, Delta, ds = p["kappa"], p["Delta"], p["delta_star"]
    q, h = eta + kappa, eta + p["b"]
    return ((1 + Delta) * (q * f3 + f2 - f1 / q)
            + kappa * (f * f2 - f1**2 + f * f1 / q)
            - 0.5 * ds * (q * eta * f2 + 2 * q * f1 - 0.5 * eta * f1 - 0.5 * f)
            - q * f1 / p["Da"]
            - Delta * q * g1
            - 2 * p["beta_m"] * q * th / h**4)


def reduced_momentum(eta, z, P, P1, p):
    """Eq. 14 as two lower-order equations, with a pressure-like unknown P(eta).

      Ra = Phi(eta, f..f3, g1, theta) - P        (third order in f)
      Rb = P' - 2*kappa*f1**2/q                  (first order in P)

    Equivalent to R14 = 0: if Ra = Rb = 0 then q*R14 = Phi' - 2 kappa f1^2/q = P' - 2 kappa f1^2/q = 0,
    and q = eta + kappa >= kappa > 0, so R14 = 0. Conversely, from a solution of R14 set P = Phi.
    P gets no boundary condition; B1, B2, B7, B8 still fix f, so the count of conditions is unchanged.
    Both are divided by (1+Delta)*q, so Rb equals R14/(1+Delta) (the PINN's scaling) when P = Phi.
    """
    q = eta + p["kappa"]
    s = (1 + p["Delta"]) * q
    return (momentum_flux(eta, z, p) - P) / s, (P1 - 2 * p["kappa"] * z[1] ** 2 / q) / s


def thermal_coefficient(z, p):
    """(1 + Nr*A^3)/Pr, the th2 coefficient. It must stay away from zero."""
    A = 1.0 + (p["theta_w"] - 1.0) * z[6]
    return (1.0 + p["Nr"] * A**3) / p["Pr"]


def highest_matrix(z, p):
    """M with R = M*H + R(H=0). Shape (..., 5, 5)."""
    th, ch = np.asarray(z[6]), np.asarray(z[10])
    M = np.zeros(th.shape + (5, 5))
    D = p["Delta"]
    M[..., 0, 0] = 1 + D
    M[..., 0, 1] = -D
    M[..., 1, 1] = 1 + D / 2
    M[..., 2, 2] = thermal_coefficient(z, p)
    M[..., 3, 2] = p["Nt"] / (p["Le"] * p["Pr"] * p["Nb"])
    M[..., 3, 3] = 1 / (p["Le"] * p["Pr"])
    M[..., 4, 3] = p["Pe"] * ch / p["Pr"]
    M[..., 4, 4] = 1 / p["Lb"]
    return M


def highest_derivatives(eta, z, p):
    """Solve M*H = -R(H=0) pointwise. No explicit inverse is formed."""
    z = np.asarray(z, dtype=float)
    r0 = residuals(eta, z, np.zeros((5,) + z.shape[1:]), p)
    M = highest_matrix(z, p)
    H = np.linalg.solve(M, -np.moveaxis(r0, 0, -1)[..., None])[..., 0]
    return np.moveaxis(H, -1, 0)


def rhs(eta, z, p):
    """dz/deta for solve_bvp. z is (12, N)."""
    H = highest_derivatives(eta, z, p)
    return np.stack([z[1], z[2], z[3], H[0], z[5], H[1], z[7], H[2], z[9], H[3], z[11], H[4]])


def b4_original(th0, th1_0, p):
    c = p["theta_w"] - 1.0
    return (-th1_0 - p["Bi"] * (1 - th0)
            - p["Rd"] / c * (p["theta_w"]**4 - (1 + c * th0)**4))


def b4_stable(th0, th1_0, p):
    """Same as b4_original after dividing the quartic difference by c; safe when c is small."""
    Aw = 1.0 + (p["theta_w"] - 1.0) * th0
    Qw = (1 - th0) * (p["theta_w"] + Aw) * (p["theta_w"]**2 + Aw**2)
    return -th1_0 - p["Bi"] * (1 - th0) - p["Rd"] * Qw


def far_conditions(zb, L, p, form="local"):
    """Asymptotic forms of theta(inf)=0 and phi(inf)=0, imposed at eta=L (Lentini and Keller, 1980).

    Where f1, f2 and g have died out (f -> f_L), Eqs. 16 and 17 reduce to
    theta'' + (1+m)*theta'/q = 0 and phi'' + (1+mp)*phi'/q = -(Nt/Nb)*(theta'' + theta'/q),
    with m = Pr*kappa*f_L/(1+Nr*A^3) and mp = Le*Pr*kappa*f_L. The decaying solutions are
    theta = C*q^-m and phi = E*q^-mp + P*q^-m; these two conditions keep only them.
    They need m > 0. Dropped nonlinear and source terms leave a slow drift that the L study measures.

    form "local" (the reference) takes A at theta(L); form "unit" takes A = 1, its eta -> infinity
    limit. Both impose theta, phi -> 0 to the same order. They approach the eta -> infinity values
    from opposite sides, so solving with both gives a far-field error bar (reference audit D2).
    """
    f, th, th1, ph, ph1 = zb[0], zb[6], zb[7], zb[8], zb[9]
    q = L + p["kappa"]
    if form not in ("local", "unit"):
        raise ValueError(f"unknown far-condition form {form!r}")
    A = 1.0 + (p["theta_w"] - 1.0) * th if form == "local" else 1.0
    m = p["Pr"] * p["kappa"] * f / (1.0 + p["Nr"] * A**3)
    mp = p["Le"] * p["Pr"] * p["kappa"] * f
    return th1 + m * th / q, ph1 + (mp * ph - p["Nt"] / p["Nb"] * m * th) / q


def boundary(za, zb, p, stable=True, L=None, far_form="local"):
    """B1 to B12 as an array. za is the state at eta=0, zb at eta=L."""
    return np.array(boundary_terms(za, zb, p, stable, L, far_form))


def boundary_terms(za, zb, p, stable=True, L=None, far_form="local"):
    """B1 to B12 as a list. Plain arithmetic only, so PyTorch tensors work too.

    With L given, B10 and B11 become the asymptotic conditions of far_conditions. chi keeps
    chi(L)=0: as printed, Eq. 18 has no decaying far-field solution for chi (SPEC section 8).
    """
    f, f1, f2, f3, g, g1, th, th1, ph, ph1, ch, ch1 = za
    b4 = b4_stable(th, th1, p) if stable else b4_original(th, th1, p)
    out = [
        f1 - 1 - p["N1"] * (f2 - f1 / p["kappa"]),
        f - p["fw"] / (p["Pr"] * p["Le"]) * ph1,
        g + p["ell"] * f2,
        b4,
        ph1 + p["Nt"] / p["Nb"] * th1,
        ch - 1,
        zb[1], zb[2], zb[4], zb[6], zb[8], zb[10],
    ]
    if L is not None:
        out[9], out[10] = far_conditions(zb, L, p, far_form)
    return out
