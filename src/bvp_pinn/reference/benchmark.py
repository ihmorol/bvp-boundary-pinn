"""Code checks against published and exact solutions (SPEC V11).

The locked case has no published solution, so the same code (equations.py, solver.py) is run
on limits of Eqs. 14 to 19 that do have one. These are code checks only; the locked values are
never changed.

1. Curved stretching sheet (Delta = delta* = beta_m = N1 = fw = ell = 0, no Darcy term). Eq. 14 then is
   the curved-sheet equation of Abbas et al. (2013). Their skin friction -Re^(1/2) Cf = -f''(0) + f'(0)/kappa,
   five digits, as reprinted in Table 1 of Shafiq, Nadeem and Noor, Commun. Theor. Phys. 71 (2019) 344.
2. Flat-sheet limit (kappa -> infinity) with Nr = Nt = Rd = 0. Eq. 14 becomes the derivative of Crane's
   equation, exact f = 1 - exp(-eta), and Eq. 16 becomes theta'' + Pr f theta' = 0. With xi = Pr exp(-eta),
   theta/theta(0) = P(Pr, xi)/P(Pr, Pr), P the regularized lower incomplete gamma function, so
   -theta'(0)/theta(0) = Pr^Pr exp(-Pr) / (Gamma(Pr) P(Pr, Pr)). The Robin wall condition (Bi) only
   scales theta, so theta/theta(0) is compared.
3. Flat-sheet limit with the Darcy term and wall slip at the locked Da and N1. Eq. 14 is then the
   derivative of f''' + f f'' - f'^2 - f'/Da = 0 with f'(0) = 1 + N1 f''(0), solved exactly by
   f' = c exp(-a eta) with a^2 = c + 1/Da and c = 1/(1 + N1 a).
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma, gammainc

from ..config import ROOT, load_case, load_reference_settings
from . import solver as sv

OFF = dict(Delta=0.0, B=0.0, Da=np.inf, delta_star=0.0, beta_m=0.0, N1=0.0, fw=0.0, ell=0.0,
           Nr=0.0, Nt=0.0, Rd=0.0, Pe=0.0)
ABBAS = {5: 1.15763, 10: 1.07349, 30: 1.02353, 50: 1.01405, 200: 1.00356, 1000: 1.00079}
PRANDTL = (0.72, 1.0, 3.0, 7.0, 10.0)
FLAT_KAPPA = 1e6
GUESS = {"rate": 1.0, "theta0": 0.5}


def theta_exact(eta, Pr):
    return gammainc(Pr, Pr * np.exp(-eta)) / gammainc(Pr, Pr)


def nusselt_exact(Pr):
    return Pr**Pr * np.exp(-Pr) / (gamma(Pr) * gammainc(Pr, Pr))


def slip_darcy_exact(Da, N1):
    """(a, c) of the exact f' = c exp(-a eta) for the flat sheet with Darcy drag and slip."""
    a = brentq(lambda a: a**2 - 1 / (1 + N1 * a) - 1 / Da, 1e-6, 1e3)
    return a, 1 / (1 + N1 * a)


def _solve(p, L, s, tol=1e-9):
    r = sv.solve_case(p, L, tol, s, label="benchmark", guess=GUESS)
    if not r.ok:
        raise RuntimeError(f"benchmark solve failed: kappa={p['kappa']}, L={L}, {r.message}")
    return r


def run(p0, s):
    base = dict(p0, **OFF)
    out = {"abbas": [], "flat": [], "kappa_sweep": [], "profiles": {}}
    for k, pub in ABBAS.items():
        row = {"kappa": k, "published": pub}
        for L in (10.0, 20.0, 80.0):
            r = _solve(dict(base, kappa=float(k), Pr=1.0), L, s)
            row[f"L{L:g}"] = -r.wall["f2_0"] + 1.0 / k
        out["abbas"].append(row)
    eta = np.linspace(0.0, 10.0, 201)
    for Pr in PRANDTL:
        r = _solve(dict(base, kappa=FLAT_KAPPA, Pr=Pr), 20.0, s)
        z = r.sol(eta)
        out["flat"].append({"Pr": Pr, "exact": float(nusselt_exact(Pr)),
                            "code": -r.wall["th1_0"] / r.wall["theta_0"]})
        out["profiles"][f"theta_Pr{Pr:g}"] = z[6] / z[6, 0]
        if Pr == 1.0:
            out["profiles"]["f1"] = z[1]
    out["profiles"]["eta"] = eta
    a, c = slip_darcy_exact(p0["Da"], p0["N1"])
    r = _solve(dict(base, kappa=FLAT_KAPPA, Pr=1.0, Da=p0["Da"], N1=p0["N1"]), 20.0, s)
    out["profiles"]["f1_slip_darcy"] = r.sol(eta)[1]
    out["slip_darcy"] = {"Da": p0["Da"], "N1": p0["N1"], "a": a, "c": c, "exact_f2_0": -a * c,
                         "code_f2_0": r.wall["f2_0"],
                         "f1_max_err": float(np.max(np.abs(r.sol(eta)[1] - c * np.exp(-a * eta))))}
    for k in (1e2, 1e3, 1e4, 1e5, 1e6):
        r = _solve(dict(base, kappa=k, Pr=1.0), 20.0, s)
        z = r.sol(eta)
        out["kappa_sweep"].append({
            "kappa": k,
            "f1_max_err": float(np.max(np.abs(z[1] - np.exp(-eta)))),
            "theta_max_err": float(np.max(np.abs(z[6] / z[6, 0] - theta_exact(eta, 1.0)))),
        })
    return out


def main():
    out = run(load_case()[0], load_reference_settings())
    profiles = out.pop("profiles")
    np.savez(ROOT / "reference" / "benchmarks.npz", **profiles)
    (ROOT / "reference" / "benchmarks.json").write_text(json.dumps(out, indent=2) + "\n")
    for row in out["abbas"]:
        print(f"kappa={row['kappa']:>5}: published {row['published']:.5f}, code at L=10 {row['L10']:.5f}, "
              f"L=80 {row['L80']:.5f}")
    for row in out["flat"]:
        print(f"Pr={row['Pr']:>5}: exact {row['exact']:.7f}, code {row['code']:.7f}")
    sd = out["slip_darcy"]
    print(f"slip+Darcy (Da={sd['Da']}, N1={sd['N1']}): f''(0) exact {sd['exact_f2_0']:.7f}, "
          f"code {sd['code_f2_0']:.7f}, max|f'-exact| {sd['f1_max_err']:.1e}")
    for row in out["kappa_sweep"]:
        print(f"kappa={row['kappa']:.0e}: max|f'-exp(-eta)| {row['f1_max_err']:.2e}, "
              f"max|theta-exact| {row['theta_max_err']:.2e}")
    return out


if __name__ == "__main__":
    main()
