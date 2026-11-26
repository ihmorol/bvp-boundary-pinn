"""How fast a small error grows when the locked problem is marched in eta, along reference A.

    uv run python tools/marching_growth.py

Marching window by window, as the Lorenz PINN does in time, treats the problem as an initial
value problem: each window starts from the state the window before it hands over. This tool
measures what that does to a small error in the handed-over state, on reference branch A at
L = 20 (the locked case only):

1. The eigenvalues of the Jacobian of the first-order system dz/deta = rhs(eta, z) at a few
   eta. A positive real part is a mode that grows when marching outward; a negative one grows
   when marching inward.
2. The transition matrix Phi(0 -> b), the product of exp(J dx) over the reference mesh, with J
   at the middle of each step. Its largest singular value is the largest growth of an error
   made at the wall. 1 / its smallest singular value is the growth when marching back to the
   wall; it is printed only while the matrix is well enough conditioned to resolve it.

Only reads the reference and the locked case. Nothing here feeds training. Prints its results.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bvp_pinn.config import load_case  # noqa: E402
from bvp_pinn.reference import equations as eq  # noqa: E402
from bvp_pinn.reference.io import load_reference  # noqa: E402

FOLDER = ROOT / "reference" / "locked-branchA-L20-tol1e-08"
PROBES = [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 15.0, 20.0]
MARKS = [0.5, 2.0, 6.0, 10.0, 20.0]  # the B1 window edges, and two more


def jacobian(x, y, p, h=1e-7):
    """d rhs / d z at one point, by forward differences."""
    f0 = np.asarray(eq.rhs(np.array([x]), y.reshape(12, 1), p)).reshape(12)
    J = np.zeros((12, 12))
    for j in range(12):
        yp = y.copy()
        yp[j] += h * max(1.0, abs(y[j]))
        J[:, j] = (np.asarray(eq.rhs(np.array([x]), yp.reshape(12, 1), p)).reshape(12) - f0) / (yp[j] - y[j])
    return J


def main():
    arrays, _ = load_reference(FOLDER)
    p, _ = load_case()
    eta = arrays["eta"].reshape(-1)
    z = arrays["z"] if arrays["z"].shape[0] == 12 else arrays["z"].T
    Js = [jacobian(eta[i], z[:, i], p) for i in range(len(eta))]

    print(f"Reference A, {len(eta)} mesh points on [0, {eta[-1]:g}]")
    print("eta     largest Re   smallest Re   modes growing outward   modes growing inward")
    for x in PROBES:
        i = int(np.argmin(abs(eta - x)))
        re = np.sort(np.linalg.eigvals(Js[i]).real)
        print(f"{eta[i]:6.2f}  {re[-1]:10.3f}  {re[0]:12.3f}  {int(np.sum(re > 1e-6)):12d}  {int(np.sum(re < -1e-6)):21d}")

    print("\nfrom 0 to b   outward growth of a wall error   growth when marching back to 0")
    Phi, k = np.eye(12), 0
    for i in range(len(eta) - 1):
        Phi = expm(0.5 * (Js[i] + Js[i + 1]) * (eta[i + 1] - eta[i])) @ Phi
        while k < len(MARKS) and eta[i + 1] >= MARKS[k] - 1e-12:
            s = np.linalg.svd(Phi, compute_uv=False)
            back = f"{1 / s[-1]:.2e}" if s[-1] > 1e3 * np.finfo(float).eps * s[0] else "not resolved"
            print(f"0 to {eta[i + 1]:5.2f}   {s[0]:.2e}   {back:>30}")
            k += 1


if __name__ == "__main__":
    main()
