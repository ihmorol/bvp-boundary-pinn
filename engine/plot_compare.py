"""Plot the C++ reference, a PINN prediction and the PINN error side by side.

    python engine/plot_compare.py                          # engine only, plus the check below
    python engine/plot_compare.py runs/<run_id>/pred_eval.npz
    python engine/plot_compare.py my_pinn.csv              # CSV with columns eta,f,f1,...,ch1

Reads engine/out/engine_solution.csv (run engine/bvp_engine first). If the locked branch-A
reference exists, it also prints how far the engine is from it, as a check of the engine.
This is evaluation only: nothing here feeds training (AGENTS.md section 3).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "engine" / "out"
NAMES = ["f", "f1", "f2", "f3", "g", "g1", "theta", "th1", "phi", "ph1", "chi", "ch1"]
PANELS = [("f'", "f1"), ("f", "f"), ("g", "g"), (r"\theta", "theta"), (r"\phi", "phi"), (r"\chi", "chi")]
WALL = [("f''(0)", "f2"), ("g'(0)", "g1"), ("theta(0)", "theta"), ("theta'(0)", "th1"), ("phi'(0)", "ph1"),
        ("chi'(0)", "ch1")]


def load_engine():
    data = np.genfromtxt(OUT / "engine_solution.csv", delimiter=",", names=True)
    return data["eta"], {n: data[n] for n in NAMES}


def load_pinn(path):
    path = Path(path)
    if path.suffix == ".npz":
        d = np.load(path)
        return d["eta"].ravel(), dict(zip(NAMES, d["z"]))
    d = np.genfromtxt(path, delimiter=",", names=True)
    return d["eta"], {n: d[n] for n in NAMES if n in d.dtype.names}


def rel_l2(got, want, eta):
    return float(np.sqrt(np.trapezoid((got - want) ** 2, eta) / np.trapezoid(want**2, eta)))


def check_against_locked_reference(eta, eng):
    folder = ROOT / "reference" / "locked-branchA-L20-tol1e-08"
    if not (folder / "reference.npz").exists():
        return
    ref = np.load(folder / "reference.npz")
    worst = max(np.max(np.abs(CubicSpline(eta, eng[n])(ref["eta"]) - ref["z"][i])) for i, n in enumerate(NAMES))
    wall = max(abs(eng[n][0] - ref["z"][NAMES.index(n)][0]) for _, n in WALL)
    print(f"engine vs locked solve_bvp reference: max wall difference {wall:.1e}, "
          f"max difference on the reference grid {worst:.1e} (cubic-spline interpolation)")


def main(argv):
    eta, eng = load_engine()
    check_against_locked_reference(eta, eng)
    pinn = load_pinn(argv[1]) if len(argv) > 1 else None
    rows = 2 if pinn else 1
    fig, axes = plt.subplots(rows, 6, figsize=(15, 3.2 * rows), layout="constrained", squeeze=False)
    if pinn:
        pe, pz = pinn
        print(f"{'quantity':10s} {'engine':>12s} {'PINN':>12s} {'abs error':>10s}")
        for lab, n in WALL:
            if n in pz:
                print(f"{lab:10s} {eng[n][0]:12.6f} {pz[n][0]:12.6f} {abs(pz[n][0] - eng[n][0]):10.2e}")
    kappa = 0.2  # locked kappa; log axis of eta + kappa as in the reference figures, so the wall stays visible
    for k, (lab, n) in enumerate(PANELS):
        ax = axes[0, k]
        ax.plot(eta + kappa, eng[n], "k-", lw=2, alpha=0.4, label="C++ box + Richardson")
        if pinn and n in pinn[1]:
            ref_at_pinn = np.interp(pinn[0], eta, eng[n])
            ax.plot(pinn[0] + kappa, pinn[1][n], "C3--", lw=1.2, label="PINN")
            axes[1, k].loglog(pinn[0] + kappa, np.abs(pinn[1][n] - ref_at_pinn) + 1e-16, "C3-", lw=1)
            axes[1, k].set(xlabel=r"$\eta + \kappa$", title=f"error, rel L2 {rel_l2(pinn[1][n], ref_at_pinn, pinn[0]):.1e}")
        ax.set(xlabel=r"$\eta + \kappa$", title=f"${lab}$", xscale="log")
    axes[0, 0].legend(frameon=False, fontsize=8)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "compare.png", dpi=150)
    print(f"wrote {OUT / 'compare.png'}")


if __name__ == "__main__":
    main(sys.argv)
