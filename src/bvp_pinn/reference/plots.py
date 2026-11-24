"""Figures, captions and the study table for the reference. Reads saved data only.

Rebuild all figures without solving anything:
    uv run python -m bvp_pinn.reference.plots

Main figures plot profiles against eta on a linear axis, the way similarity-solution papers show
them, and put every correctness check next to a known answer or a tolerance. Style follows Rougier,
Droettboom and Bourne (2014), "Ten simple rules for better figures" (doi:10.1371/journal.pcbi.1003833)
and Wilke, Fundamentals of Data Visualization, ch. 20: one message per figure, the Okabe-Ito
colour-blind-safe palette, a second cue (line style or marker) wherever colour separates series.
"""
from __future__ import annotations

import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OI = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73", "purple": "#CC79A7",
      "orange": "#E69F00", "sky": "#56B4E9", "black": "#000000", "grey": "#7F7F7F"}
PANELS = [(0, "f", OI["blue"]), (1, "f'", OI["blue"]), (4, "g", OI["vermillion"]),
          (6, r"\theta", OI["green"]), (8, r"\phi", OI["purple"]), (10, r"\chi", OI["orange"])]
TEX = {"f": "f", "f1": "f'", "f2": "f''", "f3": "f'''", "f4": "f''''", "g": "g", "g1": "g'", "g2": "g''",
       "theta": r"\theta", "th1": r"\theta'", "th2": r"\theta''", "phi": r"\phi", "ph1": r"\phi'",
       "ph2": r"\phi''", "chi": r"\chi", "ch1": r"\chi'", "ch2": r"\chi''"}
STYLE = {"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
         "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white",
         "pdf.fonttype": 42, "figure.dpi": 100}
MM = 1 / 25.4
WIDE = 180 * MM


def _save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _letters(axes):
    for i, ax in enumerate(np.ravel(axes)):
        ax.text(-0.02, 1.02, f"({chr(97 + i)})", transform=ax.transAxes, ha="right", va="bottom",
                fontweight="bold")


def fig_profiles(data, path, title, other=None, xmax=None):
    """f, f', g, theta, phi, chi against eta on a linear axis, as boundary-layer papers show them.

    Lines: solve_bvp. Open circles at eta = 0, 1, ..., L: Runge-Kutta shooting. Each panel prints
    the wall value and wall slope. With other given, branch B is drawn dashed instead of the circles.
    xmax zooms into the wall layer; the circles are then 20 per window.
    """
    eta, z = data["eta"], data["z"]
    xmax = xmax or eta[-1]
    step = 1.0 if xmax == eta[-1] else xmax / 20
    keep = eta <= xmax * 1.0001
    marks = np.unique(np.searchsorted(eta, np.arange(0.0, xmax + step / 2, step)).clip(0, eta.size - 1))
    fig, axes = plt.subplots(2, 3, figsize=(WIDE, 100 * MM), layout="constrained")
    for n, (ax, (i, lab, col)) in enumerate(zip(axes.flat, PANELS)):
        ax.axhline(0, color="0.8", lw=0.5, zorder=0)
        if other is not None:
            ax.plot(other["eta"], other["z"][i], color=OI["grey"], ls="--", lw=1.0)
        ax.plot(eta[keep], z[i][keep], color=col)
        if other is None:
            ax.plot(eta[marks], data["z_rk"][i][marks], ls="none", marker="o", ms=3.5, mfc="none",
                    mec="k", mew=0.6)
        ax.set_xlim(0, xmax)
        ax.set_xlabel(r"$\eta$")
        ax.set_title(f"({chr(97 + n)})  ${lab}$" + ("" if other is not None else
                     f":   ${lab}(0) = {z[i, 0]:.4f}$,  ${lab}'(0) = {z[i + 1, 0]:.4f}$"), loc="left")
    if other is None:
        handles = [plt.Line2D([], [], color="0.2", label="solve_bvp (collocation)"),
                   plt.Line2D([], [], ls="none", marker="o", mfc="none", mec="k", ms=3.5,
                              label="Runge-Kutta multiple shooting")]
    else:
        handles = [plt.Line2D([], [], color="0.2", label="branch A (target)"),
                   plt.Line2D([], [], color=OI["grey"], ls="--", label="branch B (no decay at infinity)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(title)
    _save(fig, path)


def fig_benchmarks(bench, prof, path):
    """Same code on limits with known answers: a published table and exact flat-sheet solutions."""
    from .benchmark import FLAT_KAPPA, theta_exact

    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 66 * MM), layout="constrained")
    ax = axes[0]
    k = np.array([r["kappa"] for r in bench["abbas"]], float)
    pub = np.array([r["published"] for r in bench["abbas"]])
    ax.fill_between(k, 1e-9, 1e-5 / pub, color="0.88", lw=0, label="within 1 unit of the 5th decimal")
    for key, col, mk, ls in (("L10", OI["blue"], "o", "-"), ("L80", OI["vermillion"], "^", "--")):
        dev = np.abs(np.array([r[key] for r in bench["abbas"]]) / pub - 1)
        ax.loglog(k, np.maximum(dev, 1e-9), color=col, marker=mk, ms=4, ls=ls,
                  label=rf"this code, $L={key[1:]}$")
    ax.set_ylim(1e-9, 1)
    ax.set(xlabel=r"curvature $\kappa$", ylabel="|relative difference from table|",
           title="curved sheet vs published table")
    ax.legend(frameon=False, loc="upper right")
    ax = axes[1]
    eta = prof["eta"]
    every = slice(0, None, 6)
    ax.plot(eta, np.exp(-eta), color="k", lw=0.8)
    ax.plot(eta[every], prof["f1"][every], ls="none", marker="o", ms=3.5, mfc="none", mec=OI["vermillion"],
            label=r"$f'$")
    sd = bench["slip_darcy"]
    ax.plot(eta, sd["c"] * np.exp(-sd["a"] * eta), color="k", lw=0.8)
    ax.plot(eta[every], prof["f1_slip_darcy"][every], ls="none", marker="D", ms=3, mfc="none", mec=OI["purple"],
            label=rf"$f'$, Da = {sd['Da']:g}, $N_1$ = {sd['N1']:g}")
    cols = plt.cm.viridis(np.linspace(0.0, 0.8, len(bench["flat"])))
    for r, c in zip(bench["flat"], cols):
        ax.plot(eta, theta_exact(eta, r["Pr"]), color="k", lw=0.8)
        ax.plot(eta[every], prof[f"theta_Pr{r['Pr']:g}"][every], ls="none", marker="s", ms=2.5, color=c,
                label=rf"$\theta/\theta(0)$, Pr = {r['Pr']:g}")
    ax.set_xlim(0, 6)
    ax.set(xlabel=r"$\eta$", title=rf"flat limit ($\kappa=10^{{{np.log10(FLAT_KAPPA):.0f}}}$); black: exact")
    ax.legend(frameon=False, fontsize=6.5)
    ax = axes[2]
    sweep = bench["kappa_sweep"]
    ks = np.array([r["kappa"] for r in sweep])
    ax.loglog(ks, [r["f1_max_err"] for r in sweep], color=OI["vermillion"], marker="o", ms=3.5,
              label=r"max $|f'-e^{-\eta}|$")
    ax.loglog(ks, [r["theta_max_err"] for r in sweep], color=OI["green"], marker="s", ms=3.5, ls="--",
              label=r"max $|\theta/\theta(0)-$exact$|$, Pr = 1")
    ax.loglog(ks, 0.3 / ks, color="k", lw=0.8, ls=(0, (1, 1)), label=r"slope $-1$")
    ax.set(xlabel=r"curvature $\kappa$", ylabel=r"max error on $0\leq\eta\leq 10$",
           title=r"distance to the flat exact solution")
    ax.set_ylim(1e-8, 1e-2)
    ax.legend(frameon=False, loc="lower left")
    fig.suptitle("Code checks against known answers (limits of Eqs. 14 and 16)")
    _letters(axes)
    _save(fig, path)


def _dots(ax, labels, values, ref=None, ref_label=None, lo=1e-17, hi=1e-6):
    y = np.arange(len(labels))[::-1]
    v = np.maximum(np.asarray(values, float), lo)
    ax.hlines(y, lo, v, color="0.8", lw=0.8)
    ax.plot(v, y, ls="none", marker="o", ms=4, color=OI["blue"])
    ax.set_yticks(y, labels)
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    if ref is not None:
        ax.axvline(ref, color=OI["vermillion"], ls="--", lw=0.9)
        ax.text(ref / 1.5, len(labels) - 0.5, ref_label, color=OI["vermillion"], fontsize=7, ha="right",
                va="center")


def fig_checks(data, meta, path, name):
    """Evidence that the saved curves are right, from the saved data alone."""
    from .verify import SLOPE_LINKS, slope_check

    slopes = slope_check(data["eta"], data["z"], data["H"])
    from .equations import BC_NAMES, STATE_NAMES

    bc = {k: meta["boundary_errors"][k] for k in BC_NAMES}
    rk = meta["checks"]["V10_rk_multiple_shooting"]
    st = {k: rk["max_rel_diff_by_state"][k] for k in STATE_NAMES}
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 76 * MM), layout="constrained")
    _dots(axes[0], [rf"$d{TEX[a]}/d\eta$ vs ${TEX[b]}$" for a, b in SLOPE_LINKS], list(slopes.values()),
          meta["tol"], f"solver tol {meta['tol']:g}")
    axes[0].set(xlabel="max relative mismatch", title="slope of each curve vs the next")
    _dots(axes[1], list(bc), np.abs(list(bc.values())), meta["tol"], f"bc_tol {meta['tol']:g}")
    axes[1].set(xlabel="|boundary error|", title="the 12 boundary conditions")
    _dots(axes[2], [f"${TEX[k]}$" for k in st], list(st.values()), rk["self_diff_vs_rtol_check"],
          "RK own error")
    axes[2].set(xlabel="max relative difference", title="Runge-Kutta vs collocation")
    axes[1].text(1.5e-17, -0.6, "points on the left edge are exactly 0", fontsize=6.5, color="0.4")
    fig.suptitle(f"Branch {name}: checks on the saved reference (left of the dashed line = within tolerance)")
    _letters(axes)
    _save(fig, path)
    return slopes


def fig_far_field(lst, m, kappa, path, name="A"):
    """Log-log tails of theta, phi, chi for every far-field L: shows the power-law decay."""
    keys = sorted({float(k.split("_L")[1].split("_")[0]) for k in lst if k.startswith(f"{name}_asymptotic")})
    shades = plt.cm.cividis(np.linspace(0.15, 0.85, len(keys)))
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 62 * MM), layout="constrained")
    for ax, (i, lab) in zip(axes, [(6, r"$|\theta|$"), (8, r"$|\phi|$"), (10, r"$|\chi|$")]):
        for L, c in zip(keys, shades):
            e, z = lst[f"{name}_asymptotic_L{L:g}_eta"], lst[f"{name}_asymptotic_L{L:g}_z"]
            ax.loglog(e + kappa, np.abs(z[i]), color=c, label=f"asymptotic, L = {L:g}")
        for L, ls in ((10, "--"), (keys[-1], ":")):
            k = f"{name}_dirichlet_L{L:g}_eta"
            if k in lst:
                ax.loglog(lst[k] + kappa, np.abs(lst[k.replace("_eta", "_z")][i]), color=OI["vermillion"], ls=ls,
                          lw=1.0, label=f"zero at L (old), L = {L:g}")
        ax.set_xlabel(r"$q=\eta+\kappa$")
        ax.set_title(lab)
        ax.set_ylim(1e-4, None)
    q0 = np.geomspace(3, 100, 20)
    e, z = lst[f"{name}_asymptotic_L{keys[-1]:g}_eta"], lst[f"{name}_asymptotic_L{keys[-1]:g}_z"]
    th3 = np.interp(3 - kappa, e, np.abs(z[6]))
    axes[0].loglog(q0, th3 * (q0 / 3) ** -m, color="k", lw=0.8, ls=(0, (1, 1)))
    axes[0].text(20, th3 * (20 / 3) ** -m * 1.5, rf"slope $-m=-{m:.2f}$", fontsize=7)
    fig.legend(*axes[2].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(f"Branch {name}: far-field decay for each domain length L")
    _letters(axes)
    _save(fig, path)


def fig_wall_vs_L(study, path, name="A"):
    """Each wall value against L: old zero-at-L cut versus asymptotic far conditions."""
    keys = [("f2_0", r"$f''(0)$"), ("theta_0", r"$\theta(0)$"), ("th1_0", r"$\theta'(0)$"),
            ("phi_0", r"$\phi(0)$"), ("ph1_0", r"$\phi'(0)$"), ("ch1_0", r"$\chi'(0)$")]
    sb = study[f"branch_{name}"]
    fig, axes = plt.subplots(2, 3, figsize=(WIDE, 92 * MM), layout="constrained")
    for ax, (k, lab) in zip(axes.flat, keys):
        for ladder, col, mk, ls, tag in ((sb["L_ladder"], OI["blue"], "o", "-", "asymptotic conditions"),
                                        (sb["dirichlet_L_ladder"], OI["vermillion"], "s", "--", "zero at L (old)")):
            ok = [r for r in ladder if r["status"] == 0]
            ax.plot([r["L"] for r in ok], [r["wall"][k] for r in ok], color=col, marker=mk, ls=ls, ms=3.5,
                    mfc=col if mk == "o" else "white", label=tag)
        ax.set_xscale("log", base=2)
        ax.set_xticks([10, 20, 40, 80, 160], ["10", "20", "40", "80", "160"])
        ax.set_title(lab)
        ax.set_xlabel("domain length L")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(f"Branch {name}: wall values against the far-field cut L")
    _letters(axes)
    _save(fig, path)


def fig_diagnostics(data, meta, p, path, name):
    """Where the solver works hardest: mesh spacing, residual per interval, theta'' coefficient."""
    from .equations import thermal_coefficient

    kappa = p["kappa"]
    x = data["mesh"]
    mid = 0.5 * (x[1:] + x[:-1])
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 58 * MM), layout="constrained")
    axes[0].loglog(mid + kappa, np.diff(x), color=OI["blue"], lw=0.9)
    axes[0].set_ylabel("mesh spacing h")
    axes[1].loglog(mid + kappa, data["rms_residuals"], color=OI["vermillion"], lw=0.6, marker=".", ms=1.5)
    axes[1].axhline(meta["tol"], color="k", ls="--", lw=0.8)
    axes[1].text(x[-1], meta["tol"] * 1.4, f"tol = {meta['tol']:g}", ha="right", fontsize=7)
    axes[1].set_ylabel("RMS relative residual per interval")
    axes[2].semilogx(data["eta"] + kappa, thermal_coefficient(data["z"], p), color=OI["green"])
    axes[2].set_ylim(0, None)
    axes[2].set_ylabel(r"$(1+N_r A^3)/Pr$")
    for ax in axes:
        ax.set_xlabel(r"$\eta+\kappa$")
    fig.suptitle(f"Branch {name}: solve_bvp diagnostics ({x.size} final mesh nodes)")
    _letters(axes)
    _save(fig, path)


def fig_verification(study, path):
    """Manufactured-solution order, tolerance study and where each starting guess ends up."""
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 60 * MM), layout="constrained")
    m = study["V3_mms"]
    h = m["L"] / (np.array(m["meshes"]) - 1)
    e = np.array(m["nodal_max_errors"])
    axes[0].loglog(h, e, color=OI["blue"], marker="o", ms=4)
    axes[0].loglog(h, e[-1] * (h / h[-1]) ** 4, color="k", ls=(0, (1, 1)), lw=0.8)
    axes[0].text(h[1], e[-1] * (h[1] / h[-1]) ** 4 / 3, "slope 4", fontsize=7)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(h, [f"{v:.3g}" for v in h])
    axes[0].minorticks_off()
    axes[0].set(xlabel="mesh step h", ylabel="max nodal error", title="manufactured solution")
    for key, col, mk in (("branch_A", OI["blue"], "o"), ("branch_B", OI["grey"], "s")):
        ok = [r for r in study.get(key, {}).get("tol_ladder", []) if r["status"] == 0]
        if len(ok) > 1:
            ref = ok[-1]["wall"]
            axes[1].loglog([r["tol"] for r in ok[:-1]],
                           [max(1e-16, max(abs(r["wall"][k] - ref[k]) / max(abs(ref[k]), 1e-8) for k in ref))
                            for r in ok[:-1]], color=col, marker=mk, ms=4, label=key.replace("_", " "))
    axes[1].axhline(study["settings"]["agreement_rel"], color="k", ls="--", lw=0.8)
    axes[1].set(xlabel="solver tol", ylabel="wall change vs tightest tol", title="tolerance study, L = 10")
    axes[1].legend(frameon=False)
    scan = study["branch_scan"]
    pts = {"A": [], "B": [], "failed": []}
    for name in ("A", "B"):
        meta_key = f"branch_{name}"
        if meta_key in study and "folder" in study[meta_key]:
            from .io import load_reference
            from ..config import ROOT
            pts[name] = [(g["theta0"], g["rate"]) for g in
                         load_reference(ROOT / study[meta_key]["folder"])[1]["scan_guesses_reaching_branch"]]
    pts["failed"] = [tuple(map(float, re.findall(r"r([\d.]+)-t([\d.]+)", f["label"])[0][::-1])) for f in scan["failed"]]
    for key, col, mk in (("A", OI["blue"], "o"), ("B", OI["vermillion"], "s"), ("failed", "k", "x")):
        if pts[key]:
            xs, ys = zip(*pts[key])
            axes[2].scatter(xs, ys, color=col, marker=mk, s=14, label=f"{key} ({len(xs)})")
    axes[2].set_yscale("log", base=2)
    axes[2].set(xlabel=r"guess $\theta(0)$", ylabel="guess decay rate", title="where each start converges")
    axes[2].legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.suptitle("Reference verification")
    _letters(axes)
    _save(fig, path)


CAPTIONS = """# Reference figures: captions

Generated with the figures by `uv run python -m bvp_pinn.reference.plots`. Numbers come from the saved data.
Analysis: docs/reference-solver/SPEC_reference_solutions.md section 8.5.

## Main figures (figures/)

- **fig1_profiles**: the target solution (branch A, L = {L:g}) on a linear eta axis. Lines are solve_bvp;
  open circles at eta = 0, 1, ..., {L:g} are the independent Runge-Kutta shooting solution. Each panel prints
  the wall value and wall slope.
  - Look for: a circle off its line (the two methods disagree), a field that does not settle at the far end.
- **fig2_benchmarks**: the same code on limits of Eqs. 14 and 16 that have known answers. These are code checks;
  the locked values are not changed.
  - (a) Curved sheet (only the curvature terms of Eq. 14 kept): skin friction against the published table of
    Abbas et al. (2013). At L = 10 the code gives the published value at every kappa to within one unit
    of the fifth decimal (largest difference {abbas_dev:.1e}; at kappa = 1000 the table's 1.00079 is the
    code's 1.0007993 cut to five decimals). At L = 80 the values are lower (by {abbas_shift:.1e} at kappa = 5),
    so the published table carries a domain-length error.
  - (b) Flat-sheet limit (kappa = 1e6): f' and theta/theta(0) for five Prandtl numbers against the exact
    solutions (lines), largest wall-slope difference {flat_dev:.1e} (relative); and f' with Darcy drag and
    slip at the locked Da and N1 against its exact solution, f''(0) difference {sd_dev:.1e} (relative).
  - (c) Distance to the flat exact solution against kappa: slope -1, so the curvature terms vanish at the
    rate the equations predict.
- **fig3_checks**: checks on the saved branch A reference, using the saved arrays only.
  - (a) Slope of each saved curve (quintic spline) against the next saved curve. For f''', g', theta', phi',
    chi' the target is the highest derivative that the equations demand, so this also checks that the
    curves satisfy Eqs. 14 to 18. Worst: {slope_max:.1e}.
  - (b) The 12 boundary errors against bc_tol.
  - (c) Largest Runge-Kutta minus collocation difference per state, against the RK error estimate.
- **fig4_wall_vs_L**: each wall value against the domain length L, for the asymptotic far conditions and the
  old zero-at-L cut. A flat line means the value no longer depends on where the domain is cut.

## Supplementary figures (figures/supplementary/)

- **S1_branches**: branch A against branch B (dashed), which does not decay at infinity (m = {mB:.4f}).
- **S2_far_field**: |theta|, |phi|, |chi| on log-log axes for each L. Dotted line: predicted slope
  -m = -{mA:.3f}. Asymptotic tails lie on one line; the old cut bends down to zero at L; chi depends on L.
- **S3_verification**: (a) manufactured-solution error against mesh step, slope-4 guide; (b) change of the
  wall values against solver tolerance at L = 10; (c) the branch reached from each of the 48 starting guesses.
- **S4_checks_branchB**: fig3 for branch B.
- Per branch folder, **solver_diagnostics**: mesh spacing, residual per interval against tol, and the theta''
  coefficient, which must stay away from zero.
"""


def make_all(out_root, study, exported):
    import shutil

    from ..config import load_case
    from .io import load_reference

    p = load_case()[0]
    figs = out_root / "figures"
    shutil.rmtree(figs, ignore_errors=True)
    sup = figs / "supplementary"
    sup.mkdir(parents=True)
    loaded = {name: load_reference(folder) for name, folder, _ in exported}
    bench = json.loads((out_root / "benchmarks.json").read_text())
    with plt.rc_context(STYLE):
        for name, folder, _ in exported:
            data, meta = loaded[name]
            (folder / "figures").mkdir(exist_ok=True)
            fig_diagnostics(data, meta, p, folder / "figures" / "solver_diagnostics", name)
        A, B = loaded["A"], loaded.get("B")
        fig_profiles(A[0], figs / "fig1_profiles", f"Branch A reference on $0 \\leq \\eta \\leq {A[1]['L']:g}$")
        fig_profiles(A[0], figs / "fig1b_profiles_wall", "Branch A reference, wall layer $0 \\leq \\eta \\leq 1$",
                     xmax=1.0)
        with np.load(out_root / "benchmarks.npz") as prof:
            fig_benchmarks(bench, dict(prof), figs / "fig2_benchmarks")
        slopes = fig_checks(A[0], A[1], figs / "fig3_checks", "A")
        fig_wall_vs_L(study, figs / "fig4_wall_vs_L")
        if B:
            fig_profiles(A[0], sup / "S1_branches", "Branch A (target) against branch B", other=B[0])
            fig_checks(B[0], B[1], sup / "S4_checks_branchB", "B")
        with np.load(out_root / "l_study.npz") as lst:
            fig_far_field(dict(lst), A[1]["checks"]["far_decay_exponent_m"], p["kappa"], sup / "S2_far_field")
        fig_verification(study, sup / "S3_verification")
    ab = bench["abbas"]
    (figs / "captions.md").write_text(CAPTIONS.format(
        L=A[1]["L"], mA=A[1]["checks"]["far_decay_exponent_m"],
        mB=B[1]["checks"]["far_decay_exponent_m"] if B else float("nan"),
        abbas_dev=max(abs(r["L10"] - r["published"]) for r in ab),
        abbas_shift=ab[0]["L10"] - ab[0]["L80"],
        flat_dev=max(abs(r["code"] / r["exact"] - 1) for r in bench["flat"]),
        sd_dev=abs(bench["slip_darcy"]["code_f2_0"] / bench["slip_darcy"]["exact_f2_0"] - 1),
        slope_max=max(slopes.values())))


def main():
    """Rebuild every figure from saved data (no solving)."""
    from ..config import ROOT

    out_root = ROOT / "reference"
    study = json.loads((out_root / "study.json").read_text())
    exported = [(k.split("_")[1], ROOT / v["folder"], None) for k, v in study.items()
                if k.startswith("branch_") and isinstance(v, dict) and "folder" in v]
    make_all(out_root, study, exported)
    print(f"Figures written to {out_root / 'figures'}")


def study_markdown(study):
    s = study["settings"]
    lines = [
        "# Reference study (generated)",
        "",
        "Generated by `uv run python -m bvp_pinn.reference.run`. Do not edit by hand. "
        "Spec: [docs/reference-solver/SPEC_reference_solutions.md](../docs/SPEC_reference_solutions.md). "
        "Figure captions: [figures/captions.md](figures/captions.md).",
        "",
        f"- Created: {study['created']}; git commit {study['git']['commit'][:10]}"
        f"{' (uncommitted changes present)' if study['git']['dirty'] else ''}",
        f"- Physical case SHA-256: `{study['case_sha256']}`",
        f"- Agreement bar: relative {s['agreement_rel']:g} on every wall value",
        "",
        "## V3 manufactured solution",
        "",
        "| Mesh nodes | Nodal max error | Observed order |",
        "|---:|---:|---:|",
    ]
    m = study["V3_mms"]
    for i, (n, e) in enumerate(zip(m["meshes"], m["nodal_max_errors"])):
        order = f"{m['observed_orders'][i - 1]:.2f}" if i else ""
        lines.append(f"| {n} | {e:.3e} | {order} |")
    lines += ["", f"Mesh kept fixed: {m['mesh_kept_fixed']}. Pass (last order at least "
              f"{s['mms']['min_order']}): {m['pass']}.", "", "## Branch scan", ""]
    bs = study["branch_scan"]
    lines.append(f"{sum(b['count'] for b in bs['branches'])} of "
                 f"{len(s['branch_scan']['rates']) * len(s['branch_scan']['theta0'])} guesses converged at "
                 f"L={bs['L']}, tol={bs['tol']:g}; {len(bs['failed'])} failed.")
    lines += ["", "| Branch | Guesses | theta(0) | f''(0) | g'(0) | theta'(0) | phi'(0) | chi'(0) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for b in bs["branches"]:
        w = b["wall"]
        lines.append(f"| {b['name']} | {b['count']} | {w['theta_0']:.6f} | {w['f2_0']:.6f} | {w['g1_0']:.6f} | "
                     f"{w['th1_0']:.6f} | {w['ph1_0']:.6f} | {w['ch1_0']:.6f} |")
    for key in sorted(k for k in study if k.startswith("branch_") and k != "branch_scan"):
        sb = study[key]
        name = key.split("_")[1]
        lines += ["", f"## Branch {name}", "", "| Step | Far field | L | tol | Status | Nodes | f''(0) | theta'(0) | chi'(0) |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        rows = [(r, "asymptotic") for r in sb["tol_ladder"] + sb["L_ladder"]]
        rows += [(r, "zero at L") for r in sb.get("dirichlet_L_ladder", [])]
        for r, mode in rows:
            w = r["wall"] or {"f2_0": float("nan"), "th1_0": float("nan"), "ch1_0": float("nan")}
            lines.append(f"| {r['label']} | {mode} | {r['L']:g} | {r['tol']:g} | {r['status']} | {r['nodes']} | "
                         f"{w['f2_0']:.8f} | {w['th1_0']:.8f} | {w['ch1_0']:.8f} |")
        lines += ["", "| Check | Result |", "|---|---|"]
        for k, v in sb["checks"].items():
            lines.append(f"| {k} | {v} |")
        if "folder" in sb:
            lines += ["", f"Data: `{sb['folder']}`"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
