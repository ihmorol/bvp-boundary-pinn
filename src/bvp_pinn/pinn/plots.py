"""Result figures FB1 to FB5 and table TB2 (docs/PINN_PLAN.md section 6).

    uv run python -m bvp_pinn.pinn.plots

Reads only results/summary.csv and the saved run files, after evaluate.py. Same style and
colours as the reference figures.
"""
from __future__ import annotations

import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from ..reference.io import load_reference
from ..reference.plots import MM, OI, STYLE, WIDE, _letters, _save
from .evaluate import REFERENCES, RESULTS
from .runs import RUNS

ARMS = {"B0": (OI["blue"], "-", "o"), "B1": (OI["vermillion"], "--", "s"),
        "B2": (OI["green"], "-.", "^"), "B3": (OI["purple"], ":", "D")}
PANELS = [("f", 0), ("f'", 1), ("g", 4), (r"\theta", 6), (r"\phi", 8), (r"\chi", 10)]


def load_summary():
    with open(RESULTS / "summary.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["dry_run"] != "True"]
    for r in rows:
        r["max_field_rel_l2"] = float(r["max_field_rel_l2"])
    return rows


def median_run(rows, arm):
    """The run with the median error for one arm, so a figure shows a typical seed, not the best."""
    runs = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["max_field_rel_l2"])
    return runs[(len(runs) - 1) // 2]["run_id"] if runs else None


def fig_profiles(rows, ref, path):
    """FB1: profiles of every arm's median run against the reference."""
    fig, axes = plt.subplots(2, 3, figsize=(WIDE, 100 * MM), layout="constrained")
    for ax, (lab, i) in zip(axes.flat, PANELS):
        ax.plot(ref["eta"], ref["z"][i], color="k", lw=2.2, alpha=0.35, label="reference (solve_bvp)")
        for arm, (col, ls, _) in ARMS.items():
            rid = median_run(rows, arm)
            if rid:
                pred = np.load(RUNS / rid / "pred_eval.npz")
                ax.plot(pred["eta"], pred["z"][i], color=col, ls=ls, lw=1.1, label=f"{arm} (median seed)")
        ax.set(xlabel=r"$\eta$", title=f"${lab}$", xlim=(0, ref["eta"][-1]))
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("FB1: PINN profiles against the reference")
    _letters(axes)
    _save(fig, path)


def fig_errors(rows, ref, path):
    """FB2: pointwise error of each field, log scale."""
    fig, axes = plt.subplots(2, 3, figsize=(WIDE, 100 * MM), layout="constrained")
    for ax, (lab, i) in zip(axes.flat, PANELS):
        for arm, (col, ls, _) in ARMS.items():
            rid = median_run(rows, arm)
            if rid:
                pred = np.load(RUNS / rid / "pred_eval.npz")
                ax.semilogy(pred["eta"], np.abs(pred["z"][i] - ref["z"][i]) + 1e-16, color=col, ls=ls, lw=1.0,
                            label=arm)
        ax.set(xlabel=r"$\eta$", title=f"$|{lab}_{{PINN}} - {lab}_{{ref}}|$", xlim=(0, ref["eta"][-1]))
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("FB2: pointwise error of the median run of each arm")
    _letters(axes)
    _save(fig, path)


def fig_residuals(rows, path):
    """FB3: final equation residual and boundary error per arm; dots are seeds, bars the median."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 70 * MM), layout="constrained")
    names = [("residual_mse", [f"R{i}" for i in range(14, 19)]), ("boundary_abs", [f"B{i}" for i in range(1, 13)])]
    for ax, (key, labels) in zip(axes, names):
        x = np.arange(len(labels))
        for k, (arm, (col, _, mk)) in enumerate(ARMS.items()):
            vals = np.array([[json.loads((RUNS / r["run_id"] / "metrics.json").read_text())[key][n] for n in labels]
                             for r in rows if r["arm"] == arm])
            if not len(vals):
                continue
            off = (k - 1.5) * 0.18
            ax.scatter(np.repeat(x + off, len(vals)), np.maximum(vals.T.ravel(), 1e-18), color=col, marker=mk,
                       s=10, label=arm)
            ax.hlines(np.maximum(np.median(vals, 0), 1e-18), x + off - 0.08, x + off + 0.08, color=col)
        ax.set_yscale("log")
        ax.set_xticks(x, labels)
    axes[0].set(title="mean square equation residual (evaluation grid)")
    axes[1].set(title="|boundary error|")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("FB3: where each arm misses the equations and the boundary conditions")
    _letters(axes)
    _save(fig, path)


def fig_training(rows, path):
    """FB4: training curves of seed 0: total loss, monitor loss and the five equation terms."""
    fig, axes = plt.subplots(1, 4, figsize=(WIDE, 55 * MM), layout="constrained", sharey=True)
    for ax, (arm, (col, _, _)) in zip(axes, ARMS.items()):
        rid = next((r["run_id"] for r in rows if r["arm"] == arm and r["seed"] == "0"), None)
        ax.set(title=arm, xlabel="history row")
        if not rid:
            continue
        with open(RUNS / rid / "history.csv") as fh:
            hist = list(csv.DictReader(fh))
        for name, c, ls in [("loss", "k", "-"), ("monitor", "0.5", "--")] + [
                (f"R{i}", list(OI.values())[i - 14], ":") for i in range(14, 19)]:
            y = [float(h[name]) if h.get(name) not in ("", None) else np.nan for h in hist]
            ax.semilogy(y, color=c, ls=ls, lw=0.9, label=name)
        stages = [j for j in range(1, len(hist)) if hist[j]["stage"] != hist[j - 1]["stage"]]
        for j in stages:
            ax.axvline(j, color="0.8", lw=0.6)
    axes[0].set_ylabel("loss term")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=7, frameon=False)
    fig.suptitle("FB4: training curves, seed 0 (grey lines: new stage or L-BFGS)")
    _letters(axes)
    _save(fig, path)


def fig_windows(rows, path):
    """FB5: max error of theta on each window after every stage (retention, B2 and B3)."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 60 * MM), layout="constrained", sharey=True)
    for ax, arm in zip(axes, ("B2", "B3")):
        rid = median_run(rows, arm)
        ax.set(title=f"{arm} (median seed)", xlabel="after stage")
        if not rid:
            continue
        stages = json.loads((RUNS / rid / "metrics.json").read_text())["windows_by_stage"]
        for w, mk in zip(stages[0], "osD^v"):
            ax.semilogy(range(len(stages)), [max(s[w].values()) for s in stages], marker=mk, label=f"window {w}")
        ax.legend(frameon=False, fontsize=6)
    axes[0].set_ylabel("max error over the five fields")
    fig.suptitle("FB5: does training a new window spoil the windows already learned?")
    _letters(axes)
    _save(fig, path)


def wall_table(rows):
    """TB2: wall values of every arm, median and range over seeds, against the reference."""
    keys = ["f2_0", "g1_0", "th1_0", "ph1_0", "ch1_0"]
    lines = ["| Arm | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
    ref = None
    for arm in ARMS:
        vals = [json.loads((RUNS / r["run_id"] / "metrics.json").read_text())["wall"] for r in rows
                if r["arm"] == arm]
        if not vals:
            continue
        ref = vals[0]
        cells = [f"{np.median([v[k]['pinn'] for v in vals]):.5f} ({min(v[k]['pinn'] for v in vals):.4f} to "
                 f"{max(v[k]['pinn'] for v in vals):.4f})" for k in keys]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    if ref:
        lines.append("| reference | " + " | ".join(f"{ref[k]['reference']:.5f}" for k in keys) + " |")
    (RESULTS / "wall_values.md").write_text("# TB2: wall values, median (range) over seeds\n\n" + "\n".join(lines) + "\n")


def main():
    rows = load_summary()
    ref = load_reference(REFERENCES["A"])[0]
    out = RESULTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig_profiles(rows, ref, out / "FB1_profiles")
        fig_errors(rows, ref, out / "FB2_errors")
        fig_residuals(rows, out / "FB3_residuals")
        fig_training(rows, out / "FB4_training")
        fig_windows(rows, out / "FB5_windows")
    wall_table(rows)
    print(f"Figures in {out}; table in {RESULTS / 'wall_values.md'}")


if __name__ == "__main__":
    main()
