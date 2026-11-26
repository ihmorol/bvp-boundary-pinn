"""Result figures FB1 to FB5 and FX1, tables TB2 and TX1 (docs/pinn/PINN_PLAN.md section 6).

    uv run python -m bvp_pinn.evaluation.plots                    # the campaign in configs/pinn.json
    uv run python -m bvp_pinn.evaluation.plots --campaign P4-dry

Reads only results/<campaign>/summary.csv, classical_timing.json (if built) and the saved run
files, after evaluate.py. Writes results/<campaign>/figures/, wall_values.md (TB2), accuracy.md
(TB3: per arm after Adam and final, with the paired-seed counts of rule G6) and cost.md (TX1).
Same style and colours as the reference figures (bvp_pinn/figstyle.py).
"""
from __future__ import annotations

import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from ..figstyle import MM, OI, STYLE, WIDE, letters, save
from ..pinn.runs import RUNS, read_config
from ..reference.io import load_reference
from .classical import time_to_reach
from .evaluate import REFERENCES, WALL, results_dir

ARMS = {"B0": (OI["blue"], "-", "o"), "B1": (OI["vermillion"], "--", "s"),
        "B2": (OI["green"], "-.", "^"), "B3": (OI["purple"], ":", "D")}
PANELS = [("f", 0), ("f'", 1), ("g", 4), (r"\theta", 6), (r"\phi", 8), (r"\chi", 10)]
NUMBERS = ("max_field_rel_l2", "wall_rel_max", "wall_seconds", "adam_seconds", "refine_seconds", "lm_steps",
           "parameters", "threads", "after_adam_wall_rel_max", "after_adam_max_field_rel_l2")


def load_summary(campaign):
    path = results_dir(campaign) / "summary.csv"
    if not path.exists():
        raise SystemExit(f"no {path}: campaign {campaign} has no evaluated runs yet (run bvp_pinn.evaluation.evaluate first)")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in NUMBERS:
            r[k] = float(r[k]) if r.get(k) not in ("", None) else float("nan")
    return rows


def metrics(rid):
    return json.loads((RUNS / rid / "metrics.json").read_text())


def median_run(rows, arm):
    """The run with the median error for one arm, so a figure shows a typical seed, not the best."""
    finite = lambda v: v if np.isfinite(v) else np.inf  # a failed run sorts last, never in the middle
    runs = sorted((r for r in rows if r["arm"] == arm), key=lambda r: finite(r["max_field_rel_l2"]))
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
    letters(axes)
    save(fig, path)


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
    letters(axes)
    save(fig, path)


def fig_residuals(rows, path):
    """FB3: final equation residual and boundary error per arm; dots are seeds, bars the median."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 70 * MM), layout="constrained")
    names = [("residual_mse", [f"R{i}" for i in range(14, 19)]), ("boundary_abs", [f"B{i}" for i in range(1, 13)])]
    for ax, (key, labels) in zip(axes, names):
        x = np.arange(len(labels))
        for k, (arm, (col, _, mk)) in enumerate(ARMS.items()):
            vals = np.array([[metrics(r["run_id"])[key][n] for n in labels] for r in rows if r["arm"] == arm])
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
    letters(axes)
    save(fig, path)


def fig_training(rows, path):
    """FB4: training curves of seed 0: total loss, monitor loss and the five equation terms."""
    fig, axes = plt.subplots(1, 4, figsize=(WIDE, 55 * MM), layout="constrained", sharey=True)
    for ax, (arm, _) in zip(axes, ARMS.items()):
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
        for j in range(1, len(hist)):
            if hist[j]["stage"] != hist[j - 1]["stage"]:
                ax.axvline(j, color="0.8", lw=0.6)
    axes[0].set_ylabel("loss term")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=7, frameon=False)
    fig.suptitle("FB4: training curves, seed 0 (grey lines: new stage or refine stage)")
    letters(axes)
    save(fig, path)


def fig_windows(rows, path):
    """FB5: max error on each window after every stage (retention, B2 and B3)."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 60 * MM), layout="constrained", sharey=True)
    for ax, arm in zip(axes, ("B2", "B3")):
        rid = median_run(rows, arm)
        ax.set(title=f"{arm} (median seed)", xlabel="after stage")
        if not rid:
            continue
        stages = metrics(rid)["windows_by_stage"]
        for w, mk in zip(stages[0], "osD^v"):
            ax.semilogy(range(len(stages)), [max(s[w].values()) for s in stages], marker=mk, label=f"window {w}")
        ax.legend(frameon=False, fontsize=6)
    axes[0].set_ylabel("max error over the five fields")
    fig.suptitle("FB5: does training a new window spoil the windows already learned?")
    letters(axes)
    save(fig, path)


def fig_cost(rows, timing, path):
    """FX1: largest relative wall error against wall time, PINN runs and solve_bvp."""
    fig, ax = plt.subplots(figsize=(120 * MM, 80 * MM), layout="constrained")
    for arm, (col, _, mk) in ARMS.items():
        pts = [(r["wall_seconds"], r["wall_rel_max"]) for r in rows if r["arm"] == arm]
        if pts:
            ax.scatter(*zip(*pts), color=col, marker=mk, s=16, label=f"{arm} (one dot per seed)")
    if timing:
        ok = [r for r in timing["rows"] if r.get("wall_rel_max") is not None]
        ax.scatter([r["median_seconds"] for r in ok], [r["wall_rel_max"] for r in ok], color="k", marker="x",
                   s=20, label="solve_bvp (label: tol)")
        for r in ok:
            ax.annotate(f"{r['tol']:g}", (r["median_seconds"], r["wall_rel_max"]), textcoords="offset points",
                        xytext=(4, 3), fontsize=6)
    ax.set(xscale="log", yscale="log", xlabel=f"wall time (s); PINN on {devices(rows)}, solve_bvp on one CPU thread",
           ylabel="largest relative wall error")
    ax.legend(frameon=False)
    fig.suptitle("FX1: error against wall time")
    save(fig, path)


def devices(rows):
    """The devices the runs trained on, as text, for labels."""
    names = {(r.get("device") or "unknown") + (f" ({r['gpu']})" if r.get("gpu") else "") for r in rows}
    return ", ".join(sorted(names)) or "unknown"


def accuracy_table(rows, out):
    """TB3: accuracy of each arm after Adam and after the refine chain, median and range over seeds,
    and the paired-seed counts that rule G6 needs ("arm X beats arm Y" only if it holds in every
    paired seed). A run with a missing or non-finite error never counts as the better one."""
    keys = [("wall_rel_max", "largest relative wall error"), ("max_field_rel_l2", "largest field relative L2")]
    fmt = lambda v: f"{v:.2e}" if np.isfinite(v) else "n/a"
    lines = ["| Arm | Runs | Accepted | " + " | ".join(f"{label}, after Adam | {label}, final" for _, label in keys) + " |",
             "|---|---:|---:|" + "---|---|" * len(keys)]
    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in ARMS}
    for arm, rs in by_arm.items():
        if not rs:
            continue
        cells = []
        for k, _ in keys:
            for col in (f"after_adam_{k}", k):
                v = np.array([r[col] for r in rs], dtype=float)
                v = v[np.isfinite(v)]
                cells.append("n/a" if not v.size else f"{fmt(np.median(v))} ({fmt(v.min())} to {fmt(v.max())})")
        lines.append(f"| {arm} | {len(rs)} | {sum(r['accepted'] == 'True' for r in rs)} | " + " | ".join(cells) + " |")
    pairs = ["", "## Paired seeds (rule G6)", "",
             "Each cell: in how many paired seeds the row arm has the smaller final error than the column arm, "
             "out of the seeds both arms have. A claim needs every paired seed.", ""]
    arms = [a for a, rs in by_arm.items() if rs]
    for k, label in keys:
        pairs += [f"**{label}**", "", "| Row beats column | " + " | ".join(arms) + " |", "|---|" + "---:|" * len(arms)]
        for a in arms:
            cells = []
            for b in arms:
                if a == b:
                    cells.append("")
                    continue
                ea = {r["seed"]: float(r[k]) for r in by_arm[a]}
                eb = {r["seed"]: float(r[k]) for r in by_arm[b]}
                both = sorted(set(ea) & set(eb))
                wins = sum(bool(np.isfinite(ea[s]) and (not np.isfinite(eb[s]) or ea[s] < eb[s])) for s in both)
                cells.append(f"{wins} of {len(both)}")
            pairs.append(f"| {a} | " + " | ".join(cells) + " |")
        pairs.append("")
    (out / "accuracy.md").write_text("# TB3: accuracy per arm, median (range) over seeds\n\n" + "\n".join(lines + pairs))


def wall_table(rows, out):
    """TB2: wall values of every arm, median and range over seeds, against the reference."""
    keys = list(WALL)
    lines = ["| Arm | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
    ref = None
    for arm in ARMS:
        vals = [metrics(r["run_id"])["wall"] for r in rows if r["arm"] == arm]
        if not vals:
            continue
        ref = vals[0]
        cells = [f"{np.median([v[k]['pinn'] for v in vals]):.5f} ({min(v[k]['pinn'] for v in vals):.4f} to "
                 f"{max(v[k]['pinn'] for v in vals):.4f})" for k in keys]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    if ref:
        lines.append("| reference A | " + " | ".join(f"{ref[k]['reference']:.5f}" for k in keys) + " |")
    (out / "wall_values.md").write_text("# TB2: wall values, median (range) over seeds\n\n" + "\n".join(lines) + "\n")


def cost_table(rows, timing, out):
    """TX1: cost per arm (medians over seeds) and the solve_bvp time to reach each arm's median error."""
    lines = ["| Arm | Runs | Accepted | Device | Wall s | Adam s | Refine s | LM steps | Parameters | Threads | "
             "Median wall error | solve_bvp s to the same error |", "|---|---:|---:|---|" + "---:|" * 8]
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        med = {k: float(np.nanmedian([r[k] for r in rs])) for k in NUMBERS}
        reach = time_to_reach(timing, med["wall_rel_max"]) if timing else None
        lines.append(f"| {arm} | {len(rs)} | {sum(r['accepted'] == 'True' for r in rs)} | {devices(rs)} | "
                     f"{med['wall_seconds']:.0f} | "
                     f"{med['adam_seconds']:.0f} | {med['refine_seconds']:.0f} | {med['lm_steps']:.0f} | "
                     f"{med['parameters']:.0f} | {med['threads']:.0f} | {med['wall_rel_max']:.2e} | "
                     f"{'not built' if timing is None else ('n/a' if reach is None else f'{reach:.3f}')} |")
    note = ("solve_bvp times come from classical_timing.json (median of its repeats, one CPU thread). "
            "PINN times are on the device in the Device column. Compare only times measured on the same machine, "
            "and say which device each side used.")
    (out / "cost.md").write_text("# TX1: cost per arm, median over seeds\n\n" + "\n".join(lines) + f"\n\n{note}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--campaign", default=read_config()["campaign"]["name"])
    args = ap.parse_args(argv)
    out = results_dir(args.campaign)
    rows = load_summary(args.campaign)
    ref = load_reference(REFERENCES["A"])[0]
    timing_path = out / "classical_timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else None
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig_profiles(rows, ref, figs / "FB1_profiles")
        fig_errors(rows, ref, figs / "FB2_errors")
        fig_residuals(rows, figs / "FB3_residuals")
        fig_training(rows, figs / "FB4_training")
        fig_windows(rows, figs / "FB5_windows")
        fig_cost(rows, timing, figs / "FX1_error_vs_time")
    wall_table(rows, out)
    accuracy_table(rows, out)
    cost_table(rows, timing, out)
    print(f"Figures in {figs}; tables in {out / 'wall_values.md'}, {out / 'accuracy.md'} and {out / 'cost.md'}")


if __name__ == "__main__":
    main()
