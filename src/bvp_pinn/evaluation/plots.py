"""Result figures and tables of one campaign, drawn the way the closest papers of our literature
review draw theirs (docs/pinn/RESULT_FIGURES.md maps every figure to its source, figure by figure).

    uv run python -m bvp_pinn.evaluation.plots                    # the campaign in configs/pinn.json
    uv run python -m bvp_pinn.evaluation.plots --campaign P4-dry

Reads only results/<campaign>/summary.csv, classical_timing.json (if built) and the saved run files,
after evaluate.py. Writes results/<campaign>/figures/ (FB1 to FB8, FX1, FX2 and captions.md) and the
tables wall_values.md (TB2), accuracy.md (TB3), fields.md (TB4), nodes.md (TB5) and cost.md (TX1).
Same style and colours as the reference figures (bvp_pinn/figstyle.py). The parameter-effect figures
that fill most of those papers are left out on purpose: the 22 values are locked, so there is no sweep.
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
PANELS = [("f'", 1), ("f''", 2), ("g", 4), (r"\theta", 6), (r"\phi", 8), (r"\chi", 10)]
EQUATIONS = [f"R{i}" for i in range(14, 19)]
NUMBERS = ("max_field_rel_l2", "wall_rel_max", "wall_seconds", "adam_seconds", "refine_seconds", "lm_steps",
           "parameters", "threads", "after_adam_wall_rel_max", "after_adam_max_field_rel_l2", "mean_residual_mse",
           "lbfgs_seconds", "lm_seconds")
NODES = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)  # TB5 report points, dense at the wall
PHASE_NAMES = {"adam": "Adam", "lbfgs": "L-BFGS", "bfgs": "BFGS", "lm": "LM"}


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


def present_arms(rows):
    return [arm for arm in ARMS if any(r["arm"] == arm for r in rows)]


def prediction(rid):
    with np.load(RUNS / rid / "pred_eval.npz") as data:
        return {k: data[k] for k in data.files}


def report_points(eta, xmax, n=41):
    """Indices of about n reporting points on [0, xmax], dense at the wall for the full domain. Each
    arm draws every k-th of them, offset by its own index, so the markers of different arms never sit
    on top of each other."""
    s = np.linspace(0.0, 1.0, n)
    targets = xmax * np.expm1(3.0 * s) / np.expm1(3.0) if xmax >= eta[-1] else xmax * s
    return np.unique(np.clip(np.searchsorted(eta, targets), 0, eta.size - 1))


def history(rid):
    with open(RUNS / rid / "history.csv") as fh:
        return list(csv.DictReader(fh))


def number(row, key):
    value = row.get(key)
    return float(value) if value not in ("", None) else np.nan


def fig_profiles(rows, ref, path, xmax=None):
    """FB1 (and FB1b with xmax): the reference as a line, each arm's median run as open markers on a set
    of reporting points, in the style of the squeezing-flow paper (Fig. 14), the stretching-cylinder
    paper (Fig. 3) and the hard-constrained Falkner-Skan paper (Fig. 1, with the shear profile f'')."""
    eta = ref["eta"]
    xmax = xmax or float(eta[-1])
    keep = eta <= xmax * (1 + 1e-9)
    base = report_points(eta, xmax)
    arms = present_arms(rows)
    fig, axes = plt.subplots(2, 3, figsize=(WIDE, 100 * MM), layout="constrained")
    for ax, (lab, i) in zip(axes.flat, PANELS):
        ax.axhline(0, color="0.85", lw=0.5, zorder=0)
        ax.plot(eta[keep], ref["z"][i][keep], color="k", lw=1.2, label="reference A (solve_bvp)")
        for k, arm in enumerate(arms):
            col, _, mk = ARMS[arm]
            pred = prediction(median_run(rows, arm))
            idx = base[k::len(arms)]
            ax.plot(pred["eta"][idx], pred["z"][i][idx], ls="none", marker=mk, ms=3.8, mfc="none", mec=col,
                    mew=0.9, label=f"PINN {arm}, median seed")
        ax.set(xlabel=r"$\eta$", title=f"${lab}(\\eta)$", xlim=(0, xmax))
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=5, frameon=False)
    part = "whole domain" if xmax >= eta[-1] else f"wall layer, $0 \\leq \\eta \\leq {xmax:g}$"
    fig.suptitle(f"FB1{'' if xmax >= eta[-1] else 'b'}: PINN against the reference, {part}")
    letters(axes)
    save(fig, path)


def fig_errors(rows, ref, path):
    """FB2: pointwise absolute error of each field on a log scale, one line per arm (median seed), as
    the squeezing-flow paper compares its optimisers (Fig. 15) and the Falkner-Skan paper shows its
    profile errors (Fig. 3a)."""
    fig, axes = plt.subplots(2, 3, figsize=(WIDE, 100 * MM), layout="constrained")
    eta = ref["eta"]
    marks = report_points(eta, float(eta[-1]), 16)
    for ax, (lab, i) in zip(axes.flat, PANELS):
        top = 0.0
        for arm in present_arms(rows):
            col, ls, mk = ARMS[arm]
            pred = prediction(median_run(rows, arm))
            err = np.abs(pred["z"][i] - ref["z"][i]) + 1e-16
            top = max(top, float(np.nanmax(err)))
            ax.semilogy(eta, err, color=col, ls=ls, lw=0.9, marker=mk, ms=3, mfc="none", markevery=list(marks),
                        label=f"{arm}, median seed")
        if top > 0:  # exact boundary values (error 0) would otherwise stretch the axis to 1e-16
            ax.set_ylim(top * 1e-6, top * 3)
        ax.set(xlabel=r"$\eta$", title=f"$|{lab}_{{PINN}} - {lab}_{{ref}}|$", xlim=(0, eta[-1]))
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("FB2: pointwise absolute error against reference A")
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
    """FB4: convergence history of seed 0 of each arm against wall time, with the switches between
    Adam, L-BFGS and Levenberg-Marquardt marked, as the micropolar paper marks its Adam and L-BFGS
    phases (Fig. 5) and the stretching-cylinder paper plots its loss history (Fig. 4). The monitor
    loss next to the training loss is the train-test check of the overfitting paper (Fig. 1d)."""
    arms = present_arms(rows)
    fig, axes = plt.subplots(1, max(len(arms), 1), figsize=(WIDE, 62 * MM), layout="constrained", squeeze=False)
    for ax, arm in zip(axes.flat, arms):
        rid = next((r["run_id"] for r in rows if r["arm"] == arm and r["seed"] == "0"), median_run(rows, arm))
        hist = history(rid)
        minutes = np.array([number(h, "seconds") / 60 for h in hist])
        ax.semilogy(minutes, [number(h, "loss") for h in hist], color="k", lw=1.0, label="training loss")
        mon = np.array([number(h, "monitor") for h in hist])
        ok = np.isfinite(mon)
        ax.semilogy(minutes[ok], mon[ok], color="0.45", ls="--", lw=0.9, label="monitor loss (Adam)")
        for name, col in zip(EQUATIONS, OI.values()):
            ax.semilogy(minutes, [number(h, name) for h in hist], color=col, ls=":", lw=0.8, label=name)
        span = max(float(np.nanmax(minutes) - np.nanmin(minutes)), 1e-9)
        last, level = -np.inf, 0
        for j, h in enumerate(hist):
            if j and (h["phase"] != hist[j - 1]["phase"] or h["stage"] != hist[j - 1]["stage"]):
                ax.axvline(minutes[j], color="0.6", lw=0.6, ls="--")  # new phase, or a new stage of B2, B3
            if j == 0 or h["phase"] != hist[j - 1]["phase"]:
                level = level + 1 if (minutes[j] - last) / span < 0.25 else 0  # stagger labels that crowd
                ax.text(minutes[j], 1.0 + 0.07 * level, PHASE_NAMES.get(h["phase"], h["phase"]),
                        transform=ax.get_xaxis_transform(), va="bottom", ha="left", fontsize=6, color="0.3")
                last = minutes[j]
        b = np.array([[abs(number(h, f"B{i}")) for i in range(1, 13)] for h in hist])
        bmax = np.nanmax(b) if np.isfinite(b).any() else np.nan
        seed = next((r["seed"] for r in rows if r["run_id"] == rid), "?")
        note = f"\nmax |B$_i$| logged: {bmax:.0e}" if np.isfinite(bmax) else ""
        ax.set_title(f"{arm}, seed {seed}{note}", pad=16)
        ax.set(xlabel="wall time (min)")
    axes.flat[0].set_ylabel("loss and mean square residual")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=8, frameon=False)
    fig.suptitle("FB4: convergence history (Adam and L-BFGS on the log scale, LM on the balanced scale)")
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


def fig_residual_profile(rows, path):
    """FB6: pointwise residual of each equation on the 4 001-point grid, one line per arm (median seed),
    as the Falkner-Skan paper shows its pointwise residual (Fig. 3b). The evaluation grid is not the
    training grid, so this also shows the residual between training points (overfitting paper, Fig. 3)."""
    arms = present_arms(rows)
    fig, axes = plt.subplots(2, 3, figsize=(WIDE, 100 * MM), layout="constrained")
    for arm in arms:
        rid = median_run(rows, arm)
        pred = prediction(rid)
        if "R" not in pred:
            continue
        col, ls, _ = ARMS[arm]
        R = np.abs(pred["R"]) + 1e-18
        for ax, name, r in zip(axes.flat, EQUATIONS, R):
            ax.semilogy(pred["eta"], r, color=col, ls=ls, lw=0.7, label=f"{arm}, median seed")
        axes.flat[5].semilogy(pred["eta"], R.max(0), color=col, ls=ls, lw=0.7, label=f"{arm}, median seed")
    for ax, name in zip(axes.flat, EQUATIONS + ["largest of the five"]):
        ax.set(xlabel=r"$\eta$", title=f"|{name}|" if name.startswith("R") else name, xlim=(0, 20))
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("FB6: pointwise equation residual on the 4 001-point evaluation grid")
    letters(axes)
    save(fig, path)


def fig_seed_spread(rows, path):
    """FB7: every seed of every arm, with paired seeds joined across arms and the median marked. The
    squeezing-flow paper plots each metric over its independent runs (Figs. 20 to 22), PINNacle reports
    the spread over seeds, and the Falkner-Skan paper compares variants by their final metrics (Fig. 5b).
    Hollow markers are runs the acceptance rule rejected."""
    keys = [("wall_rel_max", "wall error"), ("max_field_rel_l2", "field rel. L2"),
            ("mean_residual_mse", "residual MSE"), ("wall_seconds", "wall time (s)")]
    arms = present_arms(rows)
    x = {arm: k for k, arm in enumerate(arms)}
    seeds = sorted({r["seed"] for r in rows}, key=lambda s: int(s))
    fig, axes = plt.subplots(1, 4, figsize=(WIDE, 62 * MM), layout="constrained")
    for ax, (key, label) in zip(axes, keys):
        for seed in seeds:
            pts = sorted((x[r["arm"]], r[key]) for r in rows if r["seed"] == seed and np.isfinite(r[key]))
            if len(pts) > 1:
                ax.plot(*zip(*pts), color="0.8", lw=0.6, zorder=1)
        for arm in arms:
            col, _, mk = ARMS[arm]
            rs = [r for r in rows if r["arm"] == arm and np.isfinite(r[key])]
            for r in rs:
                filled = r["accepted"] == "True"
                ax.scatter(x[arm], r[key], marker=mk, s=18, color=col if filled else "white", edgecolor=col,
                           linewidth=0.9, zorder=3)
            if rs:
                ax.hlines(np.median([r[key] for r in rs]), x[arm] - 0.25, x[arm] + 0.25, color="k", lw=1.2, zorder=4)
        ax.set(yscale="log", title=label, xlim=(-0.6, len(arms) - 0.4))
        ax.set_xticks(range(len(arms)), arms)
    fig.suptitle("FB7: every seed (filled: accepted, hollow: rejected), paired seeds joined, median as a bar\n"
                 "wall error and field L2: the largest relative value over the wall values and the five fields")
    letters(axes)
    save(fig, path)


def fig_points(rows, path):
    """FB8: where the training points of one run sit, with the window edges, as the micropolar paper
    shows the distribution of its collocation points (Fig. 4)."""
    rid = next((r["run_id"] for r in rows if (RUNS / r["run_id"] / "points.npz").exists()), None)
    if rid is None:
        return
    with np.load(RUNS / rid / "points.npz") as pts:
        train = pts["train"].reshape(-1)
    edges = read_config()["common"]["windows"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 60 * MM), layout="constrained")
    for ax, (hi, width) in zip(axes, ((20.0, 0.25), (1.0, 0.02))):
        ax.hist(train[train <= hi], bins=np.arange(0.0, hi + width, width), color=OI["sky"], edgecolor="white", lw=0.3)
        for e in edges[1:-1]:
            if e < hi:
                ax.axvline(e, color="0.4", ls="--", lw=0.7)
        ax.set(xlabel=r"$\eta$", ylabel=f"training points per {width:g}", xlim=(0, hi))
    share = float(np.mean(train <= 0.5))
    axes[0].set_title(f"{train.size} points, {share:.0%} in [0, 0.5]")
    axes[1].set_title("wall layer (dashed: window edges of B1 to B3)")
    fig.suptitle(f"FB8: training points of run {rid}")
    letters(axes)
    save(fig, path)


def fig_time_split(rows, path):
    """FX2: median time per arm split into Adam, L-BFGS and Levenberg-Marquardt, as the micropolar
    paper profiles its training cost (Fig. 6)."""
    arms = present_arms(rows)
    parts = [("adam_seconds", "Adam", OI["orange"]), ("lbfgs_seconds", "L-BFGS", OI["sky"]),
             ("lm_seconds", "Levenberg-Marquardt", OI["purple"])]
    fig, ax = plt.subplots(figsize=(120 * MM, 34 * MM + 10 * MM * len(arms)), layout="constrained")
    for k, arm in enumerate(arms):
        left = 0.0
        for key, label, col in parts:
            v = float(np.nanmedian([r[key] for r in rows if r["arm"] == arm])) / 60
            v = v if np.isfinite(v) else 0.0
            ax.barh(k, v, left=left, color=col, edgecolor="white", label=label if k == 0 else None)
            left += v
        ax.text(left, k, f"  {left:.0f} min", va="center", fontsize=7)
    # runs that shared the machine take longer than alone; the label says how many trained at once
    ax.set_yticks(range(len(arms)), [f"{arm}\n{at_once([r for r in rows if r['arm'] == arm])} at once" for arm in arms])
    ax.invert_yaxis()
    ax.set(xlabel=f"median minutes per run ({devices(rows)})")
    fig.legend(loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("FX2: where the time of a run goes")
    save(fig, path)


def fig_cost(rows, timing, path):
    """FX1: largest relative wall error against wall time, PINN runs and solve_bvp."""
    fig, ax = plt.subplots(figsize=(120 * MM, 80 * MM), layout="constrained")
    for arm, (col, _, mk) in ARMS.items():
        pts = [(r["wall_seconds"], r["wall_rel_max"]) for r in rows if r["arm"] == arm]
        if pts:
            at = at_once([r for r in rows if r["arm"] == arm])
            ax.scatter(*zip(*pts), color=col, marker=mk, s=16, label=f"{arm} (one dot per seed, {at} at once)")
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


def at_once(rows):
    """The runs-at-once values of these runs, as text: "1", "1, 2" or "unknown"."""
    values = sorted({str(r.get("runs_at_once") or "unknown") for r in rows})
    return ", ".join(values)


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
    rel = ["", "## Relative error against reference A, median (largest) over seeds", "",
           "| Arm | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
    for arm in ARMS:
        vals = [metrics(r["run_id"])["wall"] for r in rows if r["arm"] == arm]
        if vals:
            rel.append(f"| {arm} | " + " | ".join(
                f"{np.median([v[k]['rel_error'] for v in vals]):.1e} ({max(v[k]['rel_error'] for v in vals):.1e})"
                for k in keys) + " |")
    (out / "wall_values.md").write_text("# TB2: wall values, median (range) over seeds\n\n"
                                        "Present values against the reference, as the micropolar paper (Table 1) and the "
                                        "stretching-cylinder paper (Table 7) report wall quantities.\n\n"
                                        + "\n".join(lines + rel) + "\n")


FIELD_COLUMNS = [("f", "f"), ("f1", "f′"), ("f2", "f″"), ("g", "g"), ("theta", "θ"), ("phi", "φ"), ("chi", "χ")]


def field_table(rows, out):
    """TB4: error of every field per arm, median (range) over seeds: relative L2 and max abs error, as the
    stretching-cylinder paper (Tables 2 to 6) and the Falkner-Skan paper (Table 2) report them."""
    text = ["# TB4: error per field, median (range) over seeds", "",
            "Relative L2 error over [0, 20] (trapezoid weights on the 4 001-point grid) and the largest absolute "
            "error, against reference A.", ""]
    for key, title in (("rel_l2", "Relative L2 error"), ("max_abs", "Largest absolute error")):
        text += [f"## {title}", "", "| Arm | " + " | ".join(c for _, c in FIELD_COLUMNS) + " |",
                 "|---|" + "---:|" * len(FIELD_COLUMNS)]
        for arm in ARMS:
            ms = [metrics(r["run_id"])["fields"] for r in rows if r["arm"] == arm]
            if not ms:
                continue
            cells = []
            for name, _ in FIELD_COLUMNS:
                v = np.array([m[name][key] for m in ms], dtype=float)
                cells.append(f"{np.median(v):.1e} ({v.min():.1e} to {v.max():.1e})")
            text.append(f"| {arm} | " + " | ".join(cells) + " |")
        text.append("")
    (out / "fields.md").write_text("\n".join(text))


def node_table(rows, ref, out):
    """TB5: absolute error at fixed report points, mean (smallest to largest) over the seeds of each arm,
    as the squeezing-flow paper tabulates its error node by node over independent runs (Tables 4 to 7)."""
    eta = ref["eta"]
    idx = [int(np.argmin(np.abs(eta - x))) for x in NODES]
    arms = present_arms(rows)
    text = ["# TB5: absolute error at report points, mean (smallest to largest) over seeds", ""]
    for i, name in ((1, "f′"), (6, "θ"), (8, "φ")):
        text += [f"## {name}", "", "| η | reference | " + " | ".join(arms) + " |", "|---:|---:|" + "---:|" * len(arms)]
        errs = {arm: np.array([np.abs(prediction(r["run_id"])["z"][i][idx] - ref["z"][i][idx])
                               for r in rows if r["arm"] == arm]) for arm in arms}
        for n, j in enumerate(idx):
            cells = [f"{e[:, n].mean():.1e} ({e[:, n].min():.1e} to {e[:, n].max():.1e})" for e in errs.values()]
            value = f"{ref['z'][i][j]:.5f}".replace("-0.00000", "0.00000")
            text.append(f"| {eta[j]:.3g} | {value} | " + " | ".join(cells) + " |")
        text.append("")
    (out / "nodes.md").write_text("\n".join(text))


CAPTIONS = """# Result figures and tables: what each shows and where its form comes from

Built by `python -m bvp_pinn.evaluation.plots`. Every figure follows a figure of one of the closest papers
in our literature review (docs/pinn/RESULT_FIGURES.md). Parameter-effect figures are left out: the values
are locked, so there is no parameter sweep.

| ID | What it shows | Form taken from |
|---|---|---|
| FB1 | f′, f″, g, θ, φ, χ over [0, 20]: reference A as a line, each arm's median-seed PINN as markers | squeezing flow 2026 Fig. 14; stretching cylinder 2026 Fig. 3; Falkner-Skan 2026 Fig. 1 (velocity and shear) |
| FB1b | the same, wall layer 0 ≤ η ≤ 1 | as FB1; our reference figures already split the wall layer |
| FB2 | absolute error of each field along η, log scale, one line per arm | squeezing flow 2026 Fig. 15; Falkner-Skan 2026 Fig. 3a |
| FB3 | final mean square residual per equation and error per condition, every seed | Falkner-Skan 2026 Fig. 5b; micropolar 2026 Fig. 5 (equations and conditions apart) |
| FB4 | loss, monitor loss and the five residuals against wall time, phase switches marked | micropolar 2026 Fig. 5; stretching cylinder 2026 Fig. 4; overfitting 2026 Fig. 1d (train against test) |
| FB5 | largest error per window after each stage, B2 and B3 | ours (retention question of PINN_PLAN) |
| FB6 | absolute residual of each equation along η, training points as ticks | Falkner-Skan 2026 Fig. 3b; overfitting 2026 Fig. 3 |
| FB7 | every seed of every arm for four metrics, paired seeds joined, medians | squeezing flow 2026 Figs. 20 to 22; PINNacle 2024 (spread over seeds); Falkner-Skan 2026 Fig. 5b |
| FB8 | histogram of the training points over η, window edges | micropolar 2026 Fig. 4 |
| FX1 | largest wall error against wall time, PINN runs and solve_bvp; legend gives the runs at once per arm | PINN against FEM 2024 Figs. 2 and 4 |
| FX2 | median minutes per run split into Adam, L-BFGS, Levenberg-Marquardt; label gives the runs at once (more than 1: times longer than alone) | micropolar 2026 Fig. 6 |
| TB2 | wall values per arm against reference A, with relative errors | micropolar 2026 Table 1; stretching cylinder 2026 Table 7 |
| TB3 | accuracy after Adam and final, paired-seed counts for rule G6 | PINNacle 2024 (mean and spread, failures kept) |
| TB4 | relative L2 and largest absolute error per field | stretching cylinder 2026 Tables 2 to 6; Falkner-Skan 2026 Table 2 |
| TB5 | absolute error of f′, θ, φ at report points over the seeds | squeezing flow 2026 Tables 4 to 7 |
| TX1 | cost per arm and the solve_bvp time to the same error | PINN against FEM 2024 |
"""


def cost_table(rows, timing, out):
    """TX1: cost per arm (medians over seeds) and the solve_bvp time to reach each arm's median error."""
    lines = ["| Arm | Runs | Accepted | Device | Runs at once | Wall s | Adam s | Refine s | LM steps | Parameters | "
             "Threads | Median wall error | solve_bvp s to the same error |", "|---|---:|---:|---|---|" + "---:|" * 8]
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        med = {k: float(np.nanmedian([r[k] for r in rs])) for k in NUMBERS}
        reach = time_to_reach(timing, med["wall_rel_max"]) if timing else None
        lines.append(f"| {arm} | {len(rs)} | {sum(r['accepted'] == 'True' for r in rs)} | {devices(rs)} | "
                     f"{at_once(rs)} | {med['wall_seconds']:.0f} | "
                     f"{med['adam_seconds']:.0f} | {med['refine_seconds']:.0f} | {med['lm_steps']:.0f} | "
                     f"{med['parameters']:.0f} | {med['threads']:.0f} | {med['wall_rel_max']:.2e} | "
                     f"{'not built' if timing is None else ('n/a' if reach is None else f'{reach:.3f}')} |")
    note = ("solve_bvp times come from classical_timing.json (median of its repeats, one CPU thread). "
            "PINN times are on the device in the Device column. Compare only times measured on the same machine, "
            "and say which device each side used. Runs at once: the most runs that trained on the same machine at "
            "the same time as a run (schedule.json); with more than 1 the runs shared the CPUs and the GPU, so their "
            "times are longer than one run alone takes. Compare times only between runs with the same value.")
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
        fig_profiles(rows, ref, figs / "FB1b_profiles_wall", xmax=1.0)
        fig_errors(rows, ref, figs / "FB2_errors")
        fig_residuals(rows, figs / "FB3_residuals")
        fig_training(rows, figs / "FB4_training")
        fig_windows(rows, figs / "FB5_windows")
        fig_residual_profile(rows, figs / "FB6_pointwise_residual")
        fig_seed_spread(rows, figs / "FB7_seed_spread")
        fig_points(rows, figs / "FB8_points")
        fig_cost(rows, timing, figs / "FX1_error_vs_time")
        fig_time_split(rows, figs / "FX2_time_split")
    (figs / "captions.md").write_text(CAPTIONS)
    wall_table(rows, out)
    accuracy_table(rows, out)
    field_table(rows, out)
    node_table(rows, ref, out)
    cost_table(rows, timing, out)
    print(f"Figures and captions.md in {figs}; tables wall_values.md, accuracy.md, fields.md, nodes.md and "
          f"cost.md in {out}")

if __name__ == "__main__":
    main()
