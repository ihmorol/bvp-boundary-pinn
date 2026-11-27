"""Compare finished runs with the classical reference (docs/pinn/PINN_PLAN.md section 4).

    uv run python -m bvp_pinn.evaluation.evaluate                     # the campaign in configs/pinn.json
    uv run python -m bvp_pinn.evaluation.evaluate --campaign P4-dry   # the dry-run campaign
    uv run python -m bvp_pinn.evaluation.evaluate --runs runs/<run_id> ...   # pilots: metrics.json only

Runs after training and after the acceptance rule (pinn/estimate.py), so the reference never
influences training or selection (gate G5). For each run it writes metrics.json and
pred_eval.npz in the run folder. For a campaign it then rebuilds results/<campaign>/summary.csv
and failures.md from the run folders that carry that campaign's tag, never from pilots.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import scipy
import torch

from ..config import ROOT, load_case, load_reference_settings
from ..pinn.model import predict
from ..pinn.losses import boundary_errors
from ..pinn.physics import residuals
from ..pinn.runs import RUNS, git_commit, is_finished, load_run, read_config, write_json
from ..reference.equations import HIGH_NAMES, STATE_NAMES
from ..reference.io import load_reference


def reference_folders():
    """The reference folder of each branch at the reference L and the final tolerance of
    configs/reference.json, named as bvp_pinn.reference.run names them."""
    s = load_reference_settings()
    tag = f"L{float(s['L_reference']):g}-tol{float(s['tol_values'][-1]):.0e}"
    return {b: ROOT / "reference" / f"locked-branch{b}-{tag}" for b in ("A", "B")}


REFERENCES = reference_folders()
FIELDS = {"f": 0, "g": 4, "theta": 6, "phi": 8, "chi": 10}
WALL = {"f2_0": 2, "theta_0": 6, "th1_0": 7, "phi_0": 8, "ph1_0": 9, "g1_0": 5, "ch1_0": 11}
# BVP_RESULTS_DIR moves the campaign results, for example to Google Drive on Colab.
RESULTS = Path(os.environ.get("BVP_RESULTS_DIR") or ROOT / "results")


def results_dir(campaign):
    return RESULTS / campaign


def load_references():
    """Reference arrays by branch. Branch A is the target. It must exist, sit at the reference L,
    pass its fixed-L checks (G3) and decay in the far field (m > 0), or nothing is compared with it.
    Branch B is only used to name the nearest branch, so it may be missing."""
    L = float(load_reference_settings()["L_reference"])
    out = {}
    for name, folder in REFERENCES.items():
        if name != "A" and not folder.exists():
            continue
        arrays, meta = load_reference(folder)
        if name == "A":
            checks = meta.get("checks", {})
            if float(meta["L"]) != L or not checks.get("G3_fixed_L_pass") or not checks.get("far_field_valid"):
                raise ValueError(f"{folder.name} is not a usable target: L {meta['L']} (want {L:g}), "
                                 f"G3 fixed-L pass {checks.get('G3_fixed_L_pass')}, "
                                 f"decaying far field {checks.get('far_field_valid')}")
        out[name] = arrays
    return out


def worst(values):
    """The largest of the values, or NaN if any is NaN. Python's max() is order-dependent with NaN,
    so a diverged run could otherwise show a finite "largest error"."""
    return float(np.max(np.fromiter(values, dtype=float)))


def rel_l2(got, want, eta):
    """Relative L2 error with trapezoid weights, since the reference grid is not uniform."""
    return float(np.sqrt(np.trapezoid((got - want) ** 2, eta) / np.trapezoid(want**2, eta)))


def wall_errors(z0, ref):
    """Wall values against reference A: absolute and relative. z0 holds the 12 states at eta = 0."""
    out = {}
    for k, i in WALL.items():
        want = float(ref["z"][i, 0])
        out[k] = {"pinn": float(z0[i]), "reference": want, "abs_error": float(abs(z0[i] - want)),
                  "rel_error": float(abs(z0[i] - want) / abs(want))}
    return out


def evaluate_run(folder, refs, p):
    """metrics.json content for one run; also writes pred_eval.npz."""
    cfg, model = load_run(folder)
    ref = refs["A"]
    eta = ref["eta"]
    if float(cfg["L"]) != float(eta[-1]):
        raise ValueError(f"{Path(folder).name} was trained on L = {cfg['L']}, the reference ends at {eta[-1]:g}")
    z, H = (a.numpy() for a in predict(model, torch.tensor(eta).reshape(-1, 1)))
    got, want = np.vstack([z, H]), np.vstack([ref["z"], ref["H"]])
    cols = [torch.tensor(r).reshape(-1, 1) for r in z]
    res = residuals(torch.tensor(eta).reshape(-1, 1), cols, [torch.tensor(r).reshape(-1, 1) for r in H], p)
    wall = wall_errors(z[:, 0], ref)
    metrics = {
        "fields": {n: {"rel_l2": rel_l2(g, w, eta), "max_abs": float(np.max(np.abs(g - w)))}
                   for n, g, w in zip(STATE_NAMES + HIGH_NAMES, got, want)},
        "wall": wall,
        "wall_rel_max": worst(v["rel_error"] for v in wall.values()),
        "residual_mse": dict(zip(("R14", "R15", "R16", "R17", "R18"), map(float, res.pow(2).mean(0)))),
        "boundary_abs": dict(zip([f"B{i}" for i in range(1, 13)],
                                 map(float, boundary_errors(model, p, cfg["L"]).detach().abs()))),
        "nearest_branch": nearest_branch(z, refs),
        "windows_by_stage": window_errors(folder, cfg, ref),
        "after_adam": after_adam(folder, refs),
    }
    est = Path(folder) / "error_estimate.json"  # made without the reference, by pinn/estimate.py
    if est.exists():
        wall_est = json.loads(est.read_text()).get("wall_estimated_error") or {}
        metrics["error_estimate"] = {k: {"estimated": wall_est[k], "true": wall[k]["pinn"] - wall[k]["reference"]}
                                     for k in WALL if k in wall_est}
    # R: the five residual rows on the grid, for the pointwise residual figure (FB6)
    np.savez(Path(folder) / "pred_eval.npz", eta=eta, z=z, H=H, R=res.detach().numpy().T)
    return metrics


def window_errors(folder, cfg, ref):
    """Max abs error of each field on each window, after every stage (retention, gate G4)."""
    edges = cfg["windows"]
    out = []
    for path in sorted(Path(folder).glob("stage*.pt"), key=lambda q: int(q.stem[5:])):
        _, model = load_run(folder, path.name)
        z = predict(model, torch.tensor(ref["eta"]).reshape(-1, 1))[0].numpy()
        rows = {}
        for a, b in zip(edges[:-1], edges[1:]):
            m = (ref["eta"] >= a) & (ref["eta"] <= b)
            rows[f"{a:g}-{b:g}"] = {n: float(np.max(np.abs(z[i, m] - ref["z"][i, m]))) for n, i in FIELDS.items()}
        out.append(rows)
    return out


def after_adam(folder, refs):
    """Accuracy of the state after the last Adam stage, before L-BFGS and Levenberg-Marquardt.

    Every arm ends on the same refine chain, which may hide arm differences (TRAINING_WALKTHROUGH
    section 7), so each arm is also reported here, before it."""
    stages = sorted(Path(folder).glob("stage*.pt"), key=lambda q: int(q.stem[5:]))
    if not stages:
        return None
    ref = refs["A"]
    _, model = load_run(folder, stages[-1].name)
    z = predict(model, torch.tensor(ref["eta"]).reshape(-1, 1))[0].numpy()
    wall = wall_errors(z[:, 0], ref)
    return {"checkpoint": stages[-1].name, "wall_rel_max": worst(v["rel_error"] for v in wall.values()),
            "max_field_rel_l2": worst(rel_l2(z[i], ref["z"][i], ref["eta"]) for i in FIELDS.values()),
            "nearest_branch": nearest_branch(z, refs)}


def nearest_branch(z, refs):
    """Which reference branch the PINN theta is closest to ("none" if it is not finite)."""
    d = {k: rel_l2(z[6], r["z"][6], r["eta"]) for k, r in refs.items()}
    if not all(math.isfinite(v) for v in d.values()):
        return "none"
    return min(d, key=d.get)


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def summary_row(folder, metrics):
    """One row of summary.csv: identity, acceptance, accuracy, then cost."""
    cfg = read_json(folder / "config.json")
    meta = read_json(folder / "meta.json")
    selection = read_json(folder / "selection.json")
    estimate = read_json(folder / "error_estimate.json")
    schedule = read_json(folder / "schedule.json")  # written by the campaign; missing for pilots
    f = metrics["fields"]
    stages = meta.get("refine_stages", [])
    lm_steps = sum(s.get("lm_iterations", 0) for s in stages)
    stage_seconds = lambda name: sum(s.get("refine_seconds", 0.0) for s in stages if s.get("refine") == name)
    row = {"run_id": folder.name, "arm": cfg["arm"], "seed": cfg["seed"], "campaign": cfg.get("campaign", ""),
           "result": meta.get("result"), "accepted": selection.get("accepted"),
           "reasons": "; ".join(selection.get("reasons", [])), "estimate_score": estimate.get("score"),
           "nearest_branch": metrics["nearest_branch"], "wall_rel_max": metrics["wall_rel_max"],
           "max_field_rel_l2": worst(f[n]["rel_l2"] for n in FIELDS)}
    adam = metrics.get("after_adam") or {}
    row.update({"after_adam_wall_rel_max": adam.get("wall_rel_max"),
                "after_adam_max_field_rel_l2": adam.get("max_field_rel_l2"),
                "after_adam_nearest_branch": adam.get("nearest_branch")})
    row.update({f"rel_l2_{n}": f[n]["rel_l2"] for n in FIELDS})
    row.update({f"max_abs_{n}": f[n]["max_abs"] for n in FIELDS})
    row.update({f"wall_rel_{k}": v["rel_error"] for k, v in metrics["wall"].items()})
    row["max_boundary_abs"] = max(metrics["boundary_abs"].values())
    row["mean_residual_mse"] = float(np.mean(list(metrics["residual_mse"].values())))
    row.update({"parameters": meta.get("parameters"), "threads": meta.get("threads"), "device": meta.get("device"), "gpu": meta.get("gpu"),
                "wall_seconds": meta.get("wall_seconds"), "adam_seconds": meta.get("adam_seconds"),
                "refine_seconds": meta.get("refine_seconds"), "refine_accepted": meta.get("refine_accepted",
                                                                                          meta.get("lbfgs_accepted")),
                "lm_steps": lm_steps, "lbfgs_seconds": stage_seconds("lbfgs"), "lm_seconds": stage_seconds("lm"),
                # runs training on the same machine at the same time, at most: times with more than 1 are longer
                # than the run would take alone
                "runs_at_once": schedule.get("max_runs_at_once")})
    return row


def campaign_folders(name, runs=RUNS):
    """Every run folder tagged with this campaign, finished or not."""
    return sorted(x for x in Path(runs).glob("bvp-*") if read_json(x / "config.json").get("campaign") == name)


def failure_lines(rows, folders, name):
    """results/<campaign>/failures.md: every run that failed, was rejected or is missing."""
    lines = []
    for r in rows:
        why = []
        if r["result"] != "finished":
            why.append(f"result: {r['result']}")
        if r["accepted"] is False:
            why.append(f"rejected ({r['reasons']})")
        elif r["accepted"] is None:
            why.append(f"acceptance not decided ({r['reasons'] or 'no selection.json'})")
        if r["nearest_branch"] != "A":
            why.append(f"nearest branch {r['nearest_branch']}")
        if why:
            lines.append(f"- `{r['run_id']}` ({r['arm']}, seed {r['seed']}): " + "; ".join(why))
    for x in folders:
        if not is_finished(x):
            lines.append(f"- `{x.name}`: started but never finished (no meta.json)")
        elif not (x / "model.pt").exists():
            lines.append(f"- `{x.name}`: {read_json(x / 'meta.json').get('result', 'no model saved')}")
    block = read_config()["campaign"]
    if name in (block["name"], f"{block['name']}-dry"):
        have = {}
        for r in rows:
            have.setdefault((r["arm"], int(r["seed"])), []).append(r["run_id"])
        for arm in block["arms"]:
            for seed in block["seeds"]:
                ids = have.get((arm, seed), [])
                if not ids:
                    lines.append(f"- {arm}, seed {seed}: no finished run")
                elif len(ids) > 1:
                    lines.append(f"- {arm}, seed {seed}: {len(ids)} runs ({', '.join(ids)}); the config changed mid-campaign")
    return lines


def settings_record(name, folders):
    """Everything that produced this campaign's results, saved next to them (settings.json)."""
    import matplotlib
    case_sha = load_case()[1]
    refs = {b: {"folder": f.name, "sha256sums": (f / "SHA256SUMS").read_text().splitlines(),
                "case_sha256_matches": json.loads((f / "meta.json").read_text())["case_sha256"] == case_sha}
            for b, f in REFERENCES.items() if (f / "SHA256SUMS").exists()}
    return {"campaign": name, "written": datetime.datetime.now().isoformat(timespec="seconds"),
            "case_sha256": case_sha, "pinn_config": read_config(), "reference_settings": load_reference_settings(),
            "reference": refs, "git": git_commit(),
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                         "torch": torch.__version__, "matplotlib": matplotlib.__version__},
            "run_folders": [x.name for x in folders]}


def write_campaign(name, rows, lines, folders=()):
    out = results_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "settings.json", settings_record(name, folders))
    if rows:
        with open(out / "summary.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    text = [f"# Failures and rejected runs, campaign {name}", "",
            "Built by `python -m bvp_pinn.evaluation.evaluate`. Every run listed here also stays in summary.csv "
            "and in every table and figure.", ""]
    (out / "failures.md").write_text("\n".join(text + (lines or ["None."])) + "\n")
    return out


def evaluate_folders(folders, refs, p):
    rows = []
    for folder in folders:
        metrics = evaluate_run(folder, refs, p)
        write_json(folder / "metrics.json", metrics)
        rows.append(summary_row(folder, metrics))
        print(f"{folder.name}: largest wall error {metrics['wall_rel_max']:.2e} relative, "
              f"nearest branch {metrics['nearest_branch']}, accepted {rows[-1]['accepted']}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--campaign", default=read_config()["campaign"]["name"])
    ap.add_argument("--runs", nargs="*", type=Path, help="evaluate these run folders only (pilots); no summary")
    args = ap.parse_args(argv)
    torch.set_default_dtype(torch.float64)
    p = load_case()[0]
    refs = load_references()
    if args.runs:
        return evaluate_folders([f for f in args.runs if is_finished(f) and (f / "model.pt").exists()], refs, p)
    folders = campaign_folders(args.campaign)
    rows = evaluate_folders([f for f in folders if is_finished(f) and (f / "model.pt").exists()], refs, p)
    out = write_campaign(args.campaign, rows, failure_lines(rows, folders, args.campaign), folders)
    print(f"{len(rows)} runs of campaign {args.campaign} in {out / 'summary.csv'}; failures in {out / 'failures.md'}")
    return rows


if __name__ == "__main__":
    main()
