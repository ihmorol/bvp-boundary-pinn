"""Compare finished runs with the classical reference (docs/PINN_PLAN.md section 4).

    uv run python -m bvp_pinn.pinn.evaluate

Runs after training, so the reference never influences training or selection (gate G5).
Writes runs/<run_id>/metrics.json and pred_eval.npz, then rebuilds results/summary.csv
from every run folder.
"""
from __future__ import annotations

import csv
import json

import numpy as np
import torch

from ..config import ROOT, load_case
from ..reference.equations import HIGH_NAMES, STATE_NAMES
from ..reference.io import load_reference
from . import model as nets
from .losses import boundary_errors
from .physics import residuals
from .runs import RUNS

REFERENCES = {"A": ROOT / "reference" / "locked-branchA-L20-tol1e-08",
              "B": ROOT / "reference" / "locked-branchB-L20-tol1e-08"}
FIELDS = {"f": 0, "g": 4, "theta": 6, "phi": 8, "chi": 10}
WALL = {"f2_0": 2, "g1_0": 5, "th1_0": 7, "ph1_0": 9, "ch1_0": 11}
RESULTS = ROOT / "results"


def load_run(folder, checkpoint="model.pt"):
    cfg = json.loads((folder / "config.json").read_text())
    model = nets.build(cfg, load_case()[0])
    model.load_state_dict(torch.load(folder / checkpoint, map_location="cpu"))
    return cfg, model


def rel_l2(got, want, eta):
    """Relative L2 error with trapezoid weights, since the reference grid is not uniform."""
    return float(np.sqrt(np.trapezoid((got - want) ** 2, eta) / np.trapezoid(want**2, eta)))


def evaluate_run(folder, ref, p):
    cfg, model = load_run(folder)
    eta = ref["eta"]
    z, H = (a.numpy() for a in nets.predict(model, torch.tensor(eta).reshape(-1, 1)))
    got, want = np.vstack([z, H]), np.vstack([ref["z"], ref["H"]])
    names = STATE_NAMES + HIGH_NAMES
    cols = [torch.tensor(r).reshape(-1, 1) for r in z]
    res = residuals(torch.tensor(eta).reshape(-1, 1), cols, [torch.tensor(r).reshape(-1, 1) for r in H], p)
    metrics = {
        "fields": {n: {"rel_l2": rel_l2(g, w, eta), "max_abs": float(np.max(np.abs(g - w)))}
                   for n, g, w in zip(names, got, want)},
        "wall": {k: {"pinn": float(z[i, 0]), "reference": float(ref["z"][i, 0]),
                     "abs_error": float(abs(z[i, 0] - ref["z"][i, 0]))} for k, i in WALL.items()},
        "residual_mse": dict(zip(("R14", "R15", "R16", "R17", "R18"), map(float, res.pow(2).mean(0)))),
        "boundary_abs": dict(zip([f"B{i}" for i in range(1, 13)],
                                 map(float, boundary_errors(model, p, cfg["L"]).detach().abs()))),
        "windows_by_stage": window_errors(folder, cfg, ref),
    }
    np.savez(folder / "pred_eval.npz", eta=eta, z=z, H=H)
    return metrics


def window_errors(folder, cfg, ref):
    """Max abs error of each field on each window, after every stage (retention, gate G4)."""
    edges = cfg["windows"]
    out = []
    for path in sorted(folder.glob("stage*.pt"), key=lambda q: int(q.stem[5:])):
        _, model = load_run(folder, path.name)
        z = nets.predict(model, torch.tensor(ref["eta"]).reshape(-1, 1))[0].numpy()
        rows = {}
        for a, b in zip(edges[:-1], edges[1:]):
            m = (ref["eta"] >= a) & (ref["eta"] <= b)
            rows[f"{a:g}-{b:g}"] = {n: float(np.max(np.abs(z[i, m] - ref["z"][i, m]))) for n, i in FIELDS.items()}
        out.append(rows)
    return out


def nearest_branch(z, refs):
    """Which reference branch the PINN theta is closest to: a PINN can also land on branch B."""
    d = {k: rel_l2(z[6], r["z"][6], r["eta"]) for k, r in refs.items()}
    return min(d, key=d.get)


def summary_row(folder, metrics, meta):
    cfg = json.loads((folder / "config.json").read_text())
    f = metrics["fields"]
    row = {"run_id": folder.name, "arm": cfg["arm"], "seed": cfg["seed"], "dry_run": cfg.get("dry_run", False),
           "result": meta.get("result"), "lbfgs_accepted": meta.get("lbfgs_accepted"),
           "wall_seconds": round(meta.get("wall_seconds", float("nan")), 1),
           "nearest_branch": metrics["nearest_branch"],
           "max_field_rel_l2": max(f[n]["rel_l2"] for n in FIELDS)}
    row.update({f"rel_l2_{n}": f[n]["rel_l2"] for n in FIELDS})
    row.update({f"wall_abs_{k}": v["abs_error"] for k, v in metrics["wall"].items()})
    row["max_boundary_abs"] = max(metrics["boundary_abs"].values())
    row["mean_residual_mse"] = float(np.mean(list(metrics["residual_mse"].values())))
    return row


def main(runs=RUNS):
    torch.set_default_dtype(torch.float64)
    p = load_case()[0]
    refs = {k: load_reference(v)[0] for k, v in REFERENCES.items()}
    rows = []
    for folder in sorted(x for x in runs.glob("bvp-*") if (x / "meta.json").exists()):
        meta = json.loads((folder / "meta.json").read_text())
        metrics = evaluate_run(folder, refs["A"], p)
        metrics["nearest_branch"] = nearest_branch(np.load(folder / "pred_eval.npz")["z"], refs)
        (folder / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        rows.append(summary_row(folder, metrics, meta))
        print(f"{folder.name}: max field rel L2 {rows[-1]['max_field_rel_l2']:.2e}, "
              f"nearest branch {metrics['nearest_branch']}")
    RESULTS.mkdir(exist_ok=True)
    if rows:
        with open(RESULTS / "summary.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"{len(rows)} runs in {RESULTS / 'summary.csv'}")


if __name__ == "__main__":
    main()
