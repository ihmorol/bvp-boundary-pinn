"""Cost of the classical solver, for the comparison in PINN_PLAN figure FX1 and table TX1 (TODO P5).

    uv run python -m bvp_pinn.evaluation.classical                 # results/<campaign>/classical_timing.json
    uv run python -m bvp_pinn.evaluation.classical --repeats 5 --campaign P4

solve_bvp on the locked case at L = 20 with the reference's far conditions, from the same smooth
guess the reference starts from, at tolerances 1e-3 to 1e-8. Each solve is timed `repeats` times
on one thread and the median counts. Its error uses the PINN metric (evaluate.wall_errors against
reference A), so both sides of FX1 are measured the same way.

Timings depend on the machine. Run this on the machine the PINN runs it is compared with ran on,
and the output records which machine that was.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # before NumPy loads: one thread, as the PINN runs

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402

import numpy as np  # noqa: E402

from ..config import load_case, load_reference_settings  # noqa: E402
from ..pinn.runs import read_config  # noqa: E402
from ..reference import solver as sv  # noqa: E402
from .evaluate import FIELDS, load_references, rel_l2, results_dir, wall_errors, worst  # noqa: E402

TOLERANCES = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
GUESS = {"rate": 1.0, "theta0": 0.3}  # the start that reaches branch A in the reference scan


def time_solves(repeats=5, tolerances=TOLERANCES):
    p = load_case()[0]
    settings = load_reference_settings()
    L = float(settings["L_reference"])
    ref = load_references()["A"]
    rows = []
    for tol in tolerances:
        seconds, res = [], None
        for _ in range(repeats):
            res = sv.solve_case(p, L, tol, settings, label=f"timing tol {tol:g}", guess=GUESS)
            seconds.append(res.seconds)
        row = {"tol": tol, "status": res.status, "nodes": int(res.x.size), "median_seconds": statistics.median(seconds),
               "seconds": seconds}
        if res.ok:
            z = res.sol(ref["eta"])
            wall = wall_errors(z[:, 0], ref)
            row.update({"wall_rel_max": worst(v["rel_error"] for v in wall.values()),
                        "max_field_rel_l2": worst(rel_l2(z[i], ref["z"][i], ref["eta"]) for i in FIELDS.values()),
                        "theta_0": float(z[6, 0])})
        rows.append(row)
        print(f"tol {tol:g}: status {res.status}, {row['median_seconds']:.3f} s, "
              f"largest wall error {row.get('wall_rel_max', float('nan')):.2e}", flush=True)
    return {"solver": "scipy.integrate.solve_bvp", "L": L, "guess": GUESS, "repeats": repeats,
            "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version(),
            "numpy": np.__version__, "rows": rows}


def time_to_reach(timing, error):
    """Median seconds of the cheapest solve whose wall error is at or below error, or None."""
    times = [r["median_seconds"] for r in timing["rows"] if r.get("wall_rel_max") is not None
             and r["wall_rel_max"] <= error]
    return min(times) if times else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--campaign", default=read_config()["campaign"]["name"])
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args(argv)
    out = results_dir(args.campaign)
    out.mkdir(parents=True, exist_ok=True)
    timing = time_solves(args.repeats)
    (out / "classical_timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(f"wrote {out / 'classical_timing.json'}")


if __name__ == "__main__":
    main()
