"""Which domain end L to use: the error L adds to each wall value, what it costs, and what it
does to the PINN's points and scales.

    uv run python tools/l_choice.py

The source puts B7 to B12 at eta -> infinity. Every solver cuts the domain at a finite L, and
that cut adds an error of its own, the L error. This tool measures it, on branch A of the locked
case only (no other parameter values, no other conditions):

1. Solves branch A with solve_bvp on a fine ladder of L, with the far conditions in both forms
   (equations.far_conditions: "local", the reference's, and "unit"), each solve continued from
   the one before. The L = 20 local solve must match the saved reference.
2. Takes the eta -> infinity values v_inf from the long ladder in reference/error_bars.json
   (local form, L = 2 560, 5 120, 10 240, three-level fit with ratio 2), and checks them
   against the L = 40, 80, 160 fit of that file.
3. For every L: the L error |v(L) - v_inf| / |v_inf| of each wall value except chi'(0), which
   has no limit; the change to the next L; the spread of the two forms; solve time and nodes.
4. The L each wall value needs for a target L error, from the fitted power law v = v_inf + c L^-p.
5. What L does to the PINN at the settings of configs/pinn.json, without training: the share of
   training points and of the log input range inside the wall layer [0, 0.5], the first point
   spacing, the largest log-scale factor on R14, and the point count that keeps the wall layer as
   dense as at L = 20. The Levenberg-Marquardt step time for that count is an estimate from the
   two measured times (15.2 s at 1 000 points, 24 s at 2 000, one thread).

Writes reference/l_choice.json and reference/l_choice.md. Nothing here feeds training.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from reference_error_bars import three_level  # noqa: E402

LADDER = [8, 10, 12, 15, 20, 25, 30, 40, 60, 80, 120, 160]
START = 20  # solved from the smooth guess; the rest are continued from it, up and down
KEYS = ["f2_0", "theta_0", "th1_0", "phi_0", "ph1_0", "g1_0", "f_L"]  # every wall value with a limit
WALL = ["f2_0", "theta_0", "th1_0", "phi_0", "ph1_0", "g1_0", "ch1_0"]
TARGETS = [1e-3, 1e-4, 1e-5]
WALL_LAYER = 0.5
LM_SECONDS = {1000: 15.2, 2000: 24.0}  # measured B0 LM step times, one thread (IMPLEMENTATION_NOTES 5.2, 7.3)


def solve_ladder(p, settings, tol):
    from bvp_pinn.reference import solver as sv
    out = {}
    for form in ("local", "unit"):
        s = {**settings, "far_form": form}
        rows = {}
        first = sv.solve_case(p, START, tol, s, label=f"{form} L{START}", guess={"rate": 1.0, "theta0": 0.3})
        rows[START] = first
        for part in ([L for L in LADDER if L > START], [L for L in reversed(LADDER) if L < START]):
            cur = first
            for L in part:
                x, y = sv.extend(cur, L, s)
                res = sv.solve_case(p, L, tol, s, label=f"{form} L{L}", x=x, y=y)
                rows[L] = res
                print(f"{form} L = {L}: status {res.status}, {res.seconds:.2f} s, {res.x.size} nodes", flush=True)
                if not res.ok:
                    break
                cur = res
        out[form] = {L: rows[L] for L in sorted(rows)}
    return out


def limits(saved):
    """v_inf from the three longest local solves, and the 40/80/160 fit for comparison."""
    by_L = {r["L"]: r["wall"] for r in saved["far_ladder"]["local"] if r["status"] == 0}
    longest = sorted(by_L)[-3:]
    out = {}
    for k in KEYS:
        fit = three_level(*(by_L[L][k] for L in longest))
        old = saved["L"][k].get("v_inf")
        out[k] = {"v_inf": fit["extrapolated"], "order": fit["order"], "from_L": longest,
                  "v_inf_40_80_160": old,
                  "fits_agree_rel": abs(fit["extrapolated"] - old) / abs(fit["extrapolated"]) if old is not None else None}
    return out


def needed_L(L, err, p, target):
    """L at which err(L) = c L^-p falls to target."""
    return L * (err / target) ** (1 / p) if err > target else L


def pinn_side(cfg, p, L, n_ref_wall):
    n, st, scale = cfg["n_points"], cfg["stretch"], cfg["map_scale"]
    share = math.log1p(WALL_LAYER * math.expm1(st) / L) / st
    first = L * math.expm1(st / n) / math.expm1(st)
    log_share = math.log1p(WALL_LAYER / scale) / math.log1p(L / scale)
    n_keep = math.ceil(n_ref_wall / share)
    a = (LM_SECONDS[2000] - LM_SECONDS[1000]) / 1000
    return {"wall_point_share": share, "first_spacing": first, "log_input_share": log_share,
            "log_scale_R14_at_L": (L + p["b"]) ** 4, "points_for_same_wall_density": n_keep,
            "lm_step_seconds_estimate": LM_SECONDS[1000] + a * (n_keep - 1000)}


def main():
    from bvp_pinn.config import load_case, load_reference_settings
    from bvp_pinn.pinn.runs import read_config
    from bvp_pinn.reference.io import load_reference

    p, _ = load_case()
    settings = load_reference_settings()
    tol = settings["tol_values"][-1]
    saved = json.loads((ROOT / "reference/error_bars.json").read_text())
    lim = limits(saved)
    ladder = solve_ladder(p, settings, tol)

    ref, _ = load_reference(ROOT / f"reference/locked-branchA-L{settings['L_reference']:g}-tol{tol:g}")
    z0 = ref["z"][:, 0]
    idx = {"f2_0": 2, "theta_0": 6, "th1_0": 7, "phi_0": 8, "ph1_0": 9, "g1_0": 5, "ch1_0": 11}
    here = ladder["local"][START].wall
    match = max(abs(here[k] - z0[i]) / abs(z0[i]) for k, i in idx.items())

    cfg = {**read_config()["common"]}
    share20 = math.log1p(WALL_LAYER * math.expm1(cfg["stretch"]) / 20) / cfg["stretch"]
    n_ref_wall = share20 * cfg["n_points"]

    rows = []
    Ls = [L for L in LADDER if ladder["local"].get(L) is not None and ladder["local"][L].ok]
    for i, L in enumerate(Ls):
        loc, uni = ladder["local"][L], ladder["unit"].get(L)
        w = loc.wall
        err = {k: abs(w[k] - lim[k]["v_inf"]) / abs(lim[k]["v_inf"]) for k in KEYS}
        nxt = ladder["local"][Ls[i + 1]].wall if i + 1 < len(Ls) else None
        change = {k: abs(w[k] - nxt[k]) / abs(w[k]) for k in KEYS} if nxt else None
        spread = ({k: abs(w[k] - uni.wall[k]) / abs(w[k]) for k in KEYS} if uni is not None and uni.ok else None)
        rows.append({"L": L, "seconds": round(loc.seconds, 3), "nodes": int(loc.x.size),
                     "wall": {k: w[k] for k in WALL}, "L_error_rel": err,
                     "largest_L_error": max(err.items(), key=lambda kv: kv[1]),
                     "change_to_next_L_rel": change, "form_spread_rel": spread,
                     "theta_0_branch_A": abs(w["theta_0"] - here["theta_0"]) < 0.05,
                     "pinn": pinn_side(cfg, p, L, n_ref_wall)})

    r20 = next(r for r in rows if r["L"] == 20)
    need = {k: {f"{t:g}": needed_L(20, r20["L_error_rel"][k], lim[k]["order"], t) for t in TARGETS} for k in KEYS}
    result = {"note": "Branch A, locked case. L error is relative to v_inf from the L = 2 560 to 10 240 local solves.",
              "tol": tol, "start_L": START, "match_with_saved_reference_rel": match, "limits": lim,
              "rows": rows, "L_needed_from_L20": need}
    (ROOT / "reference/l_choice.json").write_text(json.dumps(result, indent=2, default=float) + "\n")

    g = "{:.2g}".format
    md = ["# Choice of L (branch A, locked case)", "",
          "Built by `uv run python tools/l_choice.py`. Study and reading: "
          "[docs/plans/L_STUDY.md](../docs/plans/L_STUDY.md).", "",
          f"The L = {START} solve here matches the saved reference to {match:.1e} relative at the wall.",
          "v_inf: three-level fit (ratio 2) of the local solves at L = " + ", ".join(f"{L:g}" for L in lim['f2_0']['from_L'])
          + " in reference/error_bars.json.", "",
          "## Limits", "", "| Value | v_inf | Order in L | v_inf from L = 40, 80, 160 | Relative difference |",
          "|---|---:|---:|---:|---:|"]
    for k in KEYS:
        m = lim[k]
        md.append(f"| {k} | {m['v_inf']:.8g} | {m['order']:.3f} | {m['v_inf_40_80_160']:.8g} | {g(m['fits_agree_rel'])} |")
    md += ["", "chi'(0) has no limit: it keeps changing with L (SPEC section 8.2), so it is left out of the L error.", "",
           "## L error of each wall value, relative to v_inf", "",
           "| L | " + " | ".join(KEYS) + " | Largest | Solve time (s) |", "|---:|" + "---:|" * (len(KEYS) + 2)]
    for r in rows:
        k, v = r["largest_L_error"]
        md.append(f"| {r['L']:g} | " + " | ".join(g(r["L_error_rel"][q]) for q in KEYS) + f" | {g(v)} ({k}) | {r['seconds']:.2f} |")
    md += ["", "## Change to the next L, and spread of the two far-condition forms (relative)", "",
           "| L | Largest change to next L | Largest form spread | chi'(0) |", "|---:|---:|---:|---:|"]
    for r in rows:
        c = max(r["change_to_next_L_rel"].values()) if r["change_to_next_L_rel"] else None
        s = max(r["form_spread_rel"].values()) if r["form_spread_rel"] else None
        md.append(f"| {r['L']:g} | {g(c) if c is not None else 'n/a'} | {g(s) if s is not None else 'n/a'} | {r['wall']['ch1_0']:.5g} |")
    md += ["", "## L needed for a target L error, from L = 20 and the fitted order", "",
           "| Value | L error at 20 | " + " | ".join(f"target {t:g}" for t in TARGETS) + " |", "|---|---:|" + "---:|" * len(TARGETS)]
    for k in KEYS:
        md.append(f"| {k} | {g(r20['L_error_rel'][k])} | " + " | ".join(f"{need[k][f'{t:g}']:.3g}" for t in TARGETS) + " |")
    md += ["", f"## What L does to the PINN at fixed settings ({cfg['n_points']} points, stretch {cfg['stretch']:g}, "
           f"map scale {cfg['map_scale']:g}), no training", "",
           "| L | Points in [0, 0.5] | Log input range in [0, 0.5] | First spacing | (L + b)^4 on R14 | Points for the L = 20 wall density | LM step (s) **[estimate]** |",
           "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        q = r["pinn"]
        md.append(f"| {r['L']:g} | {q['wall_point_share']:.1%} | {q['log_input_share']:.1%} | {q['first_spacing']:.2g} | "
                  f"{q['log_scale_R14_at_L']:.3g} | {q['points_for_same_wall_density']} | {q['lm_step_seconds_estimate']:.0f} |")
    (ROOT / "reference/l_choice.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
