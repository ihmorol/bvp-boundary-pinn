"""Error bars for the reference wall values, from data we already have (gaps R1 b and R5).

    uv run python tools/reference_error_bars.py

1. Grid convergence index (Roache; ASME V&V 20) from the three finest levels of the C++ box
   scheme, engine/out/engine_convergence.csv (each level doubles the nodes, so r = 2).
2. Extrapolation in the domain end L from the asymptotic L study, reference/l_study.npz
   (L = 40, 80, 160): v(L) = v_inf + c L^-p. It says how far the L = 20 values are from the
   values for eta -> infinity that B7 to B12 ask for.
3. Far-field bracket (reference audit D2 and D6): a new L ladder from 20 to 10 240, solved once
   with each form of the asymptotic condition (equations.far_conditions: A at theta(L), and
   A = 1). The two forms close in on the eta -> infinity values from opposite sides. The spread
   at L = 20 is a far-field error bar; the values at L = 320 to 10 240 test the three-point fit
   of item 2, which has no spare point of its own.

Writes reference/error_bars.json and reference/error_bars.md. Nothing here feeds training.
The ladder takes a few minutes; --no-ladder reuses the ladder saved in error_bars.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LADDER = [20 * 2**k for k in range(10)]  # 20 to 10 240
FORMS = ("local", "unit")
KEYS = ["f2_0", "theta_0", "th1_0", "phi_0", "ph1_0", "g1_0", "ch1_0", "f_L"]
STATE = {"f2_0": 2, "theta_0": 6, "th1_0": 7, "phi_0": 8, "ph1_0": 9, "g1_0": 5, "ch1_0": 11}


def three_level(f1, f2, f3, r=2.0):
    """f1 coarse, f3 fine. Observed order, extrapolated value and the fine-grid GCI (safety 1.25)."""
    e32, e21 = f2 - f3, f1 - f2
    if e32 == 0 or e21 == 0 or e21 / e32 <= 0:
        return {"order": None, "extrapolated": f3, "gci_fine": None, "monotone": False}
    p = math.log(e21 / e32) / math.log(r)
    ext = f3 + (f3 - f2) / (r**p - 1)
    gci = 1.25 * abs(e32 / f3) / (r**p - 1) if f3 else None
    return {"order": p, "extrapolated": ext, "gci_fine": gci, "monotone": True}


def grid_index():
    rows = list(csv.DictReader(open(ROOT / "engine/out/engine_convergence.csv")))
    a, b, c = rows[-3:]
    return {k: {**three_level(float(a[k]), float(b[k]), float(c[k])), "finest": float(c[k]),
                "nodes": [int(a["nodes"]), int(b["nodes"]), int(c["nodes"])]} for k in KEYS}


def l_extrapolation():
    d = np.load(ROOT / "reference/l_study.npz")
    out = {}
    for k in KEYS:
        vals = {}
        for L in (10, 20, 40, 80, 160):
            z = d[f"A_asymptotic_L{L}_z"]
            vals[L] = float(z[0, -1]) if k == "f_L" else float(z[STATE[k], 0])
        fit = three_level(vals[40], vals[80], vals[160])
        ok = fit["monotone"] and fit["order"] is not None and fit["order"] > 0
        out[k] = {"values": vals, "order_in_L": fit["order"], "v_inf": fit["extrapolated"] if ok else None,
                  "L20_minus_v_inf": vals[20] - fit["extrapolated"] if ok else None, "converges": ok}
    return out


def far_ladder(tol=1e-8):
    """Wall values on L = 20 to 10 240 for each form of the far condition, each solve continued
    from the one before. Stops a form at its first failed solve and records it."""
    from bvp_pinn.config import load_case, load_reference_settings
    from bvp_pinn.reference import solver as sv
    p = load_case()[0]
    base = load_reference_settings()
    out = {}
    for form in FORMS:
        s = {**base, "far_form": form}
        rows, cur = [], None
        for L in LADDER:
            if cur is None:
                res = sv.solve_case(p, L, tol, s, label=f"{form} L{L}", guess={"rate": 1.0, "theta0": 0.3})
            else:
                x, y = sv.extend(cur, L, s)
                res = sv.solve_case(p, L, tol, s, label=f"{form} L{L}", x=x, y=y)
            rows.append({"L": L, "status": res.status, "seconds": round(res.seconds, 2), "nodes": int(res.x.size),
                         "wall": res.wall})
            print(f"{form} L = {L}: status {res.status}, {res.seconds:.1f} s, {res.x.size} nodes", flush=True)
            if not res.ok:
                break
            cur = res
        out[form] = rows
    return out


def bracket_and_fit_check(ladder, lex):
    """Spread of the two forms per L, and the item 2 fit against the new long-L values."""
    by = {form: {r["L"]: r["wall"] for r in rows if r["status"] == 0} for form, rows in ladder.items()}
    spread = {}
    for L in LADDER:
        if L in by["local"] and L in by["unit"]:
            spread[L] = {k: by["local"][L][k] - by["unit"][L][k] for k in KEYS if k in by["local"][L]}
    check = {}
    for k in KEYS:
        fit = lex[k]
        if not fit["converges"]:
            continue
        v = fit["values"]
        a = three_level(v[40], v[80], v[160])
        p_ = a["order"]
        c = (v[160] - a["extrapolated"]) * 160**p_
        pred = {L: a["extrapolated"] + c * L**-p_ for L in LADDER if L > 160}
        got = {L: by["local"][L][k] for L in pred if L in by["local"]}
        check[k] = {"max_rel_miss": max(abs(pred[L] - got[L]) / abs(got[L]) for L in got) if got else None,
                    "v_inf_fit": a["extrapolated"]}
    top = max(spread) if spread else None
    inside = {}
    if top:
        for k in check:
            lo, hi = sorted([by["local"][top][k], by["unit"][top][k]])
            inside[k] = {"bracket": [lo, hi], "fit_inside": lo <= check[k]["v_inf_fit"] <= hi}
    return {"spread_by_L": spread, "fit_check": check, "bracket_at": top, "bracket": inside}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-ladder", action="store_true", help="reuse the far-field ladder in error_bars.json")
    args = ap.parse_args()
    gci, lex = grid_index(), l_extrapolation()
    saved = ROOT / "reference/error_bars.json"
    if args.no_ladder:
        ladder = json.loads(saved.read_text())["far_ladder"]
    else:
        ladder = far_ladder()
    far = bracket_and_fit_check(ladder, lex)
    saved.write_text(json.dumps({"grid": gci, "L": lex, "far_ladder": ladder, "far": far}, indent=2) + "\n")
    fmt = lambda x, f="{:.3g}": "n/a" if x is None else f.format(x)
    lines = ["# Reference error bars (branch A)", "",
             "Built by `tools/reference_error_bars.py` from `engine/out/engine_convergence.csv`, `reference/l_study.npz` "
             "and its own far-field ladder.",
             "Grid: three finest box-scheme levels, r = 2. L: v(L) = v_inf + c L^-p fitted to L = 40, 80, 160.", "",
             "| Value | Finest grid | Observed order | GCI (fine) | L = 20 | v_inf (L fit) | L = 20 minus v_inf | Order in L |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in KEYS:
        g, l = gci[k], lex[k]
        lines.append(f"| {k} | {g['finest']:.10g} | {fmt(g['order'])} | {fmt(g['gci_fine'])} | {l['values'][20]:.8g} | "
                     f"{fmt(l['v_inf'], '{:.8g}')} | {fmt(l['L20_minus_v_inf'])} | {fmt(l['order_in_L'])} |")
    lines += ["", "chi'(0) has no limit in L: Eq. 18 as given has no decaying chi solution (SPEC section 8.2), "
              "so its fit is not a limit.", ""]
    top = far["bracket_at"]
    lines += ["## Far-field bracket (reference audit D2 and D6)", "",
              "Each L is solved twice, with A at theta(L) (local, the reference's form) and with A = 1 (unit), each form "
              "continued from its own previous L. Spread is local minus unit. Fit miss compares the L = 40, 80, 160 fit "
              "above with the solved local values at L = 320 to 10 240, points the fit never saw.", "",
              f"| Value | Spread at L = 20 | Spread at L = {top} | Bracket at L = {top} | v_inf (fit) inside | "
              "Largest relative fit miss |", "|---|---:|---:|---|---|---:|"]
    for k, chk in far["fit_check"].items():
        br = far["bracket"].get(k)
        cell = "n/a" if not br else f"{br['bracket'][0]:.8g} to {br['bracket'][1]:.8g}"
        lines.append(f"| {k} | {fmt(far['spread_by_L'].get(20, {}).get(k))} | {fmt(far['spread_by_L'].get(top, {}).get(k))} | "
                     f"{cell} | {'n/a' if not br else ('yes' if br['fit_inside'] else 'no')} | {fmt(chk['max_rel_miss'])} |")
    failed = [f"{form} at L = {r['L']}" for form, rows in ladder.items() for r in rows if r["status"] != 0]
    lines += ["", "Failed solves: " + (", ".join(failed) if failed else "none") + ".", ""]
    (ROOT / "reference/error_bars.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
