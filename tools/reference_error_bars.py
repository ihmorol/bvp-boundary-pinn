"""Error bars for the reference wall values, from data we already have (gaps R1 b and R5).

    uv run python tools/reference_error_bars.py

1. Grid convergence index (Roache; ASME V&V 20) from the three finest levels of the C++ box
   scheme, engine/out/engine_convergence.csv (each level doubles the nodes, so r = 2).
2. Extrapolation in the domain end L from the asymptotic L study, reference/l_study.npz
   (L = 40, 80, 160): v(L) = v_inf + c L^-p. It says how far the L = 20 values are from the
   values for eta -> infinity that B7 to B12 ask for.

Writes reference/error_bars.json and reference/error_bars.md. Nothing here feeds training.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
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


def main():
    gci, lex = grid_index(), l_extrapolation()
    (ROOT / "reference/error_bars.json").write_text(json.dumps({"grid": gci, "L": lex}, indent=2) + "\n")
    lines = ["# Reference error bars (branch A)", "",
             "Built by `tools/reference_error_bars.py` from `engine/out/engine_convergence.csv` and `reference/l_study.npz`.",
             "Grid: three finest box-scheme levels, r = 2. L: v(L) = v_inf + c L^-p fitted to L = 40, 80, 160.", "",
             "| Value | Finest grid | Observed order | GCI (fine) | L = 20 | v_inf (L fit) | L = 20 minus v_inf | Order in L |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda x, f="{:.3g}": "n/a" if x is None else f.format(x)
    for k in KEYS:
        g, l = gci[k], lex[k]
        lines.append(f"| {k} | {g['finest']:.10g} | {fmt(g['order'])} | {fmt(g['gci_fine'])} | {l['values'][20]:.8g} | "
                     f"{fmt(l['v_inf'], '{:.8g}')} | {fmt(l['L20_minus_v_inf'])} | {fmt(l['order_in_L'])} |")
    lines += ["", "chi'(0) has no limit in L: Eq. 18 as given has no decaying chi solution (SPEC section 8.2), so its fit is not a limit.", ""]
    (ROOT / "reference/error_bars.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
