"""The locked config must equal the supervisor receipt of 2026-11-16, symbol by symbol."""
import re

from bvp_pinn.config import PARAMETER_NAMES, load_case

# Verbatim from the receipt (docs/problem/parameters.md); re-sent by the user on 2026-11-19.
RECEIPT = """
b=0.2; bm = 0.2; B = 0.5; dt=0; Dt=0.5;
H=1; k=0.2; L =0.5; M=5; tw = 0.2; ep = 0.1;
Pe=2;Pr=8.8;Prm = 0.1;
Nb=0.1; Nr = 0.5; Nt=0.3;
Lb=1; Le=1;
Bi=1; l = 0.3; fw = 0.2; Rd = 1; N1 = 0.1;
"""
# Receipt mappings bm, dt, Dt, L, tw, M=1/Da; user-approved aliases k, l, ep; the rest share names.
TO_FIELD = {"bm": "beta_m", "dt": "delta_star", "Dt": "Delta", "L": "lambda_m", "tw": "theta_w",
            "k": "kappa", "l": "ell", "ep": "epsilon"}
UNUSED = {"H", "Prm"}  # received, but no term in Eqs. 14 to 19 uses them


def test_locked_case_equals_receipt():
    received = {k: float(v) for k, v in re.findall(r"(\w+)\s*=\s*([-\d.]+)", RECEIPT)}
    assert len(received) == 24
    expected = {"Da": 1 / received.pop("M")}
    for sym in UNUSED:
        received.pop(sym)
    expected |= {TO_FIELD.get(sym, sym): v for sym, v in received.items()}
    p, sha = load_case()
    assert set(expected) == set(PARAMETER_NAMES) and len(sha) == 64
    assert p == expected
