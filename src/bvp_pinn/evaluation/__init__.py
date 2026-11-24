"""Evaluation of finished PINN runs against the classical reference (docs/pinn/PINN_PLAN.md sections 4 and 6).

This package is the only code that reads reference data next to PINN runs. The training and
selection code in bvp_pinn.pinn never imports it (gate G5, tests/test_pinn.py).

evaluate.py     metrics per run; results/<campaign>/summary.csv and failures.md
classical.py    solve_bvp timing for the time-to-accuracy comparison (FX1, TX1)
plots.py        figures FB1 to FB5 and FX1, tables TB2 and TX1
"""
