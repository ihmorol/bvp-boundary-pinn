# src/bvp_pinn

Code for the BVP. Rules: [docs/team/CODING_RULES.md](../../docs/team/CODING_RULES.md) and the writing rule in AGENTS.md.

Three packages and two shared modules. The arrows say who may import whom:

```text
reference/  <--  pinn/  <--  evaluation/
   ^                             |
   +-----------------------------+   (evaluation reads the reference data; pinn never does)
```

`pinn/` uses only `reference/equations.py` (the equations, shared with the classical solver) and `reference/solver.py` (a mesh helper). It never imports `reference/io.py` or `evaluation/` (gate G5, tested).

- `config.py`: loads the locked physical case and the reference settings.
- `figstyle.py`: the figure style shared by the reference and PINN figures.
- `reference/`: classical reference. Spec: [docs/reference-solver/SPEC_reference_solutions.md](../../docs/reference-solver/SPEC_reference_solutions.md).
  - `equations.py`: Eqs. 14 to 19, residual and first-order form; plain arithmetic, so the PINN calls the same functions on tensors.
  - `solver.py`: meshes, guesses, solve_bvp calls and extension in L.
  - `shooting.py`: Runge-Kutta multiple shooting, the second solver.
  - `verify.py`: core-then-chi and saved-curve slope checks (locked case only).
  - `io.py`: write and load reference data with hash checks.
  - `run.py`: the full study; run `uv run python -m bvp_pinn.reference.run` from the repository root.
  - `plots.py`: reference figures and the study table.
- `pinn/`: training and acceptance. Plan: [docs/pinn/PINN_PLAN.md](../../docs/pinn/PINN_PLAN.md); file by file: [docs/pinn/IMPLEMENTATION_NOTES.md](../../docs/pinn/IMPLEMENTATION_NOTES.md) section 1. Entry points: `python -m bvp_pinn.pinn.run` (one run) and `python -m bvp_pinn.pinn.campaign` (all runs).
- `evaluation/`: finished runs against the reference: `evaluate.py` (metrics, summary, failures), `classical.py` (solve_bvp timing), `plots.py` (figures and tables).
