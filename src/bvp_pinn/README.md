# src/bvp_pinn

Code for the BVP. Rules: [docs/team/CODING_RULES.md](../../docs/team/CODING_RULES.md) and the writing rule in AGENTS.md.

- `config.py`: loads the locked physical case and the reference settings.
- `reference/`: classical reference with scipy's solve_bvp. Spec: [docs/reference-solver/SPEC_reference_solutions.md](../../docs/reference-solver/SPEC_reference_solutions.md).
  - `equations.py`: Eqs. 14 to 19 in first-order form.
  - `solver.py`: meshes, guesses, solve and extend in L.
  - `verify.py`: manufactured-solution and core-then-chi checks.
  - `io.py`: write and load reference data with hash checks. PINN training code must not import it.
  - `run.py`: the full study; run `uv run python -m bvp_pinn.reference.run` from the repository root.
  - `plots.py`: figures and the study table.

PINN modules come next ([TODO.md](../../docs/plans/TODO.md) phase P1).
