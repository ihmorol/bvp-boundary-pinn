# Strategy and gates

Date: 2026-11-24. This replaces the pre-implementation plan of 2026-11-16, which is kept in [archive/strategy-2026-11-16.md](../archive/strategy-2026-11-16.md). That plan proposed a 64-wide network, soft boundary penalties, a debug case with other parameter values and arms B0 to B6. None of that is current, and the debug case is against the fixed-problem rule in [AGENTS.md](../../AGENTS.md).

## 1. What we solve

One problem, fixed as the supervisor gave it: Eqs. 14 to 18 with boundary conditions B1 to B12 ([equations.md](../problem/equations.md)) at the 22 locked values ([parameters.md](../problem/parameters.md)). Only the way the conditions are imposed and the way the problem is solved may change. Every result is checked on the original equations.

The task (supervisor, 2026-11-19): a classical reference, a PINN trained on eta only, and a test of the PINN against the reference. The reference is for checking. It never feeds training or the choice of a run.

## 2. Gates, in order

A gate opens only when the one before it holds.

| Gate | What must hold | State on 2026-11-24 | Detail |
|---|---|---|---|
| 1 Parameters | All 22 values locked, none filled from papers | Done 2026-11-16 | [parameters.md](../problem/parameters.md) |
| 2 Software tests | `uv run pytest` passes: equations against SymPy, derivatives to order 4, built-in conditions, losses, optimizers, the acceptance rule, an end-to-end run of every arm, the no-leakage check, and the fixed-problem guards (`tests/test_fixed_problem.py`) | Pass, 57 tests | [IMPLEMENTATION_NOTES.md](../pinn/IMPLEMENTATION_NOTES.md) section 6 |
| 3 Reference | SPEC checks V1, V2, V4 to V10, V12, all at the locked case; L chosen; branches found | Pass. Audited 2026-11-23; audit defects closed 2026-11-24 | [SPEC](../reference-solver/SPEC_reference_solutions.md), [audit](../reviews/REFERENCE_AUDIT_2026-11-23.md) |
| 4 Training | Pilots P2b and P3 freeze `configs/pinn.json` with dated notes; then the campaign, 4 arms x 5 seeds | Code ready; pilots and campaign not run | [PINN_PLAN.md](../pinn/PINN_PLAN.md) sections 2, 3 and 8 |
| 5 Evaluation | Metrics, figures and the answers to PINN_PLAN section 7, every number traced to a run ID | Code ready and tested on a dry campaign; no campaign result yet | [PINN_PLAN.md](../pinn/PINN_PLAN.md) sections 4 to 7 |

The task list for each gate is [TODO.md](TODO.md).

## 3. Rules that hold at every gate

- **The problem is fixed.** Eqs. 14 to 18, B1 to B12 and the 22 supervisor values are the same in every run, tool and use case. No debug case, no continuation, no warm start from another value, no check at other values and no forced equation. The code refuses a changed case file, any setting that names a physical value and an L other than the reference's ([AGENTS.md](../../AGENTS.md) section 3).
- **No claim without a run.** Pending, blocked and failed are written down as such. A failed run stays in every table.
- **No reference in training or selection.** Training and the acceptance rule live in `src/bvp_pinn/pinn/`, which never imports the reference data or the evaluation code (gate G5, tested). Only `src/bvp_pinn/evaluation/` reads the reference, after acceptance.
- **Our choices are marked as ours.** Numerical settings, pass bars and the acceptance rule are our choices, each with a dated note in the config or the doc that uses it. Supervisor values are kept apart.
- **Claims across arms need all 5 paired seeds** (gate G6). Otherwise the result is "no consistent effect".

## 4. When something fails

| Failure | Action |
|---|---|
| A test fails | Fix the code before any run. Never loosen a test to pass it without a written reason |
| NaN or infinite value in training | The run stops and is saved with its first bad term; the refine stage rolls back to the last finite state. Nothing is masked |
| Small loss, wrong solution | The acceptance rule checks the far tail and the Newton-step error estimate, not the loss alone (run E8, IMPLEMENTATION_NOTES 7.1) |
| A run lands on branch B or no branch | Reported as such; never averaged with branch A runs |
| Every arm fails at the locked case | Report the failure with its logs. No other parameter values are tried |
| The reference fails a check | Diagnose. A failed reference is never replaced by a guess or a paper value |
