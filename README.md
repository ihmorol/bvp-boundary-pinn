# Five-variable boundary PINN

Local research repository for the FYDP five-function boundary problem: solve Eqs. 14 to 19 with a physics-informed neural network. Separate from the old Lorenz project.

**Status:** all 22 physical coefficients are complete and locked from [the user's 2026-11-16 message reporting direct supervisor values](docs/problem/parameters.md#supervisor-confirmation-2026-11-16). Da=0.2 is derived from the supplied M=5. Case status: `parameters_complete_validation_pending`.

**Fixed problem.** Eqs. 14 to 18, B1 to B12 and the locked values are used exactly as the supervisor gave them. Only the way they are imposed and solved may change (AGENTS.md section 3).

| Part | Status on 2026-11-23 |
|---|---|
| Classical reference | Done and verified on [0, 20] with asymptotic far conditions for theta and phi: solve_bvp, Runge-Kutta shooting and a C++ box scheme agree to about 1e-10. Branch A is the target. Error bars: mesh about 1e-7 relative; L = 20 against the eta -> infinity limit 8e-5 to 8e-4 per wall value; chi'(0) has no limit in L (Eq. 18 as given). See docs/reference-solver/SPEC_reference_solutions.md section 8 |
| PINN code | Done and tested (38 tests): exact boundary layer for all 12 conditions, forward-mode derivatives, start profile, residual scale, L-BFGS, BFGS and Levenberg-Marquardt refine stages, reference-free checks |
| PINN result | **B0 matches the reference.** From scratch, seeds 0 and 1: every wall value within 4e-4 relative of reference A (seed 0 within 6e-5; theta(0) 0.2501906 against 0.2501853), relative L2 1e-4 to 5e-3, balanced equation loss below 2e-6. Pipeline: log scale for Adam and L-BFGS, then Levenberg-Marquardt on the balanced scale. See docs/pinn/IMPLEMENTATION_NOTES.md sections 7.1 to 7.3 |
| Reviews | The reference audit found the reference right (docs/reviews/REFERENCE_AUDIT_2026-11-23.md); the PINN synthesis found why the PINN missed (docs/reviews/PINN_METHOD_SYNTHESIS_2026-11-23.md) |
| Arms B1 to B3, more seeds, campaign | Not started: seeds 2 to 4 at 2 000 points come next, then B1 to B3 |

**Read AGENTS.md first.** Its writing rule and honesty rules apply to every file here and to every agent that works here.

## Where things are

The root holds only rules, code, configs and data. Every document is under [docs/](docs/README.md), one folder per topic.

| Path | What it holds |
|---|---|
| AGENTS.md | Central rules: writing style, honesty, research and work rules. Read first |
| [docs/](docs/README.md) | All documents, indexed in docs/README.md |
| docs/status-log.md | What we did, newest first |
| docs/problem/ | The fixed problem: Eqs. 14 to 19, locked parameters, substituted sheet, supervisor questions |
| docs/plans/ | PRD, strategy and gates, TODO list, literature gap plan |
| docs/reference-solver/ | Classical reference: specification, checks, results, error bars, first-order form |
| docs/pinn/ | PINN plan, reading guide, implementation notes, boundary-condition design, runtime study |
| docs/reviews/ | Dated reviews |
| docs/literature/ | The 25 papers: synthesis, summaries, reading lists, verified references |
| docs/methodology/ | Method diagrams (figures/), their guide, Mermaid flowcharts, generated strategy guide |
| docs/team/ | Coding rules, skills and tools |
| docs/journal/ | Joint journal plan with the Lorenz work; no BVP tasks |
| docs/archive/ | Superseded documents, history only |
| src/bvp_pinn/ | Code: config loader, classical reference (`reference/`), PINN (`pinn/`) |
| tests/ | pytest tests; run `uv run pytest` |
| tools/ | Record check, guide and diagram builders, PINN step timer, manufactured-solution check, PINN error estimate, reference error bars |
| engine/ | Small C++ box-scheme solver of Eqs. 14 to 19, with a one-Newton-step error estimate for a PINN |
| configs/ | Locked case (`physical-case-pending.json`, hash-locked, so its old doc paths stay as written), reference settings, PINN settings, training proposal |
| reference/ | Generated reference data, figures, study tables and error bars; rebuild with `uv run python -m bvp_pinn.reference.run` |
| runs/ | PINN run folders (generated, not committed). `results/summary.csv` is rebuilt from them by `bvp_pinn.pinn.evaluate` |
| prior-work/ | FYDP paper context and the previous Lorenz PINN code, read-only |
| sources/ | Supervisor sheet, drafts and checked papers, with SHA-256 hashes. Data only |
| pyproject.toml, uv.lock | Python 3.12 environment; create it with `uv sync` |

## How we work

1. Pick a task from docs/plans/strategy.md or the status log.
2. Work behind the right gate: parameters complete, then software tests, then reference, then training, then evaluation.
3. Write what happened in docs/status-log.md. Keep it short and factual.
4. Run `python3 -I tools/check_records.py` before committing. It checks names, values, links, source hashes and the writing rule. It is not proof of solver correctness or convergence.
5. Commit with a plain message that says what changed.

## GitHub

Private remote: `ihmorol/bvp-boundary-pinn`. The supervisor documents and papers under sources/ are uploaded to that private repository only. Do not make the repository public while sources/ holds them.

## Next work

1. B0 seeds 2 to 4 with the frozen settings at 2 000 points, each accepted by the reference-free error-estimate score (campaign rule G6 needs five seeds).
2. Arms B1 to B3 with the same settings.
3. The reference audit's record and test fixes D1 to D4, D6, D7, and the manufactured-solution check with the new pipeline.

Every run is judged by the reference-free checks (decay powers m and mp, the Newton-step error estimate) before the reference is read. The supervisor said on 2026-11-19 that the physical meaning of the values is not needed: use them as given, compute a numerical reference, train the PINN on eta and test it against the reference.
