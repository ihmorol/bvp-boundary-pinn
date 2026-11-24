# Five-variable boundary PINN

Local research repository for the FYDP five-function boundary problem: solve Eqs. 14 to 19 with a physics-informed neural network. Separate from the old Lorenz project.

**Values:** all 22 physical coefficients are complete and locked from [the user's 2026-11-16 message reporting direct supervisor values](docs/problem/parameters.md#supervisor-confirmation-2026-11-16). Da=0.2 is derived from the supplied M=5. Where the work stands against the gates: [docs/plans/strategy.md](docs/plans/strategy.md) section 2.

**Fixed problem.** Eqs. 14 to 18, B1 to B12 and the locked values are used exactly as the supervisor gave them. Only the way they are imposed and solved may change (AGENTS.md section 3).

| Part | Status on 2026-11-24 |
|---|---|
| Classical reference | Done and verified on [0, 20] with asymptotic far conditions for theta and phi: solve_bvp, Runge-Kutta shooting and a C++ box scheme agree to about 1e-10. Branch A is the target; a deflation search finds only branches A and B. Error bars: mesh about 1e-7 relative; L = 20 against the eta -> infinity limit 8e-5 to 8e-4 per wall value, checked on an L study to 10 240 with two forms of the far condition; chi'(0) has no limit in L (Eq. 18 as given). The audit defects D1 to D7 are closed. See docs/reference-solver/SPEC_reference_solutions.md section 8 |
| PINN code | Complete for all four arms and tested (60 tests): exact boundary layer for all 12 conditions, Taylor-mode derivatives, start profile, residual scales, the same L-BFGS and Levenberg-Marquardt refine chain for every arm (block form for B1's window networks), an acceptance rule without the reference, a resumable campaign runner, and evaluation with every planned table and figure. A dry campaign of all 20 runs went through the whole chain |
| PINN result | **B0 matches the reference, two seeds.** From scratch at 1 000 points, seeds 0 and 1: every wall value except phi(0) within 4e-4 relative of reference A (seed 0 within 6e-5; theta(0) 0.2501906 against 0.2501853); phi(0) is 4.3e-4 (seed 0) and 6.2e-3 (seed 1) off. Relative L2 1e-4 to 5e-3. See docs/pinn/IMPLEMENTATION_NOTES.md sections 7.1 to 7.4 |
| Reviews | The reference audit found the reference right (docs/reviews/REFERENCE_AUDIT_2026-11-23.md); the PINN synthesis found why the PINN missed (docs/reviews/PINN_METHOD_SYNTHESIS_2026-11-23.md) |
| Pilots and campaign | Not run: pilots P2b (point count, budgets) and P3 (windows, replay weight), then the 20-run campaign. See docs/plans/TODO.md |

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
| src/bvp_pinn/ | Code: config loader, classical reference (`reference/`), PINN training and acceptance (`pinn/`, never reads the reference), evaluation against the reference (`evaluation/`). See src/bvp_pinn/README.md |
| tests/ | pytest tests; run `uv run pytest` |
| tools/ | Record check, guide and diagram builders, PINN step timer, PINN error estimate for older runs, reference error bars |
| engine/ | Small C++ box-scheme solver of Eqs. 14 to 19, with a one-Newton-step error estimate for a PINN and a deflated branch search |
| configs/ | Locked case (`physical-case-pending.json`, hash-locked, so its name and status string stay as written), reference settings, PINN settings with the acceptance rule and the campaign |
| reference/ | Generated reference data, figures, study tables and error bars; rebuild with `uv run python -m bvp_pinn.reference.run` |
| runs/ | PINN run folders (generated, not committed) |
| results/ | One folder per campaign, rebuilt from the tagged run folders by `bvp_pinn.evaluation`: summary.csv, failures.md, tables and figures. Dry-run campaigns are not committed |
| prior-work/ | FYDP paper context and the previous Lorenz PINN code, read-only |
| sources/ | Supervisor sheet, drafts and checked papers, with SHA-256 hashes. Data only |
| pyproject.toml, uv.lock | Python 3.12 environment; create it with `uv sync` |

## How we work

1. Pick a task from docs/plans/TODO.md or the status log.
2. Work behind the right gate: parameters complete, then software tests, then reference, then training, then evaluation.
3. Write what happened in docs/status-log.md. Keep it short and factual.
4. Run `python3 -I tools/check_records.py` before committing. It checks names, values, links, source hashes and the writing rule. It is not proof of solver correctness or convergence.
5. Commit with a plain message that says what changed.

## GitHub

Private remote: `ihmorol/bvp-boundary-pinn`. The supervisor documents and papers under sources/ are uploaded to that private repository only. Do not make the repository public while sources/ holds them.

## Next work

1. Pilot P2b: one B0 run at 2 000 points to freeze the point count and the budgets; pilot P3 for the window edges and the replay weight.
2. The campaign: `uv run python -m bvp_pinn.pinn.campaign`, then `bvp_pinn.evaluation.evaluate`, `.classical` and `.plots`.

Every run is judged by the acceptance rule in configs/pinn.json (decaying far field, no constant tail, the Newton-step error estimate) before the reference is read. The supervisor said on 2026-11-19 that the physical meaning of the values is not needed: use them as given, compute a numerical reference, train the PINN on eta and test it against the reference.
