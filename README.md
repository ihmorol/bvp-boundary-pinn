# Five-variable boundary PINN

Local research repository for the FYDP five-function boundary problem: solve Eqs. 14 to 19 with a physics-informed neural network. Separate from the old Lorenz project.

**Status:** all 22 physical coefficients are complete and locked from [the user's 2026-11-16 message reporting direct supervisor values](docs/parameters.md#supervisor-confirmation-2026-11-16). Da=0.2 is derived from the supplied M=5. Case status: `parameters_complete_validation_pending`.

Classical reference built and verified on 2026-11-19 on [0, 20] with asymptotic far conditions for theta and phi. Branch A is the target; see reference/study.md and docs/SPEC_reference_solutions.md section 8. No PINN implementation, training or evaluation has occurred.

**Read AGENTS.md first.** Its writing rule and honesty rules apply to every file here and to every agent that works here.

## Where things are

| Path | What it holds |
|---|---|
| AGENTS.md | Central rules: writing style, honesty, research and work rules |
| docs/parameters.md | Locked 22-field case, received H/Prm, mappings and historical sheet readings |
| docs/equations.md | Corrected transcription of Eqs. 14 to 19 and the first-order reference setup |
| docs/simplified-equations.md | Reading sheet: Eqs. 14 to 19 with the locked numbers substituted; equations.md stays authoritative |
| docs/strategy.md | Architecture, training flow, reference plan, tests, outputs, failure handling |
| docs/reading-list.md | Papers we checked, with what each one does and does not establish |
| docs/flowcharts.md | Editable Mermaid diagrams of the full flow |
| docs/methodology.md | Fifty small method tasks, full mathematical rules and source notes |
| bvp-methodology.pdf | Complete vector methodology diagram using the supplied flowchart theme |
| bvp-methodology.svg | Vector SVG copy with outlined text for portable mathematics |
| bvp-methodology.html | Editable, self-contained diagram with local fonts and MathML |
| docs/lorenz-to-bvp-methodology.md | Verified comparison with the live Lorenz PINN and the BVP adaptation |
| bvp-pinn-architecture.pdf | Lorenz-style BVP computational graph, training loop and complete math panels |
| bvp-pinn-architecture.svg | Vector architecture copy with outlined text |
| bvp-pinn-architecture.html | Editable architecture diagram with shared-network and optimizer-feedback paths |
| TODO.md | Ordered BVP task list, from the first reference step to results |
| docs/PRD.md | BVP problem statement, goal, scope, deliverables, success metrics |
| docs/SPEC_reference_solutions.md | Classical reference: solver, settings, verification, data format |
| docs/PINN_PLAN.md | BVP PINN arms, run rules, metrics, gates, figures, dates |
| docs/CODING_RULES.md | Layout, versions, style, configs, seeds, tests, logging, commits |
| docs/SKILLS_AND_TOOLS.md | Installed agent skills and core library documentation |
| docs/RESEARCH_NOTES.md | Verified references for the BVP, the reference solver and PINNs |
| journal/ | Joint journal plan aligning this BVP work with the separate Lorenz work; no BVP tasks |
| docs/status-log.md | What we did, newest first |
| docs/questions-for-supervisor.md | Open questions to ask the supervisor |
| configs/physical-case-pending.json | Numerically complete locked case; old filename kept for link compatibility |
| configs/training-proposal.json | Proposed training settings, not validated choices |
| tools/ | Record checks and the guide builder |
| src/bvp_pinn/ | Code: config loader and the classical reference package (reference/) |
| tests/ | pytest tests; run `uv run pytest` |
| pyproject.toml, uv.lock | Python 3.12 environment; create it with `uv sync` |
| configs/reference.json | Numerical settings of the classical reference (our choices) |
| reference/ | Generated reference data, figures and study tables; rebuild with `uv run python -m bvp_pinn.reference.run` |
| prior-work/ | FYDP paper context and the previous Lorenz PINN code, read-only reference |
| docs/archive/ | Superseded documents kept for history |
| sources/ | Supervisor sheet, drafts and checked papers, with SHA-256 hashes |
| strategy-guide.html | Generated readable guide; rebuild with tools/build_guide.py |

## How we work

1. Pick a task from docs/strategy.md or the status log.
2. Work behind the right gate: parameters complete, then software tests, then reference, then training, then evaluation.
3. Write what happened in docs/status-log.md. Keep it short and factual.
4. Run `python3 -I tools/check_records.py` before committing. It checks names, values, links, source hashes and the writing rule. It is not proof of solver correctness or convergence.
5. Commit with a plain message that says what changed.

## GitHub

Private remote: `ihmorol/bvp-boundary-pinn`. The supervisor documents and papers under sources/ are uploaded to that private repository only. Do not make the repository public while sources/ holds them.

## Next work and open input

The physical-value gate is complete. Next come implementation and software validation under docs/strategy.md, then reference, training and evaluation. Physical values cannot be tuned or changed automatically; an explicit user or supervisor update is required. Optional debug overrides stay in a separate researcher-proposed case and never change the locked target.

The supervisor answered on 2026-11-19 that the physical meaning of the values is not needed: use them as given, compute a numerical reference, train the PINN on eta and test it against the reference. The definition questions in docs/questions-for-supervisor.md are closed. The numerical/training settings in configs/training-proposal.json are unchanged, including eight nulls.
