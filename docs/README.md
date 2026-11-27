# Docs index

Start with [status-log.md](status-log.md) for what happened last, and [plans/TODO.md](plans/TODO.md) for what comes next. The rules for every file are in [AGENTS.md](../AGENTS.md).

| Folder | What it holds | Start with |
|---|---|---|
| [problem/](problem/) | The fixed problem: Eqs. 14 to 19 as given, the locked parameter values, a substituted reading sheet, and the closed supervisor questions | [equations.md](problem/equations.md), [parameters.md](problem/parameters.md) |
| [plans/](plans/) | What we build and in which order: problem statement, the gates and where each stands, task list, the literature gap plan, and the study of which L to use | [strategy.md](plans/strategy.md), [TODO.md](plans/TODO.md), [PRD.md](plans/PRD.md) |
| [reference-solver/](reference-solver/) | The classical reference: specification, checks, results, and the first-order form it solves | [SPEC_reference_solutions.md](reference-solver/SPEC_reference_solutions.md) |
| [pinn/](pinn/) | The PINN: plan and arms, a reading guide, a code-level walk through one training run, the hard terms and logic with the four arms and the experiments, implementation notes, boundary-condition design, the runtime study, and the result figures with the papers each follows | [PINN_PLAN.md](pinn/PINN_PLAN.md), [TRAINING_WALKTHROUGH.md](pinn/TRAINING_WALKTHROUGH.md), [CODE_EXPLAINED.md](pinn/CODE_EXPLAINED.md), [PINN_GUIDE.md](pinn/PINN_GUIDE.md) |
| [reviews/](reviews/) | Dated reviews of the code and the runs: the reference audit (the reference is right), the PINN method synthesis (why the PINN missed, and the plan), the first PINN review, the list of what is left before the campaign, the main-run readiness review with the Colab run, and two runs at once with a second Colab account | [PARALLEL_RUNS_2026-11-28.md](reviews/PARALLEL_RUNS_2026-11-28.md), [MAIN_RUN_READINESS_2026-11-26.md](reviews/MAIN_RUN_READINESS_2026-11-26.md), [BOUNDARY_CONDITIONS_2026-11-26.md](reviews/BOUNDARY_CONDITIONS_2026-11-26.md), [IMPLEMENTATION_GAP_2026-11-24.md](reviews/IMPLEMENTATION_GAP_2026-11-24.md), [REFERENCE_AUDIT_2026-11-23.md](reviews/REFERENCE_AUDIT_2026-11-23.md), [PINN_METHOD_SYNTHESIS_2026-11-23.md](reviews/PINN_METHOD_SYNTHESIS_2026-11-23.md) |
| [literature/](literature/) | The 25 papers: synthesis, short summaries, reading lists and verified references | [literature-synthesis.md](literature/literature-synthesis.md), [papers-tldr.md](literature/papers-tldr.md) |
| [methodology/](methodology/) | The method diagrams (one page each, and a six-page detailed methodology with every step and the reduced equations), one explainer file per step with a glossary ([steps/](methodology/steps/README.md)), and their guide, the Mermaid flowcharts, and the generated strategy guide. Figures are in [methodology/figures/](methodology/figures/) | [methodology.md](methodology/methodology.md) |
| [team/](team/) | Coding rules and the agent skills and tools we use | [CODING_RULES.md](team/CODING_RULES.md) |
| [journal/](journal/) | The joint journal plan with the separate Lorenz work. It sets no BVP tasks | [README.md](journal/README.md) |
| [archive/](archive/) | Superseded documents, kept for history only: the first methodologies, the 2026-11-16 strategy and training proposal, and the Lorenz comparison | [README.md](archive/README.md) |

Generated files: `methodology/strategy-guide.html` (rebuild with `python3 -I tools/build_guide.py`) and the diagrams in `methodology/figures/` (rebuild with `tools/build_diagrams.py`). Edit their sources, not the outputs.
