# Reading shortlist: what to read seriously

The 25 saved papers are already summarized in `analysis.md` files with page references, and [papers-tldr.md](papers-tldr.md) gives the two-minute version. A full human read is worth it only for the papers below. Everything else: rely on the analysis files and dip in when a decision needs the detail.

## Tier 1: read seriously, in this order

| # | Paper | Why | What to read |
|---|---|---|---|
| 1 | falkner-skan-hard-bc-2026 | Closest structural template: hard endpoints, regional weights, staged training with rollback | Sections 4 to 7, the ablation section, the limits section (p. 27) |
| 2 | gradient-pathologies-2021 | The B4 annealing rule we copy, and the failure mechanism behind it | Sections 2.2 to 2.4, Algorithm 1 (p. 10) |
| 3 | al-pinn-2023 | The AL arm for our twelve boundary terms | Sections 2 to 4, Appendix B hyperparameters |
| 4 | optimizer-optimizing-2025 | L-BFGS settings, the BFGS arm, stopping rules, float64 need | Sections 2.1 to 2.2, Section 4, Appendices A and B |
| 5 | micropolar-2026 | Closest physical precedent with our BC family | Eqs. 11 to 16, residuals and losses 22 to 30, Table 1 |

## Tier 2: read the key sections when the matching arm starts

| Paper | Why | What to read |
|---|---|---|
| acta-numerica-survey-2024 | Evaluation theory: loss versus error | Section 3.2, Theorem 5.3, Sections 7.3 to 7.5 |
| bc-comparison-2023 | B6 evidence and the fairness protocol | Section 3, Sections 4.1 to 4.2 |
| fi-pinn-sampling-2023 | B3 enrichment algorithm | Sections 3 to 4, Algorithms 1 and 2 |
| residual-decay-weights-2025 | Third B4 arm | Section 3.2, Section 4.1 |
| failure-modes-2021 | Failure taxonomy for our results section | Sections 3 to 5 |
| overfitting-failure-2026 | Monitor and fresh set design | Sections 3 to 4 |
| ntk-failure-2022 | Second B4 arm and its cost | Sections 6 to 7 |

## Tier 3: rely on analysis.md, no full read needed

pinn-foundations-2019 (keep its intro quotes), sciml-survey-2022, pinnacle-benchmark-2024, pinn-vs-fem-2024, pinn-pitfalls-report-2022, self-adaptive-2023, ntk-sketching-2025, bc-3d-comparison-2025, hpinn-hard-constraints-2021, beams-2024, lap-joint-2021, cylinder-nano-2026, squeezing-2026 (journal un-audited; our analysis covers the caveats).

Notes: reading-list records R12 to R21 have no local PDF; cite them at record level only. Preprints stay labeled as preprints in anything we write. Nothing here changes the parameter rule: missing coefficient values still wait on the supervisor.
