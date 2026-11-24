# Paper analysis tracker

One agent reads one paper in full and writes `analysis.md` inside that paper's folder under `sources/literature/`. Two agents run at a time. This file tracks progress and holds the one-line result of each analysis.

How to use: check the Status column before trusting a folder's `analysis.md`. Status values: pending, running, done, failed (with a note). The Takeaway column is a one-line summary. The `analysis.md` file holds the detail with page references.

Rules the agents follow: read the whole PDF, quote only what the paper reports with a page or section reference, write in short plain sentences, no banned words from AGENTS.md, no em dashes, no invented numbers. Author-reported results stay author-reported in every file. Each analysis.md now starts with a short TL;DR record; the two-minute overview of all papers is [papers-tldr.md](papers-tldr.md) and the tiered full-read list is [reading-shortlist.md](reading-shortlist.md).

## Queue

| # | Folder | Role | Pages | Status | Takeaway |
|---|---|---|---|---|---|
| 1 | micropolar-2026 | R1, closest physical precedent (4 fields, 9 BCs) | 18 | done | Plain sum of PDE and BC losses, no weights or scaling; 8D parametric input; Adam to 50k then L-BFGS plus one-shot RAR near wall; only wall shear validated. |
| 2 | squeezing-2026 | R2, fourth-order momentum precedent (4,2,2) | 41 | done | Sigmoid nets with closed-form fourth derivatives; PSO-NNA without settings; bvp4c-like reference to 1e-9; headline errors are best runs, not averages. |
| 3 | falkner-skan-hard-bc-2026 | R39, closest structural BVP precedent | 30 | done | Hard lifting removes three linear BCs; ARRB regional weights from proxy gradient norms with EMA and floor; staged Adam then monitor-rolled-back L-BFGS; single seed. |
| 4 | cylinder-nano-2026 | R15, curved stretching cylinder PINN | 25 | done | No wall-shift curvature terms despite the reading-list hope; first-order residuals only, inconsistent claims; usable as equal-weight baseline and per-field error table template. |
| 5 | lap-joint-2021 | Prior pass 3, two fourth-order fields, code supplement | 5 | done | Chained tf.gradients to u_xxxx, two cross-coupled residuals, plain sum of ten squared terms at weight one; endpoint targets partly hidden; no optimizer details. |
| 6 | beams-2024 | Prior pass 4, coupled fourth-order beam PDE | 15 | done | Dimensional coefficients (4e6) fail at loss 1e14 while the nondimensional version reaches 5.3e-4 percent error; strongest external case for fixed scales first; L-BFGS only. |
| 7 | pinn-foundations-2019 | R22, founding PINN paper | 22 | done | Loss is MSE_u + MSE_f (eq. 4) extended by MSE_b for slope BCs (eq. 6), all unit weights; L-BFGS full batch only; Latin hypercube 10k to 20k points; no data-free claim for forward problems. |
| 8 | gradient-pathologies-2021 | R24, gradient pathologies and annealing | 28 | done | Per-term gradient histograms show boundary gradients pile near zero; annealing lambda = max|grad residual| over mean|grad term i| with EMA alpha 0.9 every 10 Adam steps; no cap; Adam only, L-BFGS never used. |
| 9 | ntk-failure-2022 | R25, NTK view of training failure | 29 | done | Residual kernel dominates boundary kernel; weight update lambda_m = sum of traces over Tr(K_m), no bound, warn large weights destabilize; trace = sum of squared per-point grads, so 17 terms need 17 backwards; L-BFGS never discussed. |
| 10 | failure-modes-2021 | R26, PINN failure modes taxonomy | 20 | done | Failures are optimization-driven, not expressivity; lambda sweep alone fails (negative control); curriculum warm-start ramp drops error 7.5e-1 to 9.8e-3 at beta=20; seq2seq is time marching, does not transfer to our two-ended BVP. |
| 11 | self-adaptive-2023 | R27, self-adaptive pointwise weights | 24 | done | One trainable weight per point via sigmoid mask, gradient ascent with separate rate, no cap (violates our bound rule as-is); weights held constant in L-BFGS; wave case failed without resampling; strip weights for honest loss logging. |
| 12 | residual-decay-weights-2025 | R28, balanced residual decay weights | 64 | done | Pointwise weight = R^2 over sqrt(EMA R^4), mean-one pool shared by PDE and BC points, EMA smoothing, under 10 percent cost; Adam only, L-BFGS dropped; multi-component extension stated but never shown. |
| 13 | al-pinn-2023 | R32, augmented Lagrangian BC constraints | 19 | done | AL loss adds linear multiplier term per condition, simultaneous descent-ascent each step, fixed beta beats growing beta by 40x; theory covers linear trace constraints only; Adam only. |
| 14 | hpinn-hard-constraints-2021 | R33, hard constraints with AL | 29 | done | AL updates per-point multipliers from signed residuals between outer rounds, hard-coding applies only to Dirichlet/periodic BCs, Neumann stays soft; never applies AL to BC residuals; Adam then L-BFGS each round. |
| 15 | bc-comparison-2023 | R34, Dirichlet enforcement comparison | 22 | done | Exact lifting with normalized distance function beats tuned penalties; penalty collapses at wrong lambda (rate 3.66 to 0.01); Nitsche never better; all tests Dirichlet only, Neumann and Robin excluded. |
| 16 | bc-3d-comparison-2025 | R36, BC enforcement comparative study | 34 | done | AL best on 2D and two of three 3D tests; fixed penalty fails both ways yet tuned beta=100 beats annealing; abstract claims Neumann tests but none run; split interior/boundary error metric worth copying. |
| 17 | fi-pinn-sampling-2023 | R38, failure-informed sampling | 25 | done | Failure probability from |r| minus threshold, truncated-Gaussian importance sampling, union enrichment keeps coverage, stop when PF below epsilon; error bound needs a linear operator, so it does not cover our nonlinear system. |
| 18 | overfitting-failure-2026 | R42, failure as overfitting | 15 | done | Residual dips to 1e-11 at points while 1e5 between them; penalize squared first input-derivative of each loss term (needs order-5 autograd for our f); train/fresh-set divergence is the failure detector; lambda_r never reported. |
| 19 | ntk-sketching-2025 | R40, NTK weight convergence and sketching | 29 | done | Convergence proven for plain GD with per-block trace-ratio weights; sketch costs 2 extra forwards plus 1 backward (factor 2.5), no matrix storage; theory frame admits 17 components but step bound shrinks with stiff fourth-order residual. |
| 20 | optimizer-optimizing-2025 | R30, quasi-Newton study for PINNs | 43 | done | SSBroyden with Strong Wolfe wins; L-BFGS tested once and was worst quasi-Newton, no settings given; rescaling tau = min(1, 1/b_k) gives 2 to 3 orders gain on full-memory BFGS; float64 required; no steady BVP benchmark. |
| 21 | pinn-vs-fem-2024 | R43, honest PINN vs FEM comparison | 27 | done | Wall-clock solution and evaluation time on one machine, 10-run averages, log-log accuracy versus time; PINN 1 to 6 orders slower; loss scaling rescales one term by 1000 and stages training; no L-BFGS speedup number in the PDF. |
| 22 | pinn-pitfalls-report-2022 | R44, honest negative results report | 9 | done | 32h versus 20s cost claim with config disclaimers; cylinder failure diagnosed via drag coefficient; low loss never proves a right solution; failure report with Zenodo DOI is the template for our failed-run reporting. |
| 23 | pinnacle-benchmark-2024 | R45, PINNacle benchmark | 50 | done | 3 seeds with mean and std, NaN logged as non-convergence, 10 percent L2RE threshold, max-error metric; boundary weight 100 helps elliptic but hurts steady NS; NTK reweighting strongest; L-BFGS NaN on 7 of 22 tasks (verified against the PDF); some table cells disagree internally. |
| 24 | acta-numerica-survey-2024 | R46, error analysis survey | 81 | done | Error decomposition eq. 3.7; interior residual enters squared, boundary residual at power one via trace inequality; functions with tiny residual and huge error exist (Thm 5.14); L*L for our fourth order is order 8, so conditioning is the real fight; hard BCs condition better in their toy case. |
| 25 | sciml-survey-2022 | R47, SciML survey | 85 | done | Soft BCs lack accuracy guarantee and weight theory; keep loss terms one order of magnitude apart; Adam then L-BFGS is the reported standard; stiffness and poor BC-to-interior propagation named as key open issues. |

## Synthesis steps

| Step | Status |
|---|---|
| Read all 25 analysis.md files | done |
| Write docs/literature/literature-synthesis.md | done |
| Update docs/status-log.md, run tools/check_records.py | done |
