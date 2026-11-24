# All 25 papers in a few minutes

One short block per paper. Read this once and you know what each paper is and what it gives us. Full detail with page references lives in `sources/literature/<folder>/analysis.md` (each now starts with the same TL;DR). Start from [the tracker](paper-analysis-tracker.md) for status or [the synthesis](literature-synthesis.md) for decisions.

**Arm names (note of 2026-11-24):** B4, B5 and B6 in this file are the variants of the archived 2026-11-16 strategy (B4 gradient-based weights, B5 a twelve-output first-order network, B6 exact endpoint conditions). They are not the current arms. The current arms are B0 to B3 of [PINN_PLAN.md](../pinn/PINN_PLAN.md); the weighting and exact-condition ideas became shared settings ([IMPLEMENTATION_NOTES.md](../pinn/IMPLEMENTATION_NOTES.md) section 5.1).

Relevance: high means it shapes our pipeline choices. Medium means background or template only.

## Physical and structural precedents

**Micropolar nanofluid PINN, 2026 (R1).** Four fields with orders (3,2,2,2), nine wall and far-field conditions, cutoff eta=7, Adam then L-BFGS then one-shot RAR near the wall. For us: the closest physical precedent; we take the BC loss assembly and the float64 and SiLU evidence, not the eta=7 or the 8D parametric input. Only wall shear was validated.

**Squeezing flow PINN, 2026 (R2).** Fourth-order momentum (4,2,2) with eight plain BCs, constant coefficients, metaheuristic optimizer. For us: the penalty tuning rule (drive BC errors down to the interior level) and seed-spread reporting. Headline error 9.3e-9 is the best of 100 runs; typical runs are 3e-6. Journal un-audited.

**Hard-constrained Falkner-Skan, 2026 (R39).** One third-order BVP on a truncated domain; boundary lifting removes all three BCs; regional residual balancing; staged Adam then monitor-checked L-BFGS. For us: the closest structural template for the B6 and B4 arms. Wall error 6.7e-7, single seed.

**Stretching cylinder PINN, 2026 (R15).** Three fields, third-order flow, simple decay BCs, equal loss weights. For us: the equal-weight control arm and the per-field error table format. No wall-shift curvature terms, so it does not cover our stiffness.

**Lap joint supplement, 2021.** Code only: two coupled fourth-order fields, chained derivatives, plain unweighted loss. For us: the chained autodiff pattern for f'''' and the one-term-per-BC assembly.

**Coupled beams PINN, 2024.** Fourth-order beam PDEs, linear, analytic solutions, L-BFGS only. For us: the strongest scaling evidence we have: dimensional coefficients (4e6) fail at loss 1e14 while the nondimensional version reaches 5.3e-4 percent error.

## Foundations and surveys

**PINN foundations Part I, 2019 (R22).** The founding paper: residual plus data loss, L-BFGS full batch. For us: our 5 residual plus 12 boundary loss is a direct extension of its eqs. 4 and 6. Intro quotes on pages 6, 7 and 10.

**SciML survey, 2022 (R47).** Field review. For us: intro anchors. Soft BCs have no accuracy guarantee and no weight theory (p. 23). All loss terms must sit at one order of magnitude (pp. 58-59). Stiffness is a named open problem (pp. 57 and 61).

**Acta Numerica survey, 2024 (R46).** Error theory. For us: loss is not error. E cannot be computed during training (p. 657). Error scales about like the square root of training loss (p. 687). Our fourth-order residual makes the operator square order 8, so conditioning is the fight.

## Why PINN training fails

**Gradient pathologies, 2021 (R24).** Boundary gradients vanish while residual gradients stay orders larger; annealing fixes it. For us: the B4 rule we copy exactly (max over mean gradient ratio, EMA 0.9, every 10 steps) and the gradient histogram diagnostic.

**NTK failure view, 2022 (R25).** The residual kernel dominates the boundary kernel; trace-ratio weights rebalance it. For us: the weight rule and its cheap cost (one backward per term, 17 total). Large weights can destabilize the kernel, so cap them.

**Failure modes, 2021 (R26).** Six failure modes; the cause is optimization, not expressivity. For us: the failure taxonomy for our paper, the Hessian landscape diagnostic, the curriculum ramp idea, and the warning that weight sweeps alone do not rescue hard regimes.

**Failure as overfitting, 2026 (R42).** The residual dips at collocation points and stays large between them. For us: monitor-set divergence as the failure detector, fresh-set-only final metrics, and an optional gradient penalty that costs a fifth derivative for f.

## Loss weights

**Self-adaptive soft attention, 2023 (R27).** One trainable weight per point through a saturating sigmoid. For us: the only bounded pointwise construction, and its L-BFGS freeze rule matches ours. Lambda itself is unbounded; the wave case needed resampling to work.

**Balanced residual decay weights, 2025 (R28).** Pointwise weights from residual decay rates, one mean-of-one pool shared by equation and boundary points, under 10 percent cost. For us: the most concrete B4 arm. The multi-component extension is claimed but never shown.

**NTK sketching, 2025 (R40).** Convergence theory for NTK weights plus a cheap sketch estimator. For us: NTK weights become affordable: factor 2.5 cost, 17 block traces, no matrix storage. The theorem covers plain gradient descent only.

## Boundary enforcement

**Augmented Lagrangian PINNs, 2023 (R32).** BCs as constraints with learnable multipliers, one fixed beta, no inner loop. For us: the AL arm for our 12 BCs. Fixed beta beats a growing beta by 40x. Theory covers linear constraints only.

**hPINN hard constraints, 2021 (R33).** AL rounds for inverse design; exact construction only for Dirichlet and periodic BCs. For us: the AL round schedule (warm starts, beta growth 2, lambda over mu flattening in 6 to 9 rounds). AL on BC residuals is untested there.

**Dirichlet enforcement comparison, 2023 (R34).** Four routes compared; exact lifting beats tuned penalties. For us: the B6 lifting pattern and the penalty-scale warning (wrong lambda collapses the rate from 3.66 to 0.01). Value conditions only.

**BC enforcement study, 2025 (R36).** Newest side-by-side, including AL and annealing, on 2D and 3D tests. For us: AL wins on most tests, the split interior/boundary error metrics, and evidence that tuned beta=100 can beat annealing. Min-max Lagrange never converged. Despite the abstract, no Neumann-data case was run.

## Sampling

**Failure-informed sampling, 2023 (R38).** Enriches points where a residual-based failure probability is high. For us: the B3 algorithm: union enrichment keeps coverage, stop on the failure probability, settings N1=300, p0=0.1, N2=1000, at most 10 rounds. The error bound needs a linear operator, so it is not ours.

## Optimizers

**Optimizing the optimizer, 2025 (R30).** Quasi-Newton variants compared. For us: Strong Wolfe with c1=1e-4 and c2=0.9, a full-memory BFGS arm (L-BFGS was the worst quasi-Newton in its one test: 2.05e-3 versus 7.57e-8), and float64 for refinement. No steady BVP benchmark; the no-enhancements claim is not clean.

## Honest comparisons and reporting

**PINN versus FEM, 2024 (R43).** Systematic comparison; PINNs did not win on time or accuracy. For us: the cost protocol (train and evaluation time separately, same hardware), the no-superiority citation, and the staged warm-up pattern (a 1000x weight on one hard term made training possible).

**Pitfalls and frustration, 2022 (R44).** Published failure report: 32 hours of PINN training to match a 20-second finite difference run; a cylinder case never converged. For us: the failure-reporting template: config disclaimers, judgment calls logged, results with a DOI. Loss fell while the solution stayed wrong.

**PINNacle benchmark, 2024 (R45).** 22 PDE tasks, about a dozen method columns, three seeds each. For us: evaluation practice (NaN logged, 10 percent threshold, max error), boundary weight 100 helps elliptic and hurts steady NS, NTK reweighting strongest, L-BFGS NaN on 7 of 22 tasks yet best where it converged.
