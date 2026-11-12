# Reading list and research strategy

## Research brief

Questions:

1. Which prior works directly help with five physical outputs, a fourth-order equation, and coupled boundary penalties?
2. Which training failures should our parameter choices and tests address?
3. Which improvements can we compare fairly, and what would count as a useful research result?

This is a focused reading plan, not proof that no similar five-field paper exists. It combines earlier verified primary sources with the new subagent searches. Recent does not automatically mean stronger evidence or a better method for our BVP.

## Read these first

### R1. Micropolar fluid and zero-flux boundaries, 2026

Hamid Reza Soltani Motlagh et al., *Physics-Informed Neural Networks for Dissipative Micropolar Nanofluid Flow with Microrotation Dynamics and Zero Nanoparticle Mass Flux*. Modelling 7(4), 145; published 22 July 2026.

- DOI: https://doi.org/10.3390/modelling7040145
- [Local full article](../sources/literature/micropolar-2026/article.pdf)
- Evidence: full article inspected previously and key method passages checked again locally on November 12.
- Read Eqs. 11–16, input/output definitions, Figure 2 and residual/loss Eqs. 22–30.
- Why useful: four fields, third-order flow plus three second-order equations, nine mixed wall/far-field conditions, including wall flow linked to concentration slope and zero nanoparticle flux.
- Method: coordinate plus seven given parameters as inputs, four outputs, JAX automatic differentiation, SiLU and float64. Figure 2 explicitly names Adam → L-BFGS → 8D-RAR. Equation and boundary losses are added without labeled solution data.
- Limits: this is a parametric network, not our one-case network. It has no microorganism output and no fourth-order flow equation. It fixes eta-infinity=7; that number is not a justified domain for our problem. Table 1 validates wall shear against earlier finite-element values, not independent full-profile accuracy for every field.
- Our action: reuse the boundary-error assembly idea. Keep one-case input eta only first. Check every field, the domain and wall derivatives ourselves.

### R2. Nonlinear fourth-order fluid equation, 2026

Umar Daraz and Khaleel Ahmad, *A Heuristic Physics-Informed Neural Network for Unsteady Magnetohydrodynamic Squeezing Flow through a Darcy-Forchheimer Porous Channel with Brownian Motion, Thermophoresis, Thermal Radiation, and Heat Generation*. International Scientific Spectrum; available online 26 September 2026.

- DOI: https://doi.org/10.66972/iscis202617
- [Local full article](../sources/literature/squeezing-2026/article.pdf)
- Evidence: full article and relevant derivative/loss paragraphs inspected locally; journal quality and reported error claims have not been independently audited.
- Read Eqs. 16–18, Section 6's reference, and Section 7's network/loss Eqs. 39–42.
- Why useful: three fields with orders (4,2,2), eight BCs, smooth direct physical outputs and explicit fourth derivatives.
- Method: smooth sigmoid network functions, boundary penalties, PSO–NNA and least-squares refinement. It also has a separate supervised surrogate later in the article.
- Limits: finite planar squeezing channel, different boundaries and no micropolar/microorganism field. The PDF retains Volume 00, Issue 00. Its theoretical certificates and optimizer choices do not transfer automatically to our system. Supervised regression figures are not evidence for physics-only training.
- Our action: inspect high-order residual construction and classical conversion. Do not import its optimizer or proof as our own.

### R3. Why loss size can mislead, 2026 preprint

Endalew and Zhang, *Adaptive Physics-Informed Neural Networks for the Blasius Boundary-Layer Problem*, arXiv:2609.22185.

- Source: https://arxiv.org/abs/2609.22185
- Evidence: abstract inspected by the recent-BVP subagent; not independently reproduced, full text not inspected in this pass. Author metadata should be checked at the record before a formal bibliography submission.
- Why useful: the inspected abstract reports that the deepest network obtained the smallest weighted loss but worse wall-shear accuracy than shallower choices. It uses adaptive weighting and Adam/L-BFGS for a simple boundary-layer case.
- Limits: preprint and a much smaller problem. Reported improvements are authors' benchmark results, not promises for ours.
- Our action: select runs using original residuals, BC errors and derivative-sensitive reference quantities, not weighted loss alone. Verify metadata and detailed algorithm before reproducing it.

### R4. Optimizer comparison, 2024

Pratik Rathore et al., *Challenges in Training PINNs: A Loss Landscape Perspective*. ICML, PMLR 235, 42159–42191.

- Source: https://proceedings.mlr.press/v235/rathore24a.html
- Evidence: primary abstract verified earlier; full-text settings not checked here.
- Finding: differential operators can make the optimization poorly conditioned; Adam+L-BFGS performed favorably compared with either alone on the paper's benchmarks.
- Our action: keep Adam-only and Adam→L-BFGS controls under documented budgets. No universal superiority claim.

### R5. Loss scaling and training guide, 2023

Sifan Wang, Shyam Sankaran, Hanwen Wang and Paris Perdikaris, *An Expert's Guide to Training Physics-informed Neural Networks*, arXiv:2308.08468, inspected v1 August 2023.

- Source: https://arxiv.org/abs/2308.08468
- Evidence: full text checked by earlier subagent.
- Read Sections 3, 5.2 and 6. Loss scales can differ even after nondimensionalization. It gives gradient-norm weighting and a moving average, with no gradient through the weights. It warns about fitting a fixed residual-point set too closely.
- Limits: its published grouping and numerical defaults are not our seventeen-component loss; its causal-weighting discussion concerns time. The inspected v1 does not prescribe L-BFGS for this BVP.
- Our action: fixed scaling baseline, then separate random-point and adaptive-weight tests. Do not use eta as a time-causality coordinate.

### R6. Adaptive sampling, 2023

Chenguang Wu et al., *A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks*. CMAME 403, 115671.

- DOI: https://doi.org/10.1016/j.cma.2022.115671
- Preprint: https://arxiv.org/abs/2207.10289
- Evidence: title and primary abstract verified earlier, including benchmark comparison of RAD/RAR-D. Detailed formulas not checked in this pass.
- Our action: test residual-driven point enrichment while retaining domain coverage and fixed final evaluation. Call a custom top-error enrichment method our method, not RAD/RAR-D unless their exact algorithm is obtained and implemented.

## Additional sources to study, not automatic upgrades

### R7. Direct versus derivative-helper outputs

Gladstone et al., *FO-PINNs: A First-Order formulation for Physics Informed Neural Networks*, arXiv:2210.14320, inspected revision 2023.

- Source: https://arxiv.org/html/2210.14320v2
- Evidence: full text verified earlier.
- Benefit to test: lower differentiation order through derivative outputs and compatibility residuals.
- Limits: tested examples are second-order PDEs, and some comparisons also change soft boundaries to exact boundaries. A helper called f-double-prime need not be the actual physical derivative when compatibility is only approximate.
- Our action: direct five-output baseline versus twelve-output first-order variant, same soft BCs and comparable budgets. Check the original high-order equations and physical-output derivatives as well as helper agreements.

### R8. Exact boundaries

Sukumar and Srivastava, *Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks*. CMAME, 2022; DOI https://doi.org/10.1016/j.cma.2021.114333; preprint https://arxiv.org/abs/2104.08426.

Lagaris, Likas and Fotiadis, *Artificial Neural Networks for Solving Ordinary and Partial Differential Equations*, 1998; DOI https://doi.org/10.1109/72.712178.

- Evidence: full-text constructions checked earlier.
- Benefit: exact simple endpoint conditions can remove their penalty error.
- Limit: generic linear Dirichlet/Neumann/Robin constructions are not proof of exact nonlinear coupled wall constraints.
- Our action: optional exact chi endpoint construction as a separate experiment, never a claim that all twelve BCs are exact.

### R9. Recent dimensional loss weights, 2025

Yi En Chou, Te Hsin Liu and Chao-An Lin, *Impact of Loss Weight and Model Complexity on Physics-Informed Neural Networks for Computational Fluid Dynamics*, arXiv:2509.21393v1, posted 24 September 2025.

- Source: https://arxiv.org/abs/2509.21393v1
- Evidence: primary arXiv metadata and abstract rechecked by the audit agent.
- Finding: dimensional-analysis-based weighting schemes; authors report improvements over equal weighting on heat/convection–diffusion/cavity benchmarks.
- Limits: abstract only, no full algorithm audited; no venue or DOI established. Do not endorse its claim about traditional-solver failure without numerical evidence. No complexity result can be inferred from the title alone.
- Our action: read full text before deciding whether its scaling fits our nondimensional coupled residuals.

### R10. Second-order methods at scale, 2025

Andrea Bonfanti et al., *PINN Balls: Scaling Second-Order Methods for PINNs with Domain Decomposition and Adaptive Sampling*, arXiv:2510.21262v1, posted 24 October 2025.

- Source: https://arxiv.org/abs/2510.21262v1
- Evidence: primary metadata and abstract verified by the audit agent.
- Finding: local expert/ensemble/domain-decomposition methods aim to make second-order training scalable.
- Limits: author metadata mentions an accepted conference paper without establishing its venue here. Benchmarks are not listed in the inspected abstract. No direct validation of our 1D BVP was confirmed.
- Our action: lower-priority reading. Do not replace a small MLP with this architecture before measuring our own bottleneck.

### R11. Hard initial conditions in stiff time problems

Baoli Hao et al., *Stability in Training PINNs for Stiff PDEs: Why Initial Conditions Matter*, arXiv:2404.16189v4, first posted 2024 and updated 15 August 2026.

- Source: https://arxiv.org/abs/2404.16189v4
- Evidence: primary metadata and abstract verified. Linked DOI https://doi.org/10.3934/fods.2026016 was not independently checked at publisher level in the audit.
- Finding: initial-condition enforcement can stabilize their stiff time-dependent PDE benchmarks.
- Limit: this does not prove the same for our steady nonlinear two-ended BVP. Do not import time marching.

## Parameter-value research is not authorization to fill missing values

Shi et al., *Modelling and numerical computation for flow of micropolar fluid towards an exponential curved surface: a Keller box method*, Scientific Reports 11, 16351, 2021; DOI https://doi.org/10.1038/s41598-021-95859-x, gives related curvature and microrotation ranges. A subagent inspected table values, but its parameter symbols and equations differ from ours.

Do not equate a paper's Eckert number with lambda_m, magnetic M with beta_m, or curvature symbol with a different convention without a derivation. The user explicitly chose to wait for missing supervisor values. No literature-filled Case 1 is authorized.

## Pros, cons and failure cases that shape the strategy

| Choice | Why start/test it | Main risk | Check |
|---|---|---|---|
| One five-output tanh MLP, float64 | Simple and traceable; smooth derivatives | High-order graphs and poorly balanced gradients | Analytic derivative tests, timing, per-equation losses |
| All-soft BC penalties | Handles given nonlinear/coupled relations | Small total loss can ignore one wall constraint | All twelve unscaled/scaled BC values |
| Adam then L-BFGS | Worth testing from optimizer evidence | Changing objective during line search; high memory | Frozen points/scales/weights; function-evaluation logs |
| Uniform plus wall coverage | Simple coverage in inverse-power region | Uniform oversampling or fixed-grid fitting | Dense independent residuals, sample-count sensitivity |
| Residual enrichment | Adds points where equations are poorly met | Residual does not equal solution error | Reference profiles and unchanged final grid |
| Adaptive loss weights | Helps detect neglected components | Tiny gradient gives extreme weight; trivial shrinkable weights | Stop-gradient weight updates, bounds and diagnostics |
| Derivative helpers | Lower derivatives in training objective | Original derivatives can remain inaccurate | Original residuals and BCs, not helpers alone |
| Exact selected endpoints | Removes known simple endpoint error | Mistaking helper exactness for physical exactness | Algebraic endpoint check and actual physical derivatives |

## A defensible research contribution

Our proposed question is whether a tested combination of sampling, loss scaling and derivative representation can solve this **specific** five-field (4,2,2,2,2) problem reliably, including the requested near-boundary coefficients.

A useful result could be a reproducible benchmark with a converged reference, corrected source transcription, per-condition error reporting, boundary-sensitive accuracy, measured cost and controlled method comparisons. If a tested change improves these, report the measured improvement for the tested parameter cases.

Do not claim “first five-field PINN,” “new best optimizer,” guaranteed stability, uniqueness or a speed advantage before a broader novelty search and the experiments. A plain PINN applied to another model does not by itself establish a new algorithm.

## Retrieval record and honest limits

Three subagents performed the initial recent-paper searches. Follow-up audits narrowed claims from abstracts and removed unsupported transfer claims. One application follow-up repeatedly failed at provider capacity. Its earlier report is not upgraded to a new full-text check. Two available local full texts were inspected directly instead. Public metadata was used for the latest generic sources; no private source documents were uploaded to research services.

Detailed training formulas from abstracts were not invented. Full text or a source implementation must be obtained before reproducing an advanced algorithm. Source-paper results remain author-reported until reproduced. The strategy's practical settings are testable engineering choices, not findings attributed to those papers.
