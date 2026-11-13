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

## Earlier saved reading list

Four coupled-BVP papers were saved in the first literature pass on 2026-11-06, with provenance notes in sources/literature/READING_LIST-prior.md. R1 and R2 above come from that set. The lap-joint code supplement and the coupled-beams paper complete it.

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

## Second literature pass, November 12, 2026

Four parallel searches ran on November 12 (physical precedents, training methodology, boundary and sampling methods, benchmarks and surveys). Every candidate was then re-verified in-session against the arXiv API or the Crossref DOI registry before it entered this list. Titles, venues and years come from those fetched records. No entry rests on memory alone. Entries marked "no open PDF" are verified records without a readable full text; their claims stay at record level. Author-reported results stay author-reported; nothing from these papers was copied into configs/ or used to fill the missing parameters.

### Role map for the new entries

| Entries | Papers | Where they help us |
|---|---|---|
| R12, R13, R14 | Bioconvection PINNs with microorganism fields | Intro precedent for our chi field; how others weight a microorganism residual |
| R15 | Curved stretching cylinder PINN | Curved-wall coordinate handling next to our curvature terms |
| R16 | Free-convective derivative network with FD reference | Derivative representation arm; honest failure reporting |
| R17, R18 | Micropolar PINNs, channel and paraboloid | Microrotation field precedent; PINN paired with an independent check |
| R19, R20, R21 | Boundary-layer and stretching-flow PINNs | Application framing; PINN-as-checker precedent |
| R22, R23 | Founding PINN papers | Introduction anchors for the residual-penalty formulation |
| R24, R25, R26, R27 | Training failure and weighting foundations | Loss scaling and the B4 adaptive weight comparison |
| R28, R29, R30, R31 | Recent weighting and optimizer studies | Fixed versus adaptive weights; Adam and L-BFGS schedule |
| R32, R33, R34, R35, R36 | Boundary enforcement methods | Soft penalties versus exact imposition, comparison arm B6 |
| R37, R39 | Auxiliary outputs; hard-constrained Falkner-Skan | First-order arm B5; closest structural BVP precedent |
| R38, R40 | Failure-informed sampling; NTK weight theory | Residual enrichment arm B3 |
| R41, R42 | Large-coefficient failure remedy; overfitting view | Stiff-wall diagnosis; fresh-point evaluation design |
| R43, R44, R45 | PINN versus classical comparisons and benchmarks | The no-superiority-claim stance and evaluation practice |
| R46, R47, R48 | Error analysis survey, PINN survey, critical review | Intro positioning; residual-to-error reasoning |

### Closest physical precedents

#### R12. Bioconvection PINN with our exact optimizer chain, 2026

Azari, Behrouzifar, Mahboobtosi and Ganji, *Results in Surfaces and Interfaces*, art. 101043, 2026. DOI https://doi.org/10.1016/j.rsurfi.2026.101043. No open PDF found; record verified via Crossref.

- Evidence: Crossref record fetched in this session. The abstract was not readable, so the content claim is title-based.
- Claim (title): a TDF bioconvection nanofluid with gyrotactic microorganisms is modeled with a PINN trained with the Adam-LBFGS chain named in the title.
- Use here: the only verified precedent that combines a microorganism transport field with our planned two-stage optimizer schedule.
- Limits: microfluidics geometry, not a curved boundary layer; new paper without a citation record; title-based evidence only.

#### R13. Unified PINN for bioconvection over a stretching surface, 2024

Chandra and Das, *Engineering Applications of Artificial Intelligence* 133, art. 108647, 2024. DOI https://doi.org/10.1016/j.engappai.2024.108647. No open PDF found; record verified via Crossref and OpenAlex.

- Evidence: Crossref record and OpenAlex keywords fetched in this session ("microorganism density" keyword matched 0.82).
- Claim (record): one unified network with an interior point algorithm studies bioconvection nanofluid flow over an inclined stretching surface.
- Use here: same architectural choice as ours, one network carrying several fields and residuals; documents an interior-point training alternative.
- Limits: closed access; abstract not read; 26 citations per OpenAlex are a search-side count.

#### R14. PINN predicting a microorganism-density field, 2026

Mathur, Gupta, Mishra and Singh, *International Journal of Thermofluids* 35, art. 101690, 2026. DOI https://doi.org/10.1016/j.ijft.2026.101690. No open PDF found; record verified via Crossref; abstract read via OpenAlex.

- Evidence: Crossref record fetched in this session. One earlier web result mislabeled the journal; the Crossref container title is International Journal of Thermofluids.
- Claim (abstract): a PINN-based predictive framework for MHD stagnation-point bioconvective transport predicts a microorganism-density field among the coupled outputs.
- Use here: precedent for a PINN carrying our fifth field; when we choose weights for R18, we can compare how they handled the same residual.
- Limits: gold OA journal, but the PDF endpoint blocked scripted download; Casson hybrid nanofluid model differs from ours.

#### R15. PINN over a stretching cylinder, 2026

Khan, Khan, Ahammad et al., *Discover Nano* 21, art. 502, 2026. DOI https://doi.org/10.1186/s11671-026-04962-3. [Local PDF](../sources/literature/cylinder-nano-2026/stretching_cylinder_pinn_discover_nano_10.1186_s11671-026-04962-3.pdf)

- Evidence: Crossref record and OpenAlex abstract fetched in this session; full PDF downloaded from the Springer page.
- Claim (abstract): a local non-similarity PINN solves MHD mixed convection of a Jeffrey fluid over a stretching horizontal cylinder in a Darcy porous medium, predicting momentum, thermal and solutal fields.
- Use here: a curved wall with coordinate terms, closest published setup to our eta+kappa curvature handling.
- Limits: no microorganism field and no fourth-order momentum; brand new with no citation record.

#### R16. Physics-informed derivative networks with a finite difference reference, 2025

Cong, Li, Sun and Ren, *Scientific Reports* 15, art. 18766, 2025. DOI https://doi.org/10.1038/s41598-025-03918-4. Publisher and PMC PDF endpoints are CAPTCHA-gated, so no local PDF; record verified via Crossref and the PMC page metadata.

- Evidence: Crossref record and OpenAlex abstract fetched in this session; PMC page metadata confirmed in a browser session.
- Claim (abstract): physics-informed derivative networks output first derivatives and integrate numerically, solving the coupled Kuiken free-convective similarity ODEs across Prandtl numbers 0.1 to 10 where a same-size standard PINN fails; checked against a finite difference reference.
- Use here: published precedent for our derivative representation arm (direct outputs versus helper states) and for reporting where PINN training fails.
- Limits: free convection, not micropolar or nanofluid; mid-tier venue; abstract-level evidence for the failure claim.

#### R17. Micropolar MHD channel PINN with a semi-analytical check, 2025

Mirzagoli Ganji, Shateri and Ganji, *Physics of Fluids* 37(11), art. 113115, 2025. DOI https://doi.org/10.1063/5.0301887. No open PDF found; record verified via Crossref; abstract read via OpenAlex.

- Evidence: Crossref record and OpenAlex abstract fetched in this session.
- Claim (abstract): steady MHD micropolar flow in a porous channel with microrotation, spin-gradient viscosity, micro-inertia and convective heat and mass transfer is solved by a PINN paired with the semi-analytical Akbari-Ganji method as a benchmark.
- Use here: a microrotation field inside a coupled nonlinear BVP, and the same checking pattern as our classical reference: the PINN is judged against a separate method, not against itself.
- Limits: channel flow, not a boundary layer; closed access; 28 citations per OpenAlex.

#### R18. Micropolar flow over a paraboloidal surface, 2025

Vaseem, Uddin and Upreti, *International Communications in Heat and Mass Transfer* 169, art. 109645, 2025. DOI https://doi.org/10.1016/j.icheatmasstransfer.2025.109645. No open PDF found; record verified via Crossref.

- Evidence: Crossref record fetched in this session. The record spells the third author's name slightly differently than the search result; the record spelling is authoritative.
- Claim (title): a PINN study of phase-change micropolar fluid flow over a paraboloidal surface for solar thermal applications.
- Use here: a curved wall with a microrotation field; relevant to how curvature stiffens the wall region.
- Limits: title-based evidence; closed access; phase-change physics does not appear in our equations.

#### R19. PINNs for boundary-layer thermal-fluid problems, 2022

Bararnia and Esmaeilpour, *International Communications in Heat and Mass Transfer* 132, art. 105890, 2022. DOI https://doi.org/10.1016/j.icheatmasstransfer.2022.105890. No open PDF found; record verified via Crossref.

- Evidence: Crossref record fetched in this session; OpenAlex lists Falkner-Skan flow among its concepts.
- Claim (record): PINNs applied to viscous and thermal boundary-layer problems including Falkner-Skan-type flows.
- Use here: the established early reference that PINNs can solve nonlinear similarity BVPs; later weighting papers measure against works like this.
- Limits: closed access; abstract not read; 2022-era PINN practice; 153 citations per OpenAlex is a search-side count.

#### R20. PINN for MHD porous-media stretching flow, 2026

Jan, Khan and Farooq, *International Communications in Heat and Mass Transfer* 172, art. 110592, 2026. DOI https://doi.org/10.1016/j.icheatmasstransfer.2026.110592. No open PDF found; record verified via Crossref; keywords read via OpenAlex.

- Evidence: Crossref record and OpenAlex record fetched in this session.
- Claim (record): a PINN models MHD boundary-layer flow with heat transfer over a linearly stretching boundary in porous media.
- Use here: a smaller sibling of our wall-flow problem that combines magnetic, porous and dissipative effects; shows the mixed wall condition setup in a strong heat transfer journal.
- Limits: closed access; fewer fields than ours; abstract text unavailable.

#### R21. PINN as a checker on squeezing-flow ODEs, 2026

Gaur and Srinivasacharya, *Physics of Fluids* 38(3), art. 032012, 2026. DOI https://doi.org/10.1063/5.0314754. No open PDF found; record verified via Crossref; abstract read via OpenAlex.

- Evidence: Crossref record and OpenAlex abstract fetched in this session.
- Claim (abstract): for 3D squeezing flow of a rotating channel with a CNT hybrid nanofluid, similarity ODEs are solved by Haar wavelet collocation checked against bvp4c, and a PINN verifies solution accuracy and enforces physical constraints.
- Use here: the inverse role to ours: a classical solver first, PINN as verifier. This matches our classical-reference-first plan and shows the pattern is accepted practice.
- Limits: the PINN is not the primary solver there; closed access.

### Foundations for the introduction and the loss design

#### R22. The founding PINN paper, 2019

Raissi, Perdikaris and Karniadakis, *Journal of Computational Physics* 378, 686-707, 2019. DOI https://doi.org/10.1016/j.jcp.2018.10.045. [Local PDF](../sources/literature/pinn-foundations-2019/pinn_part1_arxiv_1711.10561.pdf) is the arXiv Part I companion (arXiv:1711.10561); the journal version is paywalled and the arXiv title differs from the journal title.

- Evidence: Crossref record with exact volume and pages fetched in this session; arXiv Part I record fetched separately.
- Claim (record): PDE residuals are embedded as penalties in the loss so networks solve forward and inverse problems without labeled solution data.
- Use here: the canonical source for our residual-sum loss; our 5 equation terms plus 12 boundary terms extend exactly this pattern.
- Limits: PDE setting; our BVP adds twelve coupled boundary penalties that this paper does not treat.

#### R23. Physics-informed machine learning survey, 2021

Karniadakis, Kevrekidis, Lu, Perdikaris, Wang and Yang, *Nature Reviews Physics* 3, 422-440, 2021. DOI https://doi.org/10.1038/s42254-021-00314-5. No open PDF found; record verified via Crossref.

- Evidence: Crossref record fetched in this session. An earlier guessed DOI ending in -z returned 404 and was discarded.
- Claim (record): a review of physics-informed learning across forward, inverse and hybrid settings.
- Use here: the standard survey citation for positioning a PINN-for-BVP work in the introduction.
- Limits: review scope only; no transferable training recipe for our problem.

#### R24. Gradient flow pathologies and learning rate annealing, 2021

Wang, Teng and Perdikaris, *SIAM Journal on Scientific Computing* 43(5), A3055-A3081, 2021. DOI https://doi.org/10.1137/20M1318043. [Local PDF](../sources/literature/gradient-pathologies-2021/gradient_flow_pathologies_arxiv_2001.04536.pdf) (arXiv:2001.04536; the arXiv title omits the word Flow).

- Evidence: arXiv record and Crossref record with exact pages fetched in this session.
- Claim (abstract): soft PDE constraints act as domain-specific regularizers; unbalanced back-propagated gradients from numerical stiffness cause training failure, and learning rate annealing of loss terms mitigates it.
- Use here: the theory-backed reason our scaled, weighted loss needs care; the annealing scheme is the standard adaptive baseline for our B4 comparison.
- Limits: PDE benchmarks; reported gains are author-reported; annealing interacts with our L-BFGS stage and needs its own controlled test.

#### R25. Why PINNs fail to train, NTK view, 2022

Wang, Yu and Perdikaris, *Journal of Computational Physics* 449, art. 110768, 2022. DOI https://doi.org/10.1016/j.jcp.2021.110768. [Local PDF](../sources/literature/ntk-failure-2022/pinn_ntk_failure_arxiv_2007.14527.pdf) (arXiv:2007.14527; the arXiv title uses the PINNs short form).

- Evidence: arXiv record and Crossref record fetched in this session.
- Claim (abstract): the neural tangent kernels of different loss terms converge to different limits with imbalanced eigenvalues, which limits PINN training; an NTK-eigenvalue-based weighting scheme is proposed.
- Use here: explains why our 17 loss components converge at different speeds; the principled rule behind any reweighting of equations versus boundaries.
- Limits: NTK computation grows with point count and derivative order; a fourth-order residual makes full NTK weighting costly for us.

#### R26. Characterizing PINN failure modes, 2021

Krishnapriyan, Gholami, Zhe, Kirby and Mahoney, *Advances in Neural Information Processing Systems 34* (NeurIPS 2021), arXiv:2109.01050. [Local PDF](../sources/literature/failure-modes-2021/failure_modes_neurips2021_arxiv_2109.01050.pdf)

- Evidence: arXiv record with journal-ref NeurIPS 2021 fetched in this session.
- Claim (abstract): soft physics constraints make the loss landscape hard to optimize, so PINNs fail on moderately complex problems; curriculum regularization and sequence-to-sequence training reduce error by one to two orders of magnitude.
- Use here: the standard taxonomy for our failure analysis; the stiff-coefficient regime we face near the wall is exactly their failing regime.
- Limits: time-stepping benchmarks; their curriculum ideas do not transfer to a two-ended BVP without adaptation.

#### R27. Self-adaptive weights as soft attention, 2023

McClenny and Braga-Neto, *Journal of Computational Physics* 474, art. 111722, 2023. DOI https://doi.org/10.1016/j.jcp.2022.111722. [Local PDF](../sources/literature/self-adaptive-2023/self_adaptive_soft_attention_arxiv_2009.04544.pdf) (arXiv:2009.04544 v5)

- Evidence: arXiv record with journal-ref and DOI fields fetched in this session.
- Claim (abstract): trainable per-point self-adaptive weights act as soft attention and force the network to fit stubborn stiff regions, with reported accuracy gains over other PINN methods.
- Use here: the fine-grained extension of our fixed 17 weights; its pointwise weights target exactly the near-wall region where our coefficients stiffen.
- Limits: time-dependent PDE examples; pointwise weights on a 1D BVP need bounds to avoid trivial shrinking, which our strategy already requires for any adaptive weights.

### Weighting, optimizer, boundary and sampling choices

#### R28. Balanced residual decay rate weights, 2025

Chen, Howard and Stinis, *Journal of Computational Physics* 542, art. 114226, 2025. DOI https://doi.org/10.1016/j.jcp.2025.114226. [Local PDF](../sources/literature/residual-decay-weights-2025/balanced_residual_decay_weights_arxiv_2407.01613.pdf) (arXiv:2407.01613)

- Evidence: arXiv record with journal-ref fetched in this session; journal DOI verified via Crossref.
- Claim (abstract): pointwise adaptive weights balance residual decay rates across training points and loss terms, with bounded weights and low cost on PINN and DeepONet benchmarks.
- Use here: a concrete published weighting rule for our multi-term loss, implementable per term over the 5 plus 12 components with bounded growth.
- Limits: their examples are PDEs and operator benchmarks; decay-rate balancing on a 1D BVP is our own test to run.

#### R29. Multi-objective loss balancing study, 2025

Bischof and Kraus, *Computer Methods in Applied Mechanics and Engineering* 439, art. 117914, 2025. DOI https://doi.org/10.1016/j.cma.2025.117914. CC BY 4.0 per the Crossref license record, but no script-accessible PDF was found; record verified via Crossref.

- Evidence: Crossref record with license field fetched in this session.
- Claim (record and title): a study of multi-objective loss balancing methods for physics-informed deep learning.
- Use here: a CMAME-level benchmark of fixed versus adaptive balancing schemes; direct precedent for our fixed-weights baseline plus the B4 comparison.
- Limits: abstract not read in this session; read the full text before borrowing any specific scheme.

#### R30. Optimizing the optimizer for PINNs, 2025 preprint

Kiyani, Shukla, Urbán, Darbon and Karniadakis, arXiv:2501.16371, v6 February 2026. No journal record verified yet; cite as a preprint. [Local PDF](../sources/literature/optimizer-optimizing-2025/optimizing_the_optimizer_arxiv_2501.16371.pdf)

- Evidence: arXiv record fetched in this session; OpenAlex shows quasi-Newton and line search keywords with about 123 citations of the preprint.
- Claim (abstract and keywords): quasi-Newton optimizers, including self-scaled BFGS and Broyden variants and L-BFGS with different line searches, are evaluated for PINNs; rescaling updates with historical gradient information brings order-of-magnitude accuracy gains without adaptive weights.
- Use here: directly informs our L-BFGS refinement stage, including line search choice and history size, and whether rescaled updates handle our stiff fourth-order residual better than plain L-BFGS.
- Limits: preprint, not peer-reviewed; PDE and operator benchmarks; settings must be re-tuned for our 1D problem.

#### R31. How accurate and competitive can PINNs be, 2025

Urbán, Stefanou and Pons, *Journal of Computational Physics* 523, art. 113656, 2025 (online December 2024). DOI https://doi.org/10.1016/j.jcp.2024.113656. No open PDF found; record verified via Crossref; abstract read via OpenAlex.

- Evidence: Crossref record and OpenAlex abstract fetched in this session.
- Claim (abstract): better optimization algorithms and small loss adjustments cut errors by orders of magnitude, so that small networks can match finite differences on their tests.
- Use here: supports our plan to test optimizer choice and loss conditioning first, before architecture changes; also a citation for the Adam-then-L-BFGS schedule question.
- Limits: closed access; claims are author-reported; their test problems differ from a nonlinear similarity BVP.

#### R32. Augmented Lagrangian relaxation for boundary conditions, 2023

Son, Cho and Hwang, *Neurocomputing* 548, art. 126424, 2023. DOI https://doi.org/10.1016/j.neucom.2023.126424. [Local PDF](../sources/literature/al-pinn-2023/augmented_lagrangian_PINN_arxiv_2205.01059.pdf) (arXiv:2205.01059)

- Evidence: arXiv record fetched in this session; journal DOI verified via Crossref.
- Claim (abstract): initial and boundary conditions are treated as constraints and enforced with augmented Lagrangian relaxation, giving learnable multipliers that adaptively balance the loss terms, with convergence proofs.
- Use here: the main published alternative to fixed soft penalty weights for our twelve boundary terms; a candidate arm for comparison B4 or B6.
- Limits: their examples are time-dependent PDEs; the AL loop adds inner iterations and its proofs cover their setting, not ours.

#### R33. Hard constraints for inverse design, hPINNs, 2021

Lu, Pestourie, Yao, Wang, Verdugo and Johnson, *SIAM Journal on Scientific Computing* 43(6), B1105-B1132, 2021. DOI https://doi.org/10.1137/21M1397908. [Local PDF](../sources/literature/hpinn-hard-constraints-2021/hpinn_hard_constraints_arxiv_2102.04626.pdf) (arXiv:2102.04626)

- Evidence: arXiv record fetched in this session; journal DOI verified via Crossref.
- Claim (abstract): hPINNs combine penalty terms with an augmented Lagrangian method as hard constraints, tested on Helmholtz and Stokes problems.
- Use here: the standard citation for AL-based hard constraints; frames comparison arm B6 beyond the distance function method in R8.
- Limits: hard-constraint demonstrations are on linear or simple problems; our twelve coupled nonlinear boundary relations are a different setting.

#### R34. Comparing ways to enforce Dirichlet boundary conditions, 2023

Berrone, Canuto, Pintore and Sukumar, *Heliyon* 9(8), e18820, 2023. DOI https://doi.org/10.1016/j.heliyon.2023.e18820. [Local PDF](../sources/literature/bc-comparison-2023/dirichlet_bc_enforcement_PINN_arxiv_2210.14795.pdf) (arXiv:2210.14795)

- Evidence: arXiv record with journal-ref fetched in this session; journal DOI verified via Crossref.
- Claim (paper): four Dirichlet imposition routes are compared for PINNs and VPINNs; modifying the network output to match boundary values beats tuned penalty scaling, with an approximate distance function performing best and Nitsche imposition worst in their tests.
- Use here: the closest published analogue of our planned comparison B6, soft penalties versus exact endpoint construction, done for BVP-style problems with penalty scaling like ours.
- Limits: linear Dirichlet conditions; our boundary set is nonlinear, coupled and includes flux-type conditions that this study does not cover.

#### R35. Trial solution for exact boundary imposition, 2024

Manavi, Fattahi and Becker, *Engineering Applications of Artificial Intelligence* 127, art. 107236, 2024. DOI https://doi.org/10.1016/j.engappai.2024.107236. No open PDF found; record verified via Crossref.

- Evidence: Crossref record fetched in this session.
- Claim (record and title): a Lagaris-style trial solution construction satisfies Dirichlet boundary conditions exactly inside the network, without boundary penalty terms.
- Use here: the modern ansatz reference for comparison arm B6 beyond R8's distance functions.
- Limits: metadata verified only; abstract not read; closed access; Dirichlet-only focus.

#### R36. Boundary condition enforcement comparative study, 2025 preprint

Rowan, Hampleman, Maute and Doostan, arXiv:2512.14941, December 2025. No journal record verified yet; cite as a preprint. [Local PDF](../sources/literature/bc-3d-comparison-2025/bc_enforcement_comparative_arxiv_2512.14941.pdf)

- Evidence: arXiv record fetched in this session.
- Claim (abstract): existing boundary enforcement techniques for PINNs are compared and a framework for arbitrary 3D geometries is proposed, with linear and nonlinear Dirichlet, Neumann and Robin tests.
- Use here: the newest side-by-side BC study we can cite when we justify keeping soft weighted penalties as our baseline.
- Limits: preprint; 3D geometries, not 1D similarity BVPs; conclusions must be re-checked on our conditions.

#### R37. Auxiliary outputs for high-order problems, A-PINN, 2022

Yuan, Ni, Deng and Hao, *Journal of Computational Physics* 462, art. 111260, 2022. DOI https://doi.org/10.1016/j.jcp.2022.111260. No open PDF found and no arXiv version was located in this session; record verified via Crossref.

- Evidence: Crossref record fetched in this session.
- Claim (record): auxiliary output variables added to the network reformulate high-order and integro-differential problems for forward and inverse PINN training.
- Use here: the published journal precedent for our twelve-output first-order variant, the mechanism behind comparison arm B5, complementing the FO-PINN preprint in R7.
- Limits: closed access, metadata verified only; some search-side sources list different author orders; trust the Crossref record order.

#### R38. Failure-informed adaptive sampling, 2023

Gao, Yan and Zhou, *SIAM Journal on Scientific Computing* 45(4), A1971-A1994, 2023. DOI https://doi.org/10.1137/22M1527763. [Local PDF](../sources/literature/fi-pinn-sampling-2023/failure_informed_sampling_arxiv_2210.00279.pdf) (arXiv:2210.00279)

- Evidence: arXiv record fetched in this session; journal DOI verified via Crossref.
- Claim (abstract): FI-PINNs define an effective failure probability from the PDE residual, use it as a posterior error indicator, and enrich collocation points in the failure region, with error bounds under their assumptions.
- Use here: the main published successor to RAR for our enrichment comparison arm B3; positions our custom top-error scoring against a named, checked method.
- Limits: PDE examples; their error bounds depend on assumptions we have not verified for a fourth-order nonlinear system.

#### R39. Hard-constrained Falkner-Skan with regional residual balancing, 2026 preprint

Endalew and Zhang, arXiv:2609.23493, September 2026. No journal record verified yet; cite as a preprint. [Local PDF](../sources/literature/falkner-skan-hard-bc-2026/falkner_skan_hard_constrained_arxiv_2609.23493.pdf)

- Evidence: arXiv record fetched in this session (v1, 20 September 2026).
- Claim (abstract): the generalized Falkner-Skan BVP is solved on a truncated finite domain with an exact boundary-admissible trial representation that removes boundary penalty terms, plus region-wise residual balancing over wall, middle and tail bands, and multi-stage Adam and L-BFGS training.
- Use here: the closest structural BVP precedent to our setup: a semi-infinite boundary-layer ODE, a finite eta=L far-field endpoint, hard wall conditions, regional weighting and our exact optimizer chain. It touches comparison arms B4, B6 and our eta=L choice at once. Same author group as R3.
- Limits: preprint; a third-order single-field problem, not five coupled fields; search-side notes say the ablations are single-seed; verify details in the full text before reuse.

#### R40. Convergence and sketching for NTK weights, 2025 preprint

Hirsch and Pichi, arXiv:2511.15530, November 2025. No journal record verified yet; cite as a preprint. [Local PDF](../sources/literature/ntk-sketching-2025/ntk_weights_sketching_arxiv_2511.15530.pdf)

- Evidence: arXiv record fetched in this session.
- Claim (abstract): convergence of gradient descent with adaptive NTK-based loss weights is proven under suitable conditions, and a randomized sketching method computes such weights cheaply.
- Use here: theory plus a feasible estimator if our fixed weights underperform; full NTK on a fourth-order residual would be too costly without sketching.
- Limits: preprint; assumptions must be checked against our nonsymmetric coupled system.

#### R41. A remedy for failure with large coefficients, 2025

Farhani, Hosseini Dashtbayaz, Kazachek and Wang, *Neural Networks* 183, art. 106963, 2025. DOI https://doi.org/10.1016/j.neunet.2024.106963. No open PDF found; record verified via Crossref; an SSRN preprint version exists.

- Evidence: Crossref record fetched in this session. The claim below was reported by the search agent from a search result, not read from the article, and is marked accordingly.
- Claim (reported, not independently read): addresses PINN convergence failures when PDE coefficients grow large or the equation becomes strongly nonlinear, and proposes a remedy.
- Use here: the large-coefficient regime matches our near-wall inverse-power terms; a candidate reference for the stiff-wall arm after we read the full text.
- Limits: second-hand claim at this point; closed access; do not cite its mechanism until read.

#### R42. PINN failure modes as overfitting, 2026 preprint

Andersen and Matsubara, arXiv:2605.30910, May 2026. No journal record verified yet; cite as a preprint. [Local PDF](../sources/literature/overfitting-failure-2026/pinn_failure_overfitting_arxiv_2605.30910.pdf)

- Evidence: arXiv record fetched in this session.
- Claim (abstract): PINN failure modes come from overfitting, the residual loss is minimized at collocation points but not elsewhere; regularization with extended double backpropagation reaches state-of-the-art results on four standard failure-mode equations with up to 23 times fewer collocation points.
- Use here: a counterpoint to the pure optimization view; strengthens our design of separate monitoring and fresh final evaluation sets, because endpoint or training-point agreement is not interior accuracy.
- Limits: preprint; their benchmarks are standard failure-mode equations, not BVPs; double backpropagation triples differentiation cost on our fourth-order graph.

### Benchmarks, honest comparisons and surveys

#### R43. Can PINNs beat the finite element method, 2024

Grossmann, Komorowska, Latz and Schoenlieb, *IMA Journal of Applied Mathematics* 89(1), 143-174, 2024. DOI https://doi.org/10.1093/imamat/hxae011 (verified via Crossref in this session); preprint arXiv:2302.04107; the published CC BY version is also on PMC (PMC11197852). [Local PDF](../sources/literature/pinn-vs-fem-2024/pinn_vs_fem_arxiv_2302.04107.pdf)

- Evidence: arXiv record fetched in this session; the search agent verified the published version and venue on the PMC record.
- Claim (abstract): a systematic cost and accuracy comparison on Poisson, Allen-Cahn and semilinear Schrodinger problems finds PINNs have not outperformed the finite element method; a simple loss scaling accelerates L-BFGS.
- Use here: the canonical honest head-to-head citation. It legitimizes our classical-reference-first plan and our explicit no-superiority-claim rule, and its scaling observation supports comparison arm B4.
- Limits: PDE problems, not similarity ODEs; conclusions are author-reported.

#### R44. Honest negative results in fluid PINNs, 2022

Chuang and Barba, *Proceedings of the Python in Science Conference* (SciPy 2022), 28-36, 2022. DOI https://doi.org/10.25080/majora-212e5952-005. [Local PDF](../sources/literature/pinn-pitfalls-report-2022/pinn_pitfalls_frustration_scipy2022_10.25080_majora-212e5952-005.pdf)

- Evidence: proceedings record and PDF fetched in this session.
- Claim (abstract): a data-free PINN on the Taylor-Green vortex took about 32 hours to match a 16x16 finite difference run that took under 20 seconds, and a cylinder case failed to produce shedding.
- Use here: the clearest published example of reporting failure and cost honestly against classical solvers; supports saving and publishing our failed runs as results.
- Limits: fluid PDEs; conference venue; 2022 software environment.

#### R45. PINNacle benchmark, 2024

Hao, Yao, Su et al., "PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs", *Advances in Neural Information Processing Systems 37*, Datasets and Benchmarks track, 2024. arXiv:2306.08827. [Local PDF](../sources/literature/pinnacle-benchmark-2024/pinnacle_benchmark_neurips2024_arxiv_2306.08827.pdf)

- Evidence: arXiv record fetched in this session; the search agent verified the NeurIPS 2024 proceedings entry by fetching it.
- Claim (abstract): a benchmark suite of over 20 PDE problems and about 10 PINN variants, with analysis of domain decomposition and loss reweighting across failure modes.
- Use here: defines the evaluation and reporting practice our evaluation design can cite, and motivates testing loss weighting on our own problem instead of trusting aggregate leaderboards.
- Limits: PDE suite; no similarity-ODE BVP class; venue is NeurIPS 2024, not 2023 as some summaries claim.

#### R46. Numerical analysis of PINNs, Acta Numerica survey, 2024

De Ryck and Mishra, *Acta Numerica* 33, 633-713, 2024. DOI https://doi.org/10.1017/s0962492923000089. [Local PDF](../sources/literature/acta-numerica-survey-2024/numerical_analysis_pinn_survey_10.1017_s0962492923000089.pdf) (publisher CC BY PDF)

- Evidence: Cambridge record and PDF fetched in this session.
- Claim (record): a rigorous survey of error and convergence analysis for PINNs and related physics-informed models.
- Use here: the theoretical anchor for our evaluation design, in particular how residual metrics relate to true error, which is the reason we report per-field errors against a converged reference rather than loss values.
- Limits: mathematical survey; cite selectively; it does not prescribe a training recipe for our system.

#### R47. Scientific machine learning through PINNs survey, 2022

Cuomo, Schiano Di Cola, Giampaolo, Rozza, Raissi and Piccialli, *Journal of Scientific Computing* 92, art. 88, 2022. DOI https://doi.org/10.1007/s10915-022-01939-z. [Local PDF](../sources/literature/sciml-survey-2022/sciml_pinn_survey_arxiv_2201.05624.pdf) (arXiv:2201.05624)

- Evidence: Crossref record fetched in this session; arXiv PDF downloaded.
- Claim (record): a widely cited review of PINN variants, loss choices, architectures and training strategies, noting that theoretical issues remain unresolved.
- Use here: framing survey for the introduction's coverage of loss structure and training choices.
- Limits: review; 2022, so it predates the recent weighting and BC studies above.

#### R48. Critical review of PINN claims, 2026

Naser, *Computers & Industrial Engineering* 212, art. 111704, 2026. DOI https://doi.org/10.1016/j.cie.2025.111704. No open PDF found; record verified via Crossref.

- Evidence: Crossref record fetched in this session. The publisher page was not readable in this session.
- Claim (record and title): a critical review arguing PINNs carry structural limitations in engineering systems and that claimed advantages need case-specific evidence.
- Use here: a citable reference for the honest framing that a PINN result must be demonstrated per problem, which is exactly our research question.
- Limits: single-author review outside computational mathematics; abstract not read in this session; treat claims as the author's position.

### What this pass did not find, and what it corrected

- No verified paper solves an exact five-field (4,2,2,2,2) similarity system with a PINN. The closest verified physical matches carry a microorganism field (R12, R14), a microrotation field (R17, R18) or a curved wall (R15, R18), each with fewer fields than ours. Our reading-list caution from the first pass stands: we cannot claim "first five-field PINN" without a dedicated novelty search, and this pass still does not prove that none exists.
- No verified PINN paper claims a fourth-order squeezing-flow momentum equation besides R2. The squeezing paper in this pass (R21) uses a PINN only as a checker.
- RAR attribution corrected: the frequently cited "Shin, Darbon and Karniadakis RAR paper" does not exist as a sampling paper. arXiv:2004.01806 is their convergence paper for linear second-order PDEs. The RAR and RAR-D definitions we may follow remain in R6 (Wu et al., CMAME 2023).
- Title corrected: the Grossmann et al. comparison (R43) is against the finite element method, not the finite difference method.
- Venue corrected: PINNacle (R45) appeared in the NeurIPS 2024 Datasets and Benchmarks track, not 2023.
- Access limits recorded: ScienceDirect, AIP, Nature and PMC PDF endpoints returned bot walls or CAPTCHAs to scripted downloads. Those entries carry metadata-only status with verified Crossref or OpenAlex records. No paywalled content was circumvented.
- All local PDFs are arXiv versions or publisher open-access copies, recorded with SHA-256 hashes in sources/manifest.json. The R22 local PDF is the arXiv Part I companion whose title differs from the 2019 journal title.
- Citation counts quoted above are search-side estimates, not verified publisher counts. They are context only, not claims.
