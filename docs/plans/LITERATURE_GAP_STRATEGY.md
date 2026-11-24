# Literature against our method: gaps and the plan to close them

Date: 2026-11-22. Branch: `speedup/taylor-jets-and-reference-engine`. Status: analysis and plan, updated 2026-11-23. Eqs. 14 to 18, B1 to B12 and the locked values are fixed as the supervisor gave them. Proposals that would change them were removed on 2026-11-23 (the chi model options in R2, the parameter continuation R4, the re-derivation R7 and the curriculum P6). Section 10 records what was then built and run.

Related: [literature-synthesis.md](../literature/literature-synthesis.md) (what each paper teaches), [PINN_REVIEW_2026-11-21.md](../reviews/PINN_REVIEW_2026-11-21.md) (findings H1, H2), [RUNTIME_STRATEGY.md](../pinn/RUNTIME_STRATEGY.md) (speed), [SPEC_reference_solutions.md](../reference-solver/SPEC_reference_solutions.md) (reference checks), [PINN_PLAN.md](../pinn/PINN_PLAN.md) (arms and gates).

Tags used below: **measured** (run here), **author-reported** (a paper's own number, not reproduced), **inference** (our reasoning, not tested), **fixed problem**: the equations, the boundary conditions and the locked values never change. Every fix below is a way to impose or solve them, and every result is checked on the original R14 to R18 and B1 to B12.

## 1. Short answer

- **The reference solver is already at or above the level of every application paper we hold.** Two independent methods plus the C++ box scheme agree to about 1e-10, and we use asymptotic far-field conditions where the papers simply cut the domain at eta = 7 or 12 without an L study.
- **The reference has three real gaps.** The far-field condition keeps only the leading linear terms, and the terms it drops are still 2.5 % of the kept ones at eta = 20 (**measured**). This is the likely cause of the phi(0) drift of 3.9e-4 per doubling of L. The benchmark checks do not cover the micropolar, dipole, radiation and nanoparticle terms. And chi has no decaying far-field solution: a property of Eq. 18 as given, which we report next to every chi number.
- **The PINN gap is mainly in training, not in the equations or the boundary layer.** The literature from 2024 to 2026 agrees on one point: for PINNs the optimization error dominates, and the cure is better conditioning (scaling, hard constraints, lower derivative order) plus second-order optimizers (full BFGS, Gauss-Newton or Levenberg-Marquardt). Our training is still Adam plus limited-memory L-BFGS, with loss weights that blow up and a starting point that sits near a "cold wall" state.
- **The plan is a one-change-at-a-time ladder on B0** (section 6). No step changes an equation, a boundary condition or a parameter value. A later step, an exact reduced-order rewrite of Eq. 14, is only a coding choice: it has the same solutions, and results are still checked on the original Eq. 14.

## 2. What was read

- All 25 papers in `sources/literature/` were read in full text in this session, with notes on method, algorithm, settings, results and relevance for each one. The method and algorithm sections were read closely. For the two long surveys (Acta Numerica 2024 and the SciML survey 2022) every section was read; the application catalogues were read quickly.
- Because the newest local papers are from mid-2026 and some key methods are newer than our reading list, seven online checks were made for recent published work (section 9.2). For these only abstracts or listings were checked, not full text, so they are cited as pointers.
- Every number from a paper is **author-reported**. None is our proof.

## 3. Reference solutions: literature against ours

### 3.1 Comparison

| Item | Application papers we hold | Ours today | Verdict |
|---|---|---|---|
| Solver | bvp4c-type collocation (squeezing, cylinder, Falkner-Skan); FEM table reused (micropolar) | `solve_bvp` at tol 1e-8, plus RK multiple shooting (DOP853, 40 segments), plus C++ Keller box with Richardson | Ours is stronger: three independent discretizations |
| First-order rewrite | 8 states, standard (squeezing) | 12 states, `M*H = -r0` solved, never inverted, diagonal watched | Same practice, better documented |
| Far field | Cut at eta = 7 (micropolar), 12 (Falkner-Skan), unstated (cylinder); no L study in any | Asymptotic conditions for theta and phi at L = 20 (Lentini and Keller 1980), L study to 160 | Ours is stronger, but the condition is leading order only (gap R1) |
| Verification | One published table (micropolar, Falkner-Skan), CAS check of the similarity reduction (squeezing) | SymPy transcription test, manufactured solution (order 3.4 to 3.8), tolerance ladder, RK against collocation, slope check, three published or exact limits | Ours is stronger, but the limits miss most physical terms (gap R3) |
| Error bound | A posteriori certificate from a Green-function fixed point, factor 1/(1 - rho) = 1.37 (squeezing) | Agreement between methods and tolerances; no formal bound | Gap R5 |
| Branches | Not discussed | Guess scan found branches A and B; B excluded because theta has no decaying mode | Adequate for the locked case |

**Measured on reference branch A (2026-11-22):** the terms that the far-field condition drops from Eq. 16, as a share of the linear terms it keeps.

| eta | Nr 3cA^2 theta'^2 | Nt theta'^2 | Nb theta' phi' | dipole source | theta | phi |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 3.4 % | 1.0 % | 0.06 % | 0.008 % | 0.095 | 0.190 |
| 10 | 2.9 % | 0.84 % | 0.09 % | 0.011 % | 0.080 | 0.189 |
| 20 | 2.5 % | 0.71 % | 0.19 % | 0.005 % | 0.068 | 0.183 |

The share falls only slowly with eta. This fits the theory: theta' is about -m theta/q, so a quadratic term over a linear term is about m theta, and theta itself decays only like q^-0.24. So a larger L alone cannot make these terms small at any affordable cost. **[inference]**

### 3.2 Reference gaps and fixes

| ID | Gap | Evidence | Fix | Test that decides | Effort | Approval |
|---|---|---|---|---|---|---|
| R1 | Far-field condition is the leading linear term only | Table above; phi(0) moves 3.9e-4 per doubling of L, the other wall values 4e-5 to 7e-5 (reference/study.md, V5) | (a) Derive a corrected asymptotic condition that keeps the quadratic terms Nt theta'^2, Nb theta' phi', 3 Nr c A^2 theta'^2 and the local A(theta), with SymPy, as a next-order expansion of the decaying mode. (b) Independently, extrapolate the wall values in L: fit value(L) = v_inf + c L^-p over L = 20 to 160 and report v_inf with the fit spread | phi(0) change per doubling of L below the 1e-4 bar of V5; (a) and (b) agree within that bar | M | No: same equations and BCs, a better numerical stand-in for eta -> infinity. The PINN keeps the L = 20 problem it is compared with until the supervisor agrees to a change |
| R2 | chi has no decaying solution | SPEC 8.2: Eq. 18 as printed has no chi convection term, so chi'(0) runs from -2.24 to -1.08 over L = 10 to 160 | None needed: Eq. 18 and B12 stay as given. B12 is imposed at eta = L, the same end as the other far conditions, and every chi number carries the note that chi'(0) depends on L | n/a | S | No |
| R3 | Benchmarks miss micropolar, dipole, radiation and nanoparticle terms | SPEC 8.5 "what this evidence does not cover" | Add published limits one at a time, each only after a written check that a parameter limit of Eqs. 14 to 19 gives exactly the paper's equations: magnetic dipole on a flat stretching sheet (Andersson and Valnes, Acta Mechanica 128, 1998); micropolar flow on a curved stretching sheet (Naveed, Abbas and Sajid, J. Appl. Fluid Mech., 2016); zero nanoparticle flux, the passive concentration condition behind our B5 (Kuznetsov and Nield 2013). If no exact limit exists, record that and do not force a comparison | Code reproduces the published table at the paper's own domain to its printed digits | M per paper | No |
| R5 | No formal error bar on wall values | Agreement only | Grid convergence index from the five box-scheme levels (ASME V&V 20 style; the convergence-study skill) | Observed order near 2 on all wall values, asymptotic range check passes | S | No |
| R6 | No reference-free error estimate for the PINN | Acta Numerica section 5: error <= C * residual when the linearized problem has a bounded inverse; squeezing paper's certificate; FI-PINN bound | Use the linearized BVP as an error estimator: start the box-scheme Newton solver from the PINN state and report the first Newton correction (this is the error estimate J^-1 R). Also report a bound from the smallest singular value of the discrete Jacobian | Estimate matches the true PINN error on branch A within a factor of 2 on the pilots | M | No: it evaluates; it never feeds training |

## 4. PINN: literature against ours

### 4.1 Comparison

| Part | What the literature does | Ours today | Gap |
|---|---|---|---|
| Network | tanh MLP is standard; modified MLP with U, V gates (Wang 2021, BRDR 2025); input injection and SiLU (micropolar); 1 to 2 wide layers (Falkner-Skan 100x2) | tanh MLP 40 x 4, float64, linear or log input map | Small; architecture comes after training fixes |
| Derivatives | Forward mode for one input (micropolar, JAX) | Forward Taylor jets (`pinn/jets.py`), same values as nested autograd to 1e-14 | None; ours matches the best practice |
| BCs | Hard lifting plus mask with zero-init last layer (Falkner-Skan); AL best among soft methods (AL-PINN, Rowan 2025); hard BCs lower the condition number (Acta 7.5.2) | All 12 exact by our TFC layer; soft and linear modes as controls | Ours goes further than any paper; the start state is not chosen (gap P5) |
| Loss weights | Bounded schemes: BRDR (pointwise, mean 1), ARRB (regions, EMA, safeguard blend); theory needs bounded weights (Hirsch and Pichi 2025) | Gradient-norm weights that reached 3e9 to 7e10 in pilots (H2) | P2 |
| Loss as integral | Region-length weighted physical loss (Falkner-Skan); quadrature weights (Acta 2.4; Rowan; Modulus) | Plain mean over wall-stretched points | P4 |
| Scaling | Nondimensional, O(1) coefficients first (beams: dimensional version stuck at loss 1e14) | Each residual divided by its highest-derivative coefficient; R14 still spans 1e3 at the wall to 1e-6 far out | P3 |
| Sampling | Fresh stratified points every Adam step (Falkner-Skan, Grossmann); one-shot RAR after Adam (micropolar); failure-probability enrichment and stop rule (FI-PINN) | 2000 fixed wall-stretched points, 1000 monitor points | P7 |
| Optimizer | Full BFGS / self-scaled Broyden in float64, 1e-8 errors where L-BFGS gave 1e-3 (Kiyani 2025); Gauss-Newton natural gradient near single precision (Jnini et al. 2024); NysNewton-CG after Adam plus L-BFGS (Rathore et al. 2024); SOAP as an approximate Newton preconditioner (Wang et al. 2025); trust-region Levenberg-Marquardt polish (squeezing) | Adam 20 000 then one L-BFGS stage, history 50, rollback if the monitor worsens | P1 |
| Curriculum | Train at an easier coefficient, warm-start the hard one (Krishnapriyan 2021; Kiyani's Allen-Cahn kappa 1 then 5) | None: the locked values are fixed | Not adopted (section 7) |
| Accuracy ceiling | Multi-stage networks fit the remaining residual with a new network, near 1e-16 (Wang and Lai, JCP 504, 2024) | One network | P11, later |
| Overfitting | Train loss against dense test loss is the failure sign; gradient regularization fixes it (Andersen and Matsubara 2026) | Separate monitor set already | P8, small |
| Stopping | Best checkpoint on a fixed monitor, plateau schedule, staged L-BFGS on growing grids with rollback (Falkner-Skan) | Fixed budgets; one L-BFGS stage | P9 |
| Formulation order | Formulation matters as much as the optimizer (lid-driven cavity, Wang 2021); conditioning grows with operator order (Acta 7.3) | f'''' in Eq. 14; an exact rewrite to f''' plus a pressure-like unknown is proven but unused (RUNTIME_STRATEGY 2.3) | P10 |
| Evaluation | Seeds and failure counts (PINNacle), time-to-accuracy (Grossmann; Chuang and Barba), max error as well as L2 | Planned in PINN_PLAN sections 4 and 6 | P12, small |

### 4.2 Why our pilots fail, read through the literature

The pilots sit near theta(0) = 0 and theta'(0) = 0, so B4 is missed by about 2.25, and f''(0) has the wrong sign (review H1). Four mechanisms from the papers fit this, all **inference** until tested:

1. **Conditioning.** For a linearized model, gradient descent converges at a rate set by the condition number of L*L composed with the tangent kernel (Acta 7.2 to 7.3). Our f'''' makes L*L eighth order, and the wall coefficients (250 for the dipole term, 125 for 1/q^3) widen the spread further. Adam cannot fix this; Newton-type steps can, because for linear models the Hessian is exactly that matrix (Acta 7.5.3).
2. **Imbalance across space, not across loss terms.** With all 12 BCs exact, the imbalance left is between the wall region and the outer region of the same residual. That is what ARRB and BRDR target; our gradient-norm weights cannot see it.
3. **Starting point and branch.** A random start is a cold, flat profile. Falkner-Skan starts training from a lifting that already satisfies the BCs; our reference reaches branch A from the smooth guess with theta(0) = 0.3.
4. **Weight blow-up.** Weights of 1e10 make L-BFGS work on a badly scaled problem (H2), and the convergence theory for adaptive weights needs bounded weights (Hirsch and Pichi 2025).

### 4.3 PINN gaps and fixes

| ID | Gap | Fix, with source | Our adaptation for this problem | Effort | Approval |
|---|---|---|---|---|---|
| P1 | Weak optimizer for an ill-conditioned least-squares loss | Full BFGS or self-scaled Broyden (Kiyani 2025); Levenberg-Marquardt / Gauss-Newton (squeezing; Jnini et al. 2024) | Our network has about 5 000 weights, so a dense 5 000 x 5 000 inverse Hessian is 200 MB in float64, and a residual Jacobian for 2 000 points x 5 equations is about 400 MB. Both fit this laptop and Colab. In exact mode the loss is a pure sum of squares, which suits Levenberg-Marquardt. Add `refine = lbfgs / bfgs / lm`, keep L-BFGS as the control, use full batch and fixed points for the refine stage | M | No |
| P2 | Loss weights unbounded | BRDR: weights proportional to the inverse residual decay rate, normalized to mean 1 (Chen, Howard and Stinis 2025). ARRB: EMA of inverse proxy-gradient norms per region, blended with uniform weights (gamma 0.8) and mixed 50:50 with the physical loss (Endalew and Zhang 2026) | Weight the five equations and three regions (wall [0, 0.5], middle, outer), not the boundary terms, which are exact. Freeze before the refine stage. Meanwhile cap the current weights at 1e3 times the equation weight (review H2) | S to M | No |
| P3 | Residual sizes differ by about 1e9 across eta | Nondimensionalize to O(1) coefficients (beams 2024); preconditioning (Acta 7.4) | Keep the zero set, change only the fixed scale: multiply each residual by a declared function of eta built from coefficient sizes, for example (eta + b)^4 for R14, whose wall terms scale like 1/(eta + b)^4. Never build scales from current residuals or from the reference (strategy.md section 5). Record the scale function in the config | S | No |
| P4 | Objective is not an integral | Region-length or quadrature weights (Falkner-Skan; Acta 2.4) | Trapezoid weights in eta for the stretched points, so the loss approximates the integral of R^2 over [0, L]. Any deliberate wall emphasis then sits in P2 or P3, where it is visible | S | No |
| P5 | Random start near a cold wall | Lifting plus zero-init last layer (Falkner-Skan) | Add the reference smooth guess (rate 1, theta(0) = 0.3; it satisfies B1, B2, B3, B5, B6 and reaches branch A in the reference scan) as a fixed base profile, with the network output starting at zero, then the TFC layer on top. This is not reference data: it is the same analytic starting guess both reference solvers use | S | No, but state it in every result |
| P7 | Fixed points | Per-step stratified resampling (Falkner-Skan); one-shot RAR (micropolar); FI-PINN stop rule | Resample stratified wall-stretched points during Adam; after Adam add the top residual points from a dense candidate grid of 20 000 (cheap in 1D); fix the set for the refine stage. Use the failure share P(|R| > eps) on the monitor set as a stop rule | S | No |
| P8 | Overfitting not tested formally | Train against dense-test divergence; gradient regularization (Andersen and Matsubara 2026) | Log monitor loss against training loss at every log step; if they diverge, add the term mean((dR/deta)^2) with a small fixed weight (needs jets to order 5) | S | No |
| P9 | Fixed budgets, one refine stage | Best checkpoint, plateau schedule, staged refine on 2 000 then 4 000 points with rollback (Falkner-Skan) | Restore best monitor checkpoint after Adam; two refine stages with rollback; stop on a monitor target | S | No |
| P10 | Fourth-order f | Lower order improves conditioning (Wang 2021; Acta 7.3) | Use the proven rewrite of Eq. 14 into Phi - P = 0 and P' - 2 kappa f'^2/q = 0 (RUNTIME_STRATEGY 2.3; SymPy test). Adds a sixth network output; P has no BC | M | No: an exact rewrite with the same solutions (SymPy proof in tests). Results are still reported with the original R14 |
| P11 | Accuracy ceiling of one network | Multi-stage networks (Wang and Lai 2024) | Only after B0 reaches the branch-A wall values: train a second small network on the correction equations with the first one frozen | M | No, but report it as its own result (B0 plus correction), not as plain B0 |
| P12 | Reporting | PINNacle, Grossmann, Chuang and Barba | Add the R6 error estimate, max errors, failure counts and time-to-accuracy curves to `evaluate.py` | S | No |
| P13 | Architecture | Modified MLP (Wang 2021); asymptotic input features (hPINN extra features) | Only if P1 to P5 leave a gap: try the modified MLP, then add log(q) and exp(-eta) as extra inputs, which match the known tail forms | S | No |

## 5. What we already do well

These match or go beyond the papers and stay as they are:

- Equations checked term by term against the source images, with an independent SymPy transcription.
- Two independent reference methods plus a third engine, and the asymptotic far field with an L study. No application paper we hold checks L at all.
- All 12 BCs exact for any weights, including the three nonlinear ones (our own step; no paper does this).
- Forward-mode derivatives, float64, separate monitor points, rollback of a worse L-BFGS stage, and a test that training never reads the reference.
- Honest records of failed checks (V5 far field, branch B, pilots).

## 6. Strategy: the order of work

Rules for every step: one change at a time; B0 only; seeds 0, 1, 2; a change is kept only if the monitor equation residual improves in all three seeds (the G6 style rule); the reference is read only after the decision, by `evaluate.py`. Each step is a dated row in IMPLEMENTATION_NOTES section 7.

### Track A: PINN (B0 diagnostic ladder)

| Step | Change on top of the previous kept steps | Gaps | Pass sign (training side) | If it fails |
|---|---|---|---|---|
| A0 | Code only: capped weights, trapezoid weights option, smooth-guess base profile option, best-checkpoint restore, staged refine, `refine = bfgs / lm` options, failure-share stop rule. Unit tests for each | P2, P4, P5, P9, P1 | `pytest` passes; a dry run shows identical results when every option is off | Fix before any run |
| A1 | Current B0, exact mode, log input map, 3 seeds, as the baseline | review H1 | Recorded, whatever it shows | Baseline stays |
| A2 | + smooth-guess base profile | P5 | B4 is exact by construction, so watch R16 near the wall and theta(0) staying away from 0 | Drop P5, keep going |
| A3 | + residual scale functions and trapezoid weights | P3, P4 | Monitor residual of R14 and R16 falls in all seeds | Try P3 and P4 apart |
| A4 | + bounded region weights (ARRB-style, then BRDR if ARRB helps) | P2 | Wall-region residual falls without the outer region rising | Keep fixed weights |
| A5 | + Levenberg-Marquardt or full BFGS refine instead of L-BFGS | P1 | Monitor loss several orders below the L-BFGS control | Keep L-BFGS; record it |
| A6 | + resampling during Adam and one-shot RAR before the refine | P7 | Monitor and training loss stay together (P8 check) | Back to fixed points |
| A7 | If B0 still misses branch A: P10, the exact reduced-order rewrite of Eq. 14 | P10 | As A3 | Report the failure with all logs |
| A8 | Freeze settings; then arms B1 to B3 with the same settings; then P11 for the best arm, reported as its own result | P11 | PINN_PLAN section 7 questions | Report |

Cost check (estimate from RUNTIME_STRATEGY: about 0.15 s per jets step): one A step with 3 seeds and 20 000 Adam steps is about 2.5 hours on this laptop, plus the refine stage. A5 with Levenberg-Marquardt costs about one 400 MB Jacobian per iteration; measure it on 200 iterations before a full run.

### Track B: reference (runs alongside Track A)

| Step | Work | Gap |
|---|---|---|
| B-1 | Grid convergence index from the five box-scheme levels | R5 |
| B-2 | L extrapolation of wall values from the existing L study | R1 (b) |
| B-3 | Corrected nonlinear far-field condition in SymPy, then the L study again | R1 (a) |
| B-4 | Error estimator from one box-scheme Newton step at the PINN state; validate on the A1 pilots against the true error | R6 |
| B-5 | Published-limit checks: dipole, then micropolar curved sheet, then zero-flux, each after the written limit check | R3 |

B-4 should be ready before A2, so each later step can report the estimate next to the true error. If the estimate is reliable, it becomes the reference-free number the paper can claim.

## 7. What we do not adopt, and why

| Idea | Source | Why not |
|---|---|---|
| Parametric input (eta plus parameters) | micropolar 2026 | Supervisor decision: input eta only, no sweep |
| Particle swarm and neural network algorithm search | squeezing 2026 | Built for 1-layer, 10-neuron nets; our net has 5 000 weights. We keep only its Levenberg-Marquardt polish |
| Supervised surrogate on reference data | squeezing 2026 | AGENTS.md: the reference never feeds training |
| Time marching, seq2seq, causal weights | failure-modes 2021, Kiyani KS runs | Both ends of [0, L] carry conditions; there is no direction to march (PINN_PLAN section 2) |
| Domain decomposition networks as the main fix | PINNacle; Acta 7.5.4 | Already arm B1; not a fix for B0 |
| Variational or weak-form loss | bc-comparison 2023, Rowan 2025 | Needs a test space; Rowan shows a coarse test space can spoil a good network. Revisit only if A3 fails |
| Augmented Lagrangian for BCs | AL-PINN 2023, Rowan 2025 | All 12 BCs are exact; nothing is left for it. Kept as the fallback if exact mode is ever dropped |
| SA-PINN pointwise ascent weights | McClenny 2023 | Weights grow without a bound; BRDR gives the same pointwise idea with mean 1 |
| Curriculum through other kappa, b, Da values | Krishnapriyan 2021; Kiyani 2025 | It trains on parameter values other than the locked case. The problem is fixed |
| Asking for a different chi term in Eq. 18 | our SPEC section 8.2 | The equations are fixed as given. We report chi's behaviour instead |
| Re-deriving Eqs. 14 to 19 from the PDEs | squeezing 2026 | Its only use would be to question the given equations |

## 8. Questions for the supervisor

Only two questions remain, and neither changes the problem:

1. **R1:** the corrected far condition imposes the same B10 and B11 more closely. If it moves the wall values, should the reported reference values be the L = 20 ones the PINN is compared with now, or the L-extrapolated ones?
2. **P11:** if a correction network is ever added, should it be reported next to B0 or in place of it?

## 9. Sources

### 9.1 Local papers (full text read on 2026-11-22)

All under `sources/literature/<folder>/`: micropolar-2026, squeezing-2026, cylinder-nano-2026, falkner-skan-hard-bc-2026, residual-decay-weights-2025, self-adaptive-2023, gradient-pathologies-2021, ntk-failure-2022, ntk-sketching-2025, optimizer-optimizing-2025, fi-pinn-sampling-2023, failure-modes-2021, overfitting-failure-2026, al-pinn-2023, hpinn-hard-constraints-2021, bc-comparison-2023, bc-3d-comparison-2025, pinnacle-benchmark-2024, pinn-vs-fem-2024, pinn-pitfalls-report-2022, pinn-foundations-2019, beams-2024, lap-joint-2021, acta-numerica-survey-2024, sciml-survey-2022. Citations with page numbers are in each folder's `analysis.md`.

### 9.2 Recent work checked online (abstract or listing only, 2026-11-22)

- Jnini, Vella, Zeinhofer, "Gauss-Newton natural gradient descent for physics-informed computational fluid dynamics", [arXiv 2402.10680](https://arxiv.org/abs/2402.10680) (2024). Reports errors near single precision.
- Rathore, Lei, Frangella, Lu, Udell, "Challenges in training PINNs: a loss landscape perspective", ICML 2024, [PMLR 235](https://proceedings.mlr.press/v235/rathore24a.html). Ill-conditioning from the differential operator; quasi-Newton improves conditioning by 1000x or more; NysNewton-CG.
- Wang and Lai, "Multi-stage neural networks: function approximator of machine precision", J. Comput. Phys. 504 (2024) 112865, [arXiv 2307.08934](https://arxiv.org/pdf/2307.08934).
- Wang, Bhartari, Li, Perdikaris, "Gradient alignment in physics-informed neural networks: a second-order optimization perspective", NeurIPS 2025, [arXiv 2502.00604](https://arxiv.org/abs/2502.00604). SOAP as an approximate Newton preconditioner.
- Andersson and Valnes, "Flow of a heated ferrofluid over a stretching sheet in the presence of a magnetic dipole", Acta Mechanica 128 (1998) 39 to 47, [record](https://portal.mardi4nfdi.de/wiki/Publication:1389801). Benchmark candidate for R3.
- Naveed, Abbas, Sajid, "MHD flow of micropolar fluid due to a curved stretching sheet with thermal radiation", J. Appl. Fluid Mech., [record](https://civilica.com/doc/1384919/). Benchmark candidate for R3; table values not yet checked.
- Kuznetsov and Nield (2013), the revised zero-flux nanofluid model, Int. J. Heat Mass Transfer 65, seen through [later papers](https://arxiv.org/pdf/1902.09346). Benchmark candidate for R3.
- Weidman, Kubitschek, Davis, "The effect of transpiration on self-similar boundary layer flow over moving surfaces", Int. J. Eng. Sci. 44 (2006) 730 to 737, seen through [reference lists](https://www.ukm.my/jsm/english_journals/vol43num8_2014/vol43num7_2014p1239-1247.html). Stability of dual solutions, for R4.

Not re-opened in this session and cited from memory in earlier files: Lentini and Keller (1980), Griewank and Walther (2008), Müller and Zeinhofer (2023).

## 10. What was built and run, 2026-11-23

Full numbers and run IDs: [IMPLEMENTATION_NOTES.md](../pinn/IMPLEMENTATION_NOTES.md) section 7.1. One seed only, so these are signals, not claims.

| Gap | Built | Measured |
|---|---|---|
| P5 start | `start = smooth`, `zero_last` (`pinn/start.py`) | Alone it did not move the result (E2 against E1) |
| P3 scale | `residual_scale = log`: R_k times (eta + b)^order_k | The decisive change: with it the far field decays (m > 0) and the state moves to branch A; without it m < 0 |
| P4 integral weights | Not built | The wall-stretched points are close to uniform in log(eta + c), c = 0.14, so the plain mean already weights the loss like an integral in the log variable that P3 uses. Trapezoid weights in eta would undo that |
| P2 bounded weights | `weight_cap` | Matters only in soft and linear modes; exact mode has one loss term, so its weight stays 1 |
| P1 optimizer | `refine = bfgs` (`pinn/bfgs.py`) and `refine = lm` (`pinn/lm.py`) | Full BFGS no better than L-BFGS per iteration, 1.5 times slower. Levenberg-Marquardt too slow on this laptop (one step over 10 minutes) |
| P7, P9 | `resample_every`, `best_checkpoint` | Built and tested, not yet compared in a run |
| P8 overfitting | Monitor against training loss | No gap: 0.0333 against 0.0336 |
| P12, R6 error estimate | `engine/bvp_engine --estimate`, `tools/pinn_error_estimate.py` | Within 10 to 20 % of the true wall errors once a run is near the solution |
| Missing item: manufactured check | `tools/pinn_manufactured.py` | Failed at the budget tried; the forced wall residual was still large |
| Missing item: branch check | `pinn/selfcheck.py` (m, mp, wall values, failure share) | Tells the non-decaying states of E1 and E2 apart from E3 onward without the reference |
| R1 b, R5 | `tools/reference_error_bars.py` | Mesh error 1e-7 relative; L = 20 against the L-extrapolated limit 8e-5 to 8e-4 per wall value; chi'(0) has no limit |

Status: the PINN no longer falls into the cold wall and heads to branch A, but it has not converged. Final run by the training-side rule: E7, theta(0) 0.197 against 0.250 and f''(0) -3.65 against -4.20. The two weightings tried fail at opposite ends: the log scale stalls with the wall residual large (E7); the plain residual reaches a tiny loss but drifts away from the solution (E8). Next, in this order: bounded per-region weights to balance them (P2), a wider network as a capacity check, a faster Gauss-Newton step (forward-mode Jacobian or a GPU), then seeds 1 and 2.

**Update, later on 2026-11-23:** solved for B0. The two reviews (reference audit, PINN synthesis) led to a coefficient-based balanced scale, a per-point Levenberg-Marquardt Jacobian and selection by the error estimate. With the log scale for Adam and L-BFGS and Levenberg-Marquardt on the balanced scale, both seeds run from scratch match the reference to within 4e-4 relative at the wall ([IMPLEMENTATION_NOTES.md](../pinn/IMPLEMENTATION_NOTES.md) section 7.3). Gap P1 is closed by Levenberg-Marquardt; P2 and P3 by the balanced scale.
