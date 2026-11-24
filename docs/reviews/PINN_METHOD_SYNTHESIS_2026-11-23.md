# Why the PINN misses the reference, and what to change

Date: 2026-11-23. Branch: `speedup/taylor-jets-and-reference-engine` (working tree, uncommitted work of other sessions present). Scope: the PINN only; the reference is checked in a separate session. Eqs. 14 to 18, B1 to B12 and the locked values are fixed; every change below is a way of imposing or solving them.

Tags: **measured** (run or computed here, on saved run folders), **author-reported** (a paper's own number), **abstract only** (link or abstract, no full text read), **inference** (our reasoning, not tested).

Status: analysis complete; the diagnostic trainings planned in section 9 were **not run** (usage limit reached during this session). All measurements below are on the seven saved run folders, with scripts kept outside the repo. No existing file was edited.

## 1. Short answer

1. **The loss is measured in the wrong norm at one end or the other.** Under the plain residual the far field gets almost no gradient (E8: gradient norm 5.7e-5 to 2.1e-3 in [4, 20] against 0.36 at the wall), so the tail drifts: theta(L) is -1.02 in E8 and -1.09 in E1, against +0.068 (**measured**). Under the log scale (eta + b)^order, the far-field R14 and R15 rows are weighted up to 1.7e5 and 408, and they carry 6.1 and 6.8 of gradient norm against at most 0.27 anywhere else (E7, **measured**). So the optimizer spends its steps on tiny f and g wiggles far out.
2. **The residual Jacobian is nearly rank deficient.** At E5 and E7 its singular values fall from about 4e4 to 2e-6 by rank 100; only 74 of 5 205 directions lie above 1e-8 of the largest (**measured**). The same shape appears under every scale, so it comes from the smooth tanh network on one input. The log scale adds 8 decades of row weights on top. That is why L-BFGS and full BFGS stall in the same place (E4, E5, E7).
3. **A small loss is not a small error in either scale.** E5 to E7 cut the log-scaled loss by 3 times while theta(L) went from 0.066 to -0.060 and theta(0) moved away from the reference (**measured**). Run selection must use the reference-free Newton-step error estimate, not the loss.
4. **A coefficient-based balanced scale helps the Gauss-Newton model.** At E5, a scale that keeps q^4 for R14 near the wall but stops at the Darcy term (q^4/(1 + 3.33 q^2)), q^2/(1 + 0.4 q^2) for R15 and q^2 for R16 to R18 lowers the largest singular value from 3.85e4 to 519 and lets a damped Gauss-Newton linear model drop its own loss from 2.0e-2 to 3.5e-3 at damping 1e-6, where the log scale gives no drop (**measured**, linear model only; the full nonlinear step overshoots, so a trust region is needed).
5. **Levenberg-Marquardt is slow only because of how the Jacobian is built.** `lm.py` builds it with one reverse pass per residual row. Each residual depends on one point and the two ends, so a per-point `vmap(jacrev)` gives the same matrix (difference 7e-15) in 0.5 s instead of 43 s on a 20 x 3 network, and in 13 to 19 s for the full 40 x 4 network on 1 000 points (**measured**, 1 thread).

Top three changes, in order: the balanced scale (section 8, P-A), per-point Jacobian Levenberg-Marquardt with a trust region (P-B), and selection by the error estimate with a tail check (P-C).

## 2. What was read

| Corpus | Access | Use |
|---|---|---|
| 25 papers in `sources/literature/` | Each folder's `analysis.md` (full-text notes with page locators, written 2026-11-22); PDFs not re-opened this session | Evidence table, methods comparison |
| `docs/literature/` (synthesis, notes, reading list, TLDR) and `LITERATURE_GAP_STRATEGY.md` section 9 | Read | Link-only papers marked abstract only |
| Lorenz work: `../lorenz1960-pinn/docs` (causal vs batch, three-run comparison, repair note, long-horizon survey, short version), `src/pinn/pinn.py` | Read the docs; code skimmed | Section 6 |
| `../fydp_workspace/All Paper txt` | First page of four ODE-ANN papers | Low relevance, abstract level |
| `dt-nb-noise-removal/paper/iceeict2027-submission` | Abstract and methods | Evaluation protocol only (it is a decision tree study, not a PINN study) |
| Our code `src/bvp_pinn/pinn/`, `configs/pinn.json`, `tools/pinn_*.py`, runs E1 to E8 | Read every PINN file; analysed every saved run | Sections 7 and 8 |

Limits: the paper notes are model-written extracts; numbers quoted from them were not re-checked against the PDFs in this session. No run labelled E6 exists in `runs/` or the docs; the ladder goes from E5 to E7.

## 3. Evidence table (question: what reaches high accuracy on stiff, high-order, coupled ODE BVPs)

| ID | Problem | Method | Raw result | Locator | Kind |
|---|---|---|---|---|---|
| falkner-skan-hard-bc-2026 | One third-order ODE, L = 12 | Hard lifting plus mask, ARRB regional weights, Adam 15 000, two L-BFGS stages with rollback, float64, 100 x 2 | Wall f''(0) error 6.691e-7; residual MSE 6.357e-9; Blasius 1.106e-7; one seed | Table 2 p. 17; p. 23 | author-reported |
| squeezing-2026 | Orders (4, 2, 2), smooth constant coefficients, [0, 1] | PSO-NNA then Levenberg-Marquardt polish, 30 weights per field | Best velocity max error 9.3e-9; 100-run mean near 3e-6, max near 4.5e-5 | Tables 3 to 5, pp. 26 to 32 | author-reported |
| micropolar-2026 | Orders (3, 2, 2, 2), eta at 7, 8D parametric input | Adam, L-BFGS, one-shot RAR, float64, SiLU | PDE MSE 1.12e-5; -f''(0) within 0.0038 % of FEM | Fig. 5 p. 9; Table 1 p. 10 | author-reported |
| cylinder-nano-2026 | Three fields, third order | Adam, equal weights, plateau decay | L2 3.5e-3 (not defined); Cf gaps up to 0.004 | p. 1; Table 7 p. 21 | author-reported |
| beams-2024 | Linear fourth order | L-BFGS on nondimensional form | 5.3e-4 % relative L2; dimensional form stuck at loss 1e14 | pp. 3 to 4 | author-reported |
| optimizer-optimizing-2025 | Burgers, Allen-Cahn, Stokes, Lorenz-63 | BFGS, self-scaled Broyden, L-BFGS after short Adam, float64 | Burgers: SSBroyden 7.57e-8, BFGS 1.50e-5, L-BFGS 2.05e-3 | Table 1 p. 8 | author-reported |
| residual-decay-weights-2025 | Helmholtz, Allen-Cahn, Burgers | Bounded pointwise decay-rate weights, Adam only | Allen-Cahn 2.51e-5; Helmholtz 1.73e-4 | Table 3 p. 26 | author-reported |
| acta-numerica-survey-2024 | Theory | Stability bounds, conditioning of L*L | Error about sqrt(training error); GD rate set by kappa(A); quasi-Newton acts as a preconditioner | pp. 675, 687, 697, 705 | theory |
| failure-modes-2021, pinnacle-benchmark-2024, pinn-vs-fem-2024, pinn-pitfalls-report-2022 | PDE benchmarks | Various | Failures come from optimization; L-BFGS best where it converges, NaN on 7 of 22 tasks; FEM faster and more accurate | as in each analysis | author-reported |
| Lorenz Run B (ours, other problem) | Lorenz-1960 IVP, one orbit | 27 causal windows, warm start, float64 | RMSE 8.4e-5; Run A (one network) RMSE 2.24 with loss 2.8e-4 | lorenz1960-pinn `causal-vs-batch-error-propagation.md` sec. 1 | measured (other project) |
| Wang and Lai 2024; Jnini et al. 2024; Rathore et al. 2024 | PDEs | Multi-stage networks; Gauss-Newton natural gradient; NysNewton-CG | Near 1e-16; near single precision; 14x over Adam | LITERATURE_GAP_STRATEGY 9.2; Lorenz survey sec. 7 | abstract only |

Other local papers (gradient-pathologies, ntk-failure, ntk-sketching, self-adaptive, al-pinn, hpinn, bc-comparison, bc-3d-comparison, fi-pinn, overfitting-failure, pinn-foundations, sciml-survey, lap-joint) address loss weights, soft boundary terms or sampling. In exact mode our loss has one term and no boundary penalty, so their weight rules act on nothing here (**inference**, consistent with E1 to E8 weights all at 1).

## 4. Best accuracy the literature shows for problems like ours

All author-reported. No paper in the corpus has a wall coefficient of order 1e2 to 1e3 (ours: 1/q^4 = 625 at the wall, dipole factor about 250), a fourth-order momentum equation and slow power-law tails together.

| Class | Best reported | Source |
|---|---|---|
| Single boundary-layer ODE (Falkner-Skan, Blasius) | Wall error 1e-7 to 7e-7, residual MSE 6e-9 | falkner-skan-hard-bc-2026 |
| Fourth-order coupled, smooth coefficients (squeezing) | Typical 3e-6, best 9e-9 | squeezing-2026 |
| Micropolar coupled, stretching sheet | Wall shear within 4e-5 relative | micropolar-2026 |
| Quasi-Newton on smooth PDEs | 1e-7 to 1e-8 relative L2 | optimizer-optimizing-2025 |

Our expectation (**inference**): with the conditioning fixed and a Gauss-Newton type optimizer, wall values within 1e-3 relative are reachable; 1e-5 or better would need a multi-stage correction network. Calibration from our runs (**measured**): under the log scale, a summed scaled MSE of 1e-2 gives wall errors of 5 to 20 %. Errors scale like the square root of the loss (Acta p. 687), so wall errors near 1e-3 need a scaled MSE near 1e-7 to 1e-8 per equation in a well-conditioned scale. A loss target of 1e-3 to 1e-4 is not enough by itself, and E8 shows it means nothing in the plain scale.

## 5. Methods comparison and disagreements

| Family | What it controls | Fits our exact-mode B0? |
|---|---|---|
| Loss weights between terms (gradient norm, NTK, annealing, SA, BRDR) | Balance of residual against boundary terms | No: one term left |
| Regional or pointwise weights (ARRB, BRDR pointwise) | Balance across space | Yes; but a fixed coefficient-based scale does the same job without moving weights |
| Hard constraints (lifting, TFC, distance functions) | Boundary error | Done for all 12; correction shapes still matter (section 7) |
| Optimizers (L-BFGS, BFGS, SSBroyden, Gauss-Newton, LM) | Optimization error | Main lever after scaling |
| Sampling (RAR, FI-PINN, resampling) | Error between points | No train/monitor gap measured (0.0336 against 0.0333); low priority |
| Time marching, causal weights | Ordering in time | Not applicable; see section 6 |

Disagreements, classified:

- **Optimizer matters (Kiyani) against weights matter (BRDR, ARRB).** Scope and design difference: Kiyani uses no weights and many quasi-Newton steps on small nets; BRDR uses Adam only. Both can hold. For us the one-term loss removes the weight question, but the row scaling inside that term is exactly what ARRB balances.
- **L-BFGS good (Falkner-Skan, PINNacle where it converges) against L-BFGS worst (Kiyani Table 1).** Budget and conditioning difference. Our measured spectrum explains our case: once the usable directions are spent, neither L-BFGS nor dense BFGS within 3 000 steps (fewer than the 5 205 updates a full BFGS matrix needs) can go further.
- **Small loss means small error (Falkner-Skan) against not (E8, Lorenz Run A, Chuang and Barba).** Resolved by stability: the error per unit residual is large when the residual is blind to some modes (tail modes for us, the equilibrium line for Lorenz). Unresolved residual: no paper in the corpus measures this constant for a BVP.

## 6. What carries over from the Lorenz work

| Lorenz technique or finding | Result there | Carries over to a two-point BVP? |
|---|---|---|
| Causal weights, time windows | Run B 8.4e-5 against Run A 2.24 | **No as a method**: no direction; both ends carry conditions. **Yes as a lesson**: Run A failed by sitting in a low-residual wrong state (residual and error anti-correlated). E8 and E1 are the same failure: tail modes with near-zero residual |
| Staged windows from the start | Error accumulates across joints | Arms B2 and B3 train the wall before the far field that sets f(L) and m; **inference**: they risk the E8 drift early |
| Hard trial function u0 + g(t) N | Works; scaling of g changes the needed network output (N(0) = 10 f(u0)) | Yes: our TFC correction width (0.5 at both ends) is sharp against the tail scale (q up to 20). The far condition acts only in [19, 20] |
| Warm start | Window 0 (cold) was the worst | Partly: the smooth start alone did not help (E2 against E1), because the scale dominated. Continuation runs (E5 to E7, E8) are warm starts that drift |
| float64 | float32 floor near 1e-5 | Already float64 |
| Input normalization without features | Hurt 3x (bandwidth) | Our log map is a feature map; keep it |
| Survey: second-order methods, multi-stage, random-feature least squares | Large gains, abstract level | Yes, ranked in section 8 |

The conference paper adds protocol only: fit every choice inside training, keep the evaluated data untouched, and use paired multi-seed tests with a multiplicity correction. That matches gates G5 and G6.

## 7. Pipeline audit, file by file

| File | Finding | Evidence |
|---|---|---|
| `model.py` | tanh 40 x 4 with log1p(eta/0.1) map. Correct. The usable rank of the Jacobian is about 74 to 102 at float64 precision under every scale, so width and depth buy little. Map scale 0.1 is close to, not equal to, the coefficient scale q = eta + 0.2 | measured, section 1 |
| `jets.py` | Correct (equal to autograd, tests). Allows a per-point Jacobian through `vmap(jacrev)` | measured 7e-15 |
| `constraints.py` | Exact for all 12 conditions (review checks). One decay 0.5 for every correction at both ends. The far-field modes theta = C1 + C2 q^-m both have near-zero residual; B10 is the only thing that removes C1, and it acts through a correction 0.5 wide. theta(L) ends with the wrong sign in E1, E3, E7, E8 | measured theta(L): -1.09, -0.17, -0.06, -1.02 against 0.068 |
| `physics.py` | Correct; divides each equation by its highest-derivative coefficient | review 2026-11-21 |
| `losses.py` | `residual_scale` log over-weights far-field R14 and R15 (rows up to 1.7e5); none starves the far field. The plain mean over wall-stretched points is close to an integral in log(eta + 0.14) | measured gradient shares |
| `points.py` | 1 000 training, 500 monitor points in the pilots; no overfitting gap | measured |
| `train.py` | The refine rollback compares monitor losses in the training scale, so it accepted E8 (237 to 2.3e-4) and E7 while the solution got worse. Fixed budgets, no stop on an error estimate | measured |
| `bfgs.py` | Correct update. 3 000 iterations cannot build a full 5 205-rank inverse Hessian; no gain over L-BFGS | E4 against E5 |
| `lm.py` | Correct algorithm. Jacobian by one reverse pass per residual row: 43 s against 0.5 s per-point on 20 x 3; this is the 10-minute step | measured |
| `selfcheck.py` | m, mp and failure share are useful. Missing: the far-field mode check (fit theta and phi on [L/2, L] to C1 + C2 q^-m and report C1) and the sign of theta(L) | inference |
| `tools/pinn_error_estimate.py` | Within 10 to 20 % of the true wall errors near the solution (E3); E7 estimate f''(0) 0.51 against true 0.55 | IMPLEMENTATION_NOTES 7.1 |
| `tools/pinn_manufactured.py` | Its forcing R(u*) keeps the large wall dipole term (about 1 400 in R14 at the wall), so it tests the hard part; it failed for the same reason as the real case | inference from the code |

## 8. Diagnosis of E1 to E8

| Run | What happened | Explanation | Status |
|---|---|---|---|
| Pilots (11-21) | Cold wall, B4 missed by 2.3 | Soft and linear modes left B4 to a weighted penalty whose weight stayed small while other weights blew up (review H1, H2). Exact B4 removed it | tested (E1 onward) |
| E1 | Exact mode, log map: f''(0) right sign, but f(L) -0.59, m < 0, theta(L) -1.09 | Plain residual blind to the far field (log-scale view of E1: [4, 20] R14 holds 100 % of the loss). B10 is exact for whatever f(L) the network has, so a wrong f(L) selects a growing tail | measured |
| E2 | Same as E1 with the smooth start | The start profile has f(L) about 0.6; the plain scale drove it negative anyway. The start is not the lever while the scale is wrong | inference from E1, E2 endpoints |
| E3 | Log scale: m > 0, theta(0) 0.175 | The q^2 weight makes the tail residual relative to theta, so the decaying mode is kept | measured |
| E4 | Full BFGS, no gain, 1.5x slower | Rank about 100 and 8 decades of row weights; 3 000 updates cannot fill a 5 205 matrix | measured spectrum; inference on the cause |
| E5 | L-BFGS control, best wall values (theta(0) 0.206) | Same stall family as E4 | measured |
| E7 | 10 000 more L-BFGS: loss 0.033 to 0.0106, wall values worse, theta(L) -0.060 | Far-field R14 and R15 rows dominate the gradient (6.1, 6.8 against at most 0.27); the remaining loss is nearly orthogonal to the usable Jacobian range: an exact Gauss-Newton model step cuts it only to 6.8e-3 even at damping 1e-10 | measured |
| E8 | Plain residual: MSE 2.3e-4, theta(0) 0.091, theta(L) -1.02 | Plain scale gives the far field gradient 1e-4 of the wall; the tail is free, and through f(L), m and B4 the wall values follow. Error per unit residual is about 300 to 600 in the tail in this norm | measured |
| Manufactured check | Failed (theta(0) 0.72 against 0.50) | Same wall forcing and same scale; budget too short | inference |
| LM attempt | One step over 10 minutes | Row-by-row Jacobian | measured (timing) |

## 9. Ranked plan

Rules: one change at a time, B0, seeds 0 to 2, decide on training-side numbers plus the reference-free error estimate, read the reference only afterwards.

| Rank | Change | Evidence | Expected effect | Cost laptop / Colab | Test that decides |
|---|---|---|---|---|---|
| P-A | Add `residual_scale = balanced`: R14 * q^4/(1 + c14 q^2), c14 = (1/Da)/(1+Delta); R15 * q^2/(1 + c15 q^2), c15 = 2 Delta B/(1+Delta/2); R16 to R18 * q^2. From eta, b and locked coefficients only | Gradient shares (E7, E8); balanced Jacobian max singular value 519 against 3.85e4 and a linear-model drop 2.0e-2 to 3.5e-3 at E5; ARRB idea (falkner-skan p. 9) | Removes the far-field R14 and R15 stiffness while keeping the tail of theta and phi; wall values closer than E5 | S: about 15 lines plus a test; runs as before (0.13 s per L-BFGS evaluation) | From E5 weights, 5 000 L-BFGS under balanced against log (one run each, 12 minutes): error estimate of f''(0) and theta(0), theta(L) sign, tail C1 |
| P-B | Levenberg-Marquardt with a per-point `vmap(jacrev)` Jacobian and Nielsen damping, on the balanced scale | Per-point Jacobian 13 to 19 s at 40 x 4; squeezing LM polish; Gauss-Newton gains (Jnini, abstract only); full GN steps overshoot at low damping (measured), so damping control is required | Orders of magnitude lower loss where L-BFGS stalls (inference) | M: about 40 lines in `lm.py`; about 30 to 60 s per step on 1 thread at 40 x 4 (estimate), 5 to 10 s on a 20 x 3 net; Colab CPU similar | 50 LM steps from the P-A state against 5 000 L-BFGS at equal wall time |
| P-C | Selection and stopping by the Newton-step error estimate, plus a tail mode check (C1 of theta and phi on [L/2, L], sign of theta(L)) in `selfcheck.py`; refine accepted only if the estimate improves | E7 and E8 were accepted on loss while getting worse; estimate within 10 to 20 % near the solution | Stops drift being kept | S | Re-score E1 to E8: the rule must rank E5 above E7 and reject E8 |
| P-D | Far-end correction width tied to the tail (decay near q_L, or the shape q^-m) instead of 0.5 | Tail modes C1, C2 have near-zero residual; theta(L) wrong in four runs | Lets B10 act on the whole tail | S to M | theta(L) sign and C1 across seeds |
| P-E | Smaller network (20 x 3 or 30 x 3) when LM is used | Usable rank about 100; LM cost falls with the square of the weight count | Same accuracy at much lower LM cost (inference) | S | Same as P-B on two sizes |
| P-F | Multi-stage correction network (Wang and Lai, abstract only), reported as its own result | Single networks floor near 1e-5 | Below 1e-5 if P-A to P-C work | M | Wall error estimate after stage 2 |
| P-G | Reduced-order Eq. 14 (P10, exact rewrite) | Lower operator order improves conditioning (Acta 7.3) | Smaller wall-row range | M | As P-A |

**Run first:** P-A, then P-C to read it, because both are small and they decide whether P-B is worth the code. The two training tests I planned for this session (small-network LM against an L-BFGS control, each under 15 minutes through `bvp_pinn.pinn.run`) are still pending; they need no code change and can run before P-A.

## 10. Gaps in the corpus (bounded to what was read)

1. No paper here treats a stiff wall coefficient plus power-law tails in one BVP PINN. Search terms: "PINN boundary layer algebraic decay far field", "PINN curved stretching sheet magnetic dipole".
2. No paper measures the residual-to-error stability constant for a BVP PINN; our section 4 calibration is the first number we have.
3. No paper compares correction-shape widths of hard far-field constraints.

## 11. Measurement record (this session)

Scripts in the session scratchpad: `diag_regions.py`, `diag_grad.py`, `time_jac.py`, `jac_state.py`, `gn_scales.py`. They read saved run folders, write nothing into the repo, and read the reference only for the region error table of finished runs. No new run folders were made.
