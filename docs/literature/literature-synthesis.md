# Literature synthesis: what we must know and adapt

This file turns the 25 per-paper analyses into one working guide. Each paper has a full analysis with page references at `sources/literature/<folder>/analysis.md`, tracked in [the tracker](paper-analysis-tracker.md). Roles R1 to R48 refer to [the reading list](reading-list.md).

One honesty note first: every number below is what the cited paper reports. We reproduced none of it. Preprints stay preprints. Our own conclusions are marked as ours.

## 1. Where each paper fits

| Folder | Role | For us |
|---|---|---|
| micropolar-2026 (R1) | Closest physical precedent, 4 fields, 9 BCs | BC loss assembly, far-field truncation, float64 and SiLU evidence |
| squeezing-2026 (R2) | Closest fourth-order fluid precedent, (4,2,2) | Penalty tuning rule, reference discipline, seed spread reporting |
| falkner-skan-hard-bc-2026 (R39) | Closest structural BVP precedent | Hard endpoints, regional residual balancing, staged training with rollback |
| cylinder-nano-2026 (R15) | Curved-wall family member | Equal-weight baseline, per-field error table template |
| lap-joint-2021, beams-2024 | Coupled fourth-order precedents | Chained fourth derivative, one-term-per-BC assembly, scaling failure case |
| pinn-foundations-2019 (R22) | Founding paper | Residual plus boundary loss pattern, L-BFGS practice, intro quotes |
| gradient-pathologies-2021 (R24) | Why soft losses fail | Gradient diagnostics, annealing rule for arm B4 |
| ntk-failure-2022 (R25) | NTK view | Trace-ratio weights, cost identity, instability warning |
| failure-modes-2021 (R26) | Failure taxonomy | Six failure modes, landscape diagnostic, curriculum idea |
| self-adaptive-2023 (R27) | Pointwise weights | Sigmoid mask with saturation, L-BFGS freeze rule |
| residual-decay-weights-2025 (R28) | Decay-rate weights | Concrete B4 arm, mean-one pool, under 10 percent cost |
| al-pinn-2023 (R32), hpinn-2021 (R33) | Augmented Lagrangian | AL arm for our 12 BCs, multiplier update rules |
| bc-comparison-2023 (R34), bc-3d-comparison-2025 (R36) | BC enforcement comparisons | B6 design, penalty scale evidence, split error metrics |
| fi-pinn-sampling-2023 (R38) | Failure-informed sampling | B3 enrichment algorithm, stop rule, settings |
| overfitting-failure-2026 (R42) | Overfitting view | Monitor set and fresh set validation, gradient penalty option |
| ntk-sketching-2025 (R40) | Cheap NTK weights | Sketch estimator cost and theory limits |
| optimizer-optimizing-2025 (R30) | Quasi-Newton study | Line search settings, BFGS arm, float64 for refinement |
| pinn-vs-fem-2024 (R43) | Honest head-to-head | Cost protocol, no-superiority stance, staged warm-up pattern |
| pinn-pitfalls-report-2022 (R44) | Failure reporting | Cost table, judgment-call logging, publication template |
| pinnacle-benchmark-2024 (R45) | Benchmark practice | Seeds and NaN reporting, success threshold, reweighting evidence |
| acta-numerica-survey-2024 (R46) | Error theory | Loss versus error, stability bound shape, conditioning view |
| sciml-survey-2022 (R47) | Framing survey | Soft BC drawbacks, equal-magnitude rule, stiffness open problem |

## 2. Must-know facts before coding

1. No verified paper solves a five-field (4,2,2,2,2) similarity BVP with a PINN. The closest are R1 (4 fields, orders 3,2,2,2), R2 (orders 4,2,2) and R39 (one third-order field). Keep the no-first-claim rule.
2. Small loss does not mean small error. The error survey states E cannot be computed during training (Acta Numerica p. 657), and functions with tiny residual and huge error exist (Thm 5.14, p. 680). This is the basis of our reference-based evaluation.
3. Expected relation: field error scales about like the square root of the training loss (Acta Numerica p. 687). A 100x loss drop should give at most about 10x error drop.
4. Unbalanced gradients are the core failure mode. Boundary-term gradients pile near zero while residual gradients stay orders larger (R24 p. 4), and in the NTK view the residual block usually dominates the boundary block (R25 p. 8). The direction is problem dependent, so measure, do not assume.
5. Stiffness grows with derivative order. Condition number of the operator square A drives training cost (Acta Numerica p. 697); our fourth-order f makes L*L order 8, which is our inference from their order-2 cases. R26 Appendix B scales a first-derivative operator as O(beta N) and a second-order one as O(nu N^2).
6. Nondimensionalization or fixed scales come first. Beams (2024): dimensional coefficients of 4e6 fail at loss 1e14 while the nondimensional version reaches 5.3e-4 percent error. Scale first, optimizer second.
7. Loss terms must sit at the same order of magnitude, otherwise the network minimizes only the largest (SciML survey pp. 58-59). Our 5 residuals and 12 BCs each need a scale set before training.
8. Weight sweeps alone do not rescue a hard regime: relative error stayed near 1.0 for lambda from 1e-6 to 1e1 at convection beta=30 (R26 Fig E.1, p. 18). Do not expect weight tuning to fix the wall region by itself.
9. Soft BC enforcement has no accuracy guarantee and no weight theory (SciML survey p. 23). Our per-condition checks are mandatory, not optional.
10. Penalty weight scale is a real risk in both directions. Tuned beta=100 beat annealing on interior error in R36, but too-large beta starves the physics loss (R36 p. 14), and in R34 the penalty rate collapsed from 3.66 to 0.01 when lambda dropped.
11. L-BFGS is unstable often but wins when it converges: NaN on 7 of 22 PINNacle tasks, best result on the tasks where it worked (R45 Table 3, verified against the PDF). Always keep the Adam checkpoint as fallback.
12. Plain L-BFGS was the worst quasi-Newton in the one test that compared it (R30: 2.05e-3 versus BFGS 1.50e-5 and self-scaled 7.57e-8 on Burgers). Full-memory BFGS is affordable for a small network and deserves an arm.
13. float64 is needed: high-order AD underflows in float32 (R1 p. 7), and quasi-Newton methods degrade without double precision (R30 App B). But precision alone does not fix overfitting: FP64 still failed on sparse grids (R42 p. 6).
14. Residual overfitting is real: residual dips to 1e-11 exactly at collocation points while reaching 1e5 between them (R42 Fig 3, p. 5). Fresh evaluation points are not decoration; they are the failure detector.
15. The classical reference stays outside training in every honest study we read (R43, R44, R45, R46). Our repo rule matches the field's good practice.

## 3. Pipeline decisions, by strategy section

### Loss and scales (strategy 5)

- Loss shape: one mean-square term per equation plus one term per boundary condition, extending the founding pattern of eq. 4 and eq. 6 in R22. Run one unweighted weight-1 control first; R1, R15 and the lap-joint supplement all trained with plain sums.
- Scales: fixed characteristic magnitudes declared before training, from field and derivative magnitudes, not from current residuals (our strategy). Evidence: must-know items 6 and 7.
- Tuning criterion for BC scales: choose weights so boundary residuals drop to the interior residual level (R2 p. 15), then sweep scales over orders of magnitude as R34 did with {1e-3, 1, 1e3}.
- Never adopt equal weights as the final setup: our residuals differ in scale and grow at the wall (our conclusion from R2's warning that their residuals are O(1) by construction).

### Adaptive weights, arm B4

- Primary arm: learning rate annealing, Algorithm 1 of R24. Weights start at 1, EMA alpha 0.9, update every 10 Adam steps. Log the numerator choice: max over the five separate residual gradients or over the summed residual loss. Freeze weights before L-BFGS; the paper is Adam-only, so freezing is our choice.
- Second arm: NTK trace ratios, lambda_m = sum of traces over Tr(K_m) (R25 p. 14). Cost is one backward per term, 17 total, using the identity Tr(K_m) = sum of squared per-point parameter gradients (our implementation note). Sparse updates, for example every 1000 steps. Cap the weights if they explode; the cap is our addition because large weights make the weighted kernel indefinite (R25 p. 12).
- Third arm: balanced residual decay weights (R28): irdr = R^2 over sqrt(EMA R^4), eps 1e-14, mean-one pool shared by equation and boundary points, EMA beta 0.999, expected under 10 percent overhead. Add a BRDR+ sub-arm that pins the stiff wall components with a fixed constant, as the paper did with alpha = 100.
- Optional pointwise arm: self-adaptive sigmoid mask (R27). It is the one construction with a bounded effective weight, and its L-BFGS freeze rule matches ours. Strip weights from the loss when logging.
- If NTK weights are wanted at every step, use the sketch estimator (R40): factor 2.5 cost, only 17 block traces needed, no matrix storage, moving average alpha 1e-4 to 1e-3. It needs nonzero residual entries.
- Evidence to expect: boundary weight 100 helped elliptic cases and hurt steady Navier-Stokes (R45 Table 3). Reweighting is problem dependent; report our own result.

### Boundary enforcement, arms B6 and AL

- Exact endpoint construction: use a polynomial factor that vanishes at the endpoints where a field has a value condition, per the lifting pattern of R34 eq. 3.3. In 1D the corner blow-up they report cannot appear. The R39 mask G(s) = L s^2 (1-s)^2 keeps endpoint curvature nonzero; for our fourth-order f, test the mask's higher-derivative behavior first. R2's lift fp = (3 eta - eta^3)/2 shows a single polynomial can carry four f-conditions.
- Scope: value conditions only. Slope, slip, flux and the coupled radiation conditions stay soft. R34 tested only Dirichlet; R36 ran no Neumann-data case despite its abstract; R39 handled three linear conditions. Say so in the write-up.
- AL arm: loss adds beta e_i^2 plus lambda_i e_i per BC, one fixed beta, twelve scalar multipliers updated by ascent lambda_i <- lambda_i + eta_lambda e_i (R32 p. 3). Grid search beta in 1 to 1e4 and eta_lambda in 1 to 1e-4 (R32 App B). Keep beta fixed: a growing penalty lost by 40x (R32 p. 6). AL round variant: warm starts, penalty growth beta = 2, stop when PDE and BC losses fall under tolerance, watch lambda/mu flatten (R33). Apply AL to BC residuals only as our own extension; no paper tested it there.
- Both penalty and multiplier terms are needed; the pure multiplier method failed to learn (R32 p. 6).
- Do not use min-max Lagrange multipliers (never converged, R36 pp. 14, 16) or constrained quasi-Newton solves (stall at 100 to 500 boundary points, R36 p. 16).

### Optimizer stages (strategy 9 and 10)

- Adam phase: lr 1e-3, betas (0.9, 0.999) with (0.99, 0.99) worth a test, plateau decay or step decay as options (R15, R45). Gradient clip norm 1.0 is a documented option (R39, R15). A one-time lr cut when the monitor crosses a threshold worked in R39.
- Optional staged warm-up: short Adam run on the hardest terms alone (BCs plus the f equation), then the full loss, then L-BFGS. Pattern from R43 p. 15, where a 1000x weight plus staging made Allen-Cahn learnable at all.
- L-BFGS phase: Strong Wolfe line search with c1 = 1e-4, c2 = 0.9 (R30 p. 30). Stop on gradient norm plus loss-change and step-change checks (R30 App A) and on relative train-loss change under sqrt(machine epsilon) over 100 iterations (R42 App A). History 50 stays our own untested choice; R30 reports none.
- Add a full-memory BFGS arm (R30). If it stalls, the self-scaled rescaling tau = min(1, 1/b_k) from stored (s, y) pairs is a candidate (R30 p. 6), tested by them only on full memory.
- Rollback rule: accept an L-BFGS stage only if the fixed monitor set does not get worse (R39 eq. 81). Keep the Adam result when refinement fails (our strategy; matches R45 NaN handling).
- Run refinement in float64 with fixed weights and fixed points (R30 App B; R43 freezes collocation for L-BFGS).

### Sampling, arm B3

- Baseline: stratified uniform plus wall coverage, matching our strategy 7. R15's 300 fixed uniform points is the control arm.
- Enrichment arm: failure-informed style (R38): indicator g = max over the five scaled residuals minus epsilon; union enrichment, never replacement; variable count per round; stop when the estimated failure probability drops below tolerance. Settings N1 = 300, p0 = 0.1, N2 = 1000, at most 10 rounds. In 1D the proposal must be truncated to [0, L]; near the wall a plain Gaussian spreads past the boundary.
- Call our top-error scorer our method unless we implement RAD or RAR-D exactly (reading list rule; the RAR definitions live in R6, not in the often-cited convergence paper).
- Expect small gains from RAR-style enrichment alone (R45 Table 3: 6.99E-1 versus 6.94E-1). Late one-shot enrichment after the optimizer chain, concentrated at the wall, is R1's pattern and is cheaper to test.
- Keep the eta grid dense enough that monitor points sit between train points (R42); dense grids over-determine the problem and beat sparse grids plus precision.

### Evaluation and reporting (strategy 12)

- Report per-field L2 and max errors against the converged reference on a fresh set, plus wall-band metrics separately from global ones. R45 has no wall metric at all; that gap is our opening.
- Report boundary satisfaction separately from interior accuracy: the I and B pair of R36 eq. 12, extended per BC group.
- Seeds: at least 3 for pilots, 10 for serious comparisons, mean and std, log NaN as non-convergence (R45 p. 7, R26 p. 4).
- Success threshold declared before runs, for example the 10 percent L2RE style of R45 plus our scaled-BC max target from strategy 12.
- Cost: solution or training time and evaluation time separately, same hardware for all arms, accuracy-versus-time log-log plots (R43 protocol).
- Report best-run and mean-run accuracy side by side (R2 lesson: headline 9.3e-9, typical 3e-6).
- Theil inequality coefficient next to MSE is one cheap scale-free line (R2 eq. 45).
- Loss is never the success indicator. Loss fell while the cylinder solution stayed wrong (R44 p. 34) and loss fell while error stayed constant (R34 p. 9). Monitor divergence between train and monitor loss is the failure signature (R42).
- Failure reporting template from R44: per-case configuration, results-as-experimental-data disclaimer, stopping decisions logged as judgment calls, failed runs published with a reproducibility package, result table with PINN, classical and literature values side by side.
- Track hyperparameter provenance: tuned, copied, or default (R44 p. 31 warning).

### Implementation checklist

- Chained autodiff one order at a time up to f'''' (lap-joint Listing S2 pattern), column-sliced outputs, tanh hidden layers with linear output (R22, lap-joint).
- Independent derivative audit: hand-derived tanh derivatives through order 4 versus autodiff once, like R2's symbolic check.
- Zero-initialize the final layer so training starts from the exact lifting when B6 is on (R39 eq. 41).
- SiLU versus tanh is a fair cheap test; both are C-infinity (R1 p. 7). Input injection of eta into every hidden layer is a second small variant from R1.
- Save the collocation set to file with seeds (lap-joint X_f.txt pattern).
- Keep-best-checkpoint by monitor loss (R15 p. 9, R39).
- A small L2 weight decay of 1e-6 is a documented stability option (R34 p. 7); plain weight decay also beat overfitting cheaply in R42.
- Curriculum option: ramp a wall-coefficient scale with warm starts and a well-posedness check per step (R26 Sec 5.1). This matches the continuation stages 1, 0.5, 0.1 already in strategy 2.
- Hessian-eigenvector landscape plot as an occasional diagnostic (R26 Sec 4.1); per-component gradient histograms every 10 steps (R24).

## 4. Writing guide

Intro and framing anchors, all with page references in the analysis files:

- Founding pattern and residual-as-regularization: R22 quotes on pages 6, 7 and 10, especially the well-posedness sentence on page 6.
- Soft BC drawbacks and equal-magnitude rule: R47 quotes on pages 23 and 58-59.
- Stiffness as the named open problem: R47 pages 57 and 61; R26 landscape evidence page 7.
- Why we do not report loss as error: R46 quotes on pages 657 and 687, plus the R43 and R44 loss-decoupling cases.
- Honest stance: R43 abstract quote (not outperformed), R44 page 28 cost sentence and page 34 judgment quote, R26 weight-sweep negative control.
- Adam then L-BFGS as standard practice: R47 p. 24; supported as preconditioning by R46 p. 705.

Related-work map by arm: B3 (R38 plus R6), B4 (R24, R25, R27, R28, R40, R45), B5 (R7 FO-PINN from the reading list; note R15's differentiated-slip trap), B6 (R34, R36, R39, plus R8 exact constructions), AL (R32, R33), optimizer (R30, R45, R43), physical precedents (R1, R2, R15, plus the metadata-only records R12 to R21 cited at record level only).

Claim discipline, restated from the analyses:

- Author-reported numbers stay author-reported. Preprints (R30, R36, R39, R40, R42) are labeled preprints in text.
- No superiority claim over solve_bvp before measured numbers; R43 and R44 are the canonical citations for that stance.
- The squeezing paper's venue is un-audited (Volume 00, Issue 00); cite it as a recent technical precedent, not a validation authority.
- Do not borrow parameter values from any paper; the seven missing coefficients stay null.

## 5. Do not import (cross-paper list)

| Item | Why |
|---|---|
| Parametric multi-input network | We solve one case; R1's 8D input adds cost with no benefit here |
| PSO-NNA or metaheuristic training | R2 gives no settings and no gradient baseline; not reproducible from the paper |
| Time marching, seq2seq, causal weighting | Our BVP is two-ended; no causal direction exists (R26, R38, R44) |
| Nitsche or variational forms | Needs weak form machinery; suboptimal in R34; would replace the confirmed Eqs. 14-19 |
| gPINN derivative-residual terms | Hurt PINNacle's Poisson case at every weight; raises our derivative order further |
| Min-max Lagrange, constrained quasi-Newton for BCs | Never converged, or stalls at realistic point counts (R36) |
| Unbounded trainable weights | Violates our bounded-weight rule; R25 warns of kernel indefiniteness |
| Solver snapshots as training residuals | Conflicts with the reference-never-trains rule (R44 p. 35) |
| eta-infinity = 7 or L = 12 | Choose L from our reference solve (R1, R39 both fixed theirs without testing) |
| Micropolar or squeezing equation details | Different fluid models; symbols conflict across papers; check every transcription against our sheet |
| Single-seed gain percentages as expectations | R39's ablation is one seed; treat as case study |
| FP32 | High-order AD and quasi-Newton both need float64 (R1, R30) |

## 6. Errors and gaps found inside the papers

Worth knowing so we never cite the shaky cells. Full details in each analysis file.

- micropolar-2026: dissipation term printed two ways (Eqs. 13/24 versus Figure 3); phi and varphi swapped in places; abstract contradicts its own Figure 13d paragraph; conclusion 5 has no supporting experiment; Table 1 gives 11 decimals from a 5-significant-figure reference.
- cylinder-nano-2026: solver called BVP4C in the abstract and solve_bvp in the text; second-order truncation claimed but no second-order residuals written; output count inconsistent; L2 error never defined; no base parameter table.
- squeezing-2026: Volume 00 issue state; no optimizer settings; headline error is the best of 100 runs; the reference is both judge and surrogate teacher.
- overfitting-failure-2026: convection and wave figure citations swapped; lambda_r never reported.
- optimizer-optimizing-2025: Table 16 header says SSBroyden, caption says SSBFGS; L-BFGS appears once with no settings.
- PINNacle: Table 3 disagrees with Tables 5 and 24 on single cells.
- R24 vs journal title: the arXiv v1 title omits the word Flow; cite the SIAM title.

The lesson matches our repo rule: check every transcription against the source page, and treat single numbers from tables with care.

## 7. Open choices to log or ask

1. Annealing numerator: summed residual gradient versus max over the five residuals (our decision to record in the B4 config).
2. Weight caps for any adaptive arm: our addition, absent from all four weight papers.
3. AL applied to our nonlinear coupled BCs: published nowhere; run as our extension or skip.
4. L-BFGS history size and the BFGS arm: no paper tested our setting; measure.
5. SiLU versus tanh and input injection: cheap tests, evidence from R1 only.
6. eta = L: take it from the reference solve and its L-sensitivity study, never from R1 or R39.

## 8. File map

- Two-minute version of all papers: [papers-tldr.md](papers-tldr.md)
- What to read seriously, tiered: [reading-shortlist.md](reading-shortlist.md)
- Tracker with status and one-line results: [paper-analysis-tracker.md](paper-analysis-tracker.md)
- Per-paper analyses: `sources/literature/<folder>/analysis.md`, one per row in section 1. Each starts with a short TL;DR record, then the page-referenced detail, quotes, adopt and do-not-import lists.
- Strategy being adapted: [strategy.md](../plans/strategy.md); equations: [equations.md](../problem/equations.md); open parameter questions: [questions-for-supervisor.md](../problem/questions-for-supervisor.md).
