# Analysis: Balanced residual decay weights, 2025

**TL;DR.** Pointwise weights from residual decay rates, one mean-of-one pool shared by equation and boundary points, under 10 percent overhead. **Gives us:** the most concrete B4 arm (irdr = R^2 over sqrt of the EMA fourth moment) and the BRDR+ pinning sub-arm. **Avoid:** no hard cap; the multi-component extension is claimed but never shown; they dropped L-BFGS. **Relevance:** high.

## Record

- Title: "Self-adaptive weights based on balanced residual decay rate for physics-informed neural networks and deep operator networks" (p. 1).
- Authors: Wenqian Chen, Amanda A. Howard, Panos Stinis, Pacific Northwest National Laboratory (p. 1).
- Year and venue: 2025. Journal of Computational Physics 542:114226 per the project record R28. The PDF is arXiv:2407.01613v2, dated 16 Sep 2025, and carries no journal header (p. 1).
- Pages read: all 64 pages, as extracted text. Pages 8 to 9 and 16 to 21 were also read as rendered PDF pages to verify equations and Algorithm 1. Appendix A (architectures, pp. 52 to 54) and the reference list (pp. 59 to 64) were read in text and skimmed once their role was clear. Nothing was unreadable.

## Problem

- Plain PINN training lets residuals at different points decay at very different rates. On a 1D Poisson toy problem the average inverse decay rate spans about two orders of magnitude across points, and stalled points sit near irdr = 1, so the plain PINN fails to converge (Sections 2.2 and 2.3, pp. 8 to 14).
- The paper reports two observations: residual decay rates differ by orders of magnitude across points, and the slowest rate dictates the speed of the whole solution (Section 1, p. 4; Section 2.3, pp. 12 to 13).
- Goal: assign larger weights to terms with slower residual decay so all rates match, while keeping weights bounded and cheap to compute (Section 1, p. 4; Section 3.2, p. 16).

## Method

- Core quantity, Eq. (9): irdr = R^2 / sqrt(Rbar^4 + eps), with eps = 1E-14 (p. 8). Rbar^4 is an exponential moving average of the 4th power of the residual, updated by Eq. (10) with smoothing factor beta_c (p. 8). Reading: irdr < 1 means the residual decays, irdr > 1 means it grows, irdr near 1 means it stalls (p. 9). The 4th moment was picked for sensitivity to large residuals; the paper reports no decisive gain over the 2nd moment (pp. 8 to 9).
- Weight rule: weights proportional to irdr, Eq. (18); normalized target w_n^ref = irdr_n / mean(irdr_n), Eq. (19); smoothed update w_n = beta_w w_{n-1} + (1 - beta_w) w_n^ref, Eq. (20) (p. 16). In Algorithm 1 the running mean is bias corrected, dividing by (1 - beta_c^n) since it starts at 0 (p. 21).
- Per-point or per-term: pointwise. Every collocation and boundary point carries its own weight (Section 3.2, p. 16; Eq. 29, p. 20). Per-term control exists only as fixed user constants alpha_R, alpha_B in Eq. (29); alpha = 1 is called BRDR, tuned alpha is BRDR+ (p. 20).
- Bounding: no cap and no growth rule. The mean of all pointwise weights over PDE and BC points together is constrained to 1, Eq. (16) (p. 15), and the EMA keeps weights smooth. The paper states this "helps ensure that the weights remain bounded and vary smoothly" (p. 16) and lists bounded weights as a property (Abstract, p. 1; Section 4.4, p. 35). The bound is on the average; a single weight can still exceed 1.
- Scaling factor: a separate global s. From the stability bound eta <= 2L / ||grad L||^2, Eq. (23), they derive s_max = s_{n-1} 2 L_{n-1} / (eta ||grad L_{n-1}||^2), Eq. (25), then smooth it, s_n = beta_s s_{n-1} + (1 - beta_s) s_max, Eq. (26), with beta_s = 1 - eta (pp. 17 to 18). It costs almost nothing because the loss and gradient already exist (p. 18).
- Update frequency: weights of the points in the current batch update every step, with gap-aware smoothing beta^{delta n_i} for mini-batches, Eqs. (27) and (28); the scaling factor updates every step (p. 19). With very many batches they advise beta = 0.9999 instead of 0.999 (p. 19).
- Cost: the paper reports under 10 percent extra time versus fixed weights on all PINN benchmarks, 104 to 107 percent in Table 3 (p. 26), 106 percent for operator learning in Table 4 (p. 44), and under 10 percent in the ablation (p. 56).
- Stop-gradient: not discussed by name. Weights and s are updated outside backprop and act as constants in the loss; Algorithm 1 then rescales the stored gradient by s/s0 after the s update (p. 21).
- Boundary conditions: covered. BC points carry pointwise weights in the same mean-one pool as PDE points (Eq. 16, p. 15), and BRDR+ can pin a component with a constant, for example alpha_B = 100 for Helmholtz BCs and alpha_I = 100 for the Allen-Cahn initial condition (pp. 23, 28 to 29).
- Benchmarks: PINN on 2D Helmholtz, 1D Allen-Cahn, 1D Burgers, versus fixed weights, soft attention (SA) and residual-based attention (RBA), Section 4 (pp. 22 to 39). PIDeepONet on 1D wave and 1D Burgers, versus fixed weights, NTK and conjugate kernel (CK) weights, Section 5 (pp. 40 to 50). Setup: mFCN, tanh, Adam only, float32 (p. 22). The only steady case is the 2D Helmholtz BVP with Dirichlet BCs (p. 23). Note: printed Eq. (29) repeats N_R in the BC term denominator; Eq. (15) and Algorithm 1 show N_B is intended (pp. 15, 20, 21).

## Key results (author-reported)

- Toy Poisson with growing oscillation, k = 8: plain PINN error (1.27 plus or minus 1.03) x 10^0 versus BRDR (1.07 plus or minus 0.70) x 10^-2, Table 1 (p. 12).
- 2D Helmholtz: BRDR (1.73 plus or minus 0.28) e-4, BRDR+ (4.86 plus or minus 0.18) e-5, RBA (1.95 plus or minus 0.20) e-4, SA (4.40 plus or minus 0.61) e-4, fixed (2.95 plus or minus 0.61) e-3, Table 3 (p. 26). BRDR+ reaches RBA's final error in less than half the epochs (Section 4.1, p. 25).
- 1D Allen-Cahn: BRDR (2.51 plus or minus 0.44) e-5 and BRDR+ (1.45 plus or minus 0.46) e-5 versus RBA (2.92 plus or minus 0.78) e-5, Table 3 (p. 26); BRDR and BRDR+ reach RBA's final error in under half and under a third of the epochs (Section 4.2, p. 29).
- 1D Burgers: BRDR (1.38 plus or minus 0.85) e-4 versus SA (4.80 plus or minus 1.01) e-4, fixed (7.36 plus or minus 4.90) e-4, RBA (8.22 plus or minus 2.33) e-4, Table 3 (p. 26); BRDR reaches the others' final error in less than half the epochs (Section 4.3, p. 33).
- Operator learning, Table 4 (p. 44): wave, BRDR 0.92 plus or minus 0.21 percent versus NTK 1.43 percent, CK 0.72 percent, fixed 2.84 percent, at 106 percent of fixed-weight time versus 456 percent for NTK and 144 percent for CK. Burgers: BRDR best at nu = 1e-2 (0.26 percent), comparable at nu = 1e-3 (3.40 percent), worse at nu = 1e-4 (17.61 percent versus NTK 11.18 percent).
- Ablation on Allen-Cahn, Table B.5 (p. 57): the BRDR parts cut error 5 to 15 fold over a plain FCN baseline; pointwise weights alone 2 to 10 fold; scaling factor alone 1 to 3 fold; mFCN 3 to 50 fold; Fourier features 3 to 100 fold. Cases 5 to 7 show Fourier features without mFCN fail to converge, error near 1 (pp. 55 to 57).
- One hyperparameter pair (beta_c, beta_w) = (0.999, 0.999) served all three PINN benchmarks (Table 2, p. 24; Section 4.4, p. 35).

## Adopt for our pipeline

- B4 arm implementation: keep Rbar^4 per point for all 5 equation residuals and all 12 BC residuals, Eqs. (9) and (10), eps = 1E-14, beta_c = 0.999; update weights by Eqs. (19) and (20) with one shared mean-one pool over all points; init w = s = 1 (pp. 8, 15 to 16, 20 to 21). Full-batch Adam makes this the plain case, no gap-aware EMA needed.
- Run the arm in the Adam stage only. The paper has no L-BFGS evidence (p. 22), so our L-BFGS stage needs its own rule, for example frozen weights.
- Log per-component mean weight ratios, Eq. (34) (p. 25), and irdr distributions. This shows which of our 17 components the method sees as the bottleneck, the same diagnostic the paper uses in Figures 4 to 6 (pp. 27, 31, 34).
- Keep a BRDR+ sub-arm: if plain BRDR underweights the stiff wall BCs, pin those components with fixed constants, as the paper did with alpha = 100 for the dominant constraint (pp. 23, 28 to 29).
- Test the scaling factor as a separate arm, not bundled. The paper warns standalone use can degrade results and recommends it next to pointwise weights (p. 18; Appendix B, pp. 55 to 56). It applies to our Adam stage only.
- If we subsample points, use the gap-aware EMA of Eqs. (27) and (28) and raise beta to 0.9999 for many batches (p. 19).
- Expect under 10 percent time overhead, then verify on our own runs (pp. 26, 56).

## Do not import

- No hard cap on weights: their bounding comes from the mean-one normalization plus EMA, not a cap on each weight. Our B4 arm is specified as bounded per loss term, so we must add our own cap or monitor the max weight (pp. 16, 35).
- Their rejection of L-BFGS: they dropped it for cost and instability at their problem sizes (p. 22). We keep our Adam then L-BFGS plan; the paper says nothing about weight behavior in an L-BFGS stage.
- mFCN, Fourier features, float32, and their Adam schedules: tied to their benchmarks and their ablation (pp. 22 to 24, 52 to 57). Our network and optimizer plan is set separately.
- The claim that extension to many loss components is straightforward: stated for Algorithm 1 (p. 20), never demonstrated. With 17 components in one shared pool, quiet components may get starved. This needs our own test.
- The shock-region reasoning: the paper argues decay-rate weights avoid over-focusing a discontinuity (Section 4.4, p. 36). Our wall region is a stiff layer with growing coefficients, not a shock. The analogy is a hypothesis, not a transferable result.
- The DeepONet material: we train one steady BVP network, not an operator (Section 5).
- The beta_s = 1 - eta coupling: only meaningful if we adopt the scaling factor (p. 18).

## Quotes worth keeping

- "the slowest convergence rate among all residuals predominantly dictates the overall solution convergence to the true values" (p. 4, Section 1).
- "which helps ensure that the weights remain bounded and vary smoothly" (p. 16, Section 3.2).
- "Although Algorithm 1 is specifically employed for problems with two loss components (PDE loss and BC loss), its extension to multiple components is straightforward." (p. 20, Section 3.5).
- "Therefore, we have chosen not to use the L-BFGS optimizer." (p. 22, Section 4).
- "our approach adapts the weights based on the residual decay rate rather than the residual magnitude" (p. 36, Section 4.4).

## Relevance

- High. The paper is the direct reference for our B4 candidate arm: bounded, low cost, pointwise weights that treat boundary residuals in the same rule as equation residuals.
