# Analysis: Hard-constrained Falkner-Skan PINN, 2026

**TL;DR.** Closest structural precedent: one third-order BVP on a truncated domain, boundary lifting removes all three BCs, regional residual balancing, staged Adam then monitor-checked L-BFGS. **Gives us:** the B6 endpoint lifting and mask, the regional weighting safeguards, monitor-based rollback. **Avoid:** their L=12 and band cuts, and the single-seed gain numbers. **Relevance:** high.

## Record

- Title: "Hard-Constrained Physics-Informed Neural Network with Adaptive Regional Residual Balancing for the Generalized Falkner-Skan Problem" (p. 1).
- Authors: Mehari Fentahun Endalew and Xiaoming John Zhang, BIMSA, Beijing (p. 1). Same group as our R3 entry, per our reading list.
- Status: arXiv preprint 2609.23493v1, physics.flu-dyn, 20 Sep 2026. No journal named (p. 1).
- Pages read: all 30. Text extraction was clean. I also checked eqs. 23 to 43 and Tables 1 and 2 against the PDF page images; they match.

## Problem

- One third-order nonlinear ODE: f''' + beta0 f f'' + beta1 (1 - (f')^2) = 0 on eta in [0, L] (eq. 1, p. 3).
- Three BCs: f(0)=0, f'(0)=0, f'(L)=1 (eq. 2, p. 3). The wall shear f''(0) and the terminal value f(L) are not prescribed; the equation must produce them (pp. 3, 6-7).
- Domain truncated at L=12 in every run (p. 3). The far field is the exact endpoint condition f'(L)=1, not an asymptotic decay (p. 3). The paper does not test other L values and flags L=12 adequacy as open for slower-decaying regimes (p. 27).
- Cases: Blasius (0.5, 0), favorable pressure gradient (0.75, 0.50), and Pohlhausen (0, 1), which has an analytical semi-infinite solution (eqs. 6 to 8, p. 3).

## Method

- Trial representation (eq. 26, p. 6): f_hat = b(eta) + c_theta H(s) + G(s) N_theta(x), with s = eta/L and x = 2s - 1 (eq. 21, p. 5).
- Lifting (eq. 19, p. 5): b(eta) = (eta - (1 - e^-eta)) / (1 - e^-L). A general form for f(0)=A, f'(0)=B, f'(L)=C is given in eq. 15 (p. 5).
- Hermite mode H(s) = 3s^2 - 2s^3 with H(1)=1 and zero endpoint slopes (eqs. 22 and 24, pp. 5-6). Mask G(s) = L s^2 (1-s)^2 with zero value and slope at both ends (eqs. 23 and 25, p. 6).
- Result: f_hat(0)=0, f_hat'(0)=0, f_hat'(L)=1 hold for all parameters (eq. 31, p. 6). No BC penalty terms enter the loss (p. 7).
- Far field: f'(L)=1 sits inside the lifting. The unprescribed f(L) stays free since f_hat(L) = b(L) + c_theta (eq. 39, p. 7). Wall curvature stays trainable (eq. 38, p. 7).
- The mask keeps nonzero second derivatives at the endpoints (eq. 32, p. 6), so the curvature freedom a third-order equation needs survives.
- Network [1, 100, 100, 1], tanh, 64-bit floats, CPU (eq. 20, p. 5; Table 1, p. 17). Final layer starts at zero and c_theta starts at 0, so training starts from the pure lifting b (eq. 41, p. 7).
- Residual: pointwise R from AD derivatives f', f'', f''' (eq. 43, p. 7). Regions: wall [0,3], middle (3,8), tail [8,12] (eq. 44, p. 8).
- Physical global residual: L_g = (1/4) L_w + (5/12) L_m + (1/3) L_t from region-length fractions, not point counts (eqs. 46 to 49, p. 8).
- ARRB rule: proxy parameters are the last layer plus c_theta. G_r = norm of the gradient of L_r over that subset (eq. 53, p. 9). Invalid norms (non-finite or below eps=1e-12) are replaced by the median of valid ones, or 1 if none are valid (p. 9). Then an EMA with rho=0.90 (eq. 55, p. 9), normalized inverse gradients q_r (eq. 56, p. 9), and a blend lambda_r = 0.8 q_r + 0.2/3, so each lambda_r is at least 0.0667, renormalized to sum 1 (eqs. 57 to 59, p. 9). The blend floor is the bounding; no stop-gradient is used.
- Adam objective: 0.5 L_g + 0.5 sum lambda_r L_r (eq. 61, p. 10). The physical fractions stay in the mix through w_eff = alpha omega_r + (1-alpha) lambda_r (eq. 63, p. 10). Weights update at steps 1, 51, 101, ... and freeze during L-BFGS (pp. 10-11).
- Sampling: 500 fresh stratified points per Adam step, 125/208/167 by largest remainder, with eta=0 and eta=L always included (pp. 8-9). No residual-based point enrichment (p. 9; Table 1, p. 17).
- Schedule: 15000 Adam steps, lr 1e-3, global gradient-norm clip at 1 (eqs. 68 to 70, p. 11). Monitor every 100 steps on a fixed 800-point grid; plateau scheduler (factor 0.5, patience 20 evaluations); one-time lr cut to max(0.2 lr, 1e-5) at the first monitor value at or below 1e-6 (eqs. 72 to 76, pp. 11-12). Best-monitor checkpoint restored.
- Then two L-BFGS stages on 2000- and 4000-point uniform grids, N_j = max(2000 j, 4 Nc), at most 500 iterations and 1000 evaluations per stage, history 50, strong Wolfe line search (eqs. 78 and 79, p. 12). L-BFGS minimizes L_g only (eq. 80, p. 12).
- A stage is kept only if the independent 800-point monitor does not get worse; otherwise the pre-stage model is restored (eq. 81, p. 12). Full sequence in Algorithm 1 (p. 13).
- Reference: SciPy BVP solver on the same L=12 domain, tolerance 1e-10, mesh from 1200 points, up to 30000 nodes, start guess f' near 1 - e^-eta, several wall-shear guesses if needed (p. 14). Final metrics on 5000 uniform points (p. 16).

## Key results (author-reported)

- Selected case (0.75, 0.50): f''(0) = 0.8997161394 vs finite-domain BVP 0.8997168085, wall error 6.691e-7, residual MSE 6.357e-9, max residual 7.540e-4, relative L2 in f' 2.848e-6 and in f'' 4.188e-5 (Table 2, p. 17; sec. 7.2, pp. 19-20).
- Blasius wall error 1.106e-7. Pohlhausen wall error 8.251e-6 against the analytical 2/sqrt(3) benchmark, with profile errors against the finite-domain BVP (Table 2, p. 17; sec. 5.2, p. 15).
- Hard BC diagnostics are exactly zero for all three cases (eq. 112, p. 18).
- Restored adaptive weights for the selected case: (0.7276, 0.1816, 0.0908); effective Adam coefficients including physical fractions are about (0.4888, 0.2991, 0.2121) (eqs. 129 and 130, p. 20).
- Single-seed ablation (seed 42, Table 4, p. 22): ARRB cuts residual MSE by 52.81% vs global weighting and 50.24% vs equal-regional; wall-shear error by 77.30% and 72.37% (p. 23). ARRB is not first to the 1e-6 monitor threshold: equal-regional at step 1500, global at 1800, ARRB at 2000 (p. 23).
- Rollback fired in real runs: the second L-BFGS stage was rejected for Pohlhausen and for both ablation baselines (p. 24).
- The paper does not discuss non-uniqueness of the Falkner-Skan family. Its failure handling is the rollback rule, and its stated sensitivity limits are single seed, prescribed region cuts at 3 and 8, untested L=12 adequacy, no soft-boundary comparison, and BVP tolerance not being an error certificate (p. 27).
- CPU wall times: 376 to 522 s per run, single runs, not repeated for timing statistics (eqs. 148 and 149, p. 25).

## Adopt for our pipeline

- Exact endpoint construction for our simple conditions: pure value or slope conditions at eta=0 and eta=L for f, g, theta, phi, chi can use the lifting of eq. 15 plus the mask of eq. 23 (pp. 5-6). This matches our planned exact endpoint construction and removes those penalty terms.
- Keep derivative freedom at the endpoints: their mask keeps nonzero second derivatives there (eq. 32, p. 6). Our f equation has a fourth derivative, so test the mask's higher-derivative behavior before adopting it for f.
- Test the region-wise residual balancing as one controlled comparison against our planned adaptive loss weights. Copy the safeguards: the 0.0667 blend floor, the median fix for invalid norms, and the EMA smoothing (p. 9).
- Keep two weight sets, as they do: physical fractions for monitoring, checkpointing, L-BFGS and final scoring; adaptive weights only inside Adam (pp. 8-10). This fits our separate train, endpoint, monitor and final-eval point sets.
- Use a deterministic fixed-grid monitor for checkpoint selection and for L-BFGS stage rollback (pp. 11-12). Directly testable with our monitor set.
- Fresh stratified batches per Adam step with forced endpoint points, instead of residual enrichment (pp. 8-9). A candidate alternative for our stiff wall band: sample it densely by region and let inverse-gradient weights do the balancing.
- The one-time lr cut at first threshold crossing plus plateau decay (pp. 11-12) is a cheap add-on to our Adam then L-BFGS plan.
- Report both lambda_r and the effective coefficient alpha omega_r + (1-alpha) lambda_r; the paper shows these orderings differ (p. 20). We should do the same for any regional weighting we test.

## Do not import

- Their L=12 and region cuts at 3 and 8. Our wall layer sits near eta of order kappa = 0.1 with coefficients growing like inverse powers of (eta + kappa), so band edges must come from our own solution scale (p. 27; our setup).
- Direct hard construction for our coupled conditions. Their BCs are linear and single-field. Our radiation condition couples theta and phi nonlinearly, and slip and flux conditions carry field-dependent coefficients. A lifting form will not cover those without new algebra.
- Their proxy-gradient subset (last layer plus one scalar). With five coupled outputs, gradient imbalance spans fields as well as regions, so the subset choice needs its own test.
- Their network size [1, 100, 100, 1]. Ours is 4 hidden layers of 64 with five outputs; keep ours and treat width as a separate axis.
- Their gain figures as expected gains for us. The ablation is one seed (p. 23, p. 27), so the 52.81% and 77.30% numbers are a case study, not an estimate.
- Their single-residual balancing design. They split one residual in space; we have five residuals, so region-wise and field-wise balancing interact and need a new rule.

## Quotes worth keeping

- "No boundary-condition penalty terms are included because eq. (31) holds analytically." (p. 7)
- "The physical global residual is therefore not computed as the unweighted mean over the complete stratified stochastic batch." (p. 8)
- "The adaptive coefficients are optimization-balancing quantities. A larger λr does not imply greater physical importance." (p. 22)
- "Thus, for this seed, the lower final residual achieved by ARRB is already apparent at the Adam checkpoint stage and is not produced solely by the subsequent common L-BFGS refinement." (p. 23)
- "This separation reduces the risk of accepting a refinement that improves the optimization grid while degrading the prescribed monitoring grid." (p. 27)

## Relevance

- High. It is the closest structural precedent for our truncated-domain BVP PINN: exact endpoint construction for simple conditions, spatial residual balancing for a stiff near-wall band, staged Adam and L-BFGS with monitor-based rollback, and a classical reference on the same domain that never enters training.
