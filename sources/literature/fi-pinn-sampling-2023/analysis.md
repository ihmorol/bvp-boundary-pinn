# Analysis: Failure-informed sampling, 2023

**TL;DR.** Enriches collocation points where a residual-based failure probability is high. **Gives us:** the B3 algorithm: union enrichment keeps coverage, stop on the failure probability, settings N1=300, p0=0.1, N2=1000, at most 10 rounds. **Avoid:** the error bound (needs a linear operator) and a Gaussian proposal that spills past the eta=0 wall. **Relevance:** high.

## Record

- Title: Failure-Informed Adaptive Sampling for Physics-Informed Neural Networks (FI-PINNs). Authors: Zhiwei Gao, Liang Yan, Tao Zhou (p. 1).
- Year: 2023. Venue: SIAM Journal on Scientific Computing 45(4):A1971-A1994. arXiv:2210.00279v3 (p. 1).
- Pages read: all 25 via text extraction. PDF pages 8 and 9 also read as images to confirm Assumptions 4.1 to 4.3, Theorem 4.4, Eq. 16 and Algorithm 2. Figure interiors on pages 13, 15, 17, 21 and 23 are garbled in plain text; captions and prose carry all claims used here.

## Problem

- A fixed collocation set can miss localized solution structure, so training fails there. The paper treats sampling as a reliability problem: find where the network "fails" and place points there (p. 1, p. 4).
- Failure at a point means the residual exceeds a threshold. The failure probability is the prior probability mass of that region (p. 4, Eq. 6).
- Link to error: Theorem 4.4 bounds the L2 solution error by the residual tolerance, the failure probability tolerance and the boundary error (p. 8, Eq. 16). A small failure probability implies a small error, under assumptions.

## Method

- Limit-state function: g(x) = |r(x; θ)| − εr, with εr a residual tolerance. Failure set ΩF = {x : g(x) > 0}, safe set otherwise (p. 4, Eq. 5).
- Failure probability: PF = ∫Ω ω(x) I_ΩF(x) dx under the prior ω. With uniform ω on a bounded domain this is the area ratio S_ΩF / S_Ω (p. 4 Eq. 6, p. 9 Eq. 18). The prior is uniform on bounded domains and Gaussian on unbounded ones (p. 3, Remark 1 p. 7).
- Plain Monte Carlo from ω needs candidate sets of O(10^4 to 10^6) because the failure region is small (p. 6). So the paper uses importance sampling with weight R(x) = ω(x)/h(x), Eq. 7 to 8 (p. 6), and builds h by self-adaptive importance sampling, SAIS (p. 6 to 8).
- SAIS steps (p. 7 Eqs. 10 to 12, Algorithm 2 p. 8): start h1 = ω. Draw N1 samples from hk, sort by g descending. Nη = number of samples with g > 0, Np = floor(p0 N1). While Nη < Np, refit a truncated Gaussian N_T(μ, Σ) from the top Np samples (Eq. 10). At stop, compute μopt and Σopt from those samples weighted by ω (Eq. 11). Draw N2 samples from this fitted density and estimate the failure probability by the ratio ω/h_opt (Eq. 12).
- Enrichment loop (Algorithm 1, p. 5): train, build g, estimate PF. If the estimate is below εp, stop. Else add D_adaptive = every new sample with g > 0 and set Dc = Dc ∪ D_adaptive. The base set stays, so coverage is preserved. Maximum M rounds, set to 10 (p. 11).
- Points per round: not fixed. Every drawn sample inside the failure region is added (Algorithm 2 line 17, p. 8). The paper frames RAR as the fixed-m alternative, taking m largest-g points from a big candidate set (p. 6).
- Default settings: N1 = 300, p0 = 0.1, N2 = 1000, M = 10 (p. 11). The unbounded time-dependent case uses N1 = 600, N2 = 2000, p0 = 0.05 (p. 18).
- Error bound (Theorem 4.4, p. 8): with A linear and C1||v|| ≤ ||Av|| + ||Bv|| ≤ C2||v|| (Assumption 4.1, Eq. 13), boundary error at most εb (Assumption 4.2) and bounded residual M (Assumption 4.3), the paper proves ||u − u*|| ≤ √2 C1^{-1} ( S_Ω (M² εp + εr²) + εb² )^{1/2} in L2 (Eq. 16).
- Benchmarks: Uniform (baseline PINNs) and RAR. Fair-comparison rule: one base model trained on a small uniform set, then three descendants with equal dataset sizes (p. 10). Evaluation: relative L2 error on random test points, Eq. 23 (p. 11). The paper does not document a frozen final evaluation set across rounds.
- Test problems: 2D Poisson with a sharp peak, viscous Burgers with a steep front, 9D Poisson, 2D unbounded Poisson, heat equation on R x [0,1], plus a causal Allen-Cahn variant (p. 11 to 22).

## Key results (author-reported)

- 2D peak Poisson, εp = εr = 0.1: mean relative L2 error 9.04e-2 for SAIS versus 7.36e-1 (Uniform) and 2.15e-1 (RAR). The failure probability estimate fell below tolerance after 4 updates (p. 12, Fig. 4).
- Observed convergence rates: error scales like εp^(1/2) and εr^1, matching Theorem 4.4 (p. 12, Fig. 7).
- Burgers, εp = εr = 0.01, L-BFGS then Adam then L-BFGS schedule (p. 12): SAIS error 5.59e-3, failure probability below tolerance after 4 updates (p. 12, Fig. 8). The same tolerance rates hold empirically on this nonlinear case (p. 14, Fig. 11).
- 9D Poisson, εp = 0.005, εr = 0.01: the paper reports Uniform and RAR "seems not converge" while SAIS error falls smoothly (p. 16, Fig. 12).
- 2D unbounded Poisson, εp = 1e-4: error 2.28e-2 after 4 updates (p. 17, Fig. 16).
- Time-dependent unbounded case, εp = 1e-10: relative error below 3e-3 after 3 updates, absolute error under 2e-3 (p. 18 to 19, Fig. 18).
- Causal Allen-Cahn (p. 22, Table 1 p. 24): causal FI-PINNs 6.50e-3 with fewer than 4000 points from an initial 1000 after 6 updates, against 2.42e-2 for causal PINNs, 5.03e-1 for plain FI-PINNs, 6.79e-1 for PINNs.
- Cost claim: SAIS needs only N1 plus N2 residual evaluations per round and "can be done in a very efficient way (compared to RAR and DAS)" (p. 2, p. 7). DAS instead trains a generative model whose cost can match solving the PDE (p. 2).

## Adopt for our pipeline

- B3 indicator: g(eta) = max_k |r_k(eta)| / s_k − εr, with s_k our fixed penalty scales, so the five equations of orders (4,2,2,2,2) are comparable. The paper uses Q = |r| for one equation and states the performance function can be swapped (Remark 2, p. 7).
- Enrich by union, not replacement: keep the base eta set and append failure points, as in Algorithm 1 (p. 5). This protects coverage of the stiff wall region where coefficients grow like inverse powers.
- Variable count per round: add every candidate with g > 0 from the fresh draw. This is the published difference from fixed-m RAR, and it is the version arm B3 should implement (p. 5 to 6, p. 8).
- Stop rule for B3: estimate the failure probability on a 1D candidate grid each round and stop enrichment when it drops below εp. The paper reports this estimate tracks the true L2 error on all its benchmarks (p. 12, 14, 16, Figs. 4, 8, 12).
- Settings to copy: N1 = 300 draws per proposal update, Np = floor(0.1 N1) quantile fit, N2 = 1000 evaluation draws, M = 10 rounds (p. 11). In 1D this costs about 1300 residual evaluations per round, cheap next to a retrain.
- Wall-region fit: in 1D the SAIS fit is a scalar mean and variance from the top decile of residual values, truncated to [0, L]. If the wall and the far field both fail, use two fits or the Gaussian mixture variant (Appendix A.1, p. 20).
- Validation plan: sweep εp and εr separately and plot error against each, as the paper does in Figs. 7 and 11 (p. 12, 14). This checks whether our indicator behaves like theirs before we trust the stop rule.

## Do not import

- Theorem 4.4 as a guarantee. Assumption 4.1 needs a linear operator (p. 8). Our five-equation system is nonlinear and fourth order, so the bound does not cover it. The paper itself only observes the rates on nonlinear Burgers (p. 14).
- The area-ratio identity for PF (Eq. 18, p. 9). It assumes a uniform prior on a bounded domain. Any non-uniform eta prior or near-wall weighting breaks the identity; estimate the probability by sampling instead.
- A symmetric Gaussian proposal centered on the failure region. Our failure region will hug eta = 0 where coefficients blow up, so a Gaussian fit there spreads mass past the boundary. Keep the quantile idea but handle the boundary, or use the mixture variant.
- Their training schedules (Adam 1e-4 for 10000 steps, p. 11; L-BFGS 50000 for Burgers, p. 12). These are problem-specific; we keep our own Adam then L-BFGS plan.
- Hard boundary constraints from the Allen-Cahn run (p. 22). Our design uses soft weighted penalties with fixed scales.
- Causal weighting (Appendix A.2, p. 21). It targets time-marching PDEs; our BVP has no time direction.

## Quotes worth keeping

- "the proposed FI-PINNs adopts the failure probability as the posterior error indicator to generate new training points" (p. 1, abstract).
- "If the failure probability is smaller than a given tolerance ϵp, we say that PINNs are reliable" (p. 4).
- "The residual-based adaptive refinement (RAR) method [15] proposed to select a fixed number of m new points with the largest values of LSF g in S." (p. 6).
- "In our experiments, we observe that by selecting p0 = 0.1, SAIS can self terminate rapidly with high numerical accuracy." (p. 7).
- "Although we have only established the theoretical results for the linear model. We indeed observe in Fig. 11 the convergence rates also hold for this nonlinear example." (p. 14).

## Relevance

- High. This paper defines the failure-informed enrichment that arm B3 tests, and it reports direct wins over uniform and RAR sampling with an indicator cheap enough for our 1D BVP.
