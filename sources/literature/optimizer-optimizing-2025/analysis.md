# Analysis: Optimizing the optimizer for PINNs, 2025

**TL;DR.** Compares quasi-Newton optimizers for PINNs. **Gives us:** Strong Wolfe with c1=1e-4 and c2=0.9, a full-memory BFGS arm (L-BFGS was the worst quasi-Newton in its one test, 2.05e-3 versus 7.57e-8), the rescaling rule, and float64 for refinement. **Avoid:** SSBroyden as the default (dense-Hessian evidence); the no-enhancements claim is not clean. **Relevance:** high.

## Record
- Title: "Optimizing the Optimizer for Physics-Informed Neural Networks and Kolmogorov-Arnold Networks" (p. 1).
- Authors: Elham Kiyani, Khemraj Shukla, Jorge F. Urban, Jerome Darbon, George Em Karniadakis (p. 1).
- Status: preprint, arXiv:2501.16371 v6, stamped 8 Feb 2026 (p. 1). Not peer reviewed. Code repo named on p. 1.
- Pages read: all 43. The text layer covered every page; tables and key equations were legible, so no page needed a visual re-check. Spot checks confirmed printed page numbers match PDF pages.

## Problem
- The paper targets the optimization error, the third of three errors (approximation, estimation, optimization) it lists from prior work (Sec. 1.1, pp. 2-3).
- It argues the PDE term makes the loss landscape crinkled, with local minima and saddle points that trap first-order training (Fig. 1, pp. 3-4).
- The stated goal: show that better quasi-Newton optimizers, not loss reweighting, close most of the gap (Sec. 4, p. 27).

## Method
- Variants compared: BFGS, SSBFGS, SSBroyden, L-BFGS with Strong Wolfe line search; BFGS with backtracking; BFGS and SSBFGS with trust-region; baselines Adam, RAdam, Lion, AdaBelief, and Hessian-free nonlinear CG with Polak-Ribiere updates (Sec. 2, pp. 7-23; Table 14, p. 23). SOAP appears only through numbers quoted from its own paper (Table 16, p. 29).
- SSBFGS: multiply the Hessian approximation by a scalar tau_k before the BFGS update, Eq. (6), p. 6. Aim: lower the condition number of the curvature product H_k times the true Hessian times H_k (p. 6).
- Rescaling rule: tau_k = min{1, 1/b_k}, with b_k = (s_k^T B_k s_k)/(y_k^T s_k), credited to Al-Baali (p. 6). The pair (s_k, y_k) comes from past iterations, so the update rescales itself with historical gradient data (Abstract, p. 1).
- SSBroyden: the same self-scaling applied over the whole Broyden family, with the theta_k mixing parameter, Eqs. (3), (7), (8), pp. 5-6.
- Line search settings: Strong Wolfe with c1 = 1e-4 and c2 = 0.9 (p. 30). Backtracking uses Armijo only, shrinking alpha by a factor rho (p. 30). Trust-region uses an adaptive radius for PDE runs (Appendix A, pp. 31-32).
- Two-stage training: 1000 Adam steps at learning rate 1e-3, then the quasi-Newton stage; some cases start quasi-Newton directly (Sec. 2.1.1, p. 7).
- Stopping: gradient-norm tolerances, e.g. ATOL = RTOL = 1e-8 or a 30000-step cap (Table 4, p. 11), plus optional stops on loss change, step change and max iterations (Appendix A, p. 31).
- Benchmarks: Burgers, Allen-Cahn, Kuramoto-Sivashinsky, Ginzburg-Landau, steady Stokes flow in a lid-driven wedge, the Lorenz ODE system (Appendix C, pp. 33-35), DeepONet fracture data (Sec. 3, pp. 24-28), Rosenbrock in 2D to 20D (Appendix D, pp. 36-37). Kuramoto-Sivashinsky carries a fourth-order spatial derivative, Eq. (12), p. 15, handled by plain autodiff.
- Metric: relative L2 error against a spectral or finite-element reference, plus training seconds and parameter counts (Tables 1-15). No benchmark is a steady ODE BVP; the closest are steady Stokes, a 2D PDE system (Sec. 2.6, pp. 23-25), and Lorenz, a first-order ODE initial value problem (Appendix C, pp. 33-35).
- Costs: BFGS-type methods here carry a dense inverse Hessian. The largest PINN has 33,667 parameters and 120,000 residual points (Table 14, p. 23; Sec. 2.6, p. 24). The paper reports no memory settings for L-BFGS. Self-scaled runs can cost more time than BFGS, e.g. 973 s vs 733 s on Allen-Cahn (Table 8, p. 14).

## Key results (the paper reports; we did not verify)
- Burgers, double precision: SSBroyden 7.57e-8, SSBFGS 9.62e-8, BFGS 1.50e-5, L-BFGS 2.05e-3 relative L2 error, each after 1000 Adam steps plus 50000 quasi-Newton steps (Table 1, p. 8).
- Starting quasi-Newton directly, with no Adam, still gives 1.5e-7 with SSBFGS on Burgers (Case 3, Table 1, p. 8).
- At a fixed target of 1e-4 then 1e-5 error, SSBroyden needs 1,307 and 2,864 iterations against 5,621 and 69,332 for BFGS, over threefold less time (Table 3, p. 10).
- JAX BFGS runs: backtracking beats trust-region, 3.0e-6 vs 1.3e-5 after an 11-step Adam warmup (Table 4, p. 11). The verdict table picks SSBroyden with Wolfe as winner and RAdam plus BFGS backtracking as runner-up (Table 6, p. 13).
- Allen-Cahn double precision: SSBroyden 1.15e-6 vs BFGS 7.59e-4 (Table 8, p. 14). Kuramoto-Sivashinsky, 20 windows: 6.51e-4 vs 3.65e-2 (Table 10, p. 18). Ginzburg-Landau: 3.48e-4 vs 7.19e-3 (Table 12, p. 21).
- Stokes wedge: SSBFGS with trust-region reaches loss 1e-12, backtracking 1e-10, Adam 1e-5 (Sec. 2.6, p. 24). Only the trust-region run resolves all four Moffatt eddies, down to speed 1e-8 (Sec. 2.6, p. 25).
- DeepONet fracture: SSBroyden with a 10-step Adam warmup gives the best errors, about 5.7e-3 and 6.9e-3 for displacement and damage (Table 15, p. 28).
- Against SOAP, with SOAP numbers quoted from that paper: Burgers 4.19e-8 vs 8.06e-6, Allen-Cahn 9.43e-7 vs 3.48e-6, Kuramoto-Sivashinsky 2.65e-5 on [0,1] vs 3.86e-2 on [0,0.8], Ginzburg-Landau 2.33e-3 vs 4.78e-3 (Table 16, p. 29). One mismatch: the Table 16 header says SSBroyden, its caption says SSBFGS, and 4.19e-8 matches the SSBFGS Case 4 in Table 1 (pp. 8, 29).
- Lorenz, 40 windows: SSBroyden relative L2 errors 5.94e-5 to 1.40e-4, slightly better than BFGS at similar time (Table C.20, p. 35).
- Single precision degrades quasi-Newton training, and BFGS after a long Adam phase degenerates (Appendix B, Tables B.18 and B.19, pp. 34-35; Sec. 4, p. 28).
- L-BFGS appears in results only once, as the worst quasi-Newton on Burgers. The paper gives no history size or step scaling for it. A multi-batch L-BFGS test is mentioned once, with only "comparable accuracy" to full batch and no numbers (Table 1, p. 8; Sec. 4, p. 29).

## Adopt for our pipeline
- Strong Wolfe line search for our L-BFGS refinement with c1 = 1e-4 and c2 = 0.9. This matches our plan and every winning run in the paper (p. 30; Tables 1, 6).
- Add a full-memory BFGS arm next to our L-BFGS arm. The paper reports L-BFGS two orders worse than BFGS and about six orders worse than SSBroyden on Burgers (Table 1, p. 8). A five-output ODE BVP network is small, so a dense Hessian is affordable.
- Add a self-scaled arm using tau_k = min{1, 1/b_k} computed from the (s_k, y_k) pairs our L-BFGS already stores (p. 6). The paper tests self-scaling only on full-memory updates, so an L-BFGS version is our own experiment, not their result.
- Keep the two-stage plan: short Adam warmup, then quasi-Newton. Warmup is optional in double precision; direct quasi-Newton starts match it on Burgers and Kuramoto-Sivashinsky (Table 1, p. 8; Table 10, p. 18). Keep the Adam phase short either way.
- Use gradient-norm stops plus the loss-change and step-change checks of Appendix A with an iteration cap, instead of a fixed budget alone (p. 31).
- Run the refinement stage in float64. The paper reports single-precision failure modes for these optimizers (Appendix B, pp. 32-35; Sec. 4, p. 28).
- Keep loss weights fixed during refinement and log per-term losses, to test their claim that the optimizer, not the weights, is the bottleneck (Sec. 4, p. 27).

## Do not import
- SSBroyden or SSBFGS as the default. Their evidence sits on dense Hessians with up to 33,667 parameters (Table 14, p. 23). Test only if our L-BFGS stage stalls; our pipeline decision stays L-BFGS.
- Trust-region. Mixed evidence: worse than backtracking for BFGS on Burgers (Tables 4-5, pp. 11-12), better for SSBFGS on Stokes (Table 14, p. 23). Extra complexity for an unclear gain.
- RAD resampling or residual-based attention inside optimizer benchmarks. The paper used RAD in several benchmarks (pp. 7, 16, 19) and residual attention in the Stokes case (p. 24), so its "no enhancements" claim is not clean, and mixing these would blur attribution in our runs.
- Any L-BFGS history or step-scaling numbers from this paper. It reports none. Our history of about 50 stays our own untested choice.
- The Allen-Cahn curriculum on the equation parameter kappa (p. 13). Problem-specific, and our parameter values are supervisor-given.
- Time marching and Chebyshev KAN architectures. Our problem is one steady 1D BVP on one network.
- Batch quasi-Newton training. The paper flags instability of gradient-difference updates under changing mini-batches (Sec. 4, p. 29). We train full batch.

## Quotes worth keeping
- "dynamically rescale updates based on historical gradient information" (Abstract, p. 1).
- "orders-of-magnitude accuracy improvements without the use of adaptive weights or any other enhancements" (Abstract, p. 1).
- "the big bottleneck for PINNs or PIKANs is the optimization error" (Sec. 4, p. 27).
- "the relative errors achieved are 10−8 for SSBroyden, 10−5 for BFGS, and 10−3 for L-BFGS" (p. 8).
- "For all tests we set c1 = 10−4 and c2 = 0.9, which are the standard choices in the optimization literature." (p. 30).

## Relevance
- High. The paper measures the exact stage we plan, Adam then quasi-Newton with Wolfe line search, and warns that plain L-BFGS may be the weak link, while full BFGS and self-scaled variants reach far lower error on small networks like ours.
