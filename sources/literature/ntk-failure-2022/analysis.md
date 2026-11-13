# Analysis: NTK view of PINN training failure, 2022

**TL;DR.** NTK view: the residual kernel dominates the boundary kernel, and trace-ratio weights rebalance it. **Gives us:** the weight rule (sum of traces over Tr(Km)), the cost identity (one backward per term), the fixed-weight arm justification, and the instability warning for large weights. **Avoid:** NTK parameterization assumptions; weighting cannot fix spectral bias. **Relevance:** high.

## Record

- Title: "When and why PINNs fail to train: A neural tangent kernel perspective".
- Authors: Sifan Wang, Xinling Yu, Paris Perdikaris, University of Pennsylvania.
- PDF read: arXiv:2007.14527v1 [cs.LG], preprint dated July 30, 2020, 29 pages, all pages read. Page numbers below are the printed page numbers, which match the PDF pages.
- Venue: the published version is Journal of Computational Physics 449:110768 (2022) per the project record R25. The PDF itself is the arXiv preprint and carries no journal marking.
- Code: github.com/PredictiveIntelligenceLab/PINNsNTK (p.2, p.10).

## Problem

- Setting: a PINN loss has a boundary term and a residual term. Under gradient flow the output errors evolve as d/dt(outputs) = -K(t) errors, where K(t) is the NTK of the PINN (Lemma 3.1, eq 3.8, p.4).
- The blocks (eq 3.9, p.4): (Kuu)ij = <du(xi_b)/dtheta, du(xj_b)/dtheta>, (Kur)ij = <du(xi_b)/dtheta, dLu(xj_r)/dtheta>, (Krr)ij = <dLu(xi_r)/dtheta, dLu(xj_r)/dtheta>. The inner product runs over all parameters (Remark 3.2, p.4). K(t) = [Ju; Jr][Ju^T, Jr^T] (Remark 3.3, p.4).
- Theorem 4.3 and 4.4 (p.5-6): for a 1D Poisson model problem with a one-hidden-layer NTK-parameterized network, K(0) converges to a deterministic limit kernel and K(t) stays close to K(0) during training. Assumptions: bounded parameters, bounded cumulative errors, activation 4-times differentiable with bounded derivatives (p.6).
- Rates: diagonalizing K(0), the i-th training error component decays at rate e^(-lambda_i t) (eq 5.3, p.7-8). Components aligned with larger eigenvalues are learned faster (p.8). PINN NTK eigenvalues decay fast and most sit near zero, so PINNs also show spectral bias (p.8, Figure 1 p.9).
- Imbalance: for the model problem the eigenvalues of Krr are much larger than those of Kuu, so the residual converges much faster than the boundary fit, which can block a correct solution (p.8). The direction is problem dependent; they note that in some cases Kuu can dominate Krr (p.8).
- They define an average convergence rate c = Tr(K)/n and say K1 dominates K2 if c1 >> c2 (Definition 5.1, p.8). The trace identity Tr(K) = sum of eigenvalues of Kuu plus sum for Krr splits the total rate by term (p.8).
- Wave equation case (p.14): the three blocks Ku, Kut, Kr move outward early in training then stay near static, and Kr and Kut dominate Ku. The ut and residual components then converge much faster than the boundary and initial-value fit, and the standard PINN collapses (p.14).
- Side note: from their experience the PINN NTK is always degenerate, so kernel-regression prediction from K* is not usable in practice (p.7).

## Method

- Weighted loss (eq 6.2, p.8): each term is scaled by lambda and normalized by its own batch size. The weighted kernel K(t) with rows lambda/N times the Jacobians drives the same error dynamics (eqs 6.3-6.4, p.9).
- Stability: from a forward Euler analysis, the learning rate should not exceed 2/lambda_max of the weighted kernel (p.9). Raising a weight is equivalent, in convergence rate, to raising that term's batch size (p.9).
- General nonlinear PDEs: the NTK may drift, so they track the weighted kernel during training and read convergence from its singular values, since it may be indefinite (p.9).
- Algorithm 1 (p.10): at iteration n compute lambda_b = Tr(K)/Tr(Kuu) and lambda_r = Tr(K)/Tr(Krr) from the current kernel, then take one gradient step. Weights start at 1 (p.10; p.14 states the same for the three-term case).
- Update frequency: every iteration or a user-set period, for example every 10 gradient descent steps (p.9). In the wave experiment they update every 1,000 iterations (p.14). The kernel is recomputed on the fly at those iterations, from the current parameters and the current batch (p.9-10, p.14). With resampled batches they argue eigenvalue changes stay negligible, so the scheme is stable (p.9).
- Cost: they compute traces of the NTK blocks instead of full eigendecompositions to save resources (p.9). At the 1,000-iteration update period they call the extra cost negligible (p.14). The paper gives no wall-clock times and says nothing about higher-order derivatives raising the cost.
- No stop-gradient and no bounding of the weights appear in Algorithm 1; the weights are plain scalars in the loss (p.10). They warn that large weights can make the weighted kernel indefinite, with imaginary eigenvalues and unstable training (p.12, p.16).
- M-term rule (eqs 7.13-7.15, p.14): lambda_m = (sum of all term traces) / Tr(K_m).
- Fixed-weight variant: because the NTK barely moves, weights computed from the NTK at initialization can be regarded as fixed during training (p.12).
- Benchmarks (p.10-15): 1D Poisson sin(pi x) for NTK convergence, 1D Poisson sin(4 pi x) for adaptive weights, 1D wave equation with a two-frequency solution. Networks use NTK parameterization and tanh: one hidden layer of width 100 for Poisson, five layers of 500 neurons for wave (p.10, p.14). Poisson runs use full-batch GD at lr 1e-5; the wave run uses Adam with default settings (p.11-13, p.14).

## Key results (author-reported)

- The paper reports that the relative change of weights and of the NTK goes to zero as width grows (10, 100, 500) over 10,000 full-batch GD steps at lr 1e-5, for one-layer and three-layer nets, and eigenvalues at step 0 and step 10,000 barely differ (Figures 2-3, p.11).
- Poisson sin(4 pi x), one layer, width 100, 40,000 GD steps: the paper reports baseline relative L2 error 2.40e-01 and 1.63e-03 with lambda_b = 100, lambda_r = 1, about two orders of magnitude better (Figure 4, p.13; p.12). lambda_b = 100 came from Tr(Krr(0))/Tr(Kuu(0)) (p.12).
- Manual sweep of lambda_b in [1, 500] over 10 trials: the paper reports a local error minimum near lambda_b = 100 that matches the theory value, and a large standard deviation at large lambda_b, blamed on imaginary eigenvalues of an indefinite kernel (p.12, Figure 5 p.13).
- Wave equation, Adam, 80,000 iterations: the paper reports baseline relative L2 error 4.518e-01 (above 40 percent) and 1.728e-03 with Algorithm 1, weights updated every 1,000 iterations (p.14, Figure 6 p.15).
- The paper reports that lambda_u and lambda_ut rise fast then stay near fixed while lambda_r stays near 1 (Figure 8a, p.15), and that the weighted eigenvalue spectra become balanced (Figure 8b, p.15).

## Adopt for our pipeline

- Map to our loss: 17 terms give 17 blocks. Term m has a block with entries <dr_m(x_i)/dtheta, dr_m(x_j)/dtheta>, the eq 3.9 structure (p.4). The adaptive arm of B4 can use lambda_m = (sum of all 17 traces) / Tr(K_m), the M-term rule (p.14).
- Cheap trace: we only need the trace of each block. Tr(K_m) = sum over points of the squared norm of that term's parameter gradient. This identity is our implementation note; the paper only says traces suffice (p.9). Cost is one backward pass per term per update, 17 total, with the weights held constant. At a 1,000-step update period this is small next to the Adam loop (their frequency choice, p.14); their every-10-steps option (p.9) is also worth timing.
- Fourth-order residual: the f'''' term sends each residual gradient through four nested eta-derivatives before the parameter backward pass. Autograd handles this, and the trace cost stays one extra backward per term. Their equations cover any operator (Remark 3.4, p.4; Remark 4.2, p.5), but their experiments stop at second order, so our fourth-order trace cost is a test item with no paper guarantee.
- Fixed arm of B4: compute the 17 traces once at initialization, freeze the weights, per their fixed-weight argument (p.12).
- Diagnostics to log: the 17 trace ratios over training, spectra of the 5 residual blocks versus the 12 boundary blocks at init and a few checkpoints, and weight trajectories. Their weights rise fast then plateau (p.15); a plateau is a healthy sign.
- Stability: for plain GD the learning rate must respect 2/lambda_max of the weighted kernel (p.9). The paper says nothing about L-BFGS, so keep adaptive weights in the Adam phase and run the L-BFGS phase on frozen weights unless we test otherwise.
- Guard: if any lambda_m explodes, cap it. The paper describes the indefinite-kernel failure mode (p.12, p.16) but gives no clipping rule; the cap is our addition.

## Do not import

- NTK parameterization (eqs 2.1-2.3, p.2-3): their theory and all their runs use it. Our standard layers are not NTK-parameterized, so the constancy theorem does not transfer and we use the weighting as an empirical heuristic.
- The "NTK stays constant" claim: proven only for one hidden layer and a linear model problem (p.5-6, p.9); the authors call the general case a conjecture (p.11). Do not cite it as established for our nonlinear five-field system.
- Kernel regression from K* (eq 4.10, p.7): they report the PINN NTK is degenerate in practice, so this route is closed.
- Their lr 1e-5 and full-batch GD settings (p.11-13): tied to their stability bound and batch sizes, not to our Adam then L-BFGS plan.
- The assumption that the residual block always dominates the boundary block (p.8): they say the direction is problem dependent, so measure our traces instead of assuming.
- Any claim that NTK weighting fixes spectral bias: they state the opposite (p.16). If our BVP has stiff near-wall structure, weighting alone will not fix it.

## Quotes worth keeping

- p.8: "Another key observation here is that the eigenvalues of Krr are much greater than Kuu, namely Krr dominates Kuu by definition 5.1."
- p.8: "In other words, the eigenvalues of the kernel characterize how fast the absolute training error decreases."
- p.9: "To compute the sum of eigenvalues, it suffices to compute the trace of the corresponding NTK matrices, which can save some computational resources."
- p.14: "In particular, we update these weights every 1,000 training iterations, hence the extra computational costs compared to a standard PINNs approach is negligible."
- p.16: "such methods cannot change the distribution of eigenvalues of the NTK and, thus, cannot directly resolve spectral bias."

## Relevance

High. This paper gives the exact trace-ratio adaptive weight rule our B4 comparison needs, and its trace-only, sparse-update cost story suggests the scheme is affordable for our fourth-order residual, though untested above second order.
