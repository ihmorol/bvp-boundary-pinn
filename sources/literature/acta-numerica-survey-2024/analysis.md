# Analysis: Acta Numerica PINN error analysis survey, 2024

**TL;DR.** Error analysis survey: small loss does not imply small error. **Gives us:** the evaluation rationale (E is not computable during training, p. 657; error scales about like the square root of training loss, p. 687), the conditioning view of our fourth-order residual, and the hard-versus-soft BC conditioning comparison. **Avoid:** dimension-dependent theory and probabilistic bounds with unestimable constants. **Relevance:** high.

## Record

- Title: Numerical analysis of physics-informed neural networks and related models in physics-informed machine learning.
- Authors: Tim De Ryck, Siddhartha Mishra.
- Year: 2024. Venue: Acta Numerica 33, pp. 633-713. DOI: 10.1017/S0962492923000089. Open access, CC BY.
- Pages read: all 81 pages (633-713). Full text extracted and read section by section, including the appendix on Sobolev spaces and the reference list. I spot-checked the theorem statements that matter most for us against the PDF at journal pages 657, 658, 675, 680 and 687. Long proofs were skimmed once their role was clear; most proofs in the survey are short sketches or are delegated to the cited source papers.

## Problem

- Scope: a unified numerical analysis of PINNs and related physics-informed models (VPINNs, wPINNs, DeepRitz, operator models) for forward and inverse PDE problems (Section 1, p. 634; Section 2, pp. 635-656).
- The survey frames everything through four questions (pp. 656-658): Q1 approximation, does the model class contain a network with small physics loss; Q2 stability, does small loss imply small true error; Q3 generalization, does small training loss imply small full-domain loss; Q4 optimization, can the loss be minimized well.
- The target error is E(theta) = ||u - u_theta||_X, usually the L2 norm (3.1, p. 656). The training error is the discretized loss (3.2, p. 657). The generalization error is the integral form of the loss (3.3, p. 657).
- Main themes: the total error splits into approximation error, generalization gap and optimization error (3.7, p. 658). Stability of the PDE and regularity of the solution decide whether small residuals mean small errors. Training is named the key bottleneck (abstract, p. 633; Section 8, p. 706).

## Method

- The survey proves no single big theorem. It organizes existing estimates in one abstract setting L[u] = f, B[u] = g (2.1, p. 636) and instantiates them per PDE: heat, Navier-Stokes, conservation laws, Poisson (Section 2.1, pp. 636-640).
- Loss functions: strong, weak and variational residuals (Section 2.3, pp. 646-651). Soft boundary conditions are penalty terms with weights lambda_s, lambda_t (2.45)-(2.46, p. 647). Systems of PDEs are handled by summing squared residuals (Remark 2.9, p. 648). Quadrature turns the loss into the trainable objective (2.68, p. 653).
- Error bounds: approximation results in Sobolev norms (Section 4, pp. 659-673), stability bounds with explicit constants per PDE (Section 5, pp. 674-684), quadrature and probabilistic generalization bounds (Section 6, pp. 684-690).
- Training analysis: linearized gradient descent, the matrix A, its condition number, and a preconditioning view of known fixes (Section 7, pp. 691-705).
- Boundary penalty treatment: the boundary residual enters every stability bound through a trace-type term, and its weight lambda is analysed as a conditioning parameter (Section 7.5.1, p. 703).

## Key results (author-reported)

- Approximation in the right norm: sup-norm approximation does not control the PDE residual because derivatives can be wrong (pp. 659-660). Theorem 4.4 (p. 661) gives tanh networks W^{k,inf} rates ln(cN)^k N^{-(s-k)/d} on [0,1]^d, so small residuals are attainable when u is smooth enough.
- Stability: if L is linear, or nonlinear with a linearization that has a bounded inverse, then ||u - v||_X <= C_PDE ||L(u) - L(v)||_Y (5.1, p. 674; Theorem 5.1, p. 675). C_PDE depends on the solution norms.
- Theorem 5.3 (p. 675, semilinear heat): ||u - v||^2 <= C1 (||R_PDE||^2 + ||R_t||^2) + C2 ||R_s||_{L2(boundary)}. C1 carries a Gronwall factor e^{(1+2C_f)T}; C2 involves the C1 norms of u and v on the boundary. The interior residual enters squared, the boundary residual at power one.
- Theorem 5.9 (pp. 677-678, viscous scalar conservation law): the stability constant contains |f''| times ||u_x||_inf. As viscosity goes to zero the solution norm grows like 1/sqrt(nu), the constant blows up, and the strong-residual bound no longer controls the error; the weak entropy residual restores a usable bound for shocks (Theorem 5.10, p. 678).
- Theorem 5.14 (p. 680, reported from Wang et al. 2022a): a negative result for high-dimensional HJB equations. For any p <= d/4 there is a smooth function with arbitrarily small Lp residual, matching boundary values, yet arbitrarily large W^{m,r} error. Stability needs an Lp residual norm with p ~ d.
- Example 5.15 (pp. 680-681): cPINNs are stable for some PDEs; XPINNs are not stable for Poisson or heat.
- Theorem 6.1 (p. 684): for deterministic quadrature, E_G(theta)^2 <= E_T(theta)^2 + C_quad N^{-alpha}.
- Theorem 6.2 (pp. 685-686): an a posteriori probabilistic bound, expected generalization error <= expected training error plus sqrt(2c^2(n+1)/N) ln(R L N). For PINNs the Lipschitz constant of the loss grows like (k+ell) times depth times ln(dRW), so higher derivatives make the bound worse; the constant c often grows exponentially in depth (p. 686).
- Theorem 6.5 (pp. 686-687, Navier-Stokes): E(theta)^2 <= O(E_T(theta,S) + N_t^{-2/d} + N_int^{-1/(d+1)} + N_s^{-1/d}). The survey reports E(theta) ~ sqrt(E_T) for large training sets, observed in Figures 6.1 and 6.2 (pp. 688-689).
- Corollary 7.2 (p. 693): if a global minimizer exists and the network and collocation sets are large enough, the Navier-Stokes L2 error is O(eps), an a priori statement that assumes Q4 is solved.
- Theorem 7.9 with Theorem 7.10 (pp. 696-697): gradient descent converges at rate (1 - c/kappa(A))^k, so it needs O(kappa(A) ln(1/eps)) steps. kappa(A) >= kappa(A composed with TT*), where A = L*L is the Hermitian square of the differential operator. With boundary terms, A = 1_Omega L*L + lambda 1_dOmega Id (Remark 7.11, p. 697).
- Theorem 7.12 (pp. 699-700): for Poisson with Fourier features, kappa(A) >= K^4 without preconditioning; rescaling by the eigenvalues of L*L makes kappa bounded and close to 1. For linear advection, kappa grows like beta^2 in the wave speed (p. 701).
- Section 7.5.3 (p. 705): for linear models, or networks in the NTK regime, the Hessian of the loss equals A (7.43). Quasi-Newton methods such as L-BFGS therefore approximate the inverse of A and act as preconditioners.
- Sections 7.5.1-7.5.2 (pp. 703-704): the boundary weight can be picked to minimize kappa(A(lambda)). In a toy Poisson example, optimal soft BCs give kappa about 5.83 while hard BC variants give 4 and 1, so hard imposition conditions better.

## Adopt for our pipeline

1. Evaluation design. Report per-field errors of f, g, theta, phi, chi against the converged solve_bvp reference, in L2 over [0, L] and in max norm. The survey defines the target error as ||u - u_theta|| (3.1, p. 656) and states it cannot be computed during training (p. 657). Loss values are training error, not solution error.
2. Residual-to-error reasoning. Treat our summed weighted penalties as E_T (3.2, p. 657). In the write-up, present the loss-to-error link as a stability bound of Theorem 5.3 type (p. 675): error <= C times interior residual and BC terms, with C depending on solution regularity and problem data. The exact constant shape is PDE specific, so for our nonlinear BVP we report the structure and the measured errors, never a computed C.
3. Sublinear expectation. The survey reports E(theta) ~ sqrt(E_T) when collocation sets are large (p. 687, Figures 6.1-6.2). Use it as a sanity check: a 100x drop in loss should give at most about a 10x drop in field error.
4. System handling. Remark 2.9 (p. 648) says systems are handled by summing squared residuals; our five residuals and twelve BC penalties with fixed scales fit the framework of (2.45)-(2.46) (p. 647) directly.
5. Optimizer choice. Our f residual has order 4, so L*L has order 8 and kappa(A) is expected to be large (Section 7.3, pp. 696-697). Adam then L-BFGS is supported by the survey's reading of quasi-Newton as Hessian preconditioning (p. 705). Cite this when we justify the schedule.
6. BC weights as diagnostics. Keep our fixed scales, but log them and read them through kappa(A(lambda)) (7.33, p. 703) and the hard-versus-soft comparison (p. 704). If training stalls while the interior residual is small, BC weight scaling is the first suspect inside our fixed plan.
7. Collocation counts. Boundary points can be fewer than interior points relative to their measure; the survey reports different interior-to-boundary ratios per PDE (Theorem 6.5, p. 687; Theorem 6.7, p. 688).

## Do not import

- Curse of dimensionality and Barron space theory (Section 4.3, pp. 665-673): our input is one dimensional, so CoD results add nothing.
- Operator learning theory (DeepONet and FNO, Theorems 4.22 and 4.27): we approximate one fixed solution, not an input-output operator.
- Weak and variational formulations (wPINNs, VPINNs, DeepRitz, Sections 2.3.2-2.3.3): our equations are smooth and Eqs. 14-19 are fixed by project rule; switching formulations would change the model.
- Inverse problem machinery (conditional stability, Carleman estimates, Theorems 5.17-5.23, pp. 681-684): we impose all twelve BCs, so there is no unique continuation step.
- Probabilistic generalization bounds (Theorems 6.2 and 6.9): their constants c and the Lipschitz bound are not estimable for our trained network, and the classical reference gives direct error measurement instead.
- Preconditioning implementations (Fourier feature rescaling, domain decomposition, causal training, Sections 7.4-7.5.4): outside our fixed pipeline, since our scales and architecture are set by plan. Record as future work only.

## Quotes worth keeping

1. "Note that it is not possible to compute E during the training process." (p. 657)
2. "obtaining a small generalization error E_G is a priori meaningless, given that we only care about the total error E*." (p. 657)
3. "the total error scales with the square root of the training error, E(θ) ∼ √E_T(θ, S). This sublinear relation is also observed in practice" (p. 687)
4. "Thus the complexity in training physics-informed machine learning models is entirely due to the spectral properties of the Hermitian square A of the underlying differential operator L." (p. 697)
5. "We identify training errors as a key bottleneck which can adversely affect the overall performance of various models in physics-informed machine learning." (p. 633)

## Relevance

High. The survey is the direct theoretical anchor for our evaluation design: it explains what loss values can and cannot say about true error, and it explains why a fourth-order residual makes training the hard part.
