# Analysis: hPINN hard constraints, 2021

**TL;DR.** AL rounds for inverse design; exact-by-construction covers only Dirichlet and periodic BCs. **Gives us:** the AL round schedule (warm starts, beta growth 2, lambda over mu flattening in 6 to 9 rounds). **Avoid:** the design-objective machinery; AL on BC residuals is untested there, so it would be our extension. **Relevance:** medium.

## Record

- Title: Physics-informed neural networks with hard constraints for inverse design (p. 1).
- Authors: Lu Lu, Raphael Pestourie, Wenjie Yao, Zhicheng Wang, Francesc Verdugo, Steven G. Johnson (p. 1).
- Year: 2021. arXiv:2102.04626v1, dated 9 Feb 2021 (p. 1).
- Venue (from project record R33): SIAM Journal on Scientific Computing 43(6):B1105-B1132, 2021. The PDF we read is the arXiv v1 preprint.
- Pages read: all 29 pages. Plain text extraction plus direct PDF reads of pp. 7-8 for the algorithm equations. Everything was readable; no gaps.

## Problem

- The paper targets PDE-constrained inverse design, not plain forward BVP solves. It minimizes a design objective J subject to PDE constraints, BC constraints and optional inequality constraints (Eq. 2.3, p. 4).
- Core difficulty: the objective and the PDE loss compete, so a soft-constrained PINN ends at an optimum that does not solve the PDE (Sec. 1, p. 2). Fixed penalty coefficients either leave constraints violated when small or make the loss ill-conditioned when large (Sec. 2.4, p. 6).
- Proposed fix: hPINN. Exact Dirichlet and periodic BCs by network construction, plus penalty or augmented Lagrangian (AL) handling of PDE and inequality constraints (Abstract, p. 1; Sec. 2.3, pp. 4-6).

## Method

- Three training modes on the same network setup: soft constraints with fixed mu (Eq. 2.5, p. 6), the penalty method (Algorithm 2.1, p. 7), and the AL method (Algorithm 2.2, p. 8). Penalty and AL are alternative algorithms, not sequential stages of one run.
- Every mode starts with round k=0, a soft-style training from random init with initial coefficients mu_F^0 and mu_h^0 (Algorithms 2.1-2.2, pp. 7-8).
- Penalty rounds: mu_F^k = beta_F mu_F^(k-1) and mu_h^k = beta_h mu_h^(k-1), beta > 1 (Sec. 2.5, p. 6). Each round warm starts from the previous weights and trains until the loss converges. The outer loop stops when both L_F and L_h fall below a tolerance (Algorithm 2.1, p. 7).
- AL loss (Eq. 2.7, p. 7): L^k = J + mu_F^k L_F + mu_h^k 1{h>0 or lambda_h^k>0} h^2 + (1/MN) sum_j sum_i lambda_{i,j}^k F_i + lambda_h^k h. Lagrangian terms use signed residuals, one multiplier per PDE per collocation point.
- Multiplier updates (p. 8): lambda_{i,j}^k = lambda_{i,j}^(k-1) + 2 mu_F^(k-1) F_i at the previous round's parameters; lambda_h^k = max(lambda_h^(k-1) + 2 mu_h^(k-1) h, 0). All multipliers start at 0. Note the update uses mu^(k-1), then mu is bumped to mu^k for the new loss (Algorithm 2.2, p. 8).
- Penalties still grow each AL round, but the paper states they no longer need to grow to infinity, which avoids ill-conditioning and converges better than the plain penalty method (Sec. 2.6, p. 8). Convergence signal: lambda^k/mu^k flattens after a few rounds (Sec. 2.6, p. 8).
- Hard BCs by architecture (Sec. 2.3, pp. 4-6): output becomes u_i = g(x) + l(x) N(x) with l = 0 on the Dirichlet boundary and l > 0 inside, for example l = (x-a)(b-x) on an interval. Periodicity comes from Fourier features on the input. These are exact and loss-free. Neumann, Robin and other BC types stay in the loss; the paper hard-codes only Dirichlet and periodic (p. 4).
- What makes a constraint "hard": the AL stage changes only the loss, never the network output construction (Sec. 2.6, pp. 7-8). Exactness by construction exists only for Dirichlet and periodic BCs. PDEs and inequalities become feasible only in the limit of the outer loop.
- Optimizers (Sec. 3.1.3, p. 11; Sec. 3.2.3, p. 19): Adam first, learning rate 1e-3 for holography and 1e-4 for Stokes, for 2e4 steps, then L-BFGS until the loss converges. This procedure runs inside every outer round.
- Benchmarks: 2D Helmholtz holography with PML (Sec. 3.1) and Stokes topology optimization with a fluid volume inequality (Sec. 3.2). Residual points from a Sobol sequence (Sec. 2.2, p. 4). Code in DeepXDE (Sec. 3, p. 8).

## Key results (author-reported)

All numbers below are the paper's own reports. We did not verify them.

- Forward holography check: a training loss near 1e-4 corresponds to L2 relative error below 1% against an FDFD reference (Sec. 3.1.3, p. 11, Fig. 3).
- Soft constraints, holography: best objective about 0.0547 at mu_F = 2. mu_F = 0.1 leaves PDE loss near 1e-1; mu_F = 10 reaches PDE loss near 1e-5 but the design fails through ill-conditioning (Sec. 3.1.4, pp. 11-14, Fig. 4).
- Penalty method, holography: mu_F must exceed 100 to push L_F below 1e-4. Best objective 0.0537 at outer step k=1, about 2% better than soft constraints. Cost: 7 training rounds, about 5.6e4 total iterations versus about 3.3e4 for soft, that is 70% more (Sec. 3.1.5, pp. 14-15, Fig. 5).
- AL method, holography: PDE loss below 1e-4; L2 relative error of |E|^2 is 1.2% for the final design; the objective stops worsening even for mu_F^k above 1e3. Normalized multipliers lambda^k/mu^k converge; the distributions are almost converged at k >= 6 (Sec. 3.1.6, pp. 14-17, Fig. 6).
- Tuned hPINN, holography: objectives 0.0517, 0.0525, 0.0528 from three random inits, versus 0.0523 for an FDFD adjoint method and 0.0528 for FEM. Same objective level, but the hPINN design is smoother and simpler (Sec. 3.1.6, p. 16, Fig. 7; Appendices B-C, pp. 27-28).
- Soft constraints, Stokes: final fluid volume 0.906 against the cap gamma = 0.9, that is 0.7% over. L2 relative error of |u| against a spectral element reference is 4.3%. Objective 13.73 versus 14.07 from the earlier MMA+FEM study; the paper says the lower value comes from the violated volume constraint (Sec. 3.2.3, pp. 19-21, Fig. 9).
- AL method, Stokes: about 400 extra iterations, about 2% more cost than the soft run. Volume loss drops from 1e-5 to 1e-8; final volume 0.901; L2 error of |u| improves to 2.0%; objective 14.12, consistent with the earlier study. The normalized inequality multiplier converges to 0.001 (Sec. 3.2.4, p. 21, Fig. 10).
- The paper reports no convergence theory for hPINN with AL and flags it as open work (Sec. 4, p. 22).

## Adopt for our pipeline

1. Outer loop around our current plan. Keep Adam then L-BFGS inside each round. Round 0 is our soft fixed-scale run; later rounds update multipliers and penalty scales and warm start from the previous weights (Algorithms 2.1-2.2, pp. 7-8).
2. Multipliers for our 12 BCs. Each BC is one constraint function c_a(theta) at one point, eta=0 or eta=L. The paper's rule reduces to lambda_a^k = lambda_a^(k-1) + 2 mu^(k-1) c_a(theta^(k-1)), so we would carry only 12 multipliers. The rule needs only signed residual values, so it extends in principle to our nonlinear coupled radiation conditions. The paper never tests AL on BC residuals, so this would be our extension (Eq. 2.7, p. 7; Sec. 3, pp. 8-21).
3. Same rule for the five PDE residuals: five multiplier vectors of length M over our eta collocation points, mirroring lambda_{i,j} for their N=2 and N=3 residual sets (Eq. 2.7, p. 7).
4. Penalty growth: mu^k = beta mu^(k-1) with beta = 2 and mu_0 near our current fixed soft scale. Their best cases used beta = 2 with mu_0 = 2 (holography) and mu_0 = 0.1 (Stokes) (Sec. 3.1.5, p. 14; Sec. 3.2.4, p. 21).
5. Stopping and diagnostics: stop the outer loop when both the PDE loss and the BC loss fall under tolerance, and track lambda^k/mu^k flattening. Convergence took 6 to 9 rounds in their runs (Algorithm 2.2, p. 8; Sec. 3.1.6, p. 16; Sec. 3.2.4, p. 21).
6. Scale every residual to order one first: they scale the Helmholtz operator by 1/omega and the Stokes residuals by 0.01, 0.01 and 100, then let AL tune relative weights per point on top (Sec. 3.1.2, p. 11; Sec. 3.2.2, p. 19). Our fixed penalty scales should play this role.
7. Loss-to-error calibration: build a mapping from training loss to solution error against the classical reference, used for checking only, like their Fig. 3 curve (Sec. 3.1.3, p. 11). This respects our rule that the classical solution never feeds training labels.
8. Warm starts keep the AL loop cheap: about 2% extra cost when the soft stage already nearly satisfies the constraints (Sec. 3.2.4, p. 21).
9. Output transforms for bounded quantities, if our setup grows any: sigmoid or clamped min/max on the output layer (Sec. 3.1.2, p. 11; Sec. 3.2.2, p. 19).

## Do not import

- The design objective J. We solve a feasibility problem, so our loss is pure constraint. Their objective-versus-PDE trade-off findings, like the mu_F = 2 sweet spot, do not map to our setup (Sec. 2.1, p. 3; Sec. 3.1.4, p. 13).
- Hard BC architecture for our 12 BCs. The g + l(x)N form handles value conditions only. It cannot impose slope, slip, flux or nonlinear coupled radiation conditions, and one l factor cannot satisfy 12 mixed conditions at two endpoints. A value condition at eta=0 could be tested this way; the rest cannot (Sec. 2.3, pp. 4-6).
- The inequality machinery. We have no inequality constraints, so the indicator term and lambda_h are dead weight for us (Eq. 2.7, p. 7).
- The two-network split for u and gamma. We have no design parameter network; our five fields come from one network (Sec. 2.2, p. 4).
- Sobol sampling and the 2D machinery (PML, Fourier periodic inputs). Our domain is 1D with two endpoints, so plain point sampling suffices (Sec. 2.2, p. 4; Sec. 3.1.1, p. 9).
- Their numeric targets. All reported objectives are holography or Stokes numbers, not comparable to our fields (Sec. 3, pp. 8-21).
- Trust in guaranteed convergence. The paper states it has no theory for hPINN AL convergence, so our adoption stays experimental (Sec. 4, p. 22).

## Quotes worth keeping

- p. 1: "all the constraints in PINNs are soft constraints, and hence we impose hard constraints by using the penalty method and the augmented Lagrangian method."
- p. 6: "when the penalty coefficients are too large, the optimization problem becomes ill-conditioned and hence makes it difficult to converge to a minimum"
- p. 7: "this good starting point helps to counteract the slow convergence that would otherwise arise"
- p. 16: "the augmented Lagrangian method solves the convergence issue in the penalty method"
- p. 22: "we used one multiplier for each residual location x, and thus many multipliers are required, which could be expensive and inefficient for large-scale problems"

## Relevance

Medium. The AL update rules, the two-stage schedule and the convergence diagnostics transfer directly to our five-field BVP, including a cheap 12-multiplier BC vector, but the paper applies AL only to PDE and inequality residuals in 2D inverse design, so AL on our nonlinear coupled BC conditions is our own extension, not a demonstrated result.
