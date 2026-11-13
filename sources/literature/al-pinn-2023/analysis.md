# Analysis: Augmented Lagrangian BC relaxation, 2023

**TL;DR.** Treats BCs as constraints with learnable multipliers and one fixed beta. **Gives us:** the AL arm for our 12 BCs (beta e^2 plus lambda e per condition, ascent update), and the finding that fixed beta beats a growing beta by 40x. **Avoid:** the convergence proofs (linear constraints only); the paper never tries L-BFGS. **Relevance:** high.

## Record

- Title: Enhanced Physics-Informed Neural Networks with Augmented Lagrangian Relaxation Method (AL-PINNs) (p. 1).
- Authors: Hwijae Son, Sung Woong Cho, Hyung Ju Hwang (p. 1). Project role: R32.
- Year: 2023. Venue: Neurocomputing; preprint header says "Preprint submitted to Neurocomputing", June 1, 2023 (p. 1). The project record lists Neurocomputing 548:126424.
- arXiv: 2205.01059v2 (p. 1).
- Pages read: all 19. Text extracted with pdftotext; pages 3 and 11 checked as page images for equations and Table B.5. No unreadable parts.

## Problem

- The paper targets the imbalance between the PDE residual loss and the initial and boundary condition losses (p. 1).
- Plain penalties need a large weight beta on the constraint error. Large beta destabilizes training and lets the boundary term dominate the residual term (pp. 1-2, 6).
- Their fix: state training as a constrained problem, argmin residual subject to T u = g, and relax the constraint with an augmented Lagrangian (p. 3, eqs. 4-5).
- Motivation: common loss-balancing rules (soft attention, learning rate annealing, NTK) rest on observed training pathologies and carry no convergence guarantee (p. 2, Section 1.1).

## Method

- Loss, eq. 5 (p. 3): residual term + beta times the squared constraint error + the linear multiplier term. One shared beta multiplies all squared constraint terms; each constraint gets its own multiplier, e.g. lambda^(1), lambda^(2), lambda^(3) in eq. A.4 and A.6 (p. 10), six groups including a Neumann flux term in eq. A.9 (pp. 10-11).
- Constraints are encoded pointwise: the boundary term is a sum over boundary collocation points of squared errors plus lambda(x_j) times the error at that point (p. 3). So the vector of errors and the vector of multipliers meet term by term.
- lambda is discretized to one learnable scalar per boundary collocation point (p. 3).
- Update loop (p. 3): simultaneous gradient descent-ascent, theta <- theta - eta_theta grad L, lambda <- lambda + eta_lambda grad L. Since grad_lambda L is the constraint error, the multiplier step is lambda_j <- lambda_j + eta_lambda (T u_nn(x_j) - g(x_j)).
- No inner loop. There is no alternation with full inner minimizations; both sets move at every step with predefined learning rates (p. 3). The paper reports using Adam for training (p. 5) but does not state separate optimizer handling for lambda (p. 5).
- Penalty parameter: beta is a fixed predefined constant, not a schedule, with no cap (p. 6). Chosen by grid: beta in {1, 10, 50, 100, 500, 1000, 10000}, eta_lambda in {1 down to 1e-4}, eta_theta in {1e-3 down to 1e-5}, lambda_0 = 0 (Appendix B, p. 11).
- Convergence claims (Section 3, pp. 4-5, proofs pp. 11-17): the loss sequence Gamma-prime converges and is equicoercive, so quasi-minimizers converge weakly to the unique strong solution and strongly in L2 (Theorem 3.7, pp. 4-5). Assumptions: unique strong solution, uniform boundedness of the multipliers, which holds for beta large enough (Lemma 3.5, p. 4; the proof bound on p. 16 needs beta above a term involving eta_lambda), and bounded derivatives of the boundary data (Assumption D.4, p. 15). Proven only for Helmholtz, viscous Burgers, Klein-Gordon (Theorem 3.7, p. 4).
- Benchmarks (Section 4, pp. 5-9): Helmholtz, viscous Burgers, Klein-Gordon, plus 2D Navier-Stokes with sinusoidal features. Baselines: vanilla PINNs, soft attention, learning rate annealing, NTK balancing. Four architectures M1-M4, 10 trials each (p. 7), 10,000 epochs with early stopping (pp. 7-8).

## Key results (author-reported)

- Helmholtz, Section 4.1 (pp. 5-7): AL with beta = 1 gives a relative L2 error about 40 times smaller than the penalty method (p. 6). Across fixed beta values, AL beats the penalty method by up to 70 times (p. 7). The pure Lagrange multiplier method without the penalty term fails to learn the solution (p. 6). A growing penalty beta_n = 10n made both loss and error rise (p. 6, Figure 1).
- The multiplier norm stays bounded in training for beta in {1, 10, 100, 1000} even with eta_lambda = 1 (p. 7, Figure 3), and for all three benchmark equations even with small beta (p. 7, Figure 4).
- Tables 1-3 (p. 8), mean relative L2 error over 10 trials: AL-PINNs is lowest in all 12 equation-architecture pairs. Examples: Helmholtz M1 6.44e-3 vs 5.12e-1 for vanilla PINNs; Klein-Gordon M1 1.42e-2 vs 3.86e-1; Burgers gains are smaller, M1 5.45e-2 vs 8.92e-2, M3 4.10e-2 vs 4.85e-2.
- All adaptive baselines leave severe boundary errors; AL-PINNs does not (p. 7, Figure 5).
- Navier-Stokes with sinusoidal features, Table 4 (p. 9): MSE 2.45e-06 for AL with sinusoidal features vs 2.87e-04 for plain PINNs.
- Cost, Appendix C (p. 11): per-epoch time close to plain PINNs, e.g. Helmholtz M2 23.7 ms vs 23.1 ms. Extra learnable parameters scale with boundary points, 200 for Burgers vs 10,200 for soft attention. The paper concludes the added cost is negligible (p. 11).

## Adopt for our pipeline

- Build the AL arm as: residual penalty + sum over our twelve BC errors e_i of (beta e_i^2 + lambda_i e_i), one shared beta and one multiplier per condition, as in eqs. A.4, A.6, A.9 (p. 10). Our flux and slip conditions fit this; the paper includes a Neumann flux constraint the same way (p. 10, eq. A.9).
- Each of our BCs is a point condition in 1D, so a scalar multiplier per BC is the exact analog of their discretized lambda. Twelve extra scalars, a negligible parameter cost (pp. 3, 11).
- Update lambda_i by ascent on its own BC error, lambda_i <- lambda_i + eta_lambda e_i, lambda_0 = 0, at every step next to Adam updates on theta (pp. 3, 11).
- Grid search beta in {1 ... 1e4} and eta_lambda in {1 ... 1e-4}. Their best beta ranged from 1 to 1e4 across equations and models (Appendix B, p. 11), so no single scale is safe for our five-field system.
- Keep beta fixed during a run. Their growing penalty lost to AL by about 40 times in error (p. 6).
- Run the three-way ablation of Section 4.1 for our BVP: pure penalty, pure multiplier, AL. Their finding that both terms are needed (pp. 6, 9) is the first thing our arm must reproduce.
- Log the multiplier norms per BC. Growth signals the need for larger beta or smaller eta_lambda (Lemma 3.5, p. 4; Figures 3-4, p. 7).
- The paper never tries L-BFGS. For our Adam then L-BFGS plan, freeze the multipliers when switching to L-BFGS and tune theta only, since a joint L-BFGS step would descend on lambda, the wrong sign for the max-min problem (p. 3).
- Check which update lambda actually gets: the analysis uses the plain ascent step (p. 3, p. 16), while Adam rescales per coordinate. The paper says Adam is used but does not separate the two, so we should log both variants (p. 5).
- Monitor raw BC errors for stopping, not the total loss, because the AL loss moves as lambda moves; they use early stopping themselves (pp. 7-8).

## Do not import

- The convergence proofs: they cover linear trace constraints T u = g on three specific equations with strong regularity assumptions (pp. 4-5, 12-17). Our radiation condition is nonlinear and couples fields, so the guarantee does not transfer. Use the method, cite the theory as theirs.
- The beta "large enough" condition as a tuning rule: the bound on p. 16 involves unknown quantities, and their own Figure 4 shows bounded multipliers even for small beta (p. 7). Grid search is the practical route.
- The many-point boundary discretization: it exists for 2D and 3D boundary integrals (p. 3). Our twelve point conditions need only scalar multipliers.
- The time-dependent benchmarks and the sinusoidal feature study: different problem class, nothing there bears on coupled nonlinear BCs (pp. 8-9).
- The NTK and soft attention comparison machinery: baseline work, not part of the method we would adopt (Section 4.2, pp. 7-8).

## Quotes worth keeping

1. "We treat the initial and boundary conditions as constraints for the optimization problem of the PDE residual." (p. 1, Abstract)
2. "We solve the above max-min problem by using the gradient descent-ascent algorithm for θ-λ." (p. 3)
3. "This approach avoids the instability that results from the large penalty parameter, and the locally convex structure is not required" (p. 3)
4. "In the numerical experiments, we treat λ as a discretization" (p. 3)
5. "the penalty and multiplier terms should both be considered in the loss function to obtain an accurate approximation" (p. 9)

## Relevance

High: this is the main published alternative to fixed soft penalty weights, and its AL arm maps directly onto our twelve boundary terms as twelve scalar multipliers with one fixed beta.
