# Analysis: PINN failure modes, 2021

**TL;DR.** Six failure modes demonstrated; the cause is optimization, not expressivity. **Gives us:** the failure taxonomy for our paper, the Hessian landscape diagnostic, the curriculum ramp idea, and the weight-sweep negative control. **Avoid:** sequence-to-sequence time marching; our two-ended BVP has no causal direction. **Relevance:** high.

## Record

- Title: Characterizing possible failure modes in physics-informed neural networks.
- Authors: Aditi S. Krishnapriyan, Amir Gholami, Shandian Zhe, Robert M. Kirby, Michael W. Mahoney (p.1).
- Year: 2021. Venue: NeurIPS 2021 (p.1). arXiv: 2109.01050v2.
- Pages read: all 20 pages. Text extracted with pdftotext; tables on p.7 to p.9 were re-checked against the PDF pages directly. No part was unreadable.

## Problem

The paper asks why PINNs with soft PDE constraints fail. The soft form is min L(u) + lambda_F * F(u), where L holds the initial and boundary misfit and F is the PDE residual (Eq. 3-4, p.2). The paper demonstrates six failure modes:

1. Coefficient regime failure. The vanilla PINN works only for small PDE coefficients. Convection error reaches about 100% relative error for beta > 10 (Sec 3.1, p.5). Reaction error hits 100% as rho grows (App A, p.14). The paper says this holds even after extensive hyperparameter tuning (p.3).
2. Missing sharp features. Reaction-diffusion at rho=5, nu=5 gives 93% relative error; at nu=2 it gives 50% and misses the sharp transitions (Sec 3.2, p.6).
3. Trivial output. For reaction, the trained net predicts a near-zero homogeneous field everywhere (App A, Fig A.1, p.14).
4. Boundary fit without physics. The net can fit the boundary conditions yet fail inside the domain, here past a timestep (Fig 1 caption, p.5).
5. Ill-conditioned regularization. The residual term is a differential operator that can be ill-conditioned and does not define a convex set like a norm ball (p.2-3). Appendix B derives approximate scalings: O(beta N) for a first-derivative operator and O(nu N^2) for a second-derivative operator, both squared by the L2 loss (App B, p.14-15).
6. Hard loss landscape. Larger coefficients make the landscape complex and non-symmetric, and the optimizer gets stuck in a high-loss minimum (Sec 4.1, Fig 3, p.7).

The paper argues the cause is optimization difficulty, not network expressivity (abstract p.1; Sec 5, p.7). On stiff regimes: the paper never studies boundary layers or coefficients that blow up at a wall. Its boundaries are periodic. The closest items are one related-work mention of stiff chemical kinetics ODEs (p.3) and the ill-conditioning appendix (App B, p.14-15).

## Method

- Curriculum regularization (Sec 5.1, p.7-8). Train on an easy member of the PDE family (small beta or rho), then reinitialize the net from those weights and train on a larger coefficient, step by step up to the target (Fig 4, p.8). No time stepping is involved; the ramp is on a PDE coefficient. It also lowers error variance across 10 seeds (Fig E.2, p.18) and gives a smoother loss landscape (Fig E.3, p.19).
- Sequence-to-sequence training (Sec 5.2, p.8-10). Predict one time segment of length Delta t at a time. Use the prediction at t = Delta t as the initial condition for the next segment (Fig 5, p.9). Same architecture and same total collocation count as the baseline (p.9, footnote 4). The paper's stated reason it helps: the problem is ill-conditioned, so restricting the dimensions helps, and the mapping is simpler over a smaller time span (p.10). This is time marching and needs a causal initial-value structure.
- Benchmarks: 1D convection (Eq. 5, p.5), 1D reaction (Eq. 14, p.14), 1D reaction-diffusion (Eq. 10, p.6). All have periodic boundaries (p.4).
- Setup: 4-layer fully connected net, 50 neurons per layer, tanh, random collocation points, L-BFGS, learning rates swept from 1e-4 to 2.0 (p.4). The paper reports that SGD variants underperformed L-BFGS on these problems (p.4, footnote 3).
- Metrics: L2 relative error and absolute error against the analytic solution, averaged over at least ten random seeds (p.4).

## Key results (author-reported)

All numbers below are the paper's own. We have not verified them.

- Convection baseline: relative error 7.84e-3 at beta=1 versus 9.61e-1 at beta=40 (Fig 3 table, p.7).
- Curriculum, convection: relative error drops from 7.50e-1 to 9.84e-3 at beta=20, from 8.97e-1 to 2.02e-2 at beta=30, and from 9.61e-1 to 5.33e-2 at beta=40 (Table 1, p.8).
- Curriculum, reaction: 0.1 to 0.6 orders lower error for rho=2-4, and 1-2 orders lower for rho=5-10 (Sec 5.1, p.8; Fig E.4, p.19).
- Seq2seq, reaction-diffusion at rho=5 with Delta t=0.1: relative error drops from 5.07e-1 to 1.18e-2 at nu=2, and from 9.60e-1 to 2.69e-2 at nu=6 (Table 2, p.9).
- Seq2seq, reaction: relative error drops from about 9.8e-1 to about 7e-2 for rho=5-10 (Table E.2, p.20).
- Seq2seq, convection: gains are smaller, from 7.38e-1 to 1.05e-1 at beta=30 with Delta t=0.1, using 10000 collocation points (Table E.1, p.19).
- Weight sweep at beta=30: relative error stays between 0.982 and 1.69 for lambda from 1e-6 to 1e1, so tuning the soft-constraint weight alone does not fix the failure (Fig E.1, p.18).
- Headline claim: up to 1-2 orders of magnitude lower error than regular PINN training (p.1 abstract; restated p.3).
- Landscape evidence: computed by perturbing the trained weights along the top two Hessian eigenvectors, not random directions (Sec 4.1, p.7).

## Adopt for our pipeline

- Use their failure list as the skeleton of our failure analysis section: coefficient regime, sharp features, trivial solution, boundary fit without physics, ill-conditioning, hard landscape (Sec 3-4, p.5-7; App B, p.14-15).
- Add the Hessian-eigenvector loss landscape plot as a diagnostic for our trained model, especially near the wall where coefficients grow like inverse powers (Sec 4.1, p.7).
- Report both L2 relative and absolute error, averaged over at least ten seeds, with the variance visible (p.4; Fig E.2, p.18).
- Test a BVP version of curriculum: ramp one difficulty scalar toward the supervisor's target values, warm-starting weights at each step. Candidate ramp variables for us are a nonlinearity strength or a wall-coefficient scale, not time (Sec 5.1, p.7-8).
- Keep L-BFGS in our Adam then L-BFGS plan; the paper's evidence favors L-BFGS over SGD variants for PINN residuals (p.4, footnote 3).
- Run a small lambda sweep early and log the result, since their sweep shows weight tuning alone does not rescue a hard regime (Fig E.1, p.18).
- Use their conditioning caution when we pick residual scales. Their first-derivative operator scales as O(beta N) and their second-derivative operator as O(nu N^2) (App B, p.15). Our order-4 term should be worse still; this extension is our inference, not their claim.

## Do not import

- Sequence-to-sequence training. It marches forward in time and hands the predicted state to the next segment. Our BVP is two-ended in eta with twelve conditions split between eta=0 and eta=L, so there is no causal marching direction (Sec 5.2, p.8-10).
- Periodic boundary loss. Their boundary term compares the two ends of a periodic domain (Eq. 8, p.5). Our twelve conditions are fixed wall and far-field values.
- Their scope of claims. Their benchmarks are 1D, single-output, with analytic solutions for checking. Ours is one input with five coupled outputs, derivative orders up to 4, and no analytic reference in the training loop.
- The exact ramp parameter. Their beta and rho ramps stay well posed because small coefficients keep the PDE easy. A ramp of our near-wall coefficients may pass through degenerate equations, so each step would need its own well-posedness check before we train on it.
- L-BFGS-only training. They sweep L-BFGS learning rates; they do not test an Adam then L-BFGS schedule, so our plan still needs its own validation (p.4).

## Quotes worth keeping

- "they can easily fail to learn relevant physical phenomena for even slightly more complex problems" (p.1, abstract).
- "it involves a differential operator that could actually be ill-conditioned, and it does not correspond to a nice convex set" (p.2-3).
- "the loss landscape at a relatively low β = 1 is rather smooth, but increasing β further results in a complex and non-symmetric loss landscape" (p.7).
- "Intuitively, since the problem is ill-conditioned, restricting the dimensions is expected to help." (p.10).
- "With curriculum regularization, the relative error is almost two orders of magnitude lower." (p.8).

## Relevance

High. The paper gives a tested failure taxonomy and a landscape diagnostic we can copy, its curriculum idea can transfer to a stiffness parameter in our BVP, and its seq2seq result marks the time-marching path we cannot use for a two-ended problem.
