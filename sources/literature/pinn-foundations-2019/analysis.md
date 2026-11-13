# Analysis: PINN foundations Part I, 2019

**TL;DR.** The founding PINN paper: residual plus data loss, L-BFGS full batch. **Gives us:** the loss pattern our 5 residuals plus 12 boundaries extend (eqs. 4 and 6), and intro quotes on pages 6, 7 and 10. **Avoid:** the data terms, unit weights as a final setup, discrete-time models. **Relevance:** high.

## Record

- Title: "Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations" (page 1).
- Authors: Maziar Raissi, Paris Perdikaris, George Em Karniadakis (page 1).
- Year: 2017. arXiv:1711.10561v1 [cs.AI], 28 Nov 2017; preprint footer dated 30 Nov 2017 (page 1).
- Venue: arXiv preprint. This is the Part I companion of the 2019 Journal of Computational Physics paper (Raissi, Perdikaris, Karniadakis, JCP 378:686-707). The journal version is paywalled and the arXiv title differs from the journal title.
- Pages read: all 22. Full text extracted with pdftotext; pages 10, 13 and 14 also checked as page images to verify the loss formulas.
- Scope: Part I covers data-driven solutions only. Discovery is left to Part II (page 4, page 20).

## Problem

- Forward problem: given fixed model parameters, infer the hidden solution u(t,x) from a small set of measurements plus the PDE as a constraint (section 1.1, page 4).
- General form: u_t + N[u] = 0, eq. 1 (page 4). The residual f := u_t + N[u] becomes a second network built by automatic differentiation, eq. 2 (page 4).
- Worked cases: Burgers equation with Dirichlet data, eq. 3 (page 5); Schrodinger equation with periodic value and slope conditions, complex solution split into two real outputs, eq. 5 (page 10); discrete-time Burgers and Allen-Cahn with Runge-Kutta priors (pages 13-18).
- Data assumptions: a few hundred points on the initial or boundary state, taken from an analytic solution (Burgers, page 7) or from a high-resolution Chebfun spectral reference (Schrodinger page 11, Allen-Cahn page 18).
- All cases are time-dependent initial value problems in one space dimension. There is no two-point boundary value problem and no coupled multi-equation system; the closest case is the two-output complex Schrodinger network (page 11).

## Method

- Network: plain feedforward MLP with tanh activation. Burgers: 9 layers, 20 neurons, 3021 parameters (page 7). Schrodinger: 5 layers, 100 neurons, 2 outputs for real and imaginary parts (page 11). Discrete-time models: 4 layers with 50 neurons (page 15) and 4 layers with 200 neurons and 101 outputs, one per Runge-Kutta stage (page 18).
- Derivatives: chain rule through automatic differentiation. The paper shows the TensorFlow code: u from the network, then u_t, u_x, u_xx via tf.gradients, then f assembled from those terms (page 5).
- Loss, forward continuous case: MSE = MSE_u + MSE_f, eq. 4 (page 5). MSE_u is the mean square mismatch on the given initial and boundary data points; MSE_f is the mean square of the PDE residual at collocation points (page 6).
- Loss with boundary conditions of derivative type: MSE = MSE_0 + MSE_b + MSE_f, eq. 6 (page 10). MSE_b penalizes the value and slope differences between the two ends of the periodic domain (page 10).
- Discrete-time loss: SSE = SSE_n + SSE_b, eq. 12 (page 13) and eq. 14 (page 17). SSE_b drives the boundary values, and for Allen-Cahn the boundary slopes, to the periodic condition (page 18).
- Weights and scaling: all loss terms are summed with unit weights. No per-term weight coefficients and no scaling discussion appear anywhere (pages 5-6, 10, 13).
- Optimizer: L-BFGS, full batch, for all benchmarks (page 6). Adam is cited only as an example of a mini-batch variant for larger datasets, reference [19] (page 6, page 22). No learning rate schedule or iteration counts are given.
- Sampling: random collocation via Latin Hypercube Sampling. N_f = 10,000 for Burgers (fig. 1 caption, page 8), N_f = 20,000 plus N_b = 50 boundary collocation points for Schrodinger (page 11). No resampling or adaptive sampling is used.
- Evaluation practice: relative L2 error against the exact or spectral reference; sensitivity tables over data count, collocation count, layers and neurons; training time reported (tables 1-2 page 9, fig. 1 caption page 8).

## Key results (author-reported)

- Burgers, continuous: relative L2 error 6.7e-4 with N_u = 100 data points and N_f = 10,000 collocation points; the paper reports this is about two orders below their earlier Gaussian process result (page 7, fig. 1 page 8).
- Training took about 60 seconds on one NVIDIA Titan X GPU (fig. 1 caption, page 8).
- Ablations: error drops as N_u grows once N_f is large enough, table 1 (page 9); error drops with depth and width, table 2 (page 9), best 5.6e-4 at 8 layers x 40 neurons.
- Schrodinger: relative L2 error 1.97e-3 with N_0 = 50, N_b = 50, N_f = 20,000 (page 11, fig. 2 page 12).
- Discrete-time Burgers: one step from t = 0.1 to t = 0.9 with 500 Runge-Kutta stages, relative L2 error 8.2e-4 with a 4x50 network (page 15, fig. 3).
- Allen-Cahn: relative L2 error 6.99e-3 with q = 100 stages, dt = 0.8, 4x200 network (fig. 4 page 19).
- Stage study: low-stage schemes lose accuracy at large dt, while q of 32 and above holds errors near 4e-4 to 8e-4 at dt = 0.8, table 4 (page 17).

## Adopt for our pipeline

- The residual-sum loss pattern: one mean-square term per equation plus separate terms per boundary condition, eq. 4 (page 6) and eq. 6 (page 10). Our loss of 5 equation residuals plus 12 boundary residuals is a direct extension of this pattern.
- Boundary conditions as soft penalties, including derivative conditions: the Schrodinger MSE_b penalizes slope mismatch h_x between the two ends (page 10). This is the template for our slip, flux and coupled radiation residuals.
- The two-network construction: solution network plus residual network sharing parameters, built by automatic differentiation (pages 4-5). This matches our plan of differentiating the float64 tanh MLP up to fourth order in eta.
- L-BFGS full batch as the polish optimizer (page 6). This supports our Adam then L-BFGS schedule, with L-BFGS last.
- Random or Latin hypercube collocation; 10k to 20k points for a 1-D space-time domain is a scale reference when we pick our eta grid density (fig. 1 page 8, page 11).
- Error protocol: report relative L2 against the classical reference, the same role their Chebfun data plays (pages 11, 18). Our solve_bvp reference stays outside training, like their reference data.
- Ablation tables over collocation count and network size, reported as error tables (tables 1-2, page 9). We can copy this for N_eta and hidden width.
- Report wall-clock training cost alongside accuracy (fig. 1 caption, page 8).

## Do not import

- Discrete-time Runge-Kutta models: they need a time direction, snapshot data and stage outputs per point. Our problem is steady in one variable, so section 3 (pages 12-19) does not apply.
- The initial or interior data term MSE_u / MSE_0: we have no solution measurements at all. Every one of our 12 non-equation terms is a boundary residual, not a data mismatch (pages 6, 10).
- Unit loss weights: the paper only sums comparable terms of one scalar field. Our loss mixes a fourth-order residual with second-order ones and twelve boundary terms of different scales, so our fixed per-term scales go beyond what the paper does (page 6).
- L-BFGS alone: the paper runs only L-BFGS. Our Adam warm start is an addition, not something the paper tests (page 6).
- The "small data" claim: the paper still needs N_u data points and notes that N_f = 0 reduces the model to a plain network (page 7 and footnote 1, page 8). Our setting is fully data-free in their sense, so their argument does not transfer.
- Uncertainty quantification: the authors flag it as missing and open (page 20). Out of scope for us.

## Quotes worth keeping

1. "The loss MSE_u corresponds to the initial and boundary data while MSE_f enforces the structure imposed by equation (3) at a finite set of collocation points." (page 6)
2. "MSE_b enforces the periodic boundary conditions, and MSE_f penalizes the Schrodinger equation not being satisfied on the collocation points." (page 10)
3. "we chose to optimize all loss functions using L-BFGS; a quasi-Newton, full-batch gradient-based optimization algorithm" (page 6)
4. "the term in MSE_f in equation (4) acts as a regularization mechanism that penalizes solutions that do not satisfy equation (3)" (page 7)
5. "if the given partial differential equation is well-posed and its solution is unique, our method is capable of achieving good prediction accuracy given a sufficiently expressive neural network architecture and a sufficient number of collocation points" (page 6)

## Relevance

High: this paper defines the residual-plus-boundary penalty loss and the L-BFGS training practice that our pipeline extends from one scalar field to five coupled residuals and twelve boundary residuals on a BVP domain.
