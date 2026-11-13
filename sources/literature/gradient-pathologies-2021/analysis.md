# Analysis: Gradient flow pathologies and annealing, 2021

**TL;DR.** Shows boundary-term gradients vanish while residual gradients stay orders larger, and fixes it with learning rate annealing. **Gives us:** the B4 rule we copy (max over mean gradient ratio, EMA 0.9, every 10 steps) and the per-term gradient histogram diagnostic. **Avoid:** their modified architecture; no weight cap exists in the paper, a cap is our addition. **Relevance:** high.

## Record

- Title on the local PDF (arXiv v1): "Understanding and Mitigating Gradient Pathologies in Physics-Informed Neural Networks". The journal version is cited in our role sheet as "Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks", SIAM Journal on Scientific Computing 43(5):A3055-A3081, 2021.
- Authors: Sifan Wang, Yujun Teng, Paris Perdikaris, University of Pennsylvania (p.1).
- arXiv id: 2001.04536, version v1, 13 Jan 2020. The PDF header is dated January 15, 2020.
- Pages read: all 28 PDF pages. All parts were readable. Page refs below use the printed preprint page numbers. Pages 5, 10, 11 and 15 were also checked directly on the PDF to confirm equations, the algorithm box and tables.

## Problem

- The paper treats soft PDE constraints as domain-specific regularizers of the empirical risk and asks how the regularization parameter controls training (p.2, eq 1).
- Main finding: a common failure mode caused by numerical stiffness. Stiff gradient flow dynamics give unbalanced back-propagated gradients between loss terms (p.1, abstract; p.4, sec 2.2).
- Evidence on 2D Helmholtz: boundary-term gradients are sharply peaked at zero in every layer while residual gradients stay orders of magnitude larger. The network then fits the equation but not the boundary data. The paper reports 15.7 percent relative L2 error after 40,000 Adam steps (p.4, sec 2.2, figs 1 and 2).
- The paper notes a paradox: the residual term has the deeper chain rule, yet the boundary gradients are the ones that vanish (p.5, sec 2.3).
- For a 1D Poisson test with solution sin(Cx) they derive bounds: the boundary gradient norm is at most 2 times epsilon times the parameter-gradient norm, and the residual gradient norm is at most order C to the fourth power times the same quantity (p.5, eqs 14 and 15). Larger C gives stronger imbalance and worse solutions (p.5-6, fig 3).
- Stiffness view: gradient descent is a forward Euler step on the gradient flow, so a stable learning rate must stay below 2 over the largest Hessian eigenvalue (p.7, sec 2.4). The paper reports residual-Hessian eigenvalues up to 1e5 and wild residual-loss oscillation even under full-batch Adam (p.8-9, fig 4). An anisotropic Helmholtz solution gives much larger Hessian eigenvalues than an isotropic one (p.7-9, fig 5).

## Method

- Diagnostic 1, per-term gradient histograms: they backpropagate each loss term separately and plot its gradient distribution per hidden layer, not the aggregate loss (p.4, sec 2.2; p.16, fig 12). Pathology signature: data-term gradients pile up at zero, residual gradients stay flat and wide. After annealing the distributions overlap and stay off zero (p.11, fig 7; p.17, fig 13).
- Diagnostic 2, stiffness: the largest eigenvalue of the Hessian of the total loss, monitored during training (p.7, sec 2.4; p.9, fig 5). Analysis only, not used inside the training loop.
- Weighted loss: L equals L_r plus the sum of lambda_i times L_i, with all lambda_i initialized to 1 (p.10, algorithm 1).
- Annealing update rule (p.10, eqs 40 and 41): lambda_hat_i equals the maximum over parameters theta of the absolute gradient of L_r, divided by the mean over theta of the absolute gradient of L_i. Then an exponential moving average: lambda_i becomes (1 minus alpha) times lambda_i plus alpha times lambda_hat_i. Recommended values: learning rate eta = 1e-3 and alpha = 0.9 (p.10).
- Update frequency: every step is allowed, or a user-chosen interval. The experiments update the weights every 10 Adam iterations (p.11; p.14).
- The weights are plain constants in the loss (p.10, eq 42). They are not learned parameters, so no stop-gradient trick is needed and none is mentioned. No cap, clip or bound on the weights appears anywhere in the paper; growth is slowed only by the moving average.
- Benchmarks: 2D Helmholtz, one-space-dimension Klein-Gordon with a cubic nonlinearity, and 2D lid-driven cavity flow at Re = 100 (sec 3, p.14-22). Setup: tanh activations, Glorot init, mini-batch size 128, Adam with default settings (p.14).
- Optimizer: Adam only. L-BFGS never appears in the paper (checked the full text).
- Companion contribution: a modified fully-connected architecture with multiplicative input gating and residual-style updates (p.12, eqs 43 to 47). It lowers the leading Hessian eigenvalue by roughly 3x (p.13, fig 11).

## Key results (author-reported)

- Helmholtz, 4 hidden layers by 50 units, 40,000 Adam steps: baseline 1.81e-01 relative L2 (p.4, fig 1); annealing alone 1.27e-02 (p.11, fig 6); annealing plus the new architecture 3.69e-03 (p.13, fig 10).
- Table 2, nine architectures, mean of 10 trials (p.15): baseline ranges 8.14e-02 to 3.10e-01; annealing plus new architecture ranges 1.49e-03 to 1.07e-02.
- Klein-Gordon, 5 layers by 50 units, 40,000 Adam steps (p.17, table 3): baseline 1.79e-01, annealing 1.06e-02, new architecture alone 1.97e-02, both combined 2.81e-03. Wall-clock times 200, 200, 760 and 1040 seconds. Note: the caption of fig 15 (p.18) gives 2.22e-03 for the combined model, a small mismatch with table 3.
- Cavity, velocity-pressure form: all four models fail, relative L2 between 7.48e-01 and 7.84e-01 (p.21, fig 17). Streamfunction-pressure form: baseline 2.71e-01, annealing 2.49e-01, new architecture 1.21e-01, combined 3.42e-02 (p.22, fig 18). Here the problem formulation mattered as much as the algorithm.
- Weight trajectories: the boundary weight grows from 1 to roughly 300 to 350 over 40,000 steps on Helmholtz (p.12, fig 8); boundary and initial-condition weights reach a few hundred on Klein-Gordon (p.18, fig 16).
- Low sensitivity to alpha within the range 0.5 to 0.9 (p.11). Overhead is small: the annealed model trains in the same 200 seconds as the baseline on Klein-Gordon (p.17, table 3).
- Headline claim: a 50-100x accuracy improvement across their benchmark range (p.1, abstract; p.23, sec 4).

## Adopt for our pipeline

- Copy algorithm 1 as the B4 adaptive baseline, one weight per loss component. Our loss has 17 components: 5 equation residuals plus 12 boundary terms. The paper states the method generalizes to losses with many terms and multiple boundary conditions on different variables (p.11).
- Settings to copy: all weights start at 1, alpha = 0.9, eta = 1e-3, weight update every 10 Adam steps (p.10-11, p.14).
- One adaptation decision to log: the numerator of eq 40 uses a single residual loss. We have five coupled residuals. Choose between the max parameter-gradient of the summed residual loss and the max over the five separate residual gradients, and record the choice in the B4 config. Do not pick silently.
- Diagnostics to copy: per-component gradient histograms and mean and max magnitudes every 10 steps, plus lambda trajectories. These mirror figs 2, 8, 12 and 16, and would show which of our 17 terms dominates and whether the stiff wall region starves the far-field and flux terms.
- No weight cap exists in the paper. Our near-wall inverse-power coefficients may drive some lambda_i high. If we add a cap, label it as our own extension in B4, not as part of the paper's method.
- Optimizer interaction: the paper is Adam-only. For our Adam then L-BFGS plan, run annealing during the Adam phase and freeze the weights before L-BFGS starts. State in the B4 docs that freezing is our choice, since the paper gives no L-BFGS evidence.
- Optional stiffness probe: estimate the largest Hessian eigenvalue occasionally by power iteration for an analysis plot, as in fig 5 (p.9), but keep it out of the training loop.

## Do not import

- Modified MLP architecture (p.12, eqs 43 to 47): B4 compares weight schemes on a fixed network, and this would confound the comparison. It also risks touching our frozen model equations.
- The 50-100x claim as an expectation for us: author-reported on their benchmarks, not verified by us (p.1, p.23).
- The streamfunction reformulation lesson (p.19-20): it applies to the incompressibility constraint of the cavity case. Our five equations and 12 boundary conditions are fixed by the project rules and cannot be reformed.
- Mini-batch size 128 with collocation points resampled each step (p.14): our 1D BVP uses a fixed deterministic point set, so their stochastic setup does not transfer directly.
- The full-batch Adam run with exponential decay schedule used for the stiffness study (p.8): that setup served the failure analysis, not the recommended method.
- Per-step Hessian monitoring: too costly for our five-output network at every step; keep it as an occasional diagnostic only if the supervisor wants it.

## Quotes worth keeping

- "the gradients corresponding to the boundary loss term L_{ub}(theta) in each layer are sharply concentrated around zero" (p.4, sec 2.2).
- "Paradoxically, here we observe the opposite behavior as it is the grad-theta L_{ub}(theta) gradients that appears to vanish during model training" (p.5, sec 2.3).
- "our computational studies confirm very low sensitivity on the additional hyper-parameter alpha" (p.11).
- "this adaptive method can be easily generalized to loss functions consisting of multiple terms (e.g., multi-variate problems with multiple boundary conditions on different variables)" (p.11).
- "consistently improve the predictive accuracy of physics-informed neural networks by a factor of 50-100x across a range of problems in computational physics" (p.1, abstract).

## Relevance

- High. Same soft-penalty setting, Adam training and multi-term losses as ours, and it supplies the exact gradient-statistics annealing rule that B4 needs as its standard adaptive baseline.
