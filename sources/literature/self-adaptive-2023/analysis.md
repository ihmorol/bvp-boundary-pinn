# Analysis: Self-adaptive soft attention weights, 2023

**TL;DR.** One trainable weight per point through a saturating sigmoid mask. **Gives us:** the only bounded pointwise construction, the L-BFGS freeze rule that matches ours, and the weight map as a wall-region diagnostic. **Avoid:** unbounded lambda; their wave case needed resampling to work at all. **Relevance:** high.

## Record

- Title: Self-Adaptive Physics-Informed Neural Networks using a Soft Attention Mechanism.
- Authors: Levi D. McClenny, Ulisses Braga-Neto, Texas A&M University.
- Venue: Journal of Computational Physics, Vol. 474, 111722, 2023 (p. 1 note).
- arXiv: 2009.04544v5, cover date 18 Jun 2024.
- Pages read: all 24. Text pulled with pdftotext. Pages 5-6, 10-11, and 15-16 re-checked as page images for equations, Table 1 and figure captions. Nothing was unreadable.

## Problem

Baseline PINN training latches onto some loss components and neglects others, which breaks convergence on stiff PDEs with sharp transitions (p. 3-4). Existing fixes weight whole loss terms with one scalar each. This paper weights each training point on its own, so the network finds the hard spots and is forced to fit them (p. 3-4, Sec. 2.2). The stated motivation is exactly our setting: stubborn spots in stiff problems (p. 1, abstract).

## Method

- One trainable, nonnegative weight per residual, boundary, and initial point. The sensor-data loss is left unmasked (p. 4-5, Eqs. 10-13).
- The weight enters through a mask function m(lambda): nonnegative, differentiable, strictly increasing. Candidates are polynomial masks c*lambda^q and sigmoidal (logistic) masks. The mask value multiplies each point's squared loss (p. 4-5, Eqs. 11-13).
- Update rule: gradient descent on network weights, gradient ascent on the weights, each with its own learning rate (p. 5, Eqs. 14-17). The weight gradient is m'(lambda) times the unmasked per-point loss (p. 5, Eqs. 18-20). No stop-gradient is used and none is needed: lambda is a free parameter, so its gradient never flows through the network.
- Weights rise monotonically whenever the unmasked loss is nonzero, and they rise faster where the loss is larger (p. 5).
- Frequency: weights update at every Adam step. During L-BFGS they are held constant (p. 6).
- Bounds: lambda itself has no cap and grows monotonically. The sigmoid mask saturates, so the effective mask value is bounded (p. 5, Fig. 1, p. 6). Polynomial masks must be capped by hand to avoid overflow (p. 5). Nothing else stops the total masked loss from drifting; the safeguard is that weights only increase, so the network cannot shrink them to make the loss look small.
- Points are fixed in the basic method. For SGD with resampling, a Gaussian process fit to the trained weights predicts weights at new points (p. 15).
- NTK view: the mask is a diagonal matrix Gamma on each loss block. Pointwise weights reshape the eigenvalue distribution of the NTK, not just rescale it, which equalizes convergence rates across terms and points (p. 17-19).

## Key results (author-reported)

All numbers below are what the paper reports, averaged over 10 random restarts (p. 6). We have not verified them.

- Viscous Burgers: L2 error 4.80e-4, std 1.01e-4, versus 6.7e-4 for the baseline PINN, using 20 percent of the training iterations. Large weights land on the shock at x=0 (p. 7-8, Figs. 2-4).
- Helmholtz: L2 error 3.2e-3, std 2.2e-4, versus 1.4e-1 baseline and 2.54e-3 to 2.74e-2 for learning-rate annealing from an earlier paper (p. 9).
- Allen-Cahn: L2 error 2.1e-2, std 1.21e-2, versus 8.0e-2 for a time-adaptive scheme, 49.61e-2 for nonadaptive C=100 weighting, and 96.15e-2 for the baseline (p. 10). Boundary weights were fixed at 1; initial-condition weights started in [0, 100], residual weights in [0, 1] (p. 10). The learned weight map was stable across restarts and put more weight at early times (p. 11-12).
- 2D Burgers: L2 error about 3e-3 at t=1/3 and 2/3 with only initial conditions enforced. The paper reports that SA-PINN stabilized L-BFGS convergence where the baseline L-BFGS failed (p. 13, Fig. 11).
- 1D wave with SGD resampling every 100 iterations: SA-PINN 2.95e-2 versus 2.08e-1 for matched fixed weights and 4.51e-1 baseline (Table 1, p. 16). Without SGD, SA-PINN failed at 8.1e-1 (p. 15-16). Cost rose from 880 s to 1208 s, about 14 percent overhead. The paper admits its wave error is still worse than the NTK-weighting paper it compares to (p. 15).
- Advection: baseline failed to find even the rough structure after 10k Adam iterations; SA-PINN reached about 5 percent L2 error (p. 20).
- Hyperparameter rule: small network learning rate, near 1e-5, and larger weight learning rate, 1e-3 to 1e-1 (p. 13, Sec. 4.5).

## Adopt for our pipeline

- Sigmoid mask with a saturation cap. This is the one construction here that gives the bounded weight our strategy requires: lambda grows but the effective mask value saturates (p. 5, Fig. 1, p. 6). Pick the saturation level from our fixed component scales.
- Freeze weights during L-BFGS. The paper does exactly what our strategy demands: weights train only in Adam and are constant in L-BFGS (p. 6). This is direct support for our freeze rule.
- Detached weight update. The weight gradient needs only the unmasked per-point loss value times m'(lambda) (p. 5, Eqs. 18-20). We can compute per-point residuals under no_grad and step the weights by ascent, so no gradient flows through the network into the weights.
- Scope the weights like they do. Residual and boundary points get weights; we would add one weight per point per equation, so five residual weight vectors plus 12 BC weights on top of our 17 fixed components. Fix the weights of any term we do not want adapted, as they fixed boundary weights at 1 for Allen-Cahn (p. 10).
- Weight map as a diagnostic. A scatter of trained weights over eta locates the hard region, as their Fig. 4 does for the Burgers shock (p. 8). For us this should light up the stiff wall region where coefficients grow.
- Report the unmasked loss. They strip the weights out of the loss before comparing runs (p. 12, Fig. 12). We should log our 17-component loss unmasked alongside the masked objective.
- Use a smaller learning rate for the network than for the weights, per their Sec. 4.5 rule (p. 13).

## Do not import

- Unbounded trainable lambda updated jointly in Adam. Our strategy requires bounded weights; their lambda grows monotonically with no cap (p. 5). If we copy the update, we add a hard cap and detach.
- Polynomial masks without a cap. They must be clamped to avoid overflow (p. 5). A saturating sigmoid is the safer fit for us.
- The GP resampling machinery. It exists to support SGD with new points each batch (p. 15). We use a fixed point set, and the wave result needed resampling to work at all, so the plain version is a poor match for us.
- Masking data terms. They deliberately leave sensor data unmasked (p. 5). Not our case, but it shows they treat data terms as special.
- Their per-problem hyperparameters and timings. Learning rates, inits, and V100 milliseconds are tuned per benchmark (p. 7, 10, 13, 15). We would set our own and measure.

## Quotes worth keeping

- "the adaptive weights are only updated in the Adam training steps, and are held constant during L-BFGS training, if any." (p. 6)
- "are monotonically increasing, provided that the corresponding unmasked losses are nonzero" (p. 5)
- "In practice, the polynomial mask functions have to be kept below a suitable (large) value, to avoid numerical overflow." (p. 5)
- "excessively sharp sigmoidal mask functions should be avoided." (p. 5)
- "tend to require smaller values of learning rate for the neural network weights (i.e., 1e-5), and larger values for the self-adaptive weights (i.e., 1e-3 to 1e-1)" (p. 13)

## Relevance

High: this is the per-point extension of our fixed 17-component loss, its L-BFGS freeze rule matches our strategy exactly, and the paper itself flags the two risks we care about, unbounded growth and sharp-mask vanishing gradients (p. 5).
