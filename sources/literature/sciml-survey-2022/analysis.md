# Analysis: SciML through PINNs survey, 2022

**TL;DR.** Field survey from 2019 to early 2022. **Gives us:** intro anchors: soft BC drawbacks (p. 23), the equal-magnitude rule (pp. 58-59), stiffness as a named open problem (pp. 57 and 61), Adam then L-BFGS as standard (p. 24). **Avoid:** adaptive weights as a baseline (no weight theory); ODE BVPs get passing treatment only. **Relevance:** high.

## Record
- Title: "Scientific Machine Learning through Physics-Informed Neural Networks: Where we are and What's next".
- Authors: Cuomo, Schiano Di Cola, Giampaolo, Rozza, Raissi, Piccialli.
- Year 2022. Venue: Journal of Scientific Computing 92:88. arXiv:2201.05624v4 (7 Jun 2022).
- Pages read: all 85. Main text pp. 1-63 read in full. Skimmed for role only: the Section 4.2 application catalog (pp. 46-50), the Table 3 software list (pp. 54-55) and the reference list (pp. 63-85). Text was extracted with pdftotext; pages 54 and 55 were re-checked directly in the PDF. No part was unreadable.

## Problem
- The survey maps the PINN field from its 2019 start to early 2022: what PINNs are, their parts, which equations they solved, what software exists, and what stays open (Sect. 1.2, p. 6).
- The literature base came mainly from a Scopus search on physics-informed or physics-constrained neural networks, plus the Brown group and PhILMs center output (p. 6).
- Structure: Sect. 2 building blocks, pp. 7-29; Sect. 3 catalog of equations from ODEs to stochastic PDEs, pp. 29-45; Sect. 4 data, applications, software, pp. 46-55; Sect. 5 future challenges, pp. 54-62; Sect. 6 conclusion, pp. 62-63.

## Method
What the survey catalogs, by topic:
- Loss structure: a weighted sum of equation residual, boundary or initial terms and data terms, each with its own weight (Eq. 2, p. 8; Eq. 6, p. 20). The usual form is a mean squared error over collocation, boundary and data points (pp. 21-22).
- BC handling: soft penalties in the loss versus hard encoding in the network. Soft enforcement has two stated drawbacks: BC accuracy is not guaranteed, and no theory guides the BC weight (p. 23). One data-free flow study prefers hard BCs (reported from Sun et al, p. 37).
- Alternative losses: integral or quadrature form (pp. 21, 52), energy-based and Galerkin weak forms, VPINN which lowers the operator order by parts (pp. 23-24, 31).
- Weighting practice: hand-set weights, weights from ratios of loss components (p. 37), dynamic weights against stiff gradient flow in the Stefan problem (p. 35), self-adaptive weights in TensorDiffEq (p. 53), weight annealing in NVIDIA Modulus (p. 52), gradient normalization as an open idea (p. 59).
- Sampling: collocation points uniform or uneven (p. 21); equispaced lattice, Sobol and Latin hypercube options (p. 46); clustering points near high-gradient zones (reported from Mao et al, p. 38); residual-based adaptive refinement, RAR, in DeepXDE (p. 51).
- Architectures: FFNN dominates, mostly 5 to 10 layers (Table 1, p. 19); tanh is the common activation and must be picked with care because second derivatives are needed (p. 13); CNN, RNN or LSTM, Bayesian and GAN variants exist (pp. 14-18).
- Training: minibatch Adam plus L-BFGS is the standard pair, often Adam first then L-BFGS fine-tuning (p. 24). One cited study trains with L-BFGS alone on larger samples of noisy data, and another finds L-BFGS-B better with few residual points (p. 24).
- Failure issues listed: no global minimum guarantee and unknown optimization error (p. 27); no objective rule for loss weights (p. 59); stiff gradient flow dynamics (pp. 35, 57); failure at high convection or viscosity coefficients and on multi-scale or chaotic solutions (pp. 54, 59, 61); poor information propagation from BC or IC into the interior (p. 60).

## Key results (author-reported)
- Most studies customize activation functions, optimizers, network structure or loss structure rather than add theory (p. 1; p. 62).
- The survey's own 1D nonlinear Schrodinger test, pp. 41-45: a 4 layer, 100 neuron tanh network trained with Adam at learning rate 1e-3 for 9000 epochs then L-BFGS fine-tuning gave a best MSE of 5.17e-4 (p. 43, Table 2 p. 45).
- In that test, doubling initial-condition points cut error more than doubling boundary points, and a vanilla PINN lost accuracy at later times (Table 2, p. 45).
- Dynamic loss weights reduced prediction error in the Stefan problem where the plain model hit a local minimum (reported from Cai et al, p. 35).
- Clustering training points around high-gradient regions improves local accuracy and limits error spread (reported from Mao et al, p. 38).
- The global error splits as R <= EO + 2EG + EA, and the optimization error EO is unknown because the loss is nonconvex (p. 27).
- The paper closes by calling optimal PINN training and extension to multiple equations open problems (p. 63).

## Adopt for our pipeline
- Adam then L-BFGS: the survey reports this order as standard practice and uses it in its own test (p. 24; pp. 43, 45). It matches our plan.
- Fixed scales set so every loss term is the same order of magnitude: the survey states a network mostly minimizes the largest term, so our 5 residual terms with derivative orders 4,2,2,2,2 plus 12 BC terms each need a scale chosen before training (pp. 58-59).
- Per-term weights for coupled systems: the Stefan example splits the loss per phase equation with its own weights (p. 34). Our five equations can follow the same pattern.
- tanh activation: smooth and infinitely differentiable, which our 4th-order eta derivatives need; the survey warns to choose activations with the derivative order in mind (p. 13).
- Non-dimensionalize eta and the residual terms first; the survey lists this as good practice for stability (pp. 24, 46).
- Sampling: Latin hypercube or Sobol points over [0, L] (p. 46). If residuals stay large near the wall layer, the survey's fixes are clustering points near high gradients (p. 38) or RAR refinement (p. 51).
- Point count: with our modest 1D collocation set, the cited finding that L-BFGS-B does better with few residual points supports a short Adam phase then a long L-BFGS phase (p. 24).
- Error reporting as mean plus spread over repeated runs, as done in Table 2 (p. 45).
- Classical reference for checking only: the survey's example compares the PINN against a Chebfun spectral solution (pp. 42-43). This matches our solve_bvp plan and the repo rule that the classical solve never feeds training labels.

## Do not import
- Dynamic or adaptive loss weighting (pp. 35, 53, 56, 59): our baseline fixes scales by design, and the survey itself says no theory guides weights, so adaptive schemes would add an untested choice.
- Hard BC encoding in the network (pp. 23, 37, 51): our 12 BCs include higher-order conditions at two eta values; the survey notes hard schemes fit simple Dirichlet forms and are inadequate for arbitrary bounds.
- Domain decomposition, cPINN, parareal or multi-network splits (pp. 18, 35-36, 38): built for large or discontinuous 2D and 3D domains, not one 1D interval.
- Integral and energy-based losses (pp. 21, 23-24): our system is strong-form collocation with a 4th-order term; a variational form would replace our confirmed Eqs. 14-19 structure.
- Bayesian, GAN and operator-learning variants (pp. 17-18, 41): they target uncertainty or parametric operator maps, out of scope for one deterministic BVP.
- Low-fidelity pre-training as initialization (pp. 24-25): it needs a second cheaper solve, and our repo rule keeps the classical reference out of training.

## Quotes worth keeping
1. "satisfying the BCs accurately is not guaranteed" and "no theory exists to guide determining the weights at this time" (soft BC drawbacks, p. 23).
2. "a NN will always focus on minimizing the largest loss terms in the weighted equation, therefore all loss terms must be of the same order of magnitude" (pp. 58-59).
3. "a key issue is to understand the relationship between PDE stiffness and the impact of algorithms as the gradient descent on the PINNs" (p. 57).
4. "they can fail to learn complex physical phenomena, like solutions that exhibit multi-scale, chaotic, or turbulent behavior" (p. 54).
5. "It will be important to develop a PINN methodology for stiff problems" (p. 61).

## Relevance
- High: this survey is the standard citation for soft BC penalties, equal-magnitude loss scaling, Adam then L-BFGS training, collocation sampling and the named open problem of stiffness that our five-field BVP sits in, even though it treats ODE BVPs only in passing (p. 29).
