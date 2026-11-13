# Analysis: PINN versus FEM comparison, 2024

**TL;DR.** Systematic PINN versus FEM comparison; PINNs did not win on time or accuracy. **Gives us:** the cost protocol (train and evaluation time separately, same hardware), the no-superiority citation, and the staged warm-up pattern (a 1000x weight on one hard term made training possible). **Avoid:** the blanket verdict (their FEM is FEniCS, ours is solve_bvp); the paper contains no L-BFGS speedup number. **Relevance:** high.

## Record

- Title: "Can physics-informed neural networks beat the finite element method?"
- Authors: Tamara G. Grossmann, Urszula Julia Komorowska, Jonas Latz, Carola-Bibiane Schonlieb.
- Venue: IMA Journal of Applied Mathematics 89(1):143-174, 2024; DOI 10.1093/imamat/hxae011 (journal record, not from our PDF).
- arXiv: 2302.04107, v1, 8 Feb 2023. Our PDF is this arXiv v1, 27 pages. It is the only arXiv version.
- Pages read: all 27. Text extraction was clean on every page; equation (10) and the figure pages were also checked against the rendered PDF. No unreadable parts. The published open-access text was consulted only to confirm the loss-weighting wording; every content claim below is from the arXiv v1 PDF.

## Problem

The paper asks a direct comparison question: given the same PDEs, which method wins on cost and accuracy, PINNs or FEM (Section 1.2, p. 3). Benchmarks (pp. 3, 8-21):

- Poisson equation in 1D, 2D and 3D, each with an analytical solution (Sections 4.1 to 4.3).
- Allen-Cahn equation in 1D, nonlinear parabolic, eps = 0.01; a harder eps = 0.001 case is also tried (Section 5, p. 13).
- Semilinear Schrodinger equation in 1D and 2D, complex valued, hyperbolic (Section 6).

This covers elliptic, parabolic and hyperbolic classes, linear and nonlinear (Section 1.2, p. 3).

## Method

Comparison protocol (Section 3.1, pp. 7-8):

- Cost metric: wall-clock time, in two parts. Solution time is the FEM system solve on CPU versus PINN training on GPU. Evaluation time is FEM interpolation onto a new mesh versus a PINN forward pass on new points. No FLOPs and no energy measurements.
- Accuracy metric: l2 relative error against ground truth on a shared mesh. Ground truth is analytical for Poisson; for Allen-Cahn and Schrodinger it is a fine-mesh FEM solution with implicit Euler time stepping (p. 7).
- Fairness measures: one machine (12 CPU cores, Quadro P6000 GPU), 10 repeated runs averaged, sweeps over FEM mesh sizes and 6 to 14 PINN architectures, results plotted as accuracy versus time on log-log axes (pp. 7-8; Figures 2, 4, 6, 8, 11, 13). The authors flag one asymmetry themselves: FEM runs on CPU, PINNs on GPU (p. 7).

Setups: FEM is FEniCS with P1 elements, conjugate gradient with ILU for Poisson, Newton for Allen-Cahn, gmres for Schrodinger, semi-implicit Euler in time (pp. 8, 14, 17, 20-21). PINNs are the vanilla Raissi et al. form in JAX: tanh feed-forward nets, one loss over PDE, boundary and initial residuals, Adam then L-BFGS, learning rates picked by hand per problem (pp. 6-7). Collocation points come from Latin hypercube sampling, re-sampled every Adam epoch but fixed during L-BFGS (p. 7).

Loss scaling variant (Section 5, pp. 14-15): only Allen-Cahn gets modified terms. Equation (10) multiplies the initial-condition residual by a fixed factor 1000; the authors say this weighting was found heuristically and rendered the best results (p. 14). The schedule is also staged: Adam for 7000 epochs on the initial-condition loss alone, then Adam for 50,000 epochs on the full loss, then L-BFGS (p. 15). The stated reason is that the network struggled to learn the initial condition when the optimisation ran on the full loss directly (p. 15). Important: the paper reports no speedup number for L-BFGS and does not claim that scaling accelerates L-BFGS. The scaling exists to make the problem learnable at all.

## Key results (author-reported)

- Overall verdict, in the abstract and Section 7: on solution time and accuracy, PINNs did not beat FEM in this study; the only evaluation-time win shown is 3D Poisson (pp. 1, 22-23).
- 1D Poisson: a [1,1] network cannot even satisfy the boundary conditions (p. 9). FEM is faster and more accurate overall; PINN architectures that match some FEM errors train 2-3 orders of magnitude slower (p. 10). Accuracy tracked layer width more than depth (p. 10).
- 2D Poisson: FEM solves 1-3 orders faster; PINN evaluation is 2-3 orders faster than FEM interpolation yet still less accurate (p. 11). The authors note FEM solved faster than their own FEM interpolation, possibly due to inefficient interpolation code (p. 11).
- 3D Poisson: PINN training is 1-3 orders slower but reaches equal or higher accuracy in most cases; PINN evaluation is 2-3 orders faster than FEM interpolation and also faster than the FEM solve, with similar to lower error (p. 12). This is the paper's one clear PINN win.
- Allen-Cahn (eps = 0.01): FEM solves 5-6 orders faster; the evaluation-time gap is about 1 order (p. 16). 20-node nets fail completely; 100 and 500 node nets fit the solution (pp. 15-16). For eps = 0.001, PINNs fail with tanh, softplus and ReLU, while FEM still approximates the near-discontinuous solution (pp. 13, 16-17).
- 1D Schrodinger: FEM is 2 orders faster with lower error for the modulus |h|; results for the real and imaginary parts are called inconclusive, FEM faster but hardly more accurate (p. 19).
- 2D Schrodinger: FEM wins solution time by 2-3 orders and evaluation time by 1 order; the PINN imaginary part is clearly worse; PINN evaluation edges the FEM solve time only (p. 22).
- Dimension note: PINN cost did not rise from 2D to 3D Poisson, which the authors read as a sign for high-dimensional use (p. 23).

## Adopt for our pipeline

- Two-metric cost protocol: report solve/train time and evaluation time separately, plus l2 relative error on a common eta grid scored against our solve_bvp output (Section 3.1, p. 7). Fits our arm comparison directly.
- Sweep one size knob per arm and plot accuracy versus time, log-log: interval count for solve_bvp, width and depth for our five-output network (Figures 2 and 4, pp. 9-11).
- Arm B4 (loss scaling): cite this paper for the pattern, not for a speedup. Their evidence is a fixed 1000x weight on one hard term plus staged warm-up, needed before Adam-then-L-BFGS could fit the full loss (pp. 14-15). Our test: fixed multiplicative weights on the fourth-order f residual and the 12 boundary penalties, then record plainly whether each weight was needed for convergence or made it faster.
- Staged Adam warm-up: try a short Adam phase on the hardest penalties alone (BC terms and the f equation), then the full five-equation loss, then L-BFGS (pattern from p. 15). Run it as an experiment and report the result either way.
- Collocation practice: Latin hypercube sampling in eta, re-sampled each Adam epoch, fixed set for L-BFGS (p. 7). Directly transferable to our [0, L] interval.
- Failure reporting: log failures as results, as the paper does: a network too small cannot satisfy boundary conditions (p. 9), and sharp solution features defeat the vanilla form (p. 16). Our analogues: wall-layer misfit and stalled BC residuals.

## Do not import

- The blanket verdict. Their FEM arm is mature FEniCS code; ours is scipy solve_bvp on a 1D ODE BVP. We state our own measured numbers; their claim stays theirs (Section 7, p. 22).
- The CPU-versus-GPU timing split (p. 7). If we compare arms on cost, both arms run on the same hardware or we state the split.
- Fine-mesh-FEM-as-ground-truth as a training source. Their truth role is fine, but our classical reference never feeds training labels (our repo rule). Scoring only.
- The high-dimension and operator-learning discussion (p. 23). Out of scope for one steady 1D BVP with fixed parameters.
- Any sentence of the form "loss scaling speeds up L-BFGS by X". The paper contains no such number (pp. 14-15).
- Their per-problem optimiser settings. Learning rates and epoch counts were hand-picked per PDE for second-order operators (pp. 7-8, 10, 12); our fourth-order system needs its own tuning.

## Quotes worth keeping

- p. 1: "In terms of solution time and accuracy, physics-informed neural networks have not been able to outperform the finite element method in our study."
- p. 7: "L-BFGS does not allow for re-sampling between iterations and we therefore only sample the collocation points once before the optimisation"
- p. 10: "their training time is 2-3 orders of magnitude higher than in FEM"
- p. 15: "We have found the network struggling to learn the initial condition when the optimisation is run on the full loss directly."
- p. 16: "PINNs in the vanilla form are not well equipped to handle discontinuous solutions; this may also be due to PINNs solving the strong PDE, rather than a weak form."

## Relevance

High. This is the honest head-to-head template for our PINN versus solve_bvp comparison: its protocol shapes our cost and accuracy reporting, its verdict frames our no-superiority-claim stance, and its loss-weighting evidence backs arm B4, with the no-speedup-number caveat above.
