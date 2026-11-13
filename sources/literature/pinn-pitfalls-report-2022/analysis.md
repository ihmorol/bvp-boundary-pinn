# Analysis: PINN pitfalls and frustration report, 2022

**TL;DR.** Published failure report: 32 hours of PINN training to match a 20-second finite difference run, and a cylinder case that never converged. **Gives us:** the failure-reporting template (config disclaimers, judgment calls logged, results with a DOI) and the loss-decoupling warning. **Avoid:** solver snapshots as training residuals; that conflicts with our reference rule. **Relevance:** high.

## Record

- Title: Experience report of physics-informed neural networks in fluid simulations: pitfalls and frustration (p. 28).
- Authors: Pi-Yueh Chuang and Lorena A. Barba, George Washington University (p. 28).
- Year and venue: 2022, Proc. of the 21st Python in Science Conference (SciPy 2022) (p. 28).
- DOI: 10.25080/majora-212e5952-005. Proceedings pages 28 to 36.
- Pages read: all 9 pages. Text extracted with pdftotext; figures and tables on pages 31 to 34 also checked in the PDF. Everything was readable.
- Reproducibility package: Zenodo DOI 10.5281/zenodo.6592457, cited as [Chu22] (p. 28, 36).

## Problem

- Two data-free PINN cases against the Navier-Stokes equations, published as "not-so-successful experiments" (p. 28).
- Case 1: 2D Taylor-Green vortex, Re = 100, periodic box, t in (0, 100] (p. 31). Partial success: acceptable accuracy at a cost far above the finite difference baseline (p. 28, 32).
- Case 2: 2D cylinder flow, Re = 200, uniform inflow, no-slip cylinder, p = 0 outlet (p. 33). Full failure: no vortex shedding after 64 hours of training. The paper reports the PINN behaved like a steady-flow solver (p. 28, 34).
- For the cylinder they tried two network sizes, manual learning rate changes, and two initial conditions, u = 1 and u = 0. The paper reports both initial conditions gave no reasonable results (p. 33).

## Method

- One fully-connected network predicts velocity and pressure together (p. 29).
- Loss: weighted sum of l2 norms of five residuals: continuity, momentum, initial condition, velocity BC, pressure BC. Weights are used in practice but left out of the paper's equations (p. 30).
- Network: 6 hidden layers, 256 neurons per layer, SiLU activation. Hyperparameters copied from the Modulus 3D Taylor-Green example. No hyperparameter optimization because of cost (p. 31).
- Optimizer: Adam with PyTorch defaults, learning rate decayed with factor 0.95^(1/10000). The cylinder runs added manual learning rate changes mid-run to escape local minima (p. 31, 33).
- Sampling: 18,432,000 pre-generated space-time points, 18,432 per iteration for the PDE residuals, same counts for the initial condition. Periodic BCs needed no training points. Cylinder: 40,960,000 pre-generated points, 40,960 per iteration, plus initial and boundary points (p. 31, 33).
- Compute: one DGX-A100 node, 8 A100 80GB GPUs, weak scaling, single precision. Baseline: PetIBM staggered-grid finite difference solver, 1 K40 GPU plus 6 CPU cores, double precision (p. 31).
- The paper notes quasi-Newton methods may work only when the parameter count is small, and that second-order derivatives already make the computational graph very large (p. 30).

## Key results (author-reported)

- Abstract cost claim: about 32 hours of PINN training to match the accuracy of a 16x16 finite difference run that took less than 20 seconds (p. 28).
- Our arithmetic check, not the paper's: 1 million iterations at their Table 1 rates, 85.0 to 90.1 s per 1000 iterations, is about 24 to 25 h. The paper does not spell out the accounting behind "about 32 hours" (p. 28, 32).
- Second cost point: an L2 error of 10^-3 at t = 2 takes under 1 s for PetIBM and over 8 hours for PINN (p. 32, Fig 5).
- Scaling: more GPUs and more training points did not speed convergence; all four GPU counts converged at a similar rate (p. 32, Fig 2).
- Accuracy: PINN error matches the 16x16 PetIBM run and is much worse than finer grids. Errors were measured on a 512x512 mesh against the analytical solution (p. 31, 32, Fig 4).
- Wall time per 1000 iterations: 85.0, 87.7, 89.1, 90.1 s on 1, 2, 4, 8 GPUs; efficiency 100 down to 94 percent (p. 32, Table 1).
- Fair comparison gap: they found no good way to define problem size for a PINN vs classical comparison, and say the gap is too large for a different metric to change the conclusion (p. 33).
- Cylinder: two network sizes, 8 A100s, 64 hours each. Loss never reached a steady value. No shedding in fields, drag, or lift (p. 33, 34, Fig 7, 8, 9).
- Drag coefficient 0.95 from PINN vs 1.38 from PetIBM; steady-flow literature values are 0.97 and 0.83. The paper reads this as steady-solver behavior and says it cannot currently explain why (p. 34, Table 2).
- Loss as a success indicator: a low residual is hard to reach because the minimum is likely local, and the cylinder loss dropped while the solution stayed wrong. They stopped on the judgment that it would not be correct even if converged (p. 30, 32, 34).
- Stated conclusion for one fixed flow condition: PetIBM outperforms the PINN solver (p. 32).

## Adopt for our pipeline

- Cost table: one table with our PINN wall time vs solve_bvp wall time, named hardware, and precision stated. Copy their abstract claim format plus Table 1 (p. 28, 31, 32).
- Accuracy vs wall time plot: rebuild their Fig 5 with our error against the solve_bvp reference over Adam and L-BFGS training time (p. 32).
- Judge success on solution error against the classical reference, never on loss alone. The cylinder case is the caution (p. 34).
- Log stopping decisions as judgment calls with the reason. They admit the Taylor-Green stop had no quantitative criterion, and the cylinder stop was a judgment call (p. 32, 34).
- Put a per-section disclaimer that results are experimental data under one configuration (p. 32, 34).
- Record hyperparameter provenance: what we tuned, what we copied, what stayed default. Their own justification gap is the warning (p. 31, 34).
- Test L-BFGS early. The paper warns quasi-Newton only helps for small parameter counts; our network is small, so test it and report either way (p. 30).
- Time each iteration early in development. They report a large graph already at second order; our residuals go to fourth order. This is our plan, based on their p. 30 observation.
- Publish failed runs with a reproducibility package and a DOI, as they did on Zenodo (p. 28).
- Result table format: PINN value, classical reference value, and literature values side by side, like their Table 2 (p. 34).

## Do not import

- Time-marching and initial condition residuals. Our BVP is steady; there is no t = 0 term (p. 31, 33).
- The periodic-BC shortcut of using no boundary points. Our 12 wall BCs need explicit penalty terms (p. 31).
- The sampling scale. 18M to 41M pre-generated points and 1M iterations are sized for 2D space-time PDEs. Copy the pattern, not the magnitude (p. 31, 33).
- Multi-GPU weak scaling. We run one device, and the paper found more GPUs did not help convergence anyway (p. 31, 32).
- Their suggestion to add traditional-solver solution snapshots as extra residuals (p. 35). This conflicts with our repo rule: the classical reference never feeds training labels in the baseline.
- Manual mid-run learning rate changes in our published records. They used them for the cylinder case (p. 33); we keep fixed schedules so failure records stay reproducible.
- Single precision as a cost default. They flagged the precision mismatch with the baseline but never tested its effect (p. 31).
- Copying library example defaults wholesale. They copied Modulus defaults and later could not justify the choices (p. 31, 34).
- The vortex-shedding diagnosis frame. Our likely failure modes are a wrong solution branch or weight imbalance across five equations, not missing unsteadiness (p. 34, 35).

## Quotes worth keeping

1. "About 32 hours of training were required for the PINN method's accuracy to match the accuracy of a 16 × 16 finite-difference simulation, which took less than 20 seconds." (p. 28)
2. "few reports discuss the required computing resources, the computational cost of training, the convergence properties, or the error analysis of PINN." (p. 29)
3. "They should be deemed experimental data rather than a thorough analysis of the method's characteristics." (p. 32)
4. "'It worked in another case!' is probably the best answer we can offer." (p. 34)
5. "Rather, we should honestly face all troubles, difficulties, and challenges. Knowing the problem is the first step to solving it." (p. 35)

## Relevance

- High. Its PDE class is far from our steady five-field BVP, but it is the published template for the cost comparison and failure reporting our pipeline plans to publish.
