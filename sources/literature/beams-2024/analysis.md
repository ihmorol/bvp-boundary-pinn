# Analysis: Complex beam systems PINN, 2024

**TL;DR.** Coupled fourth-order beam PDEs, linear, with analytic solutions, L-BFGS only. **Gives us:** the strongest scaling evidence we have: dimensional coefficients (4e6) fail at loss 1e14 while the nondimensional version reaches 5.3e-4 percent error, plus the point-count ablation pattern. **Avoid:** the lumped mean over all point types; L-BFGS-only training. **Relevance:** medium.

## Record

- Title: "Physics-informed neural networks for solving forward and inverse problems in complex beam systems" (page 1).
- Authors: Taniya Kapoor, Hongrui Wang, Alfredo Nunez, Rolf Dollevoet, Delft University of Technology (page 1).
- Venue: accepted for IEEE Transactions on Neural Networks and Learning Systems (header, page 1). The supervisor notes give the journal version as IEEE TNNLS 35(5) 5981-5995, 2024; this PDF is the arXiv manuscript.
- arXiv id: 2303.01055v2 [cs.LG], 29 Aug 2023 (page 1 sidebar).
- Pages read: all 15. Full text extraction plus direct page views for pages 4, 5, 8 and 9. All parts readable.

## Problem

- Four case studies, all time-dependent, all in nondimensional form (page 4).
- Single Euler-Bernoulli beam, eq. (9), page 4: u_tt + u_xxxx = f(x,t). One field, fourth order in space, second in time. Domain x in [0, pi], t in [0, 1] (page 4).
- Single Timoshenko beam, eq. (13), page 6: two coupled PDEs for rotation theta and displacement w, second order in space and time.
- Euler-Bernoulli double beam, eq. (15), page 8: two fourth-order equations m_i w_i_tt + K_i w_i_xxxx + k(w_i - w_other) = f_i. The Winkler foundation couples the beams through stiffness k times the relative displacement (page 8).
- Timoshenko double beam, eq. (16), page 10: four coupled PDEs, second order in space and time, same k(w_1 - w_2) coupling (page 10).
- BCs are simply supported: value and second derivative zero at both ends. Single EB: 4 BCs plus 2 ICs (page 4). Double EB: 8 BCs plus 4 ICs (page 9). Timoshenko cases: 4 or 8 BCs plus matching ICs (pages 6, 10).
- Forces are manufactured so every case has an analytic solution for validation (pages 4, 6, 9, 10).

## Method

- Feedforward MLP, 4 hidden layers, 20 neurons each, tanh activation (page 3). Inputs (x, t); output u for forward, plus the unknown parameter or force for inverse (Fig. 4, page 5).
- All derivatives, including u_xxxx, come from automatic differentiation (Fig. 4, page 5; derived quantities via autodiff, page 10). No modified network. Precision and hardware are not stated anywhere.
- Forward loss, eq. (7), page 4: one mean of squared residuals over all training points, with initial, boundary and interior sets lumped into one set Delta (page 3). The Fig. 4 caption says this loss comprises the PDEs plus boundary and initial conditions (page 5). No per-term weights are shown.
- A "residual parameter" scales the PDE term: 0.1 for the single EB beam (page 3), 1 for the EB double beam (page 10), 1 for the Timoshenko inverse and double cases (pages 8, 12). The formula for this parameter is never written.
- Inverse loss, eq. (11), page 5: PDE mean plus a data misfit mean, two summed terms added.
- Main scaling device is nondimensionalization, eq. (8), page 4: u = u_bar/l, x = x_bar/l, t = c t_bar/l^2, f = f_bar l^3/(EI). Coefficients become order 1.
- Optimizer: L-BFGS only, no Adam anywhere in the paper. 15000 epochs for the single EB forward (page 3), 5000 iterations for the Timoshenko single inverse (page 8), 2500 epochs for the Timoshenko double inverse (page 12), 15K epochs for the Timoshenko double forward (Table III, page 11).
- Sampling: random points, no adaptivity. Single EB: 16000 points, N_i = 2000, N_b = 4000, N_int = 10000, stated for the dimensional run (page 3). Timoshenko single inverse: 1600 points, N_i = 200, N_b = 400, N_int = 1000 (page 8). Timoshenko double forward: 16000 or 1600 points (Table III, page 11).
- Inverse data: sensor-style points. Timoshenko single parameter case: 5000 points at five x locations, 0.2, 0.8, 1.8, 2.6, 3 (page 8). Double-beam parameter case: 5000 points at one location x = 1.8 (page 12). Force case: 5000 points at each of six locations (page 13).
- Validation: relative percent error R = ||u* - u||_2 / ||u||_2 x 100 against analytic solutions (page 4); FDM, PGNN and gPINN comparisons (Table I, page 7); derived quantities against exact curves (Fig. 9, Table II, pages 9-10); noise tests (Table V, page 12).

## Key results (author-reported)

- Failure case: the dimensional EB equation with coefficients 10^2 and 4 x 10^6 does not train. After 15000 L-BFGS epochs the loss is 10^14 and the optimizer sits in a local minimum (page 3, Fig. 2 page 4).
- Single EB forward: R = 5.3e-4 percent (page 4).
- Single Timoshenko forward: w within 3.3e-4 percent, theta within 2.8e-3 percent (page 6). Table I, page 7: PINN beats FDM (0.005615 percent for w), PGNN (0.002739 percent for w, 3.486727 percent for theta) and gPINN (0.249849 percent for w, 5.498449 percent for theta).
- Single Timoshenko inverse: alpha = 1.0136 learned against true value 1; a DNN with the same architecture gives 0.6124 (page 8).
- EB double forward, Table II, page 10: displacement R 1.9348e-5 and 4.3253e-5 percent; bending moment 9.6112e-4 and 6.5506e-4; velocity 1.9043e-3 and 2.0161e-3; acceleration 1.9011e-2 and 1.4442e-2 percent.
- Timoshenko double forward, Table IV, page 11: with 16000 points, theta1 1.6038e-3, w1 3.9302e-5, theta2 1.0826e-3, w2 7.8614e-5 percent; with 1600 points, 2.6211e-3, 2.503e-4, 4.9405e-3, 3.4904e-4 percent.
- Timoshenko double inverse: parameter rho A1 predicted as 1.0208 against true 1, from data at one location (page 12). Force learning R: 4.3271e-2 percent with no noise, 4.8688e-2 with 10 percent noise, 1.1123e-1 with 20 percent noise (Table V, page 12). The conclusions state less than 0.2 percent error at 20 percent noise (page 13).
- Sensitivity, page 13: mean accuracy of the learned force is 0.14313413 with noise on displacement only and 0.204627 with noise on rotation only, so force is more sensitive to rotation data.
- No wall-clock times, hardware or precision are reported. Cost talk is qualitative only (pages 11, 13).

## Adopt for our pipeline

1. Nondimensionalize before anything else. Their dimensional case with 4e6-size coefficients fails hard; the rescaled case trains to 5.3e-4 percent (pages 3-4). Our near-wall inverse-power coefficients are the same hazard, so map eta and the five fields so equation coefficients sit near 1. This supports our fixed-scales plan.
2. Keep our Adam then L-BFGS plan. L-BFGS alone worked for them only after nondimensionalization (pages 3-4), so treat scaling as the first lever and the optimizer as the second.
3. Start with a small tanh MLP, 4 layers by 20 neurons, and grow only if residuals stall (page 3). Cheap baseline for our five-output network.
4. Do not copy their lumped mean loss. Eq. (7) averages all point types together (page 4). Our 12 BCs mix value, slope, slip, flux and nonlinear radiation terms, so keep separate fixed scales per BC group and per equation.
5. Copy the R metric, relative L2 percent error (page 4), as a report next to our classical solve_bvp reference.
6. Copy the point-count ablation: they compare 16000 and 1600 points (Tables III and IV, page 11). Run the same 10x ablation on our collocation count.
7. Validate derived quantities by autodiff without extra loss terms; they check bending moment, velocity and acceleration this way (page 10). We can check f', theta' and wall fluxes the same way.
8. Note their sensor-based inverse recipe (page 8) for any later parameter identification, but keep it out of the forward baseline.

## Do not import

1. Initial-value structure. All their problems are time-dependent with ICs (pages 4, 6, 9, 10). Ours is a steady BVP on [0, L] with 12 boundary conditions, so their IC handling does not transfer.
2. Linear physics. Every system they solve is linear with manufactured analytic solutions (pages 4, 6, 9, 10). Our five equations are nonlinear with coupled radiation BCs, so their error levels are not expectations for us.
3. Unity coefficients. They set all nondimensional parameters to 1 (pages 4, 6, 9). Our parameters come from the supervisor and stay in configs; we scale variables, we do not reset physics values.
4. L-BFGS-only training. No Adam stage or schedule is reported (page 3). Keep our two-stage plan.
5. One lumped mean over all residual types, eq. (7) (page 4). It hides BC and PDE imbalance, which is unsafe for a stiff wall region.
6. Inverse-problem machinery: data terms, noise handling, sensitivity analysis (pages 5, 8, 12-13). Out of scope for our forward baseline.
7. Any runtime or cost figure. The paper reports none (pages 11, 13).

## Quotes worth keeping

1. "the optimizer is stuck in the local minima and hence will not converge even if the number of epochs is increased" (page 3).
2. "this paper uses nondimensional PDEs instead of dimensional PDEs in the loss function" (page 2).
3. "areas where the training point concentration is low account for more error" (page 4).
4. "The high-order derivatives of the physical equations in the loss function of gPINN make it challenging for autodifferentiation" (page 7).
5. "the minimum training data points required to solve the problem should be determined empirically by gradually increasing the number of training points" (page 8).

## Relevance

Medium. The fourth-order derivative handling, the nondimensionalization evidence and the dimensional failure case transfer to our stiff coefficients, but the paper solves linear time-dependent beam problems with analytic solutions, not a nonlinear steady five-field BVP.
