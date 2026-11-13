# Analysis: Stretching cylinder PINN, 2026

**TL;DR.** Stretching-cylinder PINN: three fields, third-order flow, simple decay BCs, equal loss weights. **Gives us:** the equal-weight control arm, the per-field error table template, plateau LR decay with gradient clip 1.0. **Avoid:** no wall-shift curvature terms, so it does not cover our stiffness; solver naming and metrics are inconsistent. **Relevance:** medium.

## Record
- Title: "Physics informed neural network framework for non-similar nano-heat and solutal transport in a rheological fluid over a stretching cylinder" (p.1).
- Authors: Muhammad Naveed Khan, Hameed Ullah Khan, N. Ameer Ahammad, Nurul Amira Zainal, Nadia Ayari (p.1).
- Year 2026. Venue: Discover Nano, volume 21, article 502 (p.1). DOI: 10.1186/s11671-026-04962-3 (p.1).
- Pages read: all 25. Plain text plus rendered page images for pages 4 to 11; equations checked against the images.
- Fluid: Jeffrey fluid with nanoparticles, Buongiorno model. No microorganism field exists in this paper (pp.3-4).

## Problem
- Governing PDEs (1) to (4), p.4: continuity, momentum with Jeffrey stress, Darcy and Forchheimer drag, buoyancy with sin(x/a), Lorentz force; energy with Rosseland radiation, Brownian and thermophoresis terms, viscous dissipation, heat source; species with Arrhenius reaction.
- Physical fields: stream function f, temperature theta, concentration phi. Three fields only. Their chi is a streamwise derivative helper, not a microorganism density like ours (eq 18, p.6).
- Transformation (6), p.5: xi = x/a, eta = (y/a) sqrt(Gr), f = psi/(nu xi sqrt(Gr)).
- Curvature handling: eta has no wall shift. No (eta + kappa) denominators appear anywhere. Curvature enters only through sin(xi)/xi in the buoyancy term (eq 7, p.5) and through (xi cos xi - sin xi)/xi^2 in the differentiated momentum equation (p.6). The cylinder shape acts on the streamwise direction, not the wall-normal one.
- First-order truncation (14) to (17), p.6: momentum third order in f, energy and species second order. Seven BCs: slip f'(0) = 1 + lambda f''(0), f(0) = 0, theta(0) = 1, phi(0) = 1, then f', theta, phi decay to zero at infinity.
- Second-order truncation (18) to (24), pp.6-7: adds g = f_xi, h = g_xi, varphi = theta_xi, chi = phi_xi. Equation orders (3, 3, 2, 2, 2, 2) for f, g, theta, varphi, phi, chi. Fourteen BCs: eight at the wall (p.6), six decay conditions (24, p.7). The slip condition is differentiated too: g'(0) = lambda g''(0) (p.6).
- Domain 0 <= eta <= eta_inf (p.7). eta_inf is never given as a number. Figures plot profiles to eta = 8 (Figs. 3 and 5, pp.10-11).
- The streamwise station xi used in training is not stated (Sec 4, pp.7-8).

## Method
- PyTorch PINN with input eta only (p.7).
- Outputs: the text lists [f, theta, phi] (p.7). Fig. 2 shows five outputs f, theta, phi, g, h (p.7). The two statements do not agree.
- tanh activation, Xavier initialization (p.7). Three hidden layers of 80 neurons (p.8). No float precision stated.
- Derivatives by Autograd, first to third order (p.8).
- Residuals (26) to (28), p.8, match the first-order truncation: R_f, R_theta, R_phi only. No residual for g, h, varphi or chi appears anywhere, so the written loss trains three fields despite the second-order claim in section 4.1 (p.7).
- Loss (29) and (30), p.8: mean squared residuals over collocation points plus mean squared BC violation. Total = lambda_phys times L_physics plus lambda_BC times L_BC, with both weights equal to 1 (p.8 and Fig. 2, p.7). No per-equation or per-term weights.
- Sampling: 300 collocation points, uniform and fixed; boundary points at eta = 0 and eta = eta_inf; BCs enter as soft penalties (p.7).
- Optimizer: Adam, initial LR 5e-3, up to 15,000 epochs (p.8), though every reported table stops at 10,000 (Table 1, p.11). Gradient clipping with max norm 1.0 (p.8). ReduceLROnPlateau halves the LR after 1000 epochs without total-loss improvement (p.9). The Fig. 2 box also names RMSprop (p.7).
- Monitoring: total loss every 200 epochs; physics loss, BC loss and L2 error tracked throughout; the checkpoint with the lowest total loss is kept for all later analysis (p.9).
- Hardware: GPU through CUDA when available, otherwise CPU (p.9).

## Key results (the paper reports)
- L2 error 3.5e-3 with stable convergence through training (abstract, p.1).
- Table 1, p.11: L2 error falls from 0.261781 at 100 epochs to 0.003506 at 10,000.
- Fig. 3, p.10: f', theta and phi profiles overlap the reference already at 100 epochs.
- Fig. 4, p.10: boundary loss near 1e-7 and physics loss near 1e-5 by 10,000 epochs (read off the plot).
- Tables 2 to 6, pp.15-16: L2 errors for f'(eta), theta(eta), phi(eta) across M in {0.5, 1.5, 2.5}, Da in {0.2, 0.5, 1}, Q in {0.1, 0.2, 0.3}, Sc in {1, 1.5, 2}, Rd in {0.2, 0.6, 1}. Values run from about 1.1e-4 to 2.4e-2, and the largest errors sit on the velocity field.
- Table 7, p.21: skin friction Cf against Prasad et al. [30] for Pr from 0.5 to 7. Present values 0.4335 down to 0.2811; reference values 0.4375 down to 0.2826; gaps of at most about 0.004.
- Physics trends: M lowers velocity, Da raises it; Q raises velocity and temperature; Rd raises temperature; Sc lowers concentration; higher Q raises Cf and Sh and lowers Nu (pp.9-19).
- Missing definitions: the L2 error is never defined (which points, which norm, relative or absolute) and no base parameter table exists. Replication from the paper alone is not possible.

## Adopt for our pipeline
1. Optimizer schedule: Adam at LR 5e-3, ReduceLROnPlateau with factor 0.5 and patience 1000 epochs, gradient clip norm 1.0 (pp.8-9). Test this inside our Adam phase before L-BFGS.
2. Keep-best-checkpoint by total loss (p.9). Cheap, and it fits our monitor point set.
3. Track loss components and the L2 error against the classical solve during training, not only at the end (p.9). Same role as our monitor set.
4. Use lambda_phys = lambda_BC = 1 (Fig. 2, p.7) as the equal-weight baseline arm of our adaptive weight comparison.
5. Treat 300 uniform points (p.7) as the control arm for our resampling and residual enrichment tests. Their wall region is mild; ours is stiff, so we expect to need more points near eta = 0.
6. Report error per field and per parameter value on a fresh evaluation set, in the style of Tables 2 to 6 (pp.15-16). Matches our fresh final evaluation plan.
7. Note their differentiated slip bookkeeping, g'(0) = lambda g''(0) (p.6). If we test first-order helper outputs, every coupled BC must be differentiated consistently. This is the same trap our coupled radiation condition faces.

## Do not import
- Wall curvature handling: there is none. eta = (y/a) sqrt(Gr) has no shift (p.5). R15 does not cover our (eta + kappa) stiffness. Keep our own treatment.
- The LNS machinery: our baseline is a true ODE BVP in eta, not a non-similar PDE. The xi truncation has no target here. Revisit only if the supervisor asks for a streamwise extension.
- The second-order claim without second-order residuals (pp.7-8). If we add helper outputs, each one gets its own residual and its own BCs.
- The solver naming: "BVP4C" in the abstract and conclusions (pp.1, 22) versus "solve_bvp ... Python scipy" in section 5 (p.9). One reference solver, two names. We name our scipy reference exactly.
- Network size 3 x 80 (p.8): no reason to leave our 4 x 64 float64 plan; their precision is unstated.
- Far-field practice: decay to zero at an unstated eta_inf (pp.6-7). Our finite L with flux and nonlinear coupled radiation conditions gets no support from this paper.
- Unstated base parameters: no defaults for Pr, Ec, Nb, Nt, lambda, lambda_1, beta, Lambda, Gr, N, Kr, E or delta anywhere in the paper. Our configs keep nulls until the supervisor fills them, per repo rules.

## Quotes worth keeping
1. "A total number of 300 collocation points are used during training." (p.7, Sec 4.2)
2. "λ_BC = 1 and λ_phys = 1" (Fig. 2, p.7)
3. "Gradient clipping with the maximum norm equal to 1.0 is used to increase the numerical stability of backpropagation." (p.8, Sec 4.4)
4. "while the solve_bvp solver is employed in Python scipy as the reference numerical solution for validation" (p.9, Sec 5)
5. "experimental or high-fidelity CFD validations have not been performed" (p.22, Sec 7)

## Relevance
Medium: same geometry family (cylinder, mixed convection, nanofluid, MHD) with a plain PyTorch PINN recipe worth borrowing settings from, but a third-order f equation, three fields, no wall-shift curvature and simple decay BCs mean it touches none of our hard parts.
