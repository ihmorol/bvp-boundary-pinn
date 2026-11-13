# Analysis: Micropolar nanofluid PINN, 2026

**TL;DR.** Closest physical precedent: four fields with orders (3,2,2,2), nine wall and far-field conditions, cutoff eta=7, trained Adam then L-BFGS then one-shot RAR near the wall. **Gives us:** BC residual vector assembly, float64 and SiLU evidence, four-level error reporting. **Avoid:** their eta=7, the 8D parametric input, and their printed equations (the dissipation term is inconsistent). **Relevance:** high.

## Record
- Title: "Physics-Informed Neural Networks for Dissipative Micropolar Nanofluid Flow with Microrotation Dynamics and Zero Nanoparticle Mass Flux" (p. 1).
- Authors: H.R. Soltani Motlagh, A.M. Amer, N.I. Ghoneim, A.M. Megahed, A.M. Abdallah, S.B. Issa-Zadeh (p. 1).
- Venue: Modelling (MDPI), 2026, vol. 7, article 145. DOI: 10.3390/modelling7040145 (p. 1). Received 21 June 2026, published 22 July 2026 (p. 1).
- Pages read: all 18. Text extraction plus direct image checks of pages 5 to 10. Everything needed was readable; Figure 3's small print is the only low-confidence item.

## Problem
- Steady 2D MHD micropolar nanofluid over a stretching sheet in a Darcy porous medium. Parent system (1) to (5), p. 4: continuity, momentum, angular momentum for microrotation N, energy with viscous and Joule dissipation plus heat generation, species transport with chemical reaction.
- Similarity variables (9) to (10), p. 5: eta = y sqrt(c/nu), u = cx f'(eta), v = -sqrt(c nu) f(eta). Four fields: f (flow), phi (microrotation), theta (temperature), varphi (concentration). No microorganism field.
- Reduced ODEs (11) to (14), p. 5. Derivative orders (3,2,2,2): momentum carries f''' and couples to phi'; the other three are second order. Eleven dimensionless groups are listed on p. 6.
- Nine boundary conditions (15) to (16), p. 5. Wall, five: zero nanoparticle flux varphi'(0) = -(Gt/Gb) theta'(0); theta(0) = 1; f'(0) = 1; microrotation slip phi(0) = -omega f''(0); Stefan blowing f(0) = (delta/Le) varphi'(0). Far field, four: f', theta, phi, varphi go to zero as eta goes to infinity.
- Two wall conditions couple different fields: wall flow tied to the concentration slope, and the concentration flux tied to the wall temperature slope. The domain is semi-infinite and gets truncated at eta = 7 (p. 8).

## Method
- Parametric PINN in JAX. Input X in R^8 = [eta, chi, M, delta, Ec, Gt, Gb, Upsilon]; output [f, phi, theta, varphi] (p. 6, Figure 2 p. 7). One network covers an 8D parameter space.
- MLP with 128 neurons per hidden layer; the depth L is not stated. SiLU activation, He-Normal init, and input injection: the raw input vector is concatenated into every hidden layer, Eqs. (19) to (21), pp. 6-7.
- Float64 for all network and AD operations (p. 7). Derivatives up to third order come from forward-mode AD, picked over finite differences because there is one input variable, so it is claimed to be optimal and exact to machine precision (p. 7).
- Loss assembly, Eqs. (22) to (30), pp. 7-8. Four PDE residuals r_f, r_phi, r_theta, r_varphi, Eqs. (22) to (25). Boundary residual vectors: r_bc0 at eta = 0 with the five wall components, Eq. (26); r_bc_inf = [f', phi, theta, varphi] at eta = 7, Eq. (27). Total loss L_PDE + L_BC, Eq. (28). L_PDE is the point mean of the sum of squared residuals, Eq. (29). L_BC is the mean of the squared 2-norms of the two boundary vectors, Eq. (30).
- No explicit loss weights anywhere. Residuals enter with equal weight and no residual scaling or normalization is mentioned (p. 8).
- Sampling: Latin Hypercube over the 8D space; 50000 interior points, 5000 boundary points (Figure 3, p. 8).
- Optimizer chain (Figure 2 p. 7, Figure 5 p. 9): Adam, labeled "global search", runs to about 50000 epochs, then "L-BFGS + 8D-RAR (local fine-tuning)" to about 70000 total. No learning rates, batch sizes, tolerances, or stopping rules are reported.
- RAR is one-shot after the hybrid phase; new points are drawn across the full parameter space (p. 8). Figure 4, p. 9, shows the added points pile up at eta below 1, in the wall layer.
- Compute (Figure 6, p. 10): training 311.5 min with about 12.5 GB of GPU memory; inference 16.3 ms for 5000 points.

## Key results (author-reported)
- Final training metrics, Figure 5, p. 9. MSE: PDE 1.12e-05, total 9.41e-06, BC 1.29e-06. RMSE: PDE 3.34e-03, total 3.07e-03, BC 1.14e-03. MAE: PDE 1.98e-03, total 1.77e-03, BC 7.70e-04. Max abs error: total and PDE 1.49e-01, BC 1.46e-02. The text calls these deviations from "reference solutions" (p. 9) but names no reference solver, and the plotted curves are the loss terms.
- Table 1, p. 10: wall shear -f''(0) against the finite element results of Kumar [24] for four Omega values with omega = M = 0.5, delta = 0, chi = 0. Cases: 1.1890 vs 1.18899952147; 1.0590 vs 1.05895978902; 0.9676 vs 0.96757975839; 0.8424 vs 0.84238819547. The paper reports max relative deviation 0.0038 percent at Omega = 0.5 and min 0.00004 percent at Omega = 0.
- Reported trends: porous chi and magnetic M raise skin friction and couple stress, M and Ec lower the Nusselt number (Figure 13 discussion, pp. 15-16). The zero flux condition drives near-wall concentration negative (pp. 11-14).
- The abstract (p. 1) and conclusion 2 (p. 17) say thermophoresis lowers the Sherwood number, but the Figure 13d paragraph (p. 16) says it raises it. This is an internal contradiction.

## Adopt for our pipeline
- Far-field handling: truncate at finite L and impose a residual vector of decay conditions at eta = L, Eq. (27), p. 8. They leave f(inf) free and pin f' only; our 12 conditions fix the set per field, with L taken from our classical solve_bvp reference, not from their 7.
- One-shot residual enrichment late in the chain, after Adam then L-BFGS, with new points concentrated at the stiff wall (pp. 8-9). This matches our residual enrichment comparison and suggests testing one late round against iterative resampling.
- Report PDE and BC error groups separately at four levels (MSE, RMSE, MAE, max), Figure 5, p. 9. Cheap to add to our monitor point set.
- SiLU instead of tanh as a controlled test. Their reason is C-infinity smoothness for third-order AD (p. 7); our tanh is also C-infinity, so the swap is fair and cheap.
- Float64 for high-order AD supports our float64 plan (p. 7). Aligned evidence, not proof.
- Input injection, Eqs. (19) to (20), pp. 6-7: feed eta into every hidden layer. Testable small variant for our 1D net.
- Forward-mode AD for first derivatives of a single-input network (p. 7). We can test forward-over-reverse for the fourth-order f term.
- Their equal-weight MSE is a baseline for our soft weighted penalties: the paper reports BC MSE 1.29e-06 with plain equal weights on a coupled stiff system (Figure 5, p. 9). Run one unweighted control.

## Do not import
- Their eta-infinity of 7. Our kappa = b = Da = 0.1 wall terms change the decay rate, so pick L from the reference solve (their choice: p. 8).
- The parametric 8D network. We solve one parameter set, so it only adds training cost (p. 6).
- "Machine-precision derivatives" as an accuracy claim (p. 7). Exact AD derivatives do not make the solution accurate.
- Conclusion 5, that transfer learning cuts training time (p. 17). No transfer learning experiment appears in the body. Unsupported.
- Table 1's 11-decimal wall shear values (p. 10). The reference gives 5 significant figures, so the extra digits are noise.
- Their validation scope: one wall quantity, four cases, blowing switched off with delta = 0, and the other seven parameters for those cases are not stated (p. 10). We validate all five fields and wall quantities against solve_bvp.
- Their point counts. 50000 interior and 5000 boundary points serve an 8D parametric space (Figure 3, p. 8); our 1D problem needs far fewer.
- Equations taken at face value. Eqs. (13) and (24) print the dissipation term as Ec f''(Omega f'' + Omega phi) (pp. 5, 7), while Figure 3 (p. 8) and parent Eq. (4) (p. 4) carry a (1+Omega) f'' term. The paper also swaps phi and varphi in places (Eq. 13 p. 5, Eq. 24 p. 7, Figure 3 p. 8). Check every transcription against the enlarged source page before coding.

## Quotes worth keeping
- "the far-field is truncated at a finite boundary η∞ = 7.0 large enough to capture the asymptotic decay accurately" (p. 8).
- "we employ the Sigmoid Linear Unit (SiLU) across all PINN hidden layers to ensure surrogate model differentiability" (p. 7).
- "all network computations and automatic differentiation operations are enforced in 64-bit floating-point precision (Float64), guarding against underflow" (p. 7).
- "a one-shot Residual-based Adaptive Refinement (RAR) strategy is applied following an initial hybrid optimization phase" (p. 8).
- "a single refinement step is sufficient to capture the remaining boundary layer features without iterative resampling" (p. 8).

## Relevance
- High. Same family of stiff coupled similarity BVPs with flux, slip and blowing wall couplings, a finite far-field cutoff, and an Adam then L-BFGS then RAR chain; it stops short of our fifth field, fourth-order momentum, and coupled radiation conditions, so it is a precedent, not a template.
