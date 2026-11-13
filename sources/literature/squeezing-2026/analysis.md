# Analysis: Squeezing flow heuristic PINN, 2026

**TL;DR.** Closest fourth-order fluid example: fields (4,2,2), eight plain BCs, constant coefficients, metaheuristic optimizer. **Gives us:** the penalty tuning rule (drive BC errors down to the interior level), seed-spread reporting, the Theil coefficient, a 1e-9 collocation reference. **Avoid:** PSO-NNA (no settings given); the journal is un-audited and the headline 9.3e-9 is the best of 100 runs. **Relevance:** medium.

## Record

- Title: "A Heuristic Physics-Informed Neural Network for Unsteady Magnetohydrodynamic Squeezing Flow through a Darcy-Forchheimer Porous Channel with Brownian Motion, Thermophoresis, Thermal Radiation, and Heat Generation" (p. 1).
- Authors: Umar Daraz (University of Education Lahore), Khaleel Ahmad (The University of Lahore; Khazar University) (p. 1).
- Venue: International Scientific Spectrum, listed as "Volume 00, Issue 00 (2026) 1-41", ISSN 3104-3305. DOI: 10.66972/iscis202617 (p. 1).
- Dates: received 7 June 2026, revised 19 September 2026, accepted 23 September 2026, online 26 September 2026 (p. 1).
- Pages read: all 41. Full text was extracted with pdftotext; pages 14 to 17 were also viewed as PDF images to confirm Eqs. 39 to 44, Algorithm 1 and Section 8 wording. Nothing was unreadable.

## Problem

- Setting: unsteady squeezing of a hybrid nanofluid between parallel plates, gap 2h(t) with h(t) = H sqrt(1 - alpha t); similarity variable eta = y / (H sqrt(1 - alpha t)) (p. 4-5, Eqs. 1-7).
- Three fields: f (stream function), theta (temperature), phi (nanoparticle volume fraction). No microorganism field (p. 6, Eqs. 16-18).
- Orders (4, 2, 2). The system on eta in [0, 1] (p. 6, Eqs. 16-18):
  - f'''' - S A1 (eta f''' + 3 f'' + f' f'' - f f''') - Ha^2 A2 f'' - Kp f'' - Fr f' f'' = 0
  - (1 + Rd) theta'' + Pr S A3 (f theta' - eta theta') + Nb theta' phi' + Nt theta'^2 + Hs theta = 0
  - phi'' + Sc S (f phi' - eta phi') + (Nt / Nb) theta'' = 0
- All coefficients are constant parameter groups times O(1) values (S=1, Ha=0.5, Kp=0.5, Fr=0.3, Pr=7, Rd=0.3, Nb=0.3, Nt=0.3, Sc=5, Hs=-1, p. 17). No inverse-power wall terms, so no stiffness of the kind our problem has.
- Eight BCs, all plain Dirichlet or Neumann (p. 6, Eq. 18). At eta=0: f=0, f''=0, theta'=0, phi'=0. At eta=1: f=1, f'=0, theta=1, phi=1. No slip length, no flux condition, no condition that couples two fields.

## Method

- Network: one separate shallow net per field. Scalar input eta, one hidden layer of H=10 logistic sigmoid units, linear output weights (p. 14, Eq. 39). Figure 5 draws 21 inputs and 21 outputs, but that is the reporting grid drawn as nodes, not extra fields (p. 15).
- Derivatives: closed form. Sigmoid derivatives up to order four are polynomials in s = sigma(w eta + b): sigma' = s(1-s), sigma'' = sigma'(1-2s), sigma''' = sigma'(1-6s+6s^2), sigma'''' = sigma'(1-14s+36s^2-24s^3), combined as g^(m) = sum alpha_j w_j^m sigma^(m) (p. 14, Eqs. 39-40). The paper reports no special trick for fourth-derivative stability; the fourth derivative is bounded and analytic for this activation, the layer is small, and the domain is the unit interval. The derivative-free search also never backpropagates through the residual.
- Loss (p. 14, Eqs. 41-42): mean over Nc = 41 uniform collocation points of Rm^2 + Re^2 + Rc^2 in one bracket, plus lambda times the sum of squared boundary residuals over the eight conditions. Equal weight per residual; single penalty weight lambda with no reported value. Tuning rule: "lambda is chosen large enough to drive the boundary residual to the level of the interior residual" (p. 15).
- Optimizer: hybrid PSO then NNA then polish (p. 15-17, Eqs. 43-44, Algorithm 1). PSO moves a population of weight vectors; the neural network algorithm of Sadollah et al. [40] then forms new candidates as normalized weighted combinations of the population with bias and transfer operators; a trust-region reflective least-squares step of Levenberg-Marquardt type polishes the winner. All settings are symbolic: population size n and iteration budgets are inputs, and omega, c1, c2, beta decay get no numbers (p. 15, 17). No Adam or L-BFGS run at all; the stated reason is only that for small shallow nets a derivative-free global search is often more reliable than gradient descent (p. 2). Each field has 30 trainable weights.
- Classical reference (Section 6): first-order conversion y = (f, f', f'', f''', theta, theta', phi, phi')^T, eight states closed by the eight BCs (p. 13, Eq. 38). Solver: "residual controlled collocation method that is mathematically equivalent to the bvp4c routine", fourth-order Lobatto collocation, adaptive meshing until the maximum residual falls below 1e-9 (p. 13-14). The reference both scores the PINN and labels the surrogate (p. 14).
- Supervised surrogate (Section 8): separate 10-unit tanh net fit to the dense reference data by Levenberg-Marquardt (Eq. 47), 70/15/15 train/val/test split, inputs and targets mapped to [-1, 1], early stop after six failed validation epochs (p. 17). Used only for performance, training-state, histogram and regression plots (p. 17). The physics PINN never sees reference data.
- Fixed-point layer (Section 5): lift fp = (3 eta - eta^3)/2 with theta_p = phi_p = 1 satisfies all eight conditions exactly (p. 9, Eq. 29); Green functions K and g convert the system to a fixed point equation (p. 9, Eqs. 30-31); a defect bound gives an a posteriori error certificate with amplification about 1.37 (p. 13, Eq. 37).
- Sampling and validation: fixed uniform 41 interior points and a fixed 21-node reporting grid, no resampling. Accuracy is MSE (Eq. 46) and Theil inequality coefficient (Eq. 45) against the reference (p. 16). The hybrid is rerun 100 times from random populations for four scenarios, velocity at Ha = 0 and 0.1, temperature at Hs = -1 and -1.5, with per-node min/max/mean/std tables (p. 29-34). A symbolic algebra check confirms the reduction and that the network residuals match the manuscript equations (p. 37, Sec. 10).

## Key results (author-reported)

All numbers below are author-reported. We did not rerun anything.

- Optimizer ordering on velocity: PSO alone reaches error order 1e-4, NNA alone 1e-5, the hybrid 1e-8 to 1e-9 (p. 26, Fig. 15).
- Velocity at Ha = 0.1: hybrid maximum absolute error 9.300e-9 at eta = 1, fitness 5.633e-11 (p. 26-27, Table 3).
- Temperature: hybrid maximum error 8.500e-7 at Hs = -1 and 2.183e-7 at Hs = -1.5 (p. 27, Figs. 16-17).
- Concentration: hybrid maximum error 1.041e-6 and 1.918e-6 for the same two cases (p. 28, Figs. 18-19). No 100-run statistics for concentration.
- 100-run averages: velocity mean fitness 1.406e-7 at Ha = 0, mean Theil coefficient 2.096e-6; temperature MSE 5.272e-12 at Hs = -1; per-node average error near 1e-6 to 1e-5 across the domain (p. 29-30, Tables 4-7). Note the gap: the headline 9.3e-9 is the best run, while typical runs average about 3e-6 with maxima near 4.5e-5 (p. 31-32, Tables 4-5).
- Surrogate: regression R = 1.000000 for velocity, R = 0.999999 for temperature, validation stop at 24 epochs (p. 34-35, Figs. 23-24).
- Analysis numbers: sum of Green norms M_k = 0.77625 (p. 10, Lemma 2); global contraction modulus 31.14 at baseline, contraction only when parameters are scaled by kappa near 0.085 (p. 12); local spectral radius 0.270 at the computed solution (p. 12).
- Claimed gain: the hybrid beats either metaheuristic alone by four to five orders of magnitude (p. 37, Sec. 9.8).

## Adopt for our pipeline

- Derivative audit: we train through f'''' with autodiff. Follow their practice of an independent check: verify autodiff fourth derivatives of tanh against hand-derived formulas once, like their symbolic check of the reduction (p. 14, 37).
- Penalty tuning criterion: choose per-condition weights so boundary residuals drop to the interior residual level, not a fixed huge constant (p. 15). Fits our soft weighted penalties.
- Reference discipline: report the solve_bvp residual tolerance alongside accuracy claims, and score the PINN on fixed reporting points plus a fresh final-evaluation set (p. 13-14). They only use a 21-node reporting grid; our four point sets are stronger, keep ours.
- Theil inequality coefficient next to MSE in our result tables; it is scale-invariant and costs one line (p. 16, Eq. 45).
- Seed spread: report min/max/mean/std over several seeds for each controlled comparison. Their Tables 4-5 show why: best run 9.3e-9, typical runs 3e-6 (p. 29-34).
- Polynomial lift: fp = (3 eta - eta^3)/2 meets all four f-conditions exactly (p. 9). For us, only the simple value and slope conditions among our twelve could be built in this way; this supports our exact endpoint construction test, with slip, flux and coupled radiation conditions left to penalties.
- Surrogate labeling: if we ever fit a surrogate to reference data, mark it as reference-supervised and keep it outside the physics-only claim, as they do (p. 17). Our baseline must not use it at all.
- Report best-run and mean-run accuracy side by side; they imply it, we should state it.

## Do not import

- PSO-NNA optimizer: hyperparameters are not given, so the result is not reproducible from the paper, and no gradient-based baseline is run for comparison. Our Adam then L-BFGS plan stays.
- Per-field 10-unit sigmoid nets: no capacity margin for our five coupled fields and stiff wall layer. Keep the 4x64 float64 tanh MLP.
- Equal residual weights in one bracket (Eq. 41): their residuals are O(1) by nondimensional construction; our five residuals differ in scale and grow near the wall, so we keep fixed per-field scales.
- LM trust-region polish as the accuracy source: their 1e-9 comes from a least-squares polish on 30 parameters per field. Our network has orders of magnitude more parameters, and adding the polish would change the method under test.
- Surrogate training on reference data: conflicts with the repo rule that the classical reference never feeds training labels in the baseline.
- Green function fixed-point machinery for training: it needs a constant-coefficient biharmonic operator and simple homogeneous BCs; our inverse-power wall coefficients break the clean kernel. Keep only as a citation for residual-to-error certificates.
- Their parameter values and hybrid-nanofluid constants: different fluid model. Our configs keep supervisor-given values and nulls.

## Quotes worth keeping

- p. 14: "Rather than building the boundary data into a trial form, the eight conditions of Eq. (18) are enforced through the loss".
- p. 15: "lambda is chosen large enough to drive the boundary residual to the level of the interior residual" (lambda rendered as the symbol in the original).
- p. 2: "For the small shallow networks that arise in similarity reduced problems, a derivative free global search is often more reliable than gradient descent".
- p. 13-14: "adapts the mesh until the maximum residual falls below 10-9" (sentence spans the page break).
- p. 17: "This surrogate is used only to generate the performance, training state, error histogram and regression plots discussed below."

## Relevance

- Medium. It is the closest verified nonlinear fourth-order fluid example in our list, with a working penalty-enforced BC setup and a 1e-9 collocation reference, but its equations have smooth constant coefficients, its eight BCs are all value or slope type, and its accuracy rests on a metaheuristic plus least-squares path we do not follow.

## Quality caveats

- The "Volume 00, Issue 00" header means the article is not yet assigned to a journal issue; the venue is a new journal outside the main fluid mechanics publishers, so treat the claims as not externally validated (p. 1).
- Not audited, as far as the text shows: no code or data availability statement, no compute cost or runtime, no PSO/NNA/lambda settings, no concentration statistics over the 100 runs, and the profile matches are visual ("indistinguishable from the reference curve", p. 18; "to within the line width", p. 25; "reproduced exactly by the network", p. 22).
- The reference solver is both the accuracy judge and the surrogate's teacher, so all accuracy is relative to a bvp4c-equivalent solve, not to an exact solution (p. 14).
- Nothing in the paper addresses stiff near-wall coefficient growth, slip or flux conditions, or coupled boundary conditions, which are exactly our hard parts.
