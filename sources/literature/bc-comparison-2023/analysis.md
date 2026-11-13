# Analysis: Dirichlet BC enforcement comparison, 2023

**TL;DR.** Compares four Dirichlet enforcement routes; exact lifting beats tuned penalties. **Gives us:** the B6 lifting pattern, the penalty-scale warning (wrong lambda collapses the rate from 3.66 to 0.01), and the fairness protocol. **Avoid:** Nitsche and VPINN machinery; only value conditions were tested. **Relevance:** high.

## Record
- Title: "Enforcing Dirichlet boundary conditions in Physics-Informed Neural Networks and Variational Physics-Informed Neural Networks" (p.1).
- Authors: S. Berrone, C. Canuto, M. Pintore, N. Sukumar (p.1).
- Year: 2023. The PDF read is arXiv:2210.14795v2 [math.NA], dated 31 Jul 2023 (p.1). The project record lists the journal version as Heliyon 9(8):e18820 (2023); the PDF itself is the preprint.
- Pages read: 22 of 22. Plain text extract plus direct PDF reads of pages 6-7, 9-10, 12-13, 14-18 to recover equations and figure legends. Nothing was unreadable.

## Problem
- The paper compares four routes for imposing Dirichlet boundary conditions in PINNs and interpolated VPINNs (Sec. 1, p.2; Sec. 3, p.5).
- Test problems: a linear second-order elliptic PDE on an L-shaped domain with a simple and an oscillatory exact solution (Sec. 4.1, p.7); a nonlinear parametric elliptic problem on the unit square (Sec. 4.2, p.12); linear elasticity on an L-shaped domain (Sec. 4.3, p.13); a stabilized Eikonal equation on the same domain (Sec. 4.4, p.15); a 1D space-time convection problem with beta = 30 (Sec. 4.5, p.16).
- Neumann and Robin conditions are excluded by scope: "they can be weakly enforced by the trained VPINN due to the chosen variational formulation" (Sec. 3, p.5). Every condition tested is a value condition.

## Method
- M_A, penalty: add lambda times the summed squared misfit at N_B boundary control points to the loss, Eq. 3.1-3.2 (Sec. 3, p.5). Lambda is a fixed hyperparameter, chosen per run from {10^3, 1, 10^-3} in the experiments (Fig. 4 and Fig. 8 captions, p.8 and p.10; Fig. 15 legend, p.15).
- M_B, output modification: append a non-trainable layer Bw = g + phi w, Eq. 3.3 (Sec. 3, p.5). g is an extension of the boundary data, built by transfinite interpolation when unknown (p.6). phi is an approximate distance function (ADF) to the Dirichlet boundary, normalized to order m so that phi = 0, d(phi)/dn = 1 and higher normal derivatives vanish, Eq. 3.4 (p.6). This forces every Dirichlet value exactly for any network output.
- M_C, output modification without normalization: same lifting but phi = product of segment ADFs (Sec. 3, p.6). All derivatives stay bounded, but phi can be tiny near boundary corners.
- M_D, Nitsche: VPINN only. The test space is enlarged and the residuals gain two boundary terms, a (w - g) flux term and gamma times h^-1 times (g - w), Eq. 3.5 (Sec. 3, p.7). Gamma tested in {0.1, 1, 10} (Fig. 6 and Fig. 10 captions, p.9 and p.11).
- Protocol: one fixed architecture for all routes, 4 hidden layers of 50 tanh neurons with a linear output layer (Sec. 4, p.7). Adam with an exponentially decaying learning rate, then BFGS until iterates repeat to machine precision (p.7). PINN runs add an L2 weight regularization of 10^-6, picked by experiments (p.7). Epoch budgets: 5000 Adam epochs in Sec. 4.1 (Fig. 7, p.10), 10000 in Sec. 4.2 (Fig. 12, p.13), about 18000 in Secs. 4.3-4.5 (Figs. 15, 17, 18).
- Metrics: H1 error against the exact solution or a fine FEniCS reference (p.13-15), training loss curves, and empirical convergence rates under mesh refinement (Sec. 4.1, p.8-9). The BC error itself is never reported separately.

## Key results (author-reported)
- Simple solution: all four routes reach the theoretical VPINN rate k_int = q + 2 - k_test when the loss is well minimized; m = 1 or 2 in M_B and gamma in {0.1, 1, 10} in M_D give nearly identical rates (Sec. 4.1, p.8; Figs. 4-6).
- Oscillatory solution: only M_B and M_D keep the theoretical rate. M_A rates drop from 3.66 at lambda = 10^3 to 0.01 at lambda = 10^-3 for q = 3 (Fig. 8a, p.10). M_C rates fall to 2.36 and -0.43 (Fig. 9, p.11). M_A needs large lambda here: "large values of lambda are required to properly enforce the Dirichlet boundary conditions" (Sec. 4.1, p.10).
- Training speed: M_B converges fastest and to the most accurate solution; M_C barely trains under Adam alone and needs BFGS; M_A behaves the same under both optimizers (Sec. 4.1, p.9, Fig. 7).
- Nonlinear parametric problem: higher lambda gives lower error for M_A, and "the most stable and accurate approach remains M_B" (Sec. 4.2, p.12, Figs. 12-13).
- Elasticity: best H1 error 0.020 (VPINN), 0.070 (PINN, Delaunay points), 0.047 (PINN, uniform points); M_B is always the most efficient, and the point distribution does not change the ranking (Sec. 4.3, p.14-15, Fig. 15). Wrong lambda (10^-3 or 1) ruins accuracy and speed (p.14). M_D performs like M_A with a reasonable lambda (p.14).
- Eikonal: M_B always most accurate; best H1 error 0.021 (VPINN), 0.085 and 0.029 (PINN). lambda = 10^-3 destroys convergence and lambda = 10^3 needs the second-order optimizer; M_C and M_D show similar failures, with M_C failing outright in the uniform-point PINN (Sec. 4.4, p.16-17, Fig. 17).
- Convection: with VPINN, M_A is slightly better than M_B when lambda is chosen well; with PINN, exact imposition wins for any lambda. Best H1 error 0.077 (VPINN), 0.125 and 0.051 (PINN). M_D is not competitive (Sec. 4.5, p.16-18, Fig. 18).
- Interpolation stabilizes the VPINN: for non-interpolated VPINNs the error grows again once each hidden layer exceeds about 100 neurons, while the interpolated error stays flat and lower across network sizes (Sec. 4.1, p.10-12, Fig. 11).
- Overall verdict: M_B is best, M_C is suboptimal, M_A works but only with a well-chosen lambda, M_D sometimes matches M_A and sometimes needs a second-order optimizer (Sec. 5, p.18).
- Caveat reported by the authors: the Laplacian of the normalized ADF blows up at boundary corners, so collocation-PINNs should avoid collocation points near Dirichlet corners (Appendix A.1, p.21-22; Sec. 3, p.6).

## Adopt for our pipeline
- B6 exact construction: the non-trainable lifting Bw = g + phi w, Eq. 3.3 (p.5), is the pattern to copy. Our domain is the eta interval [0, L], so the 2D segment ADF machinery reduces to a polynomial factor. For each output field, use a factor that vanishes at the endpoints where that field carries a value condition, for example eta(L-eta)/L scaled so the endpoint slope is 1. A polynomial factor has bounded derivatives of every order, so the corner blow-up they report (Appendix A.1, p.21-22) cannot appear in 1D.
- One lifting per field: our five outputs have different BC sets, so f, g, theta, our phi field and chi each get their own factor and their own extension of the boundary data. The paper applies one network per unknown too (one or two output neurons, p.7), so this matches their construction.
- Split of our condition set: value conditions at eta = 0 and eta = L can be lifted exactly. Slope, slip, flux, and the coupled nonlinear radiation conditions stay in the soft loss. The paper itself notes the penalty route "can be easily adapted to impose other types of boundary conditions" (Sec. 3, p.5); exact lifting of derivative conditions is not tested there and would need a squared or Hermite-type factor as our own extension.
- Penalty scales: their evidence backs our fixed-scale soft weights, but only if the scales are large. They fixed lambda per run and compared {10^3, 1, 10^-3}; the smallest value stalled or ruined runs (Fig. 8, p.10; Fig. 17, p.16-17). Our tuning gate should sweep BC weight scales over orders of magnitude instead of picking one.
- Optimizer protocol: Adam with decayed learning rate then BFGS, identical across routes (p.7), matches our Adam then L-BFGS plan. Their Fig. 7 shows some routes only converge once the second-order phase starts (M_C, Sec. 4.1, p.9), so B6 and the baseline must get the same full budget with the phase split reported.
- Fairness protocol: same architecture, same collocation or interpolation points, only the BC handling changes (Sec. 2.1, p.3; Sec. 4, p.7). Copy this for B6: same network, same eta collocation set, same epoch budget, only the endpoint treatment differs.
- Metrics: judge arms on interior error against a trusted reference plus per-condition BC violation, not on the loss. The paper shows loss and error can decouple: "all the loss functions are decreasing even when the error is constant" (Sec. 4.1, p.9). Our plan to log interior residual norms and BC residuals separately covers this.
- Weight regularization: their PINN runs needed a small L2 weight penalty (10^-6) for stability (p.7). Worth one test in our tuning gate, logged as author-suggested, not supervisor-set.

## Do not import
- Nitsche route M_D: it needs a variational weak form, integration by parts, and an enlarged test-function space (Sec. 3, p.7). Our pipeline is a collocation PINN with no test functions, and the paper reports it suboptimal anyway (Sec. 5, p.18).
- M_C product ADF without normalization: the paper reports suboptimal results and outright convergence failures for it (Fig. 9, p.11; Fig. 17c, p.17; Sec. 5, p.18). In 1D the product and normalized forms nearly coincide, but there is no reason to pick the weaker variant.
- VPINN machinery: test functions, quadrature rules, interpolation operator, mesh refinement studies (Sec. 2.2, p.3-4). Our model is a collocation PINN on one fixed eta grid; none of it transfers.
- The 2D segment ADF formulas, signed distance, trimming function, transfinite interpolation (Sec. 3, p.5-6): our boundary is two points and the data there is explicit. Only the 1D analogue survives.
- The blanket claim that exact imposition beats penalties: their proof covers Dirichlet value conditions on second-order operators (Sec. 5, p.18). Our arm B6 lifts only part of twelve conditions, and our f equation is fourth order, so their ranking is evidence, not a prediction.
- Their convergence-rate-versus-mesh results (Sec. 4.1, p.8-11): we run one fixed domain length per solve, not a mesh refinement study.

## Quotes worth keeping
- "modifying the output of the neural network to exactly match the prescribed values leads to more efficient and accurate solvers." (Abstract, p.1)
- "We do not consider Neumann or Robin boundary conditions since they can be weakly enforced by the trained VPINN due to the chosen variational formulation" (Sec. 3, p.5)
- "It is important to remark that all the loss functions are decreasing even when the error is constant." (Sec. 4.1, p.9)
- "the correct values are problem dependent and can be often found only after a potentially expensive tuning." (Sec. 4.3, p.14)
- "the choice of the additional penalization parameter is crucial because wrong values can prevent convergence to the correct solution or dramatically slow down the training." (Sec. 5, p.18)

## Relevance
High: this is the closest published analogue of our B6 arm, with the same four-route framing, fixed penalty scales, Adam then BFGS, and an exact construction that beats tuned penalties; but it tests only Dirichlet value conditions on second-order problems, so it guides the B6 design without settling how it will behave on our slope, slip, flux, and coupled radiation conditions.
