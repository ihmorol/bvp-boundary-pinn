# Analysis: BC enforcement comparative study, 2025

**TL;DR.** Newest side-by-side of BC methods, including AL and annealing, on 2D and 3D tests. **Gives us:** AL wins on most tests, the split interior/boundary error metric pair, and evidence that tuned beta=100 can beat annealing. **Avoid:** min-max Lagrange (never converged); despite the abstract, no Neumann-data case was run. **Relevance:** high.

## Record

- Title: "Boundary condition enforcement with PINNs: A comparative study and verification on 3D geometries" (p. 1).
- Authors: Conor Rowan, Kai Hampleman, Kurt Maute, Alireza Doostan, Smead Aerospace Engineering Sciences, University of Colorado Boulder (p. 1).
- Year and status: arXiv:2512.14941v1 [math.NA], posted 16 Dec 2025. Preprint, v1, not peer reviewed (p. 1).
- Length: 34 pages. Main text pp. 1-28, references pp. 29-34. All pages read: full text extraction, plus direct page reads of pp. 13-19 and 22-27 for tables, equations and figures. No unreadable parts.

## Problem

- The paper compares BC enforcement techniques for PINNs trained with the strong form loss, then builds a general 3D solver framework around the winner (abstract, p. 1; Sections 1 and 4).
- Three methods are ruled out on paper: coordinate transformation, distance functions and PINN-FEM, because they need preprocessing and only handle Dirichlet conditions (Section 3, pp. 8-10, Table 1 p. 10).
- Five methods get a side-by-side test on one problem: standard penalty, learning rate annealing, SA-PINNs, min-max Lagrange multipliers and the Augmented Lagrangian (Section 3, pp. 11-14). A sixth route, solving the Lagrange stationarity system with quasi-Newton constrained optimizers, is also tested (pp. 14-16).
- BC types actually tested: inhomogeneous Dirichlet in the 2D comparison (pp. 11, 13); inhomogeneous Dirichlet in the 3D Fisher-KPP case (pp. 20-21); Dirichlet plus nonlinear Robin, flux plus sigma*u^4 radiation (pp. 22-24); clamped and inhomogeneous Dirichlet in the elasticity case (pp. 25-26).
- Honest note: the abstract claims Dirichlet, Neumann and Robin tests (p. 1), but no numerical example enforces Neumann traction data. Neumann terms exist only in the framework objective, Eq. 14 (p. 18).
- Nonlinearity sits in the PDE (Fisher-KPP reaction term, Eq. 16, p. 20) and in the BC (u^4 radiation, Eq. 18, p. 23). Nothing couples two distinct fields; the Robin condition ties the flux to the same field's value (p. 23).

## Method

- Standard penalty: adds (beta/2) times the squared BC residual to the strong form residual loss; beta is fixed by hand (Eq. 7, p. 11).
- Learning rate annealing: resets the BC weight each step to the ratio of gradient magnitudes of the PDE term and the BC term, smoothed with a moving average, alpha = 0.9 (Eq. 8, p. 11; settings p. 13).
- SA-PINNs: a trainable pointwise penalty field on the boundary, driven up by gradient ascent on the pointwise BC error (Eq. 9, p. 12).
- Min-max Lagrange multipliers: a multiplier field at boundary integration points, maximized while the network minimizes (Eq. 10, p. 12).
- Augmented Lagrangian: Adam inner loop on the penalty plus multiplier objective, then a multiplier update and penalty growth beta <- 2*beta, repeated until convergence criteria hold (Eq. 11, p. 12; Algorithm 1, p. 20).
- Test framework: manufactured solutions on a unit circle in 2D, then three 3D level-set geometries with marching cubes surface grids and no mini-batching (p. 13; Section 4, pp. 17-19).
- Metrics: relative L1 interior error I and boundary error B, both against the exact manufactured solution (Eq. 12, p. 14). So the paper checks true solution error, not only BC residual.
- Tuning protocol: the paper states each method's settings came from experimentation to maximize its performance (p. 14). 2D runs: MLP with two hidden layers of 25 tanh units, 1e4 Adam epochs, learning rate 5e-3 (1e-3 for the Lagrange method), 500 boundary points, beta in {1, 100} (p. 13). Augmented Lagrangian defaults: learning rate 1e-3 to 5e-3, beta0 = 1, lambda0 = 0, gamma = 2, plus gradient, boundary and objective convergence thresholds (Table 3, p. 19).

## Key results (author-reported)

- 2D Dirichlet test, averaged over the last 100 steps (Table 2, p. 14): penalty beta=1 gives I=2.2e-2, B=9.8e-2; penalty beta=100 gives I=4.8e-3, B=5.6e-3; LR annealing I=1.8e-2, B=9.0e-3; SA-PINN I=2.9e-3, B=8.7e-3; Lagrange multipliers I=3.2e0, B=3.8e0; Augmented Lagrangian I=1.2e-3, B=1.4e-3. The top three are LR annealing, SA-PINN and the Augmented Lagrangian (p. 14).
- Penalty magnitude: small beta leaves boundary error high; the paper warns that if beta is too large the optimizer ignores the physics loss, so beta is problem specific (p. 14). They rule out the fixed penalty method on tuning sensitivity grounds, even though tuned beta=100 beat LR annealing on interior error (p. 16).
- The min-max Lagrange method never converged, under Adam or SGD and any learning rate they tried (pp. 14, 16).
- Quasi-Newton constrained solvers (SciPy trust-constr and SLSQP) drive the BC error to zero, but the strong form loss stalls once boundary points reach realistic counts (Figure 6, pp. 15-16).
- SA-PINNs are sensitive to the penalty learning rate, and at their 3D resolution would add up to 50,000 variables, a fifty-fold increase in problem size (pp. 16-17).
- 3D Fisher-KPP: the paper reports Augmented Lagrangian B=0.89%, I=0.51% in 6385 steps, 1269.8 s (p. 21); LR annealing B=3.6%, I=2.9% in the same step count (p. 22).
- Nonlinear Robin heat transfer: Augmented Lagrangian B=0.66%, I=0.53% in 1702 steps, 464.9 s; LR annealing B=0.84%, I=0.74%, 587.3 s; the paper calls the two methods similar here (p. 24).
- Elastic pipe: Augmented Lagrangian I=0.51%, B=0.49% in 1803 steps, 1855.8 s (p. 26); LR annealing I=B=1.1%, 1788.1 s (p. 27). The inner loop converged at most 11 times, so the largest penalty reached was 2^11 = 2048 (p. 27).
- Verdict: the Augmented Lagrangian wins in 2D and in two of three 3D cases, and is the authors' preferred method (p. 28). High-frequency solutions remain an open problem for them (p. 28).

## Adopt for our pipeline

- Split metrics: adopt their relative L1 pair, one boundary error and one interior error per field against our reference solution (Eq. 12, p. 14). This matches our need to separate BC satisfaction from interior accuracy and to check true error, not only BC residual.
- Penalty scale evidence: their beta=1 versus beta=100 sweep shows a fixed scale can fail in both directions (p. 14). We keep our fixed scales, but should log B and I per BC group so a bad scale is visible, and run a small scale sweep before freezing values.
- Tuned fixed penalty can win: in their 2D test, beta=100 beat LR annealing on interior error (Table 2, p. 14). This supports our soft weighted arm, provided we document the tuning cost.
- Augmented Lagrangian arm: beta0=1, lambda0=0, gamma=2, Adam inner loop, convergence on gradient, boundary error and objective thresholds (Table 3, p. 19; Algorithm 1, p. 20). With our 12 scalar BCs the multiplier and penalty vectors stay tiny, so this is a fair third arm beside soft penalties and B6.
- Per-group LR annealing: they balanced Dirichlet and Robin terms independently (p. 24). Our analogue is per-field or per-BC-group weight updates with alpha=0.9.
- Reporting parity: the paper reports wall-clock time and step count per method (pp. 21-27). We should report the same for our Adam then L-BFGS arms.
- Their pattern, Augmented Lagrangian clearly ahead on the nonlinear PDE case but tied on the nonlinear BC case (pp. 21-24), is a testable hypothesis for our five-field nonlinear system.

## Do not import

- Level sets, marching cubes, surface grids and normals (Section 4, pp. 17-19). Our domain is the 1D interval [0, L]; the two endpoints need no geometry machinery.
- Min-max Lagrange multipliers: it failed to converge in all their trials (pp. 14, 16).
- Quasi-Newton constrained solves for BCs: they stall at realistic constraint counts (pp. 15-16). Caution note for us: our L-BFGS stays unconstrained on penalties, a different setup.
- SA-PINNs as-is: penalty learning rate sensitivity and slow penalty growth (pp. 16-17). Parameter bloat does not apply at our scale of 12 conditions, but the tuning risk does.
- Their accuracy numbers: the sub-1% results are for smooth manufactured solutions (pp. 21-27). Our problem has sharp near-wall structure in eta, so their magnitudes do not transfer.
- Their BC coverage claim: no example enforces Neumann data (see Problem), so their results are not evidence about pure flux, slip or coupled two-field conditions like ours.
- Their cost model: background grids of 75^3 and tens of thousands of integration points (pp. 21-26) are irrelevant to a 1D two-endpoint BVP.

## Quotes worth keeping

- "if the penalty is made too large, the optimizer will completely ignore the physics loss term" (p. 14).
- "The min-max formulation of the Lagrange multiplier problem does not converge within the allotted training period" (p. 14).
- "when the number of boundary integration points takes on realistic values of 100-500, the optimizer gets stuck at solutions with large values of the strong form loss" (p. 16).
- "treating BCs with SA-PINNs corresponds to a fifty-fold increase in the size of the optimization problem" (p. 17).
- "the Augmented Lagrangian method is the preferred strategy for handling BCs with PINNs in the opinion of the authors" (p. 28).

## Relevance

- High. It is the newest side-by-side of soft penalties against adaptive weighting and constrained enforcement, with true error metrics and tuned baselines, which is exactly our comparison question; the caveat is that its tests are 2D/3D single-field PDEs and no Neumann-data case is actually run.
