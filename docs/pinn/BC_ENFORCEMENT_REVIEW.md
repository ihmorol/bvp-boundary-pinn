# Boundary condition enforcement: review and recommendation

Date: 2026-11-21. Status: literature review and design sketch. Nothing here has been trained or run, except one small matrix check marked **[our check]**.

Related files: [PINN_PLAN.md](PINN_PLAN.md) (arms B0 to B3), [equations.md](../problem/equations.md) (Eq. 19 and the B errors), [literature-synthesis.md](../literature/literature-synthesis.md) section 3, [SPEC_reference_solutions.md](../reference-solver/SPEC_reference_solutions.md) section 8.

## 1. The question

How should the PINN impose the 12 boundary conditions B1 to B12: all as soft penalty terms, built exactly into the network output, a mix, or another route (augmented Lagrangian, theory of functional connections, distance functions)? The answer must fit this problem: one input eta on [0, 20], five outputs, a fourth-order f, a thin wall layer, slow power-law tails for theta and phi, and the fixed arms B0 to B3 with learning-rate annealing, then Adam, then L-BFGS.

## 2. Short answer

- Build the conditions into the output with one small constraint layer in the style of the theory of functional connections (TFC): output = network + fixed correction shapes times the amount the network misses each condition.
- The nine linear conditions (B1, B2, B3, B5, B6, B7, B8, B9, B12) fit this directly. Published TFC work covers linear combinations of values and derivatives, and conditions that couple two fields.
- The three nonlinear conditions (B4, B10, B11) can also be made exact in the same layer, because each one fixes a single slope as a function of values that the layer leaves untouched. This step is our own. We found no paper that does it.
- Recommended default: all 12 exact. Fallback, if the pilot shows trouble: the 9 linear ones exact and B4, B10, B11 soft with annealing (the usual TFC practice). Control: all 12 soft on B0 only.
- The evidence that exact beats soft is moderate in 1D and mixed in general. No study tests our problem class (coupled nonlinear conditions on a high-order similarity ODE). The pilot must check it.

## 3. Sources checked

Verified means title, authors and venue were matched against Crossref (DOI) or the arXiv API in this session. "Read" means the full text was read in this session or in an earlier local analysis.

| Short name | Source | Verification | What we read |
|---|---|---|---|
| Lagaris 1998 | Lagaris, Likas, Fotiadis, IEEE Trans. Neural Networks 9(5):987-1000, DOI 10.1109/72.712178 | Crossref and arXiv physics/9705023 | arXiv text, Sec. 3.1 to 3.2 |
| Mortari 2017 | Mortari, "The Theory of Connections: Connecting Points", Mathematics 5(4):57, DOI 10.3390/math5040057 | Crossref and arXiv 1702.06862v2 | arXiv text, pp. 8 to 10 |
| Mortari, Furfaro 2021 | "Univariate Theory of Functional Connections Applied to Component Constraints", Math. Comput. Appl. 26(1):9, DOI 10.3390/mca26010009 | Crossref | Abstract only |
| Johnston 2021 | H. Johnston, "The Theory of Functional Connections: A journey from theory to application", PhD thesis, Texas A&M, arXiv 2105.08034 | arXiv API | Secs. 2.4.4 to 2.4.6, 4.6, 4.8.2 |
| Leake, Mortari 2020 | "Deep Theory of Functional Connections", Mach. Learn. Knowl. Extr. 2(1):37-55, DOI 10.3390/make2010004 | Crossref | Metadata only |
| Schiassi 2021 (X-TFC) | Schiassi et al., Neurocomputing 457:334-356, DOI 10.1016/j.neucom.2021.06.015 | Crossref; preprint arXiv 2005.10632 | Preprint, Sec. 3 |
| Sukumar, Srivastava 2022 | CMAME 389:114333, DOI 10.1016/j.cma.2021.114333 | Crossref and arXiv 2104.08426v2 | arXiv text, Secs. 5, 9, 10 |
| Berrone 2023 | Heliyon 9(8):e18820, DOI 10.1016/j.heliyon.2023.e18820 | Crossref | Local analysis (bc-comparison-2023) |
| Lu 2021 (hPINN) | SIAM J. Sci. Comput. 43(6):B1105-B1132, DOI 10.1137/21M1397908 | Crossref | Local analysis |
| Son 2023 (AL-PINN) | Neurocomputing 548:126424, DOI 10.1016/j.neucom.2023.126424 | Crossref | Local analysis |
| Rowan 2025 | arXiv 2512.14941, preprint | arXiv API | Local analysis (bc-3d-comparison-2025) |
| Endalew, Zhang 2026 | arXiv 2609.23493, preprint | arXiv API | Local analysis (falkner-skan-hard-bc-2026) |
| Baty 2023 | "Solving higher-order Lane-Emden-Fowler type equations using PINNs: benchmark tests comparing soft and hard constraints", arXiv 2307.07302, preprint | arXiv API | Full text |
| Liu 2022 | "A Unified Hard-Constraint Framework for Solving Geometrically Complex PDEs", NeurIPS 35:20287-20299 (arXiv journal ref), arXiv 2210.03526 | arXiv API | Abstract |
| Barschkis 2023 | "Exact and soft boundary conditions in PINNs for the Variable Coefficient Poisson equation", arXiv 2310.02548, preprint | arXiv API | Results and conclusions |
| Rucker, Erickson 2026 | "Exact Boundary Enforcement Along Implicit Geometries for Physics-Informed, Deep Learning Problems in Continuum Mechanics", arXiv 2606.07579, preprint | arXiv API | Abstract, Sec. 5.2, conclusions |
| Wang, Teng, Perdikaris 2021 | SIAM J. Sci. Comput. 43(5):A3055-A3081, DOI 10.1137/20M1318043 | Crossref | Not reread (team decision already made) |

The micropolar (Modelling 2026) and squeezing-flow (2026) analyses were also read. Both use plain soft penalties and give no comparison of enforcement methods, so they add nothing to this choice beyond "soft can work" (micropolar BC MSE 1.29e-6 with equal weights, Figure 5, p. 9, author-reported).

## 4. Our 12 conditions and the options

z = [f, f', f'', f''', g, g', theta, theta', phi, phi', chi, chi']. "Free" means the condition does not fix that quantity; the network sets it.

| BC | Type | Exact, simple? | How | Fixes | Leaves free |
|---|---|---|---|---|---|
| B1 | linear, f' and f'' at 0 | yes | one TFC functional (1+N1/kappa) f'(0) - N1 f''(0) = 1 | that combination | f''(0) |
| B2 | linear, f(0) and phi'(0) | yes | f(0) target = fw/(Pr Le) phi'(0), a component constraint | f(0) | |
| B3 | linear, g(0) and f''(0) | yes | g(0) target = -ell f''(0) | g(0) | |
| B4 | nonlinear Robin at 0 | yes, our extension | theta'(0) target = F4(theta(0)); theta(0) left as the network value | theta'(0) | theta(0) |
| B5 | linear, phi'(0) and theta'(0) | yes | phi'(0) target = -(Nt/Nb) theta'(0) | phi'(0) | phi(0) |
| B6 | value at 0 | yes | chi(0) target = 1 | chi(0) | |
| B7, B8 | f'(L) = 0, f''(L) = 0 | yes | two derivative functionals at L | f'(L), f''(L) | f(L), f'''(L) |
| B9, B12 | g(L) = 0, chi(L) = 0 | yes | value functionals at L | g(L), chi(L) | |
| B10 | nonlinear far condition | yes, our extension | theta'(L) target = -m theta(L)/(L+kappa), m from f(L), theta(L) | theta'(L) | theta(L) |
| B11 | nonlinear far condition | yes, our extension | phi'(L) target = -(mp phi(L) - (Nt/Nb) m theta(L))/(L+kappa) | phi'(L) | phi(L) |

Why the nonlinear three can be exact: each is linear in exactly one slope (theta'(0), theta'(L), phi'(L)) and nonlinear only in values (theta(0), f(L), theta(L), phi(L)). If the correction shapes for theta and phi have zero value at both ends, those values stay equal to the network's own values, and each slope can be set directly. The dependencies form a chain with no loop (section 7).

Other options and why they are not the main pick:

| Option | Fit for us |
|---|---|
| All soft, annealed weights | Works in published ODE cases (micropolar, squeezing). Needs 12 weights and leaves every BC approximate. Keep as the B0 control |
| Multiplicative mask, u = B + M(eta) N (Lagaris eq. 14, Falkner-Skan 2026 eq. 26) | Fine for single-field value and slope conditions. Coupled and slip conditions need new algebra per condition. The TFC additive form covers the same cases with one rule |
| Distance functions (Sukumar 2022) | In 1D the distance function is just eta or L - eta, so it reduces to a mask. The Robin form (Sec. 5.1.3, p. 14) is single-field and linear. No gain over TFC here |
| hPINN (Lu 2021) | Exact only for Dirichlet and periodic (Sec. 2.3, pp. 4 to 6); its AL acts on PDE residuals. Not for our conditions |
| Augmented Lagrangian (Son 2023, Rowan 2025) | Best soft-type method in Rowan's tests, but adds 12 multipliers with their own updates, must be frozen at L-BFGS, and sits beside the annealing rule the team already chose. With exact enforcement nothing is left for it to act on |
| Quasi-Newton constrained solves, min-max Lagrange | Failed in Rowan 2025 (pp. 14 to 16). Do not use |

## 5. Evidence from the literature

### 5.1 Exact linear conditions with derivatives and couplings exist and are standard in TFC

- Mortari 2017 gives constrained expressions for any set of linear constraints on a function and its derivatives at points, including linear combinations across points ("relative constraints", arXiv v2 pp. 8 to 10). The coefficient matrix must be nonsingular, and this depends on the chosen support functions (p. 7).
- Johnston 2021 shows constraints that couple two dependent variables (Example 2.6, component constraints, pp. 26 to 27; Example 2.7, mixed constraints, pp. 29 to 30). This is the B2, B3, B5 case.
- Johnston 2021 Example 2.8 (pp. 31 to 33) builds the constrained expression for the Falkner-Skan conditions y(0) = 0, y'(0) = 0, y'(infinity) = 1. It is a construction example only. We found no TFC paper that reports a numerical Falkner-Skan or Blasius solution, so we do not claim one.
- For nonlinear boundary constraints, Johnston 2021 Sec. 4.8.2 (pp. 128 to 129) embeds the linear constraints and keeps the nonlinear one as a loss residual. This is the published practice behind our fallback.

### 5.2 Accuracy: exact against soft

| Source | Problem | Result (author-reported) | Weight for us |
|---|---|---|---|
| Sukumar 2022, Sec. 9.1 to 9.4, pp. 22 to 29 | 1D rod with Dirichlet, Neumann, Robin; clamped beam (4th order) | Exact about one order more accurate than soft in most cases. Beam: soft needed a boundary weight ratio of 1e4 to match (p. 29) | High relevance, but the soft baseline used equal weights, not annealing |
| Baty 2023, Secs. III to V | Lane-Emden-Fowler ODEs, orders 2, 3 and 4 | Similar accuracy in most cases; hard slightly worse in the 2nd-order case (Sec. III); hard much better in the two-point 3rd-order case with y'(1) given at the far end (Sec. IV.B) | Closest high-order ODE test. Single runs, preprint |
| Berrone 2023 | 2D elliptic, Dirichlet only | Exact lifting best; soft rate falls from 3.66 to 0.01 with a bad penalty (Fig. 8a, p. 10) | Value conditions only |
| Endalew, Zhang 2026 | Falkner-Skan, 3rd order, L = 12, slope condition at L | All three BCs exact; wall-shear error 6.7e-7 (Table 2, p. 17) | Closest structure. No soft comparison (p. 27). Preprint |
| Rowan 2025 | 2D and 3D PDEs, including a nonlinear u^4 radiation Robin condition | AL best overall; on the radiation case AL and annealing were similar (B 0.66% against 0.84%, p. 24); distance functions excluded on paper, not tested | Evidence that soft plus annealing handles a nonlinear Robin term |
| Rucker, Erickson 2026 | 2D elastodynamics with traction conditions | All-soft gave the best accuracy; all-hard was about twice as fast (Sec. 5.2, p. 23) | Counter-evidence, 2D, preprint |
| Barschkis 2023 | 2D Poisson | Exact slightly better; a poor filter choice slows training (Sec. 6) | Weak, preprint |
| X-TFC 2021 | ODE and PDE tests | Errors near 1e-16 to 1e-10 (preprint Sec. 3) | Least-squares training of one layer, not gradient training. Does not transfer |

Reading: in 1D, exact enforcement is at least as good as soft in almost every reported case, and clearly better when the soft weights are not tuned. Against tuned or adaptive weights the gap shrinks, and one 2D study found soft better. No study compares exact against annealed soft weights on a coupled nonlinear ODE BVP.

## 6. Interactions with our setup

- **Nested autograd to order 4.** The correction shapes are fixed functions of eta, and the correction amounts are scalars computed from the network at eta = 0 and eta = L. Derivatives at collocation points pass through the sum as usual. The highest order stays 4 (for f). The extra cost is one forward pass at two points with derivatives up to order 2.
- **Thin wall layer.** Exact enforcement does not resolve the layer; the network still has to learn f''' near 73 and f'''' near -1728 at the wall. It removes one source of wall error (BC misfit at 0). A multiplicative mask that vanishes at the wall would force the network to grow there; the additive TFC form does not.
- **Correction shapes.** With wall shapes built from exp(-eta/a) and far shapes from exp(-(L-eta)/a), a = 0.5: the 4 by 4 matrix for f has condition number 10, for theta 6. The fourth derivative of a correction shape is at most 64 per unit of correction, small next to the true f'''' at the wall. Far shapes are below 3e-10 at eta = 6, so far corrections do not reach the inner windows. With a = 2 they reach 0.1 at eta = 6. **[our check]**: a throwaway SymPy script in this review, not a repository test.
- **Annealing weights.** With all 12 exact, the loss has only the five residuals, so annealing balances 4 weights against R14 instead of 16. With the fallback it balances 4 plus 3.
- **B1 arm (window networks).** Wall conditions belong to window 1 and far conditions to window K; no condition crosses a window. Interface continuity stays soft and annealed. Exactness does not change the B1 design.
- **B2 and B3 arms (replay).** All 12 hold at every stage by construction, which meets the team rule literally even when the far window is not trained yet. Localized far shapes (a = 0.5) keep the untrained network value at L from moving the wall windows, which protects replay. Global polynomial shapes would not.
- **L-BFGS.** Nothing to freeze for the BCs; the loss stays smooth. AL multipliers would need freezing, exact enforcement does not.
- **Branches.** Exact enforcement does not pick branch A over branch B. Both satisfy all 12 conditions; branch B has m < 0 (SPEC section 8.4, V6). Initialization still decides.

## 7. Construction sketch

Network output N(eta) = (Nf, Ng, Nth, Nph, Nch). L = 20, q_L = L + kappa. Fixed shapes, with a = 0.5 (our choice): w(eta) = exp(-eta/a), v(eta) = exp(-(L - eta)/a).

General rule for one field u with functionals C_1..C_k, targets t_1..t_k and supports s_1..s_k:

```text
M[i][j]  = C_i[s_j]                    (k by k, computed once in float64)
phi_j    = sum_i s_i * inv(M)[i][j]    so that C_i[phi_j] = 1 if i = j, else 0
u(eta)   = N_u(eta) + sum_j phi_j(eta) * (t_j - C_j[N_u])
```

A functional whose target is the network's own value (t_j = C_j[N_u]) adds nothing but forces the other shapes to leave that value untouched.

Steps, in dependency order:

```text
1. Scalars from the network at the ends:
   th0 = Nth(0),  thL = Nth(L),  phL = Nph(L)
   th1_0 = F4(th0) = -Bi*(1 - th0) - Rd*Qw(th0)       (B4, stable form of equations.md)
   ph1_0 = -(Nt/Nb) * th1_0                            (B5)

2. f: functionals f(0); (1 + N1/kappa) f'(0) - N1 f''(0); f'(L); f''(L)
      targets      fw/(Pr*Le)*ph1_0 (B2); 1 (B1); 0 (B7); 0 (B8)
      supports     w, eta*w, (L-eta)*v, (L-eta)^2*v
   then fL = f(L) and f2_0 = f''(0) from the built f.

3. theta: functionals th(0), th'(0), th(L), th'(L)
      targets  th0, th1_0, thL, -m*thL/q_L                (B4, B10)
      with m = Pr*kappa*fL / (1 + Nr*A^3), A = 1 + (theta_w - 1)*thL
      supports w, eta*w, v, (L-eta)*v

4. phi: functionals ph(0), ph'(0), ph(L), ph'(L)
      targets  Nph(0), ph1_0, phL, -(mp*phL - (Nt/Nb)*m*thL)/q_L   (B5, B11)
      with mp = Le*Pr*kappa*fL; same supports as theta

5. g:   functionals g(0), g(L); targets -ell*f2_0 (B3), 0 (B9); supports w, v
6. chi: functionals chi(0), chi(L); targets 1 (B6), 0 (B12); supports w, v
```

Checks of the sketch:

- No loop: step 1 uses only network values; f needs only step 1; theta and phi need fL from step 2; g needs f''(0) from step 2.
- Nothing extra is imposed. f(L), f'''(L), f''(0), theta(0), theta(L), phi(0) and phi(L) stay free, as equations.md requires.
- Every function that meets all 12 conditions can be represented: set N equal to it and all corrections vanish (the TFC completeness argument, Mortari 2017).
- Fallback (B4, B10, B11 soft): drop the nonlinear targets. theta = Nth; ph1_0 = -(Nt/Nb) Nth'(0); phi drops its far slope functional; B4, B10, B11 return to the loss with annealed weights.
- Code size, estimated: one helper that builds phi_j from (supports, functionals), about 15 lines, plus the five steps, about 30 lines. The soft path costs about 15 lines, and the 12 B errors in PyTorch are needed anyway for evaluation.
- Explaining it to a supervisor: "We take the network output, measure how far it misses each boundary condition, and add a fixed smooth shape times that miss. So the conditions hold exactly for any weights, and the loss contains only the five equations."

Test to add to gate G1: for random weights, all 12 boundary errors of the built output stay below 1e-12, and the built output equals N when N already satisfies all 12.

## 8. Recommendation and trade-offs

| Mode | Accuracy evidence | Code | Cost | Main risk |
|---|---|---|---|---|
| All 12 exact (default) | Linear part: Sukumar, Baty, Berrone, TFC. Nonlinear part: none | about 45 lines | +2 boundary points per step | Our own nonlinear step; division by 1 + Nr*A^3 inside the output |
| 9 exact, 3 soft (fallback) | Same linear evidence; soft nonlinear Robin worked in Rowan 2025 | about 40 lines | same | 3 annealed point terms at the wall and far end |
| All 12 soft (control on B0) | Works in micropolar and squeezing papers; untuned weights lose in Sukumar | about 15 lines | none | BC misfit enters wall values; 12 annealed weights |
| Augmented Lagrangian | Best soft method in Rowan 2025, untested on 12 scalar conditions | about 30 lines | small | Clashes with the annealing decision; multiplier tuning |

Decision rule for the pilot (P2), declared before running: run B0 with all-exact, fallback and all-soft, 3 seeds each. Keep all-exact unless it fails to converge in any seed, gives NaN, or ends with a higher mean equation residual on the monitor grid than the fallback in all 3 seeds. Soft runs also need every scaled B error below 1e-6 to count. This uses training-side quantities only, in line with PINN_PLAN section 3. Then freeze the mode for all arms.

If the evidence stays mixed after the pilot, say so in the paper: exact enforcement is chosen for simplicity and for meeting the "all 12 at every step" rule, not because it was shown to be more accurate.

## 9. Open risks

1. **Nonlinear exact step is ours.** No paper imposes a nonlinear Robin or asymptotic condition exactly this way. Label it our extension in any write-up.
2. **Division inside the output.** m divides by 1 + Nr*A^3. With Nr = 0.5 and theta_w = 0.2 this is zero at theta(L) near 2.82. If the network wanders there, the built theta and phi blow up, not just the loss. Start with a zero final layer (Falkner-Skan 2026 eq. 41), so theta(L) starts at 0, and guard against NaN. The same term sits in the soft loss, so it is not new, but it is now in every field.
3. **B4 chain strength.** f(0) depends on theta(0) through B4, B5, B2. The factors are small (fw/(Pr Le) = 0.023), but phi'(0) = -3 theta'(0) amplifies theta(0) errors into phi. Watch the phi wall layer.
4. **Choice of a.** a = 0.5 is our choice, untested. It trades reach into the inner windows against larger high derivatives. Change it only in the pilot, then freeze it.
5. **Mixed outside evidence.** Rucker 2026 found soft more accurate in 2D. Baty found hard slightly worse in one ODE case. Sukumar's soft baselines were untuned. Exact enforcement may not improve our wall values.
6. **Exact does not mean accurate.** Boundary errors at 1e-15 say nothing about the wall layer. Keep the field and wall-value metrics of PINN_PLAN section 4 as the judge.
7. **Branch choice is unaffected.** Exact enforcement does not exclude branch B; check f(L) > 0 (m > 0) in every run.

## 10. What we could not verify

- Any TFC, X-TFC or Deep-TFC paper with a numerical Falkner-Skan, Blasius or other similarity boundary-layer solution. We found only the constraint example in Johnston 2021 (Example 2.8).
- The full text of Mortari and Furfaro 2021 and of Leake and Mortari 2020 (metadata and abstract only).
- A published comparison of exact against annealed soft enforcement for a high-order ODE with coupled or nonlinear conditions. We found none.
- The full text of Liu 2022 beyond its abstract; it targets 2D and 3D geometry, so we did not use it.
- Journal versions of Baty 2023, Barschkis 2023, Rowan 2025, Rucker 2026 and Endalew 2026. All are cited as preprints.
- Page numbers for Lagaris 1998 refer to the arXiv text (physics/9705023), not the journal pages.
