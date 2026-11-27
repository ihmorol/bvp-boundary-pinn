# SPEC: classical reference solutions

Date: 2026-11-19. Status: specification, nothing implemented. Problem definition: [PRD.md](../plans/PRD.md) section 1 (authoritative source [equations.md](../problem/equations.md)). Task order: [TODO.md](../plans/TODO.md) phases R0 to R5.

Tags: **[verified]** checked against the cited documentation; **[assumption]** a choice to confirm during implementation.

## 1. Purpose

Produce a trusted numerical solution of Eqs. 14 to 19 at the locked case. The PINN is tested against it. The reference never enters PINN training ([AGENTS.md](../../AGENTS.md) section 3).

## 2. Formulation

Rewrite the system as twelve first-order states:

```text
z = [f, f1, f2, f3, g, g1, theta, th1, phi, ph1, chi, ch1]
H = [f4, g2, th2, ph2, ch2]          highest derivatives
R = M(z)*H + r0(z, eta)              r0 = residuals R14..R18 evaluated with H = 0
```

M has a nonzero diagonal plus three off-diagonal entries (row 1 column 2, row 4 column 3, row 5 column 4). So H can be found row by row: g2, then f4, then th2, ph2, ch2. Its exact entries are in [equations.md](../problem/equations.md); a step-by-step derivation is in [first-order-form.md](first-order-form.md). At each eta, solve M*H = -r0 with a linear solve (never an explicit inverse), then return dz/deta = [f1, f2, f3, f4, g1, g2, th1, th2, ph1, ph2, ch1, ch2].

- **Diagonal check.** The diagonal entries 1+Delta, 1+Delta/2, (1+Nr*A^3)/Pr, 1/(Le*Pr) and 1/Lb must stay away from zero along the solution. Log the smallest |(1+Nr*A^3)/Pr| found.
- **Boundary function.** Returns B1 to B12 from [PRD.md](../plans/PRD.md) section 1.5, with B4 in its tested stable form.

## 3. Solver choices

| Method | Role | Why |
|---|---|---|
| `scipy.integrate.solve_bvp` | Primary reference | Built for nonlinear two-point BVPs in first-order form, which is our formulation. SciPy documents it as "a 4th order collocation algorithm with the control of residuals" following Kierzenka and Shampine (2001). Its solution is a C1 continuous cubic spline (`PPoly`), with per-interval RMS relative residuals and a status code: 0 converged, 1 node limit exceeded, 2 singular Jacobian **[verified: SciPy v1.18.0 manual]**. Our planning records already chose it ([strategy.md](../plans/strategy.md) section 6) |
| Ten-state core plus linear chi solve | Structural cross-check | Eqs. 14 to 17 and B1 to B5, B7 to B11 do not contain chi. So the core can be solved alone, then R18 solved as a linear BVP in chi on the fixed core curves. Compare with the monolithic twelve-state result ([strategy.md](../plans/strategy.md) section 6) |
| Multiple shooting with Runge-Kutta (SciPy `solve_ivp`, DOP853) | Second, independent reference (check V10) | An adaptive 8th-order Runge-Kutta integrator on 40 segments, with Newton joining the segments and enforcing the boundary conditions. It shares only the equations with solve_bvp, not the discretization. It starts from the smooth guess, not from the solve_bvp result. Single shooting is not used: some f modes grow like exp(1.8*eta), about 1e16 over [0, 20] |
| Asymptotic far-field conditions (Lentini and Keller, 1980) | Used for theta and phi at eta = L | theta and phi decay only like powers of (eta + kappa). Setting them to zero at a finite L gives wall values that keep changing with L; the asymptotic conditions keep only the decaying modes (section 8) |
| Single shooting | Not used | See the multiple-shooting row |
| Hand-written finite differences | Not used as primary | Needs its own Newton solver and is lower order than solve_bvp's collocation at similar effort **[assumption, judgment]** |

## 4. Discretization and tolerance settings

The settings used are in [configs/reference.json](../../configs/reference.json). They are our numerical choices, not supervisor values.

| Setting | Start value | Note |
|---|---|---|
| Far endpoint L | L = 20; study at 10, 20, 40, 80, 160 | Smallest L whose settled wall values agree with the next L, see section 8 |
| Far-field conditions | Asymptotic conditions for theta and phi; zero values for f1, f2, g, chi at L | `far_field` in configs/reference.json; section 8 |
| Initial mesh | 400 nodes, exponential stretching with first step 1e-3 | solve_bvp then refines it |
| `tol` | 1e-5, then 1e-6, 1e-7, 1e-8 | SciPy default is 1e-3, which is too loose here. `tol` measures the RMS relative residual per mesh interval **[verified]** |
| `bc_tol` | Equal to `tol` | SciPy default is `tol` **[verified]** |
| `max_nodes` | 100,000 | SciPy default 1,000 **[verified]**; raised so status 1 does not stop refinement early **[assumption]** |
| Initial guesses | At least 5 | Smooth decaying profiles consistent with B1 to B12, with derivative states consistent with the value states; vary the wall slope and temperature level |
| Precision | float64 | |

## 5. Accuracy verification

| Step | Test | Pass rule |
|---|---|---|
| V1 | Residual code against an independent SymPy transcription of equations.md at random artificial states. Both transcriptions start from equations.md, so this checks the code against equations.md. equations.md itself is checked against the source page by eye (AGENTS.md section 3) | Agreement to near float64 rounding |
| V2 | Algebraic rearrangement: R(H) - R(0) = M*H at random states ([equations.md](../problem/equations.md) test 3) | Agreement to near float64 rounding |
| V4 | Tolerance sequence at the locked case (section 4) | Wall values f2(0), g1(0), th1(0), ph1(0), ch1(0) agree to at least 4 significant figures at the two tightest tolerances **[assumption: 4 figures is our bar]** |
| V5 | L study, each L continued from the previous solution: L = 10 to 160 in the study of 2026-11-19, and L = 20 to 10 240 with both forms of the far condition on 2026-11-24 (section 8.6) | Wall values change by less than the V4 bar between the two largest L. Compare profiles only on the common interval |
| V6 | Multiple initial guesses (at least 5), and a deflated Newton search from 52 starts (`engine/bvp_engine --deflate`) | All converge to the same solution within the V4 bar, or each branch is saved and named |
| V7 | Boundary errors | All twelve original B1 to B12 at or below bc_tol at the solution |
| V8 | Structural cross-check (section 3) | Agreement with the monolithic solution within the V4 bar |
| V10 | Runge-Kutta multiple shooting (section 3) against solve_bvp on the evaluation grid; also RK at rtol 1e-11 against rtol 1e-9 | Every state and every wall value agrees within the V4 bar |
| V9 | Diagonal check | Smallest (1+Nr*A^3)/Pr along the solution logged and well away from zero |
| V12 | Saved-curve check: the slope of each saved curve (quintic spline of the saved points) against the next saved curve; for f''', g', theta', phi', chi' against the highest derivative the equations demand. Uses only reference.npz | Largest relative mismatch at or below the solver tol |

Rule from the 2026-11-16 strategy ([archive](../archive/strategy-2026-11-16.md), section 6): do not take f4 by differentiating the cubic spline of f four times. Use H from the linear solve. Do not call substituting H back into the same equations an independent residual check.

If the reference fails, record the attempts, guesses and messages, and report to the supervisor. A failure does not prove that no solution exists. No physical value is changed to make it pass.

## 6. Output data format

One folder per solution branch: `reference/<case_id>/`, where `case_id` = `locked-branch{X}-L{L}-tol{tol}`. Branches are named A, B, ... in order of increasing theta(0).

| File | Content |
|---|---|
| `reference.npz` | Arrays `eta` (evaluation grid, float64, ascending), `z` (12 x N, state order as in section 2), `H` (5 x N, highest derivatives from the linear solve), `mesh` (final solver mesh), `mesh_states` (12 x mesh size, solver values at the mesh), `z_rk` (12 x N, the Runge-Kutta reference on the same grid) |
| `meta.json` | Solver name and SciPy version, `tol`, `bc_tol`, `max_nodes`, L, initial and final mesh sizes, `status` and `message`, max RMS residual, B1 to B12 values, guess ID, branch label, SHA-256 of [physical-case-pending.json](../../configs/physical-case-pending.json), git hash, date |
| `SHA256SUMS` | Hashes of the two files above |

Evaluation grid: 4,001 points with exponential stretching and first step 5e-4. PINN evaluation points can be added later from the same solution. Values come from `sol(eta)`, which is the C1 cubic spline **[verified]**. H at those points comes from the linear solve on the interpolated z. Section 5 V4 to V5 decide whether the interpolated values are accurate enough. Interpolation error is part of the reference error and must be reported with it.

## 7. How the PINN stage uses the reference

1. **One loader.** `load_reference(path)` checks `SHA256SUMS` and that the parameter-config hash in `meta.json` matches the locked config. It refuses to load on any mismatch.
2. **Evaluation only.** The loader is called by evaluation scripts after model selection. Training and selection code must not import it (gate G5, [PINN_PLAN.md](../pinn/PINN_PLAN.md) section 5).
3. **What is compared.** PINN outputs and autograd derivatives against z and H on the same eta points:
   - f, f1, f2, f3, f4
   - g, g1, g2
   - theta, th1, th2
   - phi, ph1, ph2
   - chi, ch1, ch2

   Metrics and names: [PINN_PLAN.md](../pinn/PINN_PLAN.md) section 4.
4. **Same L.** The PINN uses the same L as the accepted reference.

## 8. Results, 2026-11-19 (asymptotic far-field conditions)

Code: `src/bvp_pinn/reference/`. Run `uv run python -m bvp_pinn.reference.run` from the repository root (about seven minutes, most of it the Runge-Kutta check). Full tables: [reference/study.md](../../reference/study.md). Figures: `reference/figures/` and each branch folder's `figures/`.

### 8.1 How papers set eta at infinity

| Source | Far-field handling |
|---|---|
| Micropolar nanofluid PINN (Modelling 2026), our local analysis | Cut at eta_inf = 7, decay values imposed there |
| Falkner-Skan hard-constraint PINN (arXiv 2609.23493), our local analysis | L = 12 in every run, no other L tested |
| Stretching-cylinder PINN (Discover Nano 2026), our local analysis | eta_inf never stated; plots end at eta = 8 |
| Kumar et al. 2023 (the draft's source model), Europe PMC full text PMC10406858 | RKF45 shooting; no eta_inf value found in the text |
| Lentini and Keller 1980 (numerical analysis) | Asymptotic boundary conditions: at a finite L, impose the conditions that select the decaying far-field solutions |

None of the four application papers checks that its results stop changing as eta_inf grows. Our first reference did the same thing (values set to zero at L = 10) and failed that check.

### 8.2 Far-field behaviour of these equations

Once f1, f2 and g have died out (they decay like about exp(-0.65*eta); the rate for g tends to 0.63), f tends to a constant f_inf.

- **f is taken bounded at infinity.** f' = C/(eta+kappa) solves the homogeneous Eq. 14 exactly, so f' -> 0 and f'' -> 0 at infinity allow a family of solutions, one per C. The reference takes C = 0, that is (eta+kappa) f' -> 0 and f bounded; B7 and B8 at a finite L give exactly this member. On the saved branch A, (eta+kappa) f' falls from 3e-4 at eta = 10 to 1e-6 at eta = 18 (`test_reference_takes_f_bounded_at_infinity` (tests/test_fixed_problem.py)). Reference audit D1; supervisor question 10 asks for confirmation.

- **theta and phi decay only as powers of q = eta + kappa.** Eqs. 16 and 17 reduce to theta ~ C*q^-m and phi ~ E*q^-mp + P*q^-m, where m = Pr*kappa*f_inf/(1+Nr*A^3) and mp = Le*Pr*kappa*f_inf. The 1/q factor on every convective term, together with delta_star = 0, removes any faster decay.
- **Branch A has m = 0.239.** The measured slope of log|theta'| against log q at large eta is -1.25, against a predicted -1.23 to -1.24.
- **chi has no decaying solution.** As printed, Eq. 18 has no chi convection term. Its kappa*f*phi/q term with phi ~ q^-m drives chi to grow like q^(1-m). So chi -> 0 as eta -> infinity cannot be met.

### 8.3 Effect on the wall values (branch A)

| Quantity | Zero values at L = 10 (old) | Asymptotic, L = 20 (reference) | Asymptotic, L = 160 |
|---|---:|---:|---:|
| f''(0) | -4.0893 | -4.2033 | -4.2037 |
| theta(0) | 0.2382 | 0.2502 | 0.2502 |
| theta'(0) | -1.2963 | -1.2594 | -1.2593 |
| phi'(0) | 3.8890 | 3.7783 | 3.7779 |
| chi'(0) | -2.3879 | -2.0087 | -1.0823 |

The old L = 10 reference was off by about 3% in f''(0) and 5% in theta(0) against the far-field limit.

### 8.4 Checks

- **V1, V2:** pass.
- **V4:** at most about 1e-10 relative between tol 1e-7 and 1e-8; pass.
- **V5 for f, g, theta, phi with asymptotic conditions (branch A):**
  - From L = 20, every wall slope and value changes by at most 7e-5 per doubling of L, except phi(0) at 3.9e-4, falling slowly.
  - The strict 1e-4 bar is therefore not met, because of phi(0). It is recorded as `V5_core_L_independent = false`.
  - Remaining far-field error at L = 20, estimated from the slowly falling per-doubling changes: about 2e-4 relative for wall slopes and 2e-3 for phi(0). **[estimate]**
- **V5 for chi:** no limit. chi'(0) on branch A is -2.24, -2.01, -1.77, -1.49, -1.08 at L = 10, 20, 40, 80, 160. In the reference, chi is defined by chi(L) = 0 at L = 20 and has no eta -> infinity meaning.
- **V6:** 48 guesses gave 25 to branch A, 20 to branch B and 3 failures. A second guess family of 330 starts (reference audit, check 5) found only A and B.
  - Deflation (2026-11-24, reference audit D7): `engine/bvp_engine --deflate` runs Newton on the box scheme (401 nodes, L = 20) from 52 smooth starts (rates 0.5 to 4, theta(0) from -0.5 to 2.5), each time with the roots found so far deflated (Farrell, Birkisson and Funke, SIAM J. Sci. Comput. 37, 2015). 105 Newton runs found exactly two solutions, A and B, both confirmed on 801 nodes ([engine/out/engine_branches.csv](../../engine/out/engine_branches.csv)). Deflation changes no parameter value. This is stronger than a guess scan, but it is still a finite search and not a proof that no third branch exists.
  - Branch B has f_inf = -0.0036, so m = -0.006 < 0. theta then has no decaying mode, and branch B is not a solution of the eta -> infinity problem (`far_field_valid = false`).
  - Branch A (`far_field_valid = true`) is the valid solution. Both are exported; B is flagged.
- **V7:** all boundary errors below 1e-15; pass.
- **V8:** core-then-chi agreement 2e-10 (A) and 1e-8 (B); pass.
- **V9:** smallest (1+Nr*A^3)/Pr is 0.143 (A) and 0.119 (B).
- **V10, Runge-Kutta against collocation:** pass on both branches.

  | | Branch A | Branch B |
  |---|---:|---:|
  | Largest difference over the 12 states, relative to each state's size | 2.8e-10 | 8.7e-9 |
  | Largest wall-value difference | 1.9e-10 | 9.3e-9 |
  | RK at rtol 1e-11 against rtol 1e-9 (RK's own error) | 5.1e-10 | 1.7e-10 |
  | Newton steps from the smooth guess | 4 | 7 |

  Figure: `figures/rk_vs_collocation` in each branch folder. Most of the remaining gap is the cubic interpolation of the solve_bvp result between its mesh nodes.
  - **What V10 proves:** both discretizations of the same boundary value problem give the same answer to within their own error estimates, so the numerical solution is correct to about 1e-9.
  - **What V10 does not prove:** whether the equations and boundary conditions are right. Both methods use the same equations, the same L and the same far conditions. Those are checked separately: SymPy and the source images for the equations (V1, equations.md), and sections 8.2 and 8.3 for the far field.
- **Gate G3:** passes for the fixed-L problem on both branches, now including V10 (and, since 2026-11-24, without the removed V3). The strict far-field part fails because of the phi(0) drift and chi.

**Reference for the PINN:** branch A, L = 20, asymptotic far conditions for theta and phi (`equations.far_conditions`), zero values at L for f1, f2, g and chi. The PINN must use exactly these boundary conditions, or the comparison is not like for like.

Open questions for the supervisor ([questions-for-supervisor.md](../problem/questions-for-supervisor.md)):

- Confirm branch A as the target.
- Confirm L = 20 with asymptotic far conditions.
- How should chi be treated at infinity, given that Eq. 18 as printed has no chi convection term?

### 8.5 Figures and evidence that the reference is right

Rebuilt on 2026-11-19 after review: the earlier log-axis small multiples were hard to read and did not show
any comparison with a known answer. Rebuild from saved data with `uv run python -m bvp_pinn.reference.plots`;
captions with numbers are in `reference/figures/captions.md`.

**Figure files.**

| File | Shows |
|---|---|
| `figures/fig1_profiles` | f, f', g, theta, phi, chi against eta on a linear axis [0, 20], as similarity-solution papers plot them. Lines: solve_bvp. Circles at eta = 0, 1, ..., 20: Runge-Kutta shooting. Wall value and wall slope in each title |
| `figures/fig1b_profiles_wall` | The same on [0, 1], with circles every 0.05, so the wall layer can be inspected |
| `figures/fig3_checks` | Checks on the saved data: slope consistency (V12), the 12 boundary errors (V7), RK against collocation (V10) |
| `figures/fig4_wall_vs_L` | Wall values against L, asymptotic far conditions against the old zero-at-L cut |
| `figures/supplementary/` | S1 branch A against B, S2 far-field tails (log-log), S3 tolerance study and branch scan, S4 fig3 for branch B |
| `<branch folder>/figures/solver_diagnostics` | Mesh spacing, residual per interval, theta'' coefficient |

**Evidence, from strongest to weakest.**

1. **Saved curves are self-consistent (V12, fig3 a).** The slope of every saved curve equals the next saved
   curve to at most 3.5e-9 (relative). For f''', g', theta', phi', chi' the target is the highest
   derivative from Eqs. 14 to 18, so the plotted curves satisfy the equations to that level. The largest
   mismatches sit at eta about 6, where the solve_bvp mesh spacing jumps.
2. **Boundary conditions (V7, fig3 b):** ten of twelve are exactly zero; B3 and B5 are about 1e-16.
3. **Two methods (V10, fig3 c and the circles in fig1):** RK shooting and collocation agree to 2.8e-10.
4. **A third method (C++ box scheme, engine/):** agrees with the saved reference to 2.7e-10 at the wall; an independent fourth solver written for the reference audit agrees to 1e-11.

**What this evidence does not cover.** Every check here solves the locked case: Eqs. 14 to 18, B1 to B12
and the 22 supervisor values. Checks at other parameter values (published curved-sheet and exact flat-sheet
limits, old V11, with fig2) and a manufactured solution with a forced equation (old V3) were removed on
2026-11-24 (AGENTS.md section 3). So the transcription of the equations rests on the SymPy test (V1, at the
locked values), the algebra test (V2) and the visual check against the source page in equations.md; the
numerics rest on three solvers that agree (V10, the C++ engine, the audit's fourth solver), the tolerance
study (V4) and the slope check (V12). The delta_star terms are zero at the locked values, so no check here
can see a copying error in them, and none can change a result. None of this tests that the printed
equations are physically right.

**What the figures show about the solution.**
- **Wall layer** (fig1b): f' falls from 0.386 to 0.10 by eta = 0.2; phi changes sign at eta = 0.056.
  solve_bvp puts 715 of 1543 nodes in eta < 0.3. A PINN with uniform points will likely under-resolve
  this layer. **[judgment]**
- **phi does not go to zero on [0, 20]** (fig1 e): it rises to about 0.19 and stays there; the asymptotic
  condition lets it decay only as a power of (eta + kappa) (S2). theta decays the same slow way (0.068 at
  eta = 20).
- **chi falls almost linearly to zero at L** (fig1 f): it is set by the cut, as section 8.2 explains, and
  chi'(0) changes with L (fig4 f).
- **Branch B** (S1) is a different solution: f turns negative, theta and phi level off near 0.67 and 1.36 at eta = 20.
- **Far-field choice matters** (fig4): with the old zero-at-L cut every wall value drifts with L; with
  the asymptotic conditions all but chi'(0) are flat from L = 20.

**Limits of the figures.** Circles are drawn at 21 points (fig1) or every 0.05 (fig1b), not at every saved
point; fig3 a uses spline slopes, whose own error is below 1e-9 on smooth test functions
(`test_slope_check_on_known_functions`).

### 8.6 Error bars on the wall values (2026-11-23)

Built by `uv run python tools/reference_error_bars.py` from data we already had; full table in [reference/error_bars.md](../../reference/error_bars.md). Branch A.

| Error source | How it was measured | Size |
|---|---|---|
| Mesh (discretization) | Grid convergence index from the three finest C++ box-scheme levels (1601, 3201, 6401 nodes, r = 2); observed order 2.00 for every wall value | 7e-8 to 8e-7 relative, so far below every other error |
| Domain end L = 20 | v(L) = v_inf + c L^-p fitted to the asymptotic L study at L = 40, 80, 160; order in L is about 0.36 | L = 20 minus v_inf: f''(0) 7.6e-4, theta(0) -8.0e-5, theta'(0) -2.4e-4, phi(0) -2.3e-4, phi'(0) 7.3e-4, g'(0) 4.8e-4 |
| chi'(0) | Same fit | No limit: chi'(0) grows in size with L (order -0.5), as section 8.2 predicts for Eq. 18 as given |

What it means: the L = 20 reference is correct to about 1e-6 for the L = 20 problem the PINN solves. As an estimate of the eta -> infinity values that B7 to B12 ask for, it is within 8e-4 absolute for every wall value except chi'(0). Relative, that is about 2e-4 for f''(0), theta'(0), phi'(0) and g'(0), 3e-4 for theta(0), and 1.8e-3 for phi(0), whose value is small (corrected 2026-11-26 from "about 2e-4 relative for every wall value"; the L study in [L_STUDY.md](../plans/L_STUDY.md) has the full table). The slow order 0.36 matches the power-law tails theta ~ q^-m with m = 0.24, so a longer domain helps only slowly. The fit uses three points; on 2026-11-24 it was tested on points it never saw (below).

**Two forms of the far condition, and a longer L study (2026-11-24, reference audit D2 and D6).** `tools/reference_error_bars.py` now solves L = 20, 40, ..., 10 240 twice: with A taken at theta(L), as the reference does, and with A = 1, its eta -> infinity limit (`equations.far_conditions`, `form`). Both impose theta, phi -> 0 to the same order. Every solve converged. Full table: [reference/error_bars.md](../../reference/error_bars.md).

| What | Result |
|---|---|
| Fit check | The L = 40, 80, 160 fit predicts the solved values at L = 320 to 10 240 to 5e-7 to 3e-5 relative. So the fit holds, and the L = 20 distances to the limit in the table above stand |
| Bracket at L = 10 240 | The two forms bracket every wall value; every fitted limit lies inside its bracket. f''(0) in [-4.20416, -4.20397], theta(0) in [0.250255, 0.250275], phi'(0) in [3.77748, 3.77767] |
| Spread at L = 20 | f''(0) 3.5e-3, theta(0) 3.6e-4, theta'(0) 1.1e-3, phi'(0) 3.3e-3 (local minus unit). This is a cautious far-field bar: the local form, used by the reference and the PINN, is the closer one, about 2e-4 relative from the limit |

None of this changes the L = 20 problem the PINN is compared with; it only bounds its distance from eta -> infinity.

Reference-free check for the PINN: `engine/bvp_engine --estimate` takes one Newton step of the box scheme from any state (see [engine/README.md](../../engine/README.md)). On the reference plus a known theta change of 1e-3, it recovered 99.9 % of the change at the wall. `bvp_pinn.pinn.estimate` runs it for every trained network, after training and without the reference, and `tools/pinn_error_estimate.py` does it for older run folders.
