# Audit of the classical reference, 2026-11-23

Scope: the classical reference for the locked case (branch A, L = 20, asymptotic far conditions for theta and phi), as described in [SPEC](../reference-solver/SPEC_reference_solutions.md) section 8 and saved in `reference/locked-branchA-L20-tol1e-08/`. Question: is the reference right, given that the best PINN run (E7, [IMPLEMENTATION_NOTES.md](../pinn/IMPLEMENTATION_NOTES.md) section 7.1) gives theta(0) = 0.197 and f''(0) = -3.652 against the reference 0.2502 and -4.2033?

Done by one Claude Code session working on the reference side only. A second session studies the PINN side. Nothing in the repository was edited. Scripts and raw outputs are in `runs/reference-audit-2026-11-23/` (not committed, `runs/` is ignored).

## Answer

The reference is right. It solves Eqs. 14 to 18 and B1 to B12 as printed, with the stated far-field treatment, to about 1e-9 at L = 20. Its distance from the eta -> infinity limit is about 2e-4 relative (8e-4 absolute in f''(0), 8e-5 in theta(0)). The PINN gap is about 700 times larger than that: 0.55 in f''(0) and 0.053 in theta(0). So the PINN side is the one that is wrong.

The audit found one real gap in how the far field is described (defect D1 below): the printed far conditions f' -> 0 and f'' -> 0 do not fix the solution uniquely on [0, infinity). The reference picks the natural member of the family (f bounded at infinity), and the PINN uses the same conditions at L = 20, so this does not explain the gap. It should still be written down. The other defects are about tests and stale numbers, not about the computed solution.

## Verdicts

| # | Check | Verdict | Main evidence |
|---|---|---|---|
| 1 | Transcription of Eqs. 14 to 19 | **Pass** | Source images re-rendered and read term by term; my own transcription, written from the images, agrees with `equations.py` to 2e-16 at 400 random states |
| 2 | First-order form M*H = -r0 | **Pass** | Each equation is linear in its highest derivatives; M is lower triangular; it is singular only at theta = 2.825, while the solution has theta in [0.068, 0.250] |
| 3 | Far field | **Pass for the L = 20 problem, with a documentation defect (D1)** | m and mp derived again and match the code; L study pushed to L = 10 240 with two versions of the condition, which close in on one limit from both sides; but f' = C/(eta+kappa) is an exact extra far-field solution, so a one-parameter family meets every printed far condition |
| 4 | Numerics | **Pass** | My independent solver reproduces the saved wall values to 1e-11; the C++ engine rebuilt today reproduces them to 1e-10; RK shooting re-run today; 15 reference-side tests pass; the SPEC 8.6 error bars hold out to L = 10 240 |
| 5 | Branches | **Pass for A as the target; completeness uncertain** | B has f(L) = -0.0036 for every L from 20 to 160, so m = -0.006 and theta cannot decay; a second scan of 330 starts from a different guess family found only A and B. A finite scan cannot prove that no other branch exists |
| 6 | Benchmarks and papers | **Pass, with limits** | The exact flat-sheet checks are strong and independent of L; the curved-sheet table check is weak (L chosen after the fact). None of the four application papers contradicts our choices; their problems decay fast, ours does not |

## How the audit was done

- **Methods.** Workflow 05 (methods comparison) for check 6 and workflow 07 (evidence appraisal) for the claim table at the end, from the `research-synthesis` skill. Convergence and stability reasoning follow the `convergence-study`, `numerical-stability` and `benchmark-and-mms-planner` checklists (observed order, Richardson, asymptotic range, conditioning of the highest-derivative solve).
- **Independent solver.** `indep.py` was written from the rendered source images, not from `equations.py` or `equations.md`. It uses a different first-order form (each highest derivative isolated by hand, no matrix), solves the ten-state core (f, g, theta, phi) alone and chi afterwards, uses a different start-guess family and its own far conditions derived below. The 22 values were typed from the receipt in [parameters.md](../problem/parameters.md) and checked against `configs/physical-case-pending.json` (no mismatch).
- **Limit of independence.** The independent solver was written by the same kind of agent that wrote the repository code, reading the same images. It removes shared-code errors, not shared-reading errors. A human read of the enlarged page is still the rule in [AGENTS.md](../../AGENTS.md) section 3.
- **Compute.** One thread per process, one process at a time. The full `bvp_pinn.reference.run` was not re-run, because it rewrites files under `reference/`. Its parts were re-run into the scratch folder instead.

## Check 1. Transcription: pass

**Source.** `sources/project-draft/project3Nov_2026.docx` (SHA-256 starts `e9caca1674cc74b6`, the same file as the 2026-11-19 check). The equation previews `word/media/image44.wmf` to `image49.wmf` were rendered with LibreOffice and pdftoppm at 220 dpi. Their place in `word/document.xml` puts them next to the labels (14) to (19), so they are Eqs. 14 to 19.

**Term by term against the images.**

| Eq. | Result |
|---|---|
| 14 | Match. (1+Delta)[f'''' + 2f'''/q - f''/q^2 + f'/q^3]; kappa/q[ff''' - f'f'']; kappa/q^2[ff'' - f'^2]; -kappa/q^3 ff'; -0.5 delta*[(eta f''' + 3f'') + (eta/2 f'' + f')/q]; -(1/Da)[f'' + f'/q]; -Delta(g'' + g'/q); -2 beta_m/h^4 [(1/q - 4/h) theta + theta'] |
| 15 | Match, including the sign of -kappa/q[f'g - fg'] |
| 16 | Match. The Nr bracket holds A^3 theta'/q, 3(theta_w - 1)A^2 theta'^2 and A^3 theta''; the dipole term is (2/Pr) beta_m lambda_m (theta - eps)/h^3 [kappa f/q (1 - 2/h^2) - 2f'/h] |
| 17 | Match. The image writes the first two denominators as eta + k and the convective one as eta + kappa; the approved alias k = kappa makes them all q |
| 18 | Match, including kappa/q f phi (phi, not chi') and the final -delta* eta/2 phi' |
| 19 | Match for B1 to B12. B4 as -theta' - Bi(1 - theta) - Rd/(theta_w - 1)[theta_w^4 - A^4]; the stable form in the code is the same expression after dividing theta_w^4 - A^4 = (theta_w - 1)(1 - theta)(theta_w + A)(theta_w^2 + A^2) by theta_w - 1 |

**Two transcriptions compared.** My `core_rhs` and `wall_bc`/`far_bc` against the repository `eq.rhs` and `eq.boundary` at 200 random states each, for the locked values and for a changed set (delta* = 0.37, kappa = 0.6, b = 0.4, theta_w = 1.7, which switches on the delta* terms):

| Compared | Locked values | Changed values |
|---|---:|---:|
| First-order right-hand side, rows f to phi', max relative difference | 1.9e-16 | 1.3e-16 |
| Wall conditions B1 to B5, max absolute difference | 0 | 8.9e-16 |
| Far conditions B7 to B11 at L = 20, max absolute difference | 0 | 0 |

The C++ engine is a third copy. Its converged wall values equal the solve_bvp ones to 1e-10 (check 4), which a transcription difference in any term that is non-zero on the solution would break.

**What `tests/test_equations_sympy.py` proves, and what it does not.**

- It proves that `equations.py` equals a second, SymPy copy of [equations.md](../problem/equations.md) at random states, for two parameter sets, so a typing slip in one copy is caught.
- It proves that `boundary` equals hand formulas also taken from equations.md, and that the momentum-flux identity behind `reduced_momentum` holds symbolically.
- It does not prove that equations.md matches the source. Both copies come from the same sheet, so an error in the sheet passes. The source check is visual: 2026-11-19 and again here.
- It does not test `far_conditions`, `highest_matrix` or the chi handling. Those are in `test_equations_algebra.py`, and the far-condition test there is circular (defect D3).

**Limit.** The WMF images are the previews of MathType objects. I did not decode the MathType binary (`oleObject*.bin`), so a stale preview would not be caught. This is unlikely, and the supervisor confirmed the printed form.

## Check 2. First-order form: pass

**Linearity.** Each residual is linear in the highest derivatives H = (f'''', g'', theta'', phi'', chi''): R14 in f'''' and g''; R15 in g''; R16 in theta'' (A depends on theta, not theta''); R17 in phi'' and theta''; R18 in chi'' and phi'' (through chi phi''). So R = M H + r0 is exact, with r0 = R at H = 0.

**M.** Entries checked against the images: diagonal 1 + Delta, 1 + Delta/2, (1 + Nr A^3)/Pr, 1/(Le Pr), 1/Lb; off-diagonal -Delta (row 1, g''), Nt/(Le Pr Nb) (row 4, theta''), Pe chi/Pr (row 5, phi''). M is lower triangular after ordering H as (g'', f'''', theta'', phi'', chi''), so det M is the product of the diagonal.

**Where M can be singular.** Only the third diagonal entry can vanish: 1 + Nr A^3 = 0 gives A = -(1/Nr)^(1/3) = -1.260, so theta = (1 - A)/(1 - theta_w) = 2.825 for theta_w = 0.2. On the solution theta stays in [0.068, 0.250] (branch A) and [0.59, 0.67] (branch B). The smallest diagonal is (1 + Nr A^3)/Pr = 0.1427 (A, at the wall) and 0.1193 (B, at eta = L). Both sit where theta is largest, which is the worst point. The other coefficients are bounded on [0, L] because q = eta + kappa and h = eta + b are at least 0.2. The largest is 2 beta_m/h^4 = 250 at the wall.

**Code checks.** `test_residual_is_linear_in_highest_derivatives` and `test_highest_derivatives_zero_the_residual` pass. My solver does not use M at all, and it agrees with the M-based right-hand side to 2e-16.

## Check 3. Far field: pass for L = 20, documentation defect D1

### 3a. m and mp, derived again

Far from the wall f' and f'' are small and f -> f_inf, so with delta* = 0:

- Eq. 16 keeps (1 + Nr A^3)/Pr (theta'' + theta'/q) + kappa f_inf theta'/q, plus terms that are quadratic in theta' and phi' and a dipole source of size q^-4.
- Dividing by (1 + Nr A^3)/Pr gives theta'' + (1 + m) theta'/q = 0 with **m = Pr kappa f_inf/(1 + Nr A^3)**. Its solutions are 1 and q^-m. With m > 0 the condition theta' + m theta/q = 0 removes the constant mode and keeps q^-m.
- Eq. 17 gives phi'' + (1 + mp) phi'/q = -(Nt/Nb)(theta'' + theta'/q), with **mp = Le Pr kappa f_inf**. For theta = C q^-m the right side is -(Nt/Nb) C m^2 q^(-m-2), so phi = E q^-mp + P q^-m with P = -(Nt/Nb) C m/(m - mp).
- On q^-m, phi' + mp phi/q = (mp - m) P q^(-m-1) = (Nt/Nb) m theta/q. So the condition phi' + (mp phi - (Nt/Nb) m theta)/q = 0 removes the constant mode of phi and keeps both decaying modes.

This is what `far_conditions` codes. One difference: the strict linearisation about theta = 0 has A = 1, while the code uses A at theta(L). Both are leading-order forms of the same condition. At L = 20 they differ by 8.3e-4 relative in f''(0) and 1.4e-3 in theta(0) (table below). That spread is the size of the far-field model error at L = 20. It is not a mistake.

The dipole source in Eq. 16 (the -eps part) forces a particular theta K q^-2 with K = beta_m lambda_m eps kappa f_inf/((1 + Nr)(2 - m)), about 1.5e-4. With m = 0.24 < 2 it decays faster than the kept mode. It is 4e-7 at L = 20, and the error it leaves in the theta condition shifts theta by about 3e-6. Both are below the other far-field errors (estimate, not a solve).

### 3b. Are they a valid stand-in for theta, phi -> 0?

Yes, when m > 0 and mp > 0. The measured slope d ln theta/d ln q on [15, 20] of the saved reference is -0.2393, against m = 0.2390.

I pushed the L study 64 times further than the SPEC did, with both forms of the condition, each solve continued from the previous L (core only, tol 1e-8):

| L | f''(0), A(theta_L) (code) | f''(0), A = 1 | theta(0), A(theta_L) | theta(0), A = 1 | phi'(0), A(theta_L) | phi'(0), A = 1 |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | -4.2032976 | -4.2067893 | 0.2501853 | 0.2505479 | 3.7783060 | 3.7749930 |
| 80 | -4.2035931 | -4.2054228 | 0.2502163 | 0.2504063 | 3.7780228 | 3.7762866 |
| 320 | -4.2037768 | -4.2047373 | 0.2502354 | 0.2503351 | 3.7778484 | 3.7769370 |
| 1 280 | -4.2038846 | -4.2043905 | 0.2502465 | 0.2502991 | 3.7777461 | 3.7772661 |
| 5 120 | -4.2039464 | -4.2042137 | 0.2502530 | 0.2502807 | 3.7776875 | 3.7774338 |
| 10 240 | -4.2039663 | -4.2041608 | 0.2502550 | 0.2502752 | 3.7776686 | 3.7774840 |

The two forms close in on one limit from opposite sides. At L = 10 240 they bracket it: f''(0) in [-4.20416, -4.20397], theta(0) in [0.250255, 0.250275], phi'(0) in [3.77748, 3.77767]. The SPEC 8.6 limits from a three-point fit at L = 40, 80, 160 (f''(0) -4.20406, theta(0) 0.250265, phi'(0) 3.77758) all sit inside these brackets. So the error bars there hold, though they came from an exactly determined fit (defect D6).

**Why the approach is slow** (order about 0.36 in L). The terms dropped in 3a, Nt theta'^2, Nb theta' phi', 3(theta_w - 1) Nr A^2 theta'^2 and A^3 - 1, are smaller than the kept ones only by a factor of order q^-m or q^-mp, that is L^-0.23 to L^-0.34. A longer domain therefore helps little. Running both forms of the condition at one L gives an error bar for the price of two solves.

### 3c. B7 to B9 at L = 20, and a free parameter in the far field (D1)

**f' = C/(eta + kappa) solves the homogeneous Eq. 14 exactly.** With f' = C/q, f'' = -C/q^2, f''' = 2C/q^3, f'''' = -6C/q^4 and f = C ln q + const:

- the viscous bracket is C(-6 + 4 + 1 + 1)/q^4 = 0;
- the three kappa brackets add to (kappa/q^4)(2fC + C^2 - fC - C^2 - fC) = 0 for any f;
- the Darcy bracket f'' + f'/q is 0.

The term f'' + f'/q is also the only way f forces Eq. 15, so this tail does not drive g. In the flow it is an irrotational swirl about the centre of curvature, which neither viscosity nor Darcy drag resists.

**Consequence on [0, infinity).** f' = C/q and f'' = -C/q^2 both go to zero, so B7 and B8 hold for every C. g, theta and phi still decay. For C > 0, f grows like C ln q, so the local m grows and theta and phi decay even faster. Counting modes says the same thing. The ten core states have ten far-field amplitudes. Four must vanish at infinity: the growing f and g exponentials and the constant modes of theta and phi. That leaves six free amplitudes against five wall conditions, so the solution set is a one-parameter family.

**Numerical check.** I imposed f'(L) = C/q, f''(L) = -C/q^2 instead of zero, with the asymptotic theta, phi conditions:

| C | f''(0), L = 20 | f''(0), L = 80 | theta(0), L = 20 | theta(0), L = 80 | m at L = 80 |
|---:|---:|---:|---:|---:|---:|
| -0.01 | -4.2596 | -4.2656 | 0.25289 | 0.25351 | 0.178 |
| 0 (reference) | -4.2033 | -4.2036 | 0.25019 | 0.25022 | 0.236 |
| 0.01 | -4.1509 | -4.1485 | 0.24791 | 0.24766 | 0.292 |
| 0.05 | -3.9644 | -3.9601 | 0.24129 | 0.24085 | 0.515 |
| 0.10 | -3.7569 | -3.7540 | 0.23582 | 0.23552 | 0.791 |

All solves converged. q f'(eta) is flat at C over the outer half of the domain, and the wall values hardly depend on L. So each row approximates a true eta -> infinity solution, and the wall shear moves by about 5 per unit C. With B7 and B8 imposed at a finite L, the two conditions together force C to about zero: q f' at L/2 is 3e-4 at L = 20 and below 1e-5 at L = 80. **So the reference selects the member with f bounded at infinity (C = 0).** That is the usual reading of f'(infinity) = 0 in stretching-sheet work, and it is the right default. But it is a selection, not a consequence. SPEC 8.2 ("f tends to a constant f_inf") and 8.6 ("the eta -> infinity values that B7 to B12 ask for") present it as a consequence.

The PINN imposes the same B7 and B8 at L = 20 in exact mode (`pinn/constraints.py` imports `far_conditions`), so it selects C = 0 too. No member of the family matches the PINN anyway: f''(0) = -3.65 needs C of about 0.125, which gives theta(0) of about 0.233, not 0.197.

**B9, g(L) = 0.** The saved g decays like exp(-0.66 eta) between eta = 12 and 16, tending to the linear rate (2 Delta B/(1 + Delta/2))^0.5 = 0.632. Setting g = 0 at L instead of at infinity changes the wall values by about exp(-2 x 0.63 x 20), about 1e-11. f' follows g at the same rate in that region, so B7 and B8 add the same kind of error, apart from the C mode above.

### 3d. The chi caveat: correct

As printed, Eq. 18 has no chi convection term. Far out it reads (q chi')' = -Lb q [kappa f phi/q + (Pe/Pr)(chi phi'' + chi' phi')], whose leading part is -Lb kappa f_inf phi. With phi of size q^-m this gives chi of size q^(1-m) = q^0.76, growing. The homogeneous modes 1 and ln q do not decay either. So chi -> 0 cannot be met, as SPEC 8.2 says.

The reference defines chi by chi(L) = 0 at L = 20. My linear chi solve on my own core gives chi'(0) = -2.0087347, equal to the saved value to 1e-10. chi does not appear in Eqs. 14 to 17 or in B1 to B11, so it cannot move theta(0) or f''(0).

## Check 4. Numerics: pass

| Evidence | Source | Result |
|---|---|---|
| Independent solver, my transcription and guesses, L = 20, tol 1e-8 | `run1.json` | Wall values equal the saved reference to 4e-12 to 2e-10 relative; core profiles to 2.4e-10; chi'(0) to 1e-10 |
| C++ box scheme, rebuilt today from `engine/bvp_engine.cpp` in the scratch folder | `engine/` | Observed order 2.00; Richardson f''(0) = -4.2032976026, theta(0) = 0.2501852894, against saved -4.2032976027, 0.2501852894. The convergence table is identical to the committed `engine/out/engine_convergence.csv` except the timing column |
| RK multiple shooting, re-run today with the repository code | `run5.json` | 5 Newton steps from the smooth guess, 13 s. Largest difference to the saved solve_bvp states 2.8e-10 relative (chi), wall values equal to 1e-10, and equal to the saved RK curves to 8e-12. RK's own error (rtol 1e-11 against 1e-9) 5.1e-10 |
| solve_bvp tolerance ladder | [study.md](../../reference/study.md) | Wall values unchanged to nine digits from tol 1e-5 to 1e-8 (L = 10) |
| Mesh error | [error_bars.md](../../reference/error_bars.md) | Box-scheme GCI 7e-8 to 8e-7 relative; the level-to-level changes above (5.0e-6 in f''(0) from 3 201 to 6 401 nodes, order 2) agree |
| Manufactured solution, solve_bvp | `test_manufactured_solution_order` | Passed today (orders 3.38, 3.83, 3.63 recorded) |
| Reference-side tests | `tests/test_equations_sympy.py`, `test_equations_algebra.py`, `test_reference_verify.py`, `test_config.py` | 15 passed in 47.8 s; no repository file changed |
| Far-field error at L = 20 | 3b above | About 8e-4 absolute in f''(0), 8e-5 in theta(0), 2e-4 in phi'(0) |

**Error budget at L = 20 against eta -> infinity (with C = 0).** Discretization about 1e-9 to 1e-7. Far field about 2e-4 relative. The PINN gap is 0.55 in f''(0) and 0.053 in theta(0), roughly 700 times the far-field error and more than a million times the discretization error.

**How sensitive the wall values are to each far condition** (L = 20, central difference, target 0 moved by plus or minus 1e-5):

| Condition at L = 20 | d f''(0) | d theta(0) | d theta'(0) | d phi'(0) |
|---|---:|---:|---:|---:|
| B7, f'(L) | 106.5 | -4.85 | -14.8 | 44.3 |
| B8, f''(L) | -58.7 | 2.67 | 8.13 | -24.4 |
| B9, g(L) | 4.79 | -0.22 | -0.66 | 1.98 |
| B10, theta' + m theta/q | -77.7 | 8.06 | 24.6 | -73.7 |
| B11, phi condition | 0 | 0 | 0 | 0 |

Each entry is the change in the wall value per unit error in that condition. Two readings:

- The reference meets each condition to about 1e-15 (V7), so these factors do not matter for it. They matter for any method that meets the conditions only approximately. For example, an error of 5e-3 in f'(20) moves f''(0) by about 0.5. That is the size of the PINN gap, so a soft or approximate B7 would be enough to cause it. The PINN runs in section 7.1 build all twelve conditions in exactly, so this is a warning for other modes, not the cause of E7's error.
- B11 has exactly zero effect, and that is correct. Eqs. 14 to 17 and B1 to B5 use phi only through phi'. So phi is fixed only up to a constant by the core, and B11 sets that constant. It moves phi(0) and chi, not theta, f or phi'. This also explains why phi(0) converges in L more slowly than the other wall values (SPEC 8.4).

## Check 5. Branches: pass for A, completeness uncertain

**Why B is excluded.** Branch B has f(L) = -0.003575, -0.003584, -0.003586, -0.003586 at L = 20, 40, 80, 160 (`reference/study.json`), so the sign is settled and m = -0.0060, mp = -0.0063. With m < 0 the theta modes are 1 and q^0.006, and neither decays. The data agree: on branch B theta rises from 0.592 at the wall to 0.671 at L = 20, and phi(20) = 1.36. So B does not satisfy theta -> 0. The exclusion rests on the same C = 0 selection as in 3c. With a tail C > 0, f would eventually turn positive and theta could decay even on branch B. Under the stated selection, the exclusion is justified.

**Another branch?** The SPEC scan used one guess family: exponential profiles, 4 rates and 12 wall temperatures, 48 starts at L = 10. My scan used a different family at L = 20, with asymptotic conditions on the core only:

- a sech^(1/2)-shaped theta with wall values from -0.5 to 2.6;
- rates 0.3 to 6;
- f profiles scaled by 0.3, 1 and 3;
- phi with either sign;
- small random noise near the wall.

Result of my scan (`run3.json`, tol 1e-5 as in the SPEC scan, mesh capped at 8 000 nodes, 160 s in all):

| Outcome | Starts | theta(0) | f''(0) | m |
|---|---:|---:|---:|---:|
| Branch A | 176 | 0.250185 | -4.203298 | 0.239 |
| Branch B | 72 | 0.592419 | -6.954162 | -0.006 |
| Not converged (node cap reached) | 82 | | | |

Of the 82 failures, 75 started from theta(0) of 1.2 or more. No start crashed. No third branch appeared.

This makes a third branch near these starts unlikely. It is not a proof: a scan only finds branches whose basin of attraction holds one of the starts. Continuation in a parameter, used as a reference-only aid as `configs/reference.json` already allows, or deflation would test it properly (fix F5).

**Stale number.** [engine/README.md](../../engine/README.md) and the brief for this audit give branch B theta(0) = 0.7423. That value is the old zero-at-L = 10 result (`reference/study.json`, `dirichlet_L_ladder`). The saved branch B with asymptotic conditions has theta(0) = 0.5924 (defect D5).

## Check 6. Benchmarks and the application papers: pass, with limits

### 6a. SPEC 8.5 limits

`test_benchmarks_published_and_exact` passed today. Read against `reference/benchmarks.json`:

- **Strong:** exact flat-sheet solutions, with kappa = 1e6 and no L effect because the decay is exponential:
  - Crane's f' to 2.8e-7;
  - -theta'(0)/theta(0) for Pr = 0.72 to 10, to at most 4.8e-7 relative (for example Pr = 7: exact 1.8954033, code 1.8954035);
  - Darcy drag with slip at the locked Da and N1: f''(0) = -1.94167683 against the exact -1.94167675. I re-derived that exact solution: f' = c exp(-a eta) with a^2 = c + 1/Da and c = 1/(1 + N1 a).
  - The distance to the flat solution falls as 1/kappa from kappa = 1e2 to 1e6, as the curvature terms predict.
- **Weak:** the curved-sheet table of Abbas et al. (2013), via Shafiq et al. (2019). The code matches it at L = 10 only. At L = 20 it is lower by up to 0.54 % (kappa = 5: 1.15139 against 1.15763), and L = 10 was picked because it matches. So this check has one tuned knob. SPEC 8.5 says this openly. The L effect there is the same far-field softness as in 3b and 3c: for a curved sheet with no Darcy term every mode decays slowly.
- **Not covered:** none of these limits switches on Delta, B, beta_m, lambda_m, epsilon, Nr, Rd, Nt, Nb, fw, ell or chi. Those terms rest on the transcription checks (check 1) and the manufactured solution, which test the code against the printed equations, not the physics.

### 6b. Methods comparison with the application papers (workflow 05)

Read in the extracted PDF text, at the locations given, not only in the `analysis.md` notes.

| Paper (folder) | Problem and design | Far-field handling | Reference solver | Tests L? |
|---|---|---|---|---|
| micropolar-2026 | Micropolar nanofluid, flat sheet, PINN; f third order, three second-order fields | Truncated at eta_inf = 7.0, called "large enough", decay values imposed there, Eq. (27), p. 8 | FEM values of Kumar for -f''(0), section 4, Table 1, p. 10 | No |
| falkner-skan-hard-bc-2026 | Falkner-Skan, hard-constraint PINN | L = 12 in every run (p. 3); the same L for the SciPy BVP reference (p. 14); L adequacy left open for slower decay (p. 27) | SciPy BVP, same L | No, flagged as open |
| cylinder-nano-2026 | Stretching cylinder, LNS-PINN | Conditions at infinity, Eqs. (10), (17), (24); domain 0 <= eta <= eta_inf (p. 7); no number for eta_inf found in the extracted text | Named both as BVP4C and as SciPy solve_bvp (p. 9) | No |
| squeezing-2026 | Squeezing flow between plates | None: a finite physical domain [0, 1] with symmetry at the centre and the moving plate (Eq. (5), p. 5; p. 8) | Shooting or bvp4c are named as the usual tools (p. 2) | Not applicable |
| This repository | Curved sheet with curvature-damped convection (kappa/q on every convective term), delta* = 0 | B7 to B9 and B12 zero at L = 20; asymptotic conditions for theta and phi | solve_bvp, RK multiple shooting, box scheme; agree to 1e-10 | Yes, to L = 160 (SPEC), here to L = 10 240 |

**Comparability.** The three semi-infinite papers have flat or cylinder geometries. Their convective terms do not fade with eta, so their profiles decay exponentially and a cut at eta = 7 to 12 with zero values is enough there. This is my inference from the equation type; I did not re-derive their decay rates. In our equations every convective term carries kappa/(eta + kappa) and delta* = 0, so theta and phi decay only as powers of q (m = 0.239). The zero-at-L practice therefore does not transfer: at L = 20 it gives theta(0) = 0.2408 and f''(0) = -4.1122, which is 3.8 % and 2.2 % off the limit.

None of the papers contradicts our domain or far-field choice. None of them tests its own cut, so they give no evidence for or against L = 20. The micropolar paper's wall couplings (f(0) tied to the concentration slope, zero nanoparticle flux, Eq. (26), p. 8) have the same form as our B2 and B5 and are imposed the same way. Kumar et al. (2023), the draft's source model, is not in `sources/`, so its far-field handling is not accessible here.

## Defects and fixes

None of these fixes changes Eqs. 14 to 18, B1 to B12 or the locked values. They change records, tests and how the far conditions are explained or checked.

| ID | Defect | Evidence | Fix |
|---|---|---|---|
| D1 | The far field has a free parameter that the records do not mention. f' = C/(eta + kappa) solves the homogeneous Eq. 14 exactly, so B7 and B8 at infinity hold for every C. The reference selects C = 0 (f bounded) through B7 and B8 at finite L | Check 3c: the algebra, and the converged family table at L = 20 and 80 | State the selection in SPEC 8.2 and 8.6: "the reference takes f bounded at infinity, equivalently (eta + kappa) f'(eta) -> 0; this is what B7 and B8 at a finite L give". Add a check that q f' over the outer half of the saved solution is near zero. Add the point to [questions-for-supervisor.md](../problem/questions-for-supervisor.md) as a confirmation, not a change. Tell the PINN side that a soft or approximate B7 or B8 moves f''(0) along this family |
| D2 | The far-field error bar comes from one form of the asymptotic condition only | Check 3b: the A(theta_L) and A = 1 forms differ by 8e-4 in f''(0) at L = 20 and bracket the limit | Solve once with each form and report the spread, or the bracket at a large L, as the far-field error. Both forms impose B10 and B11 to the same order, so neither changes the problem |
| D3 | `test_far_conditions_vanish_on_decaying_modes` is circular: it builds the decaying modes with the same m formula it tests | `tests/test_equations_algebra.py` | Test against the ODE instead: solve the linearised theta, phi equations far out with a fixed f_inf, fit the decay power, and compare with m and mp. Or compare with a SymPy derivation of the two conditions |
| D4 | The docstring of `equations.py` and SPEC V1 suggest the SymPy test checks the transcription against the source. It checks it against equations.md | Check 1 | Say in the test and in SPEC V1 that the source check is the visual one, and keep the image record in equations.md current. A human reader of the enlarged page is still needed under AGENTS.md section 3 |
| D5 | Stale and mismatched numbers | engine/README.md: branch B theta(0) = 0.7423 (old zero-at-L = 10 value; saved B is 0.5924). SPEC section 5, V5 lists L = 10, 15, 20, 30; the study used 10 to 160. SPEC 8.2 says f1, f2, g decay like exp(-0.7 eta); the far rate for g is 0.632 | Correct the README line and the V5 row. Say "about exp(-0.65 eta), tending to 0.63" |
| D6 | The SPEC 8.6 limits come from a three-parameter fit to three points, with no spare point to test the fit | error_bars.md | Add L = 320 to 10 240 to the L study. Core solves take about 0.5 s each with continuation. Keep the fit, now with a check |
| D7 | Branch completeness rests on guess scans only | Check 5 | Track branch A and B by continuation from a simpler case (for example in Rd or beta_m from 0 to the locked value, reference-only, intermediate results never reported), and look for folds. Or run a deflated Newton solve on the box-scheme engine, which already has a banded Newton |

## Claim appraisal (workflow 07)

Criteria for computational claims: the method fits the problem; independent discretizations agree; convergence is observed, not assumed; transcription is checked against the source; known-answer checks exist; shared assumptions between the checks are visible.

| Claim | Evidence | Shared assumptions | Confidence |
|---|---|---|---|
| The saved branch A solves Eqs. 14 to 18 with B1 to B6 at eta = 0, B7 to B9 and B12 at L = 20 and the asymptotic B10, B11, to about 1e-9 | Three solvers agree to 1e-10; an independent fourth solver with its own transcription agrees to 1e-11; GCI and order 2.00; MMS | All use the same far conditions and L | **Strong** |
| Eqs. 14 to 19 in the code match the source draft | Two separate readings of the rendered images; two transcriptions agree to 2e-16 | Both readers are AI sessions; MathType binary not decoded | **Strong** for the printed images |
| The L = 20 wall values are within about 2e-4 relative of the eta -> infinity limit | Two forms of the condition bracket the limit at L = 10 240; agrees with the SPEC fit | The selection f bounded (D1) | **Strong** under that selection; the limit is not unique without it |
| Branch B is not a solution of the eta -> infinity problem | m < 0 at every L from 20 to 160; theta grows outward | Same selection | **Strong** under the selection |
| Branch A is the only valid branch | 48 + 330 starts from two guess families | Guess scans only | **Moderate** |
| The reference is not the cause of the PINN gap | The gap is about 700 times the reference error; the PINN uses the same conditions; no far-field variant (zero at L, C family) reproduces both PINN numbers | None beyond the above | **Strong** |

What would change this: a human reading of the source page that differs from the images; a third branch found by continuation; or a supervisor ruling that the far field means something other than f bounded. The first would change the problem, the last the target, and none of them changes the numerical correctness at L = 20.

## Follow-up, 2026-11-24

What was done about each defect. None changes Eqs. 14 to 18, B1 to B12 or the locked values.

| ID | State | What was done |
|---|---|---|
| D1 | Done | SPEC 8.2 states the f-bounded selection; `test_reference_takes_f_bounded_at_infinity` (tests/test_fixed_problem.py) checks (eta+kappa) f' falls on the saved reference; supervisor question 10 asks for confirmation |
| D2 | Done | `equations.far_conditions` takes `form` ("local" or "unit"); `tools/reference_error_bars.py` solves with both and reports the spread (SPEC 8.6) |
| D3 | Done | The circular test is replaced by `test_far_conditions_follow_from_eqs_16_and_17`, which finds m and mp by root finding on `residual_terms` and checks the forced phi mode |
| D4 | Done | The docstrings of `equations.py` and the SymPy test, and SPEC V1, now say the SymPy test checks against equations.md and the source check is visual |
| D5 | Done | engine/README (2026-11-23); SPEC V5 row and the far decay rate in SPEC 8.2 (2026-11-24) |
| D6 | Done | L ladder to 10 240 with both forms; the three-point fit predicts the new points to 3e-5 relative or better, and every fitted limit lies inside the two-form bracket ([reference/error_bars.md](../../reference/error_bars.md)) |
| Checks at other values | Removed | At the user's decision, the published-limit checks (V11, the audit's check 6a) and the manufactured solution (V3) were removed: they solved other parameter values or a forced equation. The audit's verdicts that rest on them now rest on the three agreeing solvers, the tolerance study and the slope check |
| D7 | Done, with its limit | `engine/bvp_engine --deflate`: 105 deflated Newton runs from 52 starts found only A and B, both confirmed on a finer mesh. Still a finite search |

## Files

- `runs/reference-audit-2026-11-23/indep.py`: independent transcription and solver.
- `run1.py`, `run1.json`: transcription comparison and the L = 20 solves (code form, A = 1 form, zero at L).
- `run2.py`, `run2.json`: L study to 10 240 and the C family.
- `run3.py`, `run3.json`, `run3.log`: branch scan, with one line per start.
- `run4.py`, `run4.json`: far-condition sensitivities.
- `run5.py`, `run5.json`: RK multiple shooting re-run.
- `engine_out/`: engine output from the rebuilt binary.
- `source_images/`: the rendered Eqs. 14 to 19 (image44, image45, and trimmed t46 to t49) used for check 1.

Run from the repository root with `uv run python runs/reference-audit-2026-11-23/<script>`. Each script writes its JSON next to itself (`run5.py` takes the output path as its argument).
