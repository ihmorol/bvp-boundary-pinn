# Reference figures: captions

Generated with the figures by `uv run python -m bvp_pinn.reference.plots`. Numbers come from the saved data.
Analysis: docs/SPEC_reference_solutions.md section 8.5.

## Main figures (figures/)

- **fig1_profiles**: the target solution (branch A, L = 20) on a linear eta axis. Lines are solve_bvp;
  open circles at eta = 0, 1, ..., 20 are the independent Runge-Kutta shooting solution. Each panel prints
  the wall value and wall slope.
  - Look for: a circle off its line (the two methods disagree), a field that does not settle at the far end.
- **fig2_benchmarks**: the same code on limits of Eqs. 14 and 16 that have known answers. These are code checks;
  the locked values are not changed.
  - (a) Curved sheet (only the curvature terms of Eq. 14 kept): skin friction against the published table of
    Abbas et al. (2013). At L = 10 the code gives the published value at every kappa to within one unit
    of the fifth decimal (largest difference 9.3e-06; at kappa = 1000 the table's 1.00079 is the
    code's 1.0007993 cut to five decimals). At L = 80 the values are lower (by 6.9e-03 at kappa = 5),
    so the published table carries a domain-length error.
  - (b) Flat-sheet limit (kappa = 1e6): f' and theta/theta(0) for five Prandtl numbers against the exact
    solutions (lines), largest wall-slope difference 4.8e-07 (relative); and f' with Darcy drag and
    slip at the locked Da and N1 against its exact solution, f''(0) difference 4.5e-08 (relative).
  - (c) Distance to the flat exact solution against kappa: slope -1, so the curvature terms vanish at the
    rate the equations predict.
- **fig3_checks**: checks on the saved branch A reference, using the saved arrays only.
  - (a) Slope of each saved curve (quintic spline) against the next saved curve. For f''', g', theta', phi',
    chi' the target is the highest derivative that the equations demand, so this also checks that the
    curves satisfy Eqs. 14 to 18. Worst: 3.5e-09.
  - (b) The 12 boundary errors against bc_tol.
  - (c) Largest Runge-Kutta minus collocation difference per state, against the RK error estimate.
- **fig4_wall_vs_L**: each wall value against the domain length L, for the asymptotic far conditions and the
  old zero-at-L cut. A flat line means the value no longer depends on where the domain is cut.

## Supplementary figures (figures/supplementary/)

- **S1_branches**: branch A against branch B (dashed), which does not decay at infinity (m = -0.0060).
- **S2_far_field**: |theta|, |phi|, |chi| on log-log axes for each L. Dotted line: predicted slope
  -m = -0.239. Asymptotic tails lie on one line; the old cut bends down to zero at L; chi depends on L.
- **S3_verification**: (a) manufactured-solution error against mesh step, slope-4 guide; (b) change of the
  wall values against solver tolerance at L = 10; (c) the branch reached from each of the 48 starting guesses.
- **S4_checks_branchB**: fig3 for branch B.
- Per branch folder, **solver_diagnostics**: mesh spacing, residual per interval against tol, and the theta''
  coefficient, which must stay away from zero.
