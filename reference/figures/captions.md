# Reference figures: captions

Generated with the figures by `uv run python -m bvp_pinn.reference.plots`. Numbers come from the saved data.
Analysis: docs/reference-solver/SPEC_reference_solutions.md section 8.5.

## Main figures (figures/)

- **fig1_profiles**: the target solution (branch A, L = 20) on a linear eta axis. Lines are solve_bvp;
  open circles at eta = 0, 1, ..., 20 are the independent Runge-Kutta shooting solution. Each panel prints
  the wall value and wall slope.
  - Look for: a circle off its line (the two methods disagree), a field that does not settle at the far end.
- **fig2** is no longer made: it showed the published-limit checks at other parameter values, removed on
  2026-11-24 (AGENTS.md section 3). The other figure names are kept.
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
- **S3_verification**: (a) change of the wall values against solver tolerance at L = 10; (b) the branch
  reached from each of the 48 starting guesses.
- **S4_checks_branchB**: fig3 for branch B.
- Per branch folder, **solver_diagnostics**: mesh spacing, residual per interval against tol, and the theta''
  coefficient, which must stay away from zero.
