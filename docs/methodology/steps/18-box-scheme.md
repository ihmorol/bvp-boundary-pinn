# Step 18 · C++ box scheme

[Index](README.md) · page 2, the classical reference · group 7, cross-checks · [← step 17](17-reference-a.md) · [step 19 →](19-branch-search.md)

**In one line:** a separate C++ solver of the same problem, used to confirm the reference and to estimate PINN errors.

## What it does
- **Box scheme:** each mesh interval is a small box. The equations hold at its midpoint, with the states averaged: (z_(j+1) − z_j)/Δη = F(η_mid, z_mid). Second-order accurate.
- With B1 to B12 this is 12 N equations for 12 N unknowns. Newton's method solves them; the Jacobian is banded, so each step costs about N operations.
- The mesh starts at 401 nodes and is halved four times (up to 6 401 nodes). Two levels give a **Richardson value** (4 × fine − coarse)/3, and the observed order (about 2.00).
- Result: matches solve_bvp at the wall to about 3 × 10⁻¹⁰.

## In the code
- [engine/bvp_engine.cpp](../../../engine/bvp_engine.cpp): its own copy of the equations (`residuals`, [line 62](../../../engine/bvp_engine.cpp#L62)) and conditions (`wall_bc`, `far_bc`, [line 102](../../../engine/bvp_engine.cpp#L102)); the discrete system (`discrete_residual`, [line 148](../../../engine/bvp_engine.cpp#L148)); `newton` ([line 208](../../../engine/bvp_engine.cpp#L208)); the five levels ([line 500](../../../engine/bvp_engine.cpp#L500)).
- Built with g++ by `ensure_engine` ([pinn/estimate.py:46](../../../src/bvp_pinn/pinn/estimate.py#L46)). Writes `engine/out/engine_convergence.csv`.

## Checked by
`test_engine_solves_the_same_problem` in [tests/test_fixed_problem.py](../../../tests/test_fixed_problem.py).

## Ready?
Yes.
