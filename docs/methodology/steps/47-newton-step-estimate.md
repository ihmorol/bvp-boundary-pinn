# Step 47 · Newton-step estimate

[Index](README.md) · page 5, acceptance and evaluation · group 14, acceptance · [← step 46](46-run-folder.md) · [step 48 →](48-score.md)

**In one line:** the C++ box scheme takes one Newton step starting from the PINN's answer; the size of that step estimates how wrong the PINN is, without looking at the reference.

## What it does
- The trained network is evaluated on 3 201 wall-clustered nodes (first step 2 × 10⁻³), giving the 12 states there.
- The engine builds its discrete equations at those states ([step 18](18-box-scheme.md)) and solves one Newton system. If the PINN were exact, the step would be zero. Near the solution, the step equals the PINN's error.
- Nothing here reads the reference: the engine reads only the locked case file.

## The equation
```text
R(z_PINN)            discrete box-scheme residual at the PINN states
J δz = R(z_PINN)     one Newton system (J: Jacobian of the box scheme)
δz ≈ z_PINN − z_solution
```
The estimate has its own floor: on an exact solve_bvp solution the step at the wall is 4 × 10⁻⁶ to 7.5 × 10⁻⁵ relative (the box scheme's own error on this mesh). Errors below that cannot be ranked.

## In the code
1. `estimate` ([pinn/estimate.py:75](../../../src/bvp_pinn/pinn/estimate.py#L75)) builds the mesh with `wall_mesh` and gets the PINN states with `predict`.
2. `engine_step` ([pinn/estimate.py:57](../../../src/bvp_pinn/pinn/estimate.py#L57)) checks the case file, builds the engine if needed (`ensure_engine`, [line 46](../../../src/bvp_pinn/pinn/estimate.py#L46)), writes the states to a temporary CSV and runs `engine/bvp_engine --estimate`.
3. The engine side: `estimate` in [engine/bvp_engine.cpp:280](../../../engine/bvp_engine.cpp#L280) forms the residual and Jacobian and does one banded solve.

## Checked by
`test_engine_solves_the_same_problem` in [tests/test_fixed_problem.py](../../../tests/test_fixed_problem.py): on a solve_bvp solution the step is at the engine's own error level, and a known change of θ is recovered to 5 %.

## Ready?
Yes. On the pilot runs it was within 10 to 20 % of the true wall error once a run was near the solution.
