# Step 8 · Branch scan

[Index](README.md) · page 2, the classical reference · group 5, solve the reference · [← step 7](07-mesh-and-start.md) · [step 9 →](09-tolerance-ladder.md)

**In one line:** solve from 48 different starts and sort what comes back into distinct solutions (branches).

## What it does
- Nonlinear problems can have more than one solution. The scan looks for all that solve_bvp can reach.
- Starts: rates r ∈ {0.5, 1, 2, 4} × wall temperatures θ0 ∈ {0.1, 0.2, …, 1.2} = 48 starts, at L = 10 and tolerance 10⁻⁵ (cheap).
- Two converged answers are the same branch if every wall value agrees to 1 %. Branches are named by θ(0), lowest first.
- Result (rebuilt 2026-11-26): branch A from 25 starts (θ(0) ≈ 0.250), branch B from 20 starts (θ(0) ≈ 0.590), 3 starts failed.

## In the code
1. `scan_branches` ([reference/run.py:35](../../../src/bvp_pinn/reference/run.py#L35)) loops over the starts.
2. Each start calls `solve_case` ([reference/solver.py:96](../../../src/bvp_pinn/reference/solver.py#L96)): mesh, smooth guess, then `scipy.integrate.solve_bvp(rhs, boundary, …)`.
3. `rel_diff` ([reference/solver.py:124](../../../src/bvp_pinn/reference/solver.py#L124)) compares wall values to group the answers.

## Ready?
Yes. A second, stronger search is [step 19](19-branch-search.md).
