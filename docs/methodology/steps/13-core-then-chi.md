# Step 13 · Core, then chi (V8)

[Index](README.md) · page 2, the classical reference · group 6, checks at L = 20 · [← step 12](12-conditions-check.md) · [step 14 →](14-runge-kutta-shooting.md)

**In one line:** solve the problem a second way, in two parts, and check the answer is the same.

## What it does
- Eqs. 14 to 17 and B1 to B5, B7 to B11 do not contain χ. So the first ten states (the core) can be solved alone.
- Then χ is solved from Eq. 18 with the core fixed. Eq. 18 is linear in χ, with χ(0) = 1 and χ(L) = 0.
- **V8:** the two-part answer must match the full 12-state answer in every wall value to 10⁻⁴. Pass.

## In the code
- `core_then_chi` ([reference/verify.py:15](../../../src/bvp_pinn/reference/verify.py#L15)): first `solve_bvp` on 10 states, then a second `solve_bvp` on χ, χ′.
- Called at [reference/run.py:165](../../../src/bvp_pinn/reference/run.py#L165).

## Ready?
Yes.
