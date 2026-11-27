# Step 9 · Tolerance ladder

[Index](README.md) · page 2, the classical reference · group 5, solve the reference · [← step 8](08-branch-scan.md) · [step 10 →](10-domain-ladder.md)

**In one line:** solve again with tighter and tighter tolerances until the answer stops changing.

## What it does
- Tolerances 10⁻⁵, 10⁻⁶, 10⁻⁷, 10⁻⁸, each starting from the previous answer. The boundary conditions use the same tolerance (`bc_tol = tol`). At most 100 000 nodes.
- **Check V4:** the two tightest answers agree in every wall value to 10⁻⁴ relative. Pass.

## Words used here
| Term | Meaning |
|---|---|
| Tolerance | The largest error solve_bvp allows in the equations between nodes |
| Relative difference | abs(a − b)/max(abs(a), abs(b)) |

## In the code
- `ladder_tol` ([reference/run.py:58](../../../src/bvp_pinn/reference/run.py#L58)) and `solve_case` ([reference/solver.py:96](../../../src/bvp_pinn/reference/solver.py#L96)). The check is stored as `V4_tol_two_tightest`.

## Ready?
Yes.
