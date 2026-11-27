# Step 2 · Substitutions

[Index](README.md) · page 1, the fixed problem · group 1, locked case · [← step 1](01-locked-values.md) · [step 3 →](03-unknowns-and-domain.md)

**In one line:** short names that make the equations readable: q = h = η + 0.2, A = 1 − 0.8θ, and δ∗ = 0.

## What it does
- q = η + κ and h = η + b. Both κ and b are 0.2, so q = h = η + 0.2.
- A = 1 + (θ_w − 1)θ with θ_w = 0.2, so A = 1 − 0.8θ.
- δ∗ = 0 removes every unsteady term. The terms stay in the code (they cost nothing), multiplied by zero.

## Why q matters
At the wall q = 0.2, so 1/q⁴ = 625. Terms with 1/q³ or 1/q⁴ are huge at the wall and tiny far away. That is the reason for the thin wall layer and for the wall-clustered points, meshes and scales later on.

## In the code
- `residual_terms` in [reference/equations.py:44](../../../src/bvp_pinn/reference/equations.py#L44) builds q at [line 50](../../../src/bvp_pinn/reference/equations.py#L50), h at line 51, c = θ_w − 1 at line 52 and A at [line 53](../../../src/bvp_pinn/reference/equations.py#L53). No number is typed in: they come from the locked values at run time.

## Ready?
Yes. The substituted forms are on [equations.md](equations.md).
