# Step 12 · Conditions check (V7, V9)

[Index](README.md) · page 2, the classical reference · group 6, checks at L = 20 · [← step 11](11-far-decay.md) · [step 13 →](13-core-then-chi.md)

**In one line:** the final solution meets all 12 conditions, and the θ″ coefficient never reaches zero.

## What it does
- **V7:** the largest |B_i| at the final solve must be at most the tolerance, 10⁻⁸. Branch A: 4.4 × 10⁻¹⁶.
- **V9:** the smallest value of (1 + 0.5A³)/8.8 over the whole grid must stay above zero, otherwise the first-order form breaks ([step 6](06-first-order-form.md)). Branch A: 0.143.

## In the code
- V7: [reference/run.py:163](../../../src/bvp_pinn/reference/run.py#L163).
- V9: `thermal_coefficient` ([reference/equations.py:126](../../../src/bvp_pinn/reference/equations.py#L126)), its minimum taken in `export` ([reference/run.py:79](../../../src/bvp_pinn/reference/run.py#L79)).

## Ready?
Yes.
