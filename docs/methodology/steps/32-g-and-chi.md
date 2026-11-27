# Step 32 · g and chi (B3, B9, B6, B12)

[Index](README.md) · page 3, the PINN model · group 9, exact conditions · [← step 31](31-far-theta-phi.md) · [step 33 →](33-taylor-jets.md)

**In one line:** g and χ get one condition at each end.

## The rows
```text
B3   g(0) = −0.3 f″(0)     (f″(0) of the built f, step 30)
B9   g(20) = 0
B6   χ(0) = 1
B12  χ(20) = 0
```
**Worked example.** f″(0) = −4.20330 gives g(0) = 1.26099, the reference value.

## The whole build order, no loops
θ(0) → θ′(0) (B4) → φ′(0) (B5) → f(0) (B2) → f with B1, B7, B8 → f″(0), f(20) → g(0) (B3) and θ′(20), φ′(20) (B10, B11). Each step uses only values already fixed, so one pass is enough.

## In the code
- [pinn/constraints.py:88 to 93](../../../src/bvp_pinn/pinn/constraints.py#L88).

## Ready?
Yes. All 12 conditions hold to about 10⁻¹⁵ in every run so far.
