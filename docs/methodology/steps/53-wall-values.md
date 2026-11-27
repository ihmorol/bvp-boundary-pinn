# Step 53 · Wall values

[Index](README.md) · page 5, acceptance and evaluation · group 15, test · [← step 52](52-field-errors.md) · [step 54 →](54-equations-and-conditions.md)

**In one line:** the seven numbers at the wall, PINN against reference, as absolute and relative errors.

## What it does
- The wall values are f″(0), θ(0), θ′(0), φ(0), φ′(0), g′(0), χ′(0).
- For each: the PINN value, the reference value, abs(PINN − reference) and that divided by abs(reference).
- `wall_rel_max` is the largest relative one; it is the main accuracy number of a run. In the two good B0 pilots it was set by φ(0) (4.3 × 10⁻⁴ and 6.2 × 10⁻³).

## In the code
- `WALL` keys ([evaluation/evaluate.py:46](../../../src/bvp_pinn/evaluation/evaluate.py#L46)) and `wall_errors` ([line 86](../../../src/bvp_pinn/evaluation/evaluate.py#L86)).

## Ready?
Yes.
