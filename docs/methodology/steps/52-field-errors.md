# Step 52 · Field errors

[Index](README.md) · page 5, acceptance and evaluation · group 15, test · [← step 51](51-same-grid.md) · [step 53 →](53-wall-values.md)

**In one line:** for each of the 17 curves, how big the error curve is compared with the reference curve, and its largest single error.

## The equations
```text
relative L2 = sqrt( ∫ (u − u_ref)² dη  /  ∫ u_ref² dη )      over [0, 20]
max abs     = max over the grid of |u − u_ref|
```
- The integrals use the trapezoid rule, because the grid is not uniform (dense at the wall). Plain averages would over-weight the wall.
- "Largest field relative L2" of a run is the largest over f, g, θ, φ, χ. It is NaN-aware: if any value is not a number, the result is not a number (Python's plain `max` would hide it depending on order).

## In the code
- `rel_l2` ([evaluation/evaluate.py:81](../../../src/bvp_pinn/evaluation/evaluate.py#L81)); `worst` ([line 75](../../../src/bvp_pinn/evaluation/evaluate.py#L75)); per-curve values in `evaluate_run` ([line 96](../../../src/bvp_pinn/evaluation/evaluate.py#L96)), saved in `metrics.json` under `fields`.

## Ready?
Yes.
