# Step 15 · Slope check (V12)

[Index](README.md) · page 2, the classical reference · group 6, checks at L = 20 · [← step 14](14-runge-kutta-shooting.md) · [step 16 →](16-export.md)

**In one line:** the saved curves must be consistent: the slope of each saved curve equals the next saved curve.

## What it does
- For each link f → f′ → f″ → f‴ → f⁽⁴⁾, g → g′ → g″, and so on, a smooth spline is passed through the saved values on the 4 001-point grid and differentiated.
- The slope is compared with the saved next state. For the last link the target is H, the highest derivative the equations demand, so passing means the saved file itself satisfies the equations.
- **V12:** max |slope − target|/max |target| ≤ 10⁻⁸ for every link. Pass.

## In the code
- `slope_check` ([reference/verify.py:51](../../../src/bvp_pinn/reference/verify.py#L51)) uses `make_interp_spline(k=5)` (a quintic spline). Links: `SLOPE_LINKS` at [line 47](../../../src/bvp_pinn/reference/verify.py#L47).
- Called at [reference/run.py:177](../../../src/bvp_pinn/reference/run.py#L177).

**Decision G3 (diamond):** `G3_fixed_L_pass` at [reference/run.py:179](../../../src/bvp_pinn/reference/run.py#L179) is V4 and V7 and V8 and V10 and V12 and the V9 coefficient above zero. Branch A passes.

## Ready?
Yes.
