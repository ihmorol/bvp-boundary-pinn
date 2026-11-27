# Step 30 · Build f (B2, B1, B7, B8)

[Index](README.md) · page 3, the PINN model · group 9, exact conditions · [← step 29](29-b5-then-b2.md) · [step 31 →](31-far-theta-phi.md)

**In one line:** f gets its four conditions, two at each end, and then hands its own wall curvature and far value to the other fields.

## The rows
```text
B2   f(0) = φ′(0)/44              (target from step 29)
B1   1.5 f′(0) − 0.1 f″(0) = 1    (one row ties slope and curvature)
B7   f′(20) = 0
B8   f″(20) = 0
```
- B1 does not fix f′(0) or f″(0) alone; only this combination. The network still decides f″(0), which is one of the results (reference: −4.20330, so f′(0) = 0.38645).
- f(20) and f‴(20) stay free.

**After f is built**, two of its numbers are needed elsewhere: f″(0) for B3 ([step 32](32-g-and-chi.md)) and f(20) for B10 and B11 ([step 31](31-far-theta-phi.md)). They are the raw network value plus the correction's value there, which is a fixed table times the misses.

## In the code
- f misses: [pinn/constraints.py:81 to 86](../../../src/bvp_pinn/pinn/constraints.py#L81).
- `f_built = fl[F].value_at` ([line 87](../../../src/bvp_pinn/pinn/constraints.py#L87)); `value_at` ([line 145](../../../src/bvp_pinn/pinn/constraints.py#L145)) multiplies the precomputed end table by the misses. f″(0) at [line 89](../../../src/bvp_pinn/pinn/constraints.py#L89), f(20) at [line 95](../../../src/bvp_pinn/pinn/constraints.py#L95).

## Ready?
Yes.
