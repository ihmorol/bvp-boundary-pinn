# Step 35 · Residual scale

[Index](README.md) · page 3, the PINN model · group 10, residuals and loss · [← step 34](34-residual-rows.md) · [step 36 →](36-loss.md)

**In one line:** a fixed weight on each residual that depends only on η, so neither the wall nor the far field swamps the loss.

## Why it is needed
Near the wall the residual terms are of size 10³ (1/q⁴ = 625 at η = 0); far out they are tiny. A plain mean square would train only the wall layer and leave the slow far tail almost free. A small loss could then hide a wrong tail (run E8).

## The two scales
```text
log:       s_k(η) = (η + 0.2)^(n_k),   n = 4, 2, 2, 2, 2   for rows 14 to 18
balanced:  row 14 × q⁴/(1 + 3.33 q²)
           row 15 × q²/(1 + 0.4 q²)
           rows 16, 17, 18 × q²                            (q = η + 0.2)
```
- **log** is the equations written in the variable s = ln(η + 0.2): each d/dη is (1/(η + 0.2)) d/ds, so an n-th derivative carries (η + 0.2)^(−n), and the scale cancels it. Wall layer and tails then both have size about one. Row 14's weight is 0.0016 at the wall and 1.7 × 10⁵ at η = 20.
- **balanced** is the same near the wall, but rows 14 and 15 stop growing where their own far-field balance takes over: 3.33 = (1/Da)/(1 + Δ) = 5/1.5 and 0.4 = 2ΔB/(1 + Δ/2) = 0.5/1.25. Row 14's weight at η = 20 is 122 instead of 1.7 × 10⁵.
- Both use only η and locked values; neither uses the reference or the current residuals. Neither changes where a residual is zero.

## Which is used when
Adam and L-BFGS use **log**; Levenberg-Marquardt uses **balanced** (setting `refine` = "lbfgs:log+lm:balanced"). Why: log alone stalled with the wall residual still large (run E7); balanced from the start left the θ tail almost flat (run X4); log first, balanced last reached the reference (runs X3, and seeds 0 and 1).

## In the code
- `residual_scale` ([pinn/losses.py:13](../../../src/bvp_pinn/pinn/losses.py#L13)); `final_scale` ([pinn/train.py:180](../../../src/bvp_pinn/pinn/train.py#L180)) reads the last stage's scale from `refine`.

## Checked by
`test_residual_scale_is_fixed_and_keeps_the_zero_set`, `test_balanced_scale_follows_the_coefficients` in [tests/test_pinn.py](../../../tests/test_pinn.py).

## Ready?
Yes.
