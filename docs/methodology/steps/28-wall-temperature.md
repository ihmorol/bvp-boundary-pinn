# Step 28 · Wall temperature first (B4)

[Index](README.md) · page 3, the PINN model · group 9, exact conditions · [← step 27](27-correction-rule.md) · [step 29 →](29-b5-then-b2.md)

**In one line:** B4 is nonlinear, but it fixes one slope from one value. So keep the wall temperature as the network gives it and compute the slope from it.

## The equation
```text
θ0 = N_θ(0)                                   (kept, not corrected)
A_w = 1 − 0.8 θ0
Q_w = (1 − θ0)(0.2 + A_w)(0.04 + A_w²)
θ′(0) target = −(1 − θ0) − Q_w
```
This is B4, −θ′ = 0.998 − θ + 1.25(1 − 0.8θ)⁴, rearranged. Q_w is the same radiation term written without the subtraction of two close numbers (the "stable form"); a test checks the two forms agree.

**Worked example.** With the reference's θ0 = 0.25019: A_w = 0.79985, Q_w = 0.50962, so θ′(0) = −0.74981 − 0.50962 = −1.25944, the reference value.

## Why this is exact even though B4 is nonlinear
At each forward pass θ0 is just a number read from the network. The target slope is then a known number too, and correcting the slope is linear. The θ shapes are built so they do not change θ(0) (the kept row), so the slope target stays consistent.

## In the code
- [pinn/constraints.py:72 to 77](../../../src/bvp_pinn/pinn/constraints.py#L72): `th0 = lo[TH, 0]`, `Aw`, `Qw`, `th1_0`, then the θ misses [0, th1_0 − N_θ′(0)].
- The reference's copy: `b4_stable` ([reference/equations.py:169](../../../src/bvp_pinn/reference/equations.py#L169)).

## Checked by
`test_b4_stable_equals_original` in [tests/test_equations_algebra.py](../../../tests/test_equations_algebra.py); the random-weight test of [step 27](27-correction-rule.md).

## Ready?
Yes.
