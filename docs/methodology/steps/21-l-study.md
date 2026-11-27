# Step 21 · L study

[Index](README.md) · page 2, the classical reference · group 7, cross-checks · [← step 20](20-error-bars.md) · [step 22 →](22-reference-for-checking.md)

**In one line:** the L error of every wall value from L = 8 to 160, and what a longer L would cost.

## What it does
- Solves branch A at L = 8, 10, 12, 15, 20, 25, 30, 40, 60, 80, 120, 160 in both far forms.
- Takes the η → ∞ value of each wall quantity from the longest solves (L = 2 560 to 10 240) and reports the L error of each L.
- Results: at L = 20 the L error is 1.6 to 3.3 × 10⁻⁴ relative, and 1.8 × 10⁻³ for φ(0). It falls slowly, like L^(−0.41). χ′(0) has no limit. Below L = 20 values swing through their limits (φ(0) is 3.9 % off at L = 8).
- It also shows what a longer L does to the PINN: fewer points in the wall layer at the same point count.

## In the code
- [tools/l_choice.py](../../../tools/l_choice.py): `solve_ladder` ([line 51](../../../tools/l_choice.py#L51)), `limits`, `needed_L`, `pinn_side`. Writes `reference/l_choice.md` and `.json`. Plan and decision: [L_STUDY.md](../../plans/L_STUDY.md).

## Ready?
Yes. L stays 20 (user's decision).
