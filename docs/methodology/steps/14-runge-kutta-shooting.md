# Step 14 · Runge-Kutta shooting (V10)

[Index](README.md) · page 2, the classical reference · group 6, checks at L = 20 · [← step 13](13-core-then-chi.md) · [step 15 →](15-slope-check.md)

**In one line:** an independent solver, Runge-Kutta multiple shooting, must give the same reference.

## What it does
- [0, 20] is cut into 40 pieces, short near the wall (first piece 0.02).
- On each piece the equations are integrated as an initial value problem with DOP853, an 8th-order Runge-Kutta method, to relative accuracy 10⁻¹¹.
- The unknowns are the 12 states at the start of each piece (480 numbers). Newton's method adjusts them until each piece ends where the next begins and B1 to B12 hold.
- It starts from the smooth guess only, never from the solve_bvp answer.
- **V10:** every state curve agrees with solve_bvp to 10⁻⁴ relative. Measured: about 3 × 10⁻¹⁰.

## Words used here
| Term | Meaning |
|---|---|
| Single shooting | Guess the wall values, integrate to L, correct. Fails here: some modes grow like e^(1.8η) over 20 units |
| Multiple shooting | The same on short pieces, so nothing grows too much on any piece |

## In the code
- `rk_reference` ([reference/shooting.py:92](../../../src/bvp_pinn/reference/shooting.py#L92)) sets up the pieces; `multiple_shooting` ([line 27](../../../src/bvp_pinn/reference/shooting.py#L27)) runs Newton with a finite-difference Jacobian and step halving.
- Called at [reference/run.py:169](../../../src/bvp_pinn/reference/run.py#L169).

## Checked by
`test_multiple_shooting_agrees_with_solve_bvp` in [tests/test_reference_verify.py](../../../tests/test_reference_verify.py).

## Ready?
Yes.
