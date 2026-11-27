# The boundary conditions B1 to B12 (page 1, group 3)

[Index](README.md) · page 1, the fixed problem · the blue and grey condition boxes · [Glossary](glossary.md)

**In one line:** six rules at the wall and six at the far end. A solution must meet all twelve. In the PINN all twelve are built into the output, so they hold exactly for any weights.

## The twelve, reduced

```text
At the wall, η = 0                          At the far end, η = L = 20 (q_L = 20.2)
B1   1.5 f′ = 1 + 0.1 f″                     B7    f′ = 0
B2   f = φ′/44                               B8    f″ = 0
B3   g = −0.3 f″                             B9    g = 0
B4   −θ′ = 0.998 − θ + 1.25(1 − 0.8θ)⁴       B10   θ′ = −m θ/20.2
B5   φ′ = −3θ′                               B11   φ′ = −(m_p φ − 3 m θ)/20.2
B6   χ = 1                                   B12   χ = 0
with m = 1.76 f(L)/(1 + 0.5A_L³),  m_p = 1.76 f(L),  A_L = 1 − 0.8θ(L)
```

## What each one says, in plain words

| BC | Says | Numbers from |
|---|---|---|
| B1 | The wall slope of f is tied to its curvature (a slip condition) | 1 + N₁/κ = 1.5, N₁ = 0.1 |
| B2 | The wall value of f is tied to the wall slope of φ | f_w/(Pr Le) = 0.2/8.8 = 1/44 |
| B3 | The wall value of g is tied to the wall curvature of f | ℓ = 0.3 |
| B4 | The wall slope of θ is set by the wall value of θ (heat exchange plus radiation) | Bi = Rd = 1, θ_w = 0.2 |
| B5 | The wall slopes of φ and θ cancel in a fixed ratio | N_t/N_b = 3 |
| B6 | χ starts at 1 | |
| B7, B8, B9, B12 | f′, f″, g and χ are zero at the far end | |
| B10, B11 | θ and φ go to zero at infinity. At L = 20 they are still 0.068 and 0.18, so we impose the slope that only a decaying tail q^(−m) has | Lentini and Keller 1980 |

**Why B10 and B11 look different.** The source says θ → 0 and φ → 0 as η → ∞. We stop at L = 20, where θ and φ have not reached zero yet (they decay slowly, like a power of q). Setting them to zero at 20 would bend the whole curve. So we keep only the decaying part: far out, θ ≈ C q^(−m), whose slope is θ′ = −m θ/q. That is B10. B11 is the same idea for φ. Reference A has m = 0.239; at η = 20 these give θ′(20) = −8.08 × 10⁻⁴ and φ′(20) = −6.48 × 10⁻⁴, the reference's own values.

## Linear or not

| Kind | Which | Why |
|---|---|---|
| Linear | B1, B2, B3, B5 to B9, B12 | Fixed multiples of values and slopes equal a number |
| Nonlinear | B4 | Contains (1 − 0.8θ)⁴, a fourth power of the unknown |
| Nonlinear as we apply them | B10, B11 | m contains f(L) and θ(L), so m θ is a product of unknowns |

B2, B3 and B5 involve two fields each. That is a **coupling**, not a nonlinearity; they are still linear.

## Hard, not soft (decision of 2026-11-26)

- **Hard** means the condition is built into the network output ([steps 27 to 32](27-correction-rule.md)). It holds to rounding (about 10⁻¹⁵) for any weights. The loss then contains only the five equations.
- **Soft** would add each B² to the loss with a weight. It holds only as well as training makes it. Our 2026-11-21 pilots with soft and part-soft conditions stuck at θ(0) ≈ 0 with B4 missed by about 2.3, and the weights grew to 10⁹ to 10¹⁰.
- The three nonlinear ones can still be hard because each fixes one slope from values the layer leaves alone: B4 sets θ′(0) from θ(0); B10 sets θ′(20) from f(20), θ(20); B11 sets φ′(20) from f(20), θ(20), φ(20). Record: [BOUNDARY_CONDITIONS_2026-11-26.md](../../reviews/BOUNDARY_CONDITIONS_2026-11-26.md).

## In the code

- All twelve, one copy: `boundary_terms` in [reference/equations.py:204](../../../src/bvp_pinn/reference/equations.py#L204). B4 uses the stable form `b4_stable` ([line 169](../../../src/bvp_pinn/reference/equations.py#L169)); B10 and B11 come from `far_conditions` ([line 176](../../../src/bvp_pinn/reference/equations.py#L176)).
- The reference passes them to solve_bvp as its boundary residuals ([step 6](06-first-order-form.md)). The PINN builds them into its output ([steps 27 to 32](27-correction-rule.md)) and logs their size every 100 steps. The C++ engine has its own copy (`wall_bc`, `far_bc` in [engine/bvp_engine.cpp:102](../../../engine/bvp_engine.cpp#L102)).
- Mode switch: `bc_mode` in [configs/pinn.json](../../../configs/pinn.json) is `exact`; `SOFT_BCS` in [pinn/constraints.py:19](../../../src/bvp_pinn/pinn/constraints.py#L19) lists which conditions each mode leaves to the loss (none for exact).
