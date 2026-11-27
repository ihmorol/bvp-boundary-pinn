# The reduced equations (page 1, group 2)

[Index](README.md) · page 1, the fixed problem · the purple boxes Eq. 14 to Eq. 18 · [Glossary](glossary.md)

**In one line:** Eqs. 14 to 18 with the 22 locked values put in. Same equations, fewer symbols. Each line equals zero for the true solution.

## How to read them

- q = η + 0.2 everywhere (from κ = b = 0.2). At the wall q = 0.2, so 1/q = 5, 1/q² = 25, 1/q³ = 125, 1/q⁴ = 625. That is why the terms are large near the wall and the wall layer is thin.
- A = 1 − 0.8θ (from θ_w = 0.2). It appears in Eq. 16 only.
- δ∗ = 0, so every term with δ∗ is gone. The equations are steady.
- "times 8.8" means the whole equation was multiplied by Pr = 8.8 to clear the fractions. Multiplying by a positive number does not change where it is zero.
- A product of two unknowns (f f‴, θ′φ′, χφ″) makes an equation **nonlinear**: doubling a small change does not double its effect. All five equations are nonlinear.

```text
Eq. 14   1.5(f⁽⁴⁾ + 2f‴/q − f″/q² + f′/q³) + 0.2(f f‴ − f′f″)/q + 0.2(f f″ − f′²)/q² − 0.2 f f′/q³
         − 5(f″ + f′/q) − 0.5(g″ + g′/q) − 0.4(θ′ − 3θ/q)/q⁴ = 0
Eq. 15   1.25(g″ + g′/q) − 0.25(f″ + f′/q + 2g) − 0.2(f′g − f g′)/q = 0
Eq. 16   (1 + 0.5A³)(θ″ + θ′/q) + (0.3 − 1.2A²)θ′² + 0.1θ′φ′ + 1.76 f θ′/q
         + 0.2(θ − 0.1)(0.2 f (1 − 2/q²) − 2f′)/q⁴ = 0                      (times 8.8)
Eq. 17   φ″ + φ′/q + 3(θ″ + θ′/q) + 1.76 f φ′/q = 0                          (times 8.8)
Eq. 18   χ″ + χ′/q + 0.2 f φ/q + (5/22)(χφ″ + χ′φ′) = 0
```

## What each number is

| Number | Comes from | Locked values |
|---|---|---|
| 1.5, 1.25 | 1 + Δ, 1 + Δ/2 | Δ = 0.5 |
| 0.2 (in Eqs. 14, 15, 18) | κ | κ = 0.2 |
| 5 | 1/Da | Da = 0.2 (from M = 5) |
| 0.25 | Δ·B | B = 0.5 |
| 0.4 and 3θ/q | 2β_m; (1/q − 4/h)θ with h = q | β_m = 0.2 |
| 0.5A³ | N_r A³ | N_r = 0.5 |
| 0.3, 1.2A², 0.1 | N_t; 3(θ_w − 1)N_r A²; N_b | N_t = 0.3, N_b = 0.1 |
| 1.76 | Pr κ | Pr = 8.8 |
| 0.2(θ − 0.1) | 2β_m λ_m (θ − ε) | λ_m = 0.5, ε = 0.1 |
| 3 | N_t/N_b | |
| 5/22 | Pe/Pr | Pe = 2 |

## The highest derivatives

| Equation | Highest derivative | So the field is |
|---|---|---|
| 14 | f⁽⁴⁾ | fourth order in f |
| 15 | g″ | second order in g |
| 16 | θ″ | second order in θ; its coefficient 1 + 0.5A³ depends on θ and must stay above zero |
| 17 | φ″ (and θ″) | second order in φ |
| 18 | χ″ | second order in χ |

So the PINN needs f up to f⁽⁴⁾ and the other four fields up to their second derivative ([step 33](33-taylor-jets.md)), and the reference needs 4 + 2 + 2 + 2 + 2 = 12 first-order states ([step 6](06-first-order-form.md)).

## In the code

- One copy of Eqs. 14 to 18 with symbols, not numbers: `residual_terms` in [reference/equations.py:44](../../../src/bvp_pinn/reference/equations.py#L44). It reads the 22 values from the locked case, so the numbers above appear only at run time.
- The reduced forms were checked against that code on 2026-11-26 at 200 random states: largest difference 3e-16.
- The SymPy test [tests/test_equations_sympy.py](../../../tests/test_equations_sympy.py) compares the code with a second, separate transcription.
