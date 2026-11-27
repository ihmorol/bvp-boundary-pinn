# Step 6 · First-order form

[Index](README.md) · page 2, the classical reference · group 5, solve the reference · [← step 5](05-guards.md) · [step 7 →](07-mesh-and-start.md)

**In one line:** the five equations become 12 first-order equations z′ = F(η, z), the form solve_bvp needs.

## What it does
- The state z holds the 12 quantities f, f′, f″, f‴, g, g′, θ, θ′, φ, φ′, χ, χ′.
- Most entries of z′ are just the next entry of z (the derivative of f is f′, and so on).
- The five new pieces, H = (f⁽⁴⁾, g″, θ″, φ″, χ″), come from the equations themselves. Each equation is linear in its highest derivatives, so R = M H + r₀, where r₀ is the residual with H set to zero. Setting R = 0 gives the small linear system M H = −r₀.

## The equation
```text
      | 1.5  −0.5     0              0         0 |
      | 0     1.25    0              0         0 |
M  =  | 0     0      (1 + 0.5A³)/8.8 0         0 |
      | 0     0       3/8.8          1/8.8     0 |
      | 0     0       0              2χ/8.8    1 |
```
- Row 1 is Eq. 14: f⁽⁴⁾ with 1 + Δ = 1.5, and g″ with −Δ = −0.5.
- Row 3 needs 1 + 0.5A³ > 0, otherwise θ″ cannot be found ([step 12](12-conditions-check.md) checks it).

## In the code
1. `highest_matrix` ([reference/equations.py:132](../../../src/bvp_pinn/reference/equations.py#L132)) builds M at every point.
2. `highest_derivatives` ([line 148](../../../src/bvp_pinn/reference/equations.py#L148)) computes r₀ by calling the residuals with H = 0, then solves M H = −r₀ with `np.linalg.solve` (no inverse is formed).
3. `rhs` ([line 157](../../../src/bvp_pinn/reference/equations.py#L157)) stacks z′. `boundary` ([line 199](../../../src/bvp_pinn/reference/equations.py#L199)) gives B1 to B12 as the 12 boundary residuals.

## Checked by
- `test_residual_is_linear_in_highest_derivatives` and `test_highest_derivatives_zero_the_residual` in [tests/test_equations_algebra.py](../../../tests/test_equations_algebra.py).

## Ready?
Yes. The PINN does not use this form; it works with the equations directly.
