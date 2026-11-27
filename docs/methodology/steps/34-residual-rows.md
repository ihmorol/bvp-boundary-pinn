# Step 34 · Residual rows

[Index](README.md) · page 3, the PINN model · group 10, residuals and loss · [← step 33](33-taylor-jets.md) · [step 35 →](35-residual-scale.md)

**In one line:** each residual is divided by the coefficient of its highest derivative, so all five are in comparable units.

## The equation
```text
row 14 = R14/1.5                     (units of f⁽⁴⁾)
row 15 = R15/1.25                    (units of g″)
row 16 = R16/((1 + 0.5A³)/8.8)       (units of θ″)
row 17 = R17/(1/8.8) = 8.8 R17       (units of φ″)
row 18 = R18/1                       (units of χ″)
```
R14 to R18 are the original residuals (not the "times 8.8" forms of page 1). Dividing by a nonzero number does not move where a residual is zero, so the solution is the same; only the relative sizes in the loss change. The θ row's divisor depends on θ and stays above zero (reference: at least 0.143).

## In the code
- `physics.residuals` ([pinn/physics.py:15](../../../src/bvp_pinn/pinn/physics.py#L15)) calls `residual_terms` and divides; the divisor of row 16 is `thermal_coefficient` ([reference/equations.py:126](../../../src/bvp_pinn/reference/equations.py#L126)).

## Ready?
Yes.
