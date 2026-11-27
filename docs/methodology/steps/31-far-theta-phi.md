# Step 31 · Far theta and phi (B10, B11)

[Index](README.md) · page 3, the PINN model · group 9, exact conditions · [← step 30](30-build-f.md) · [step 32 →](32-g-and-chi.md)

**In one line:** at the far end, keep θ(20) and φ(20) as the network gives them, and set their slopes from the decay formula.

## The equations
```text
A_L = 1 − 0.8 θ(20),   m = 1.76 f(20)/(1 + 0.5 A_L³),   m_p = 1.76 f(20)
θ′(20) target = −m θ(20)/20.2                       (B10)
φ′(20) target = −(m_p φ(20) − 3 m θ(20))/20.2       (B11)
```
**Worked example (reference values).** f(20) = 0.19317, θ(20) = 0.06831, φ(20) = 0.18254 give m = 0.23902, m_p = 0.33999, θ′(20) = −8.08 × 10⁻⁴ and φ′(20) = −6.48 × 10⁻⁴, the reference slopes.

## Limits of this step
- The layer makes B10 and B11 hold for any m, also m < 0. A tail with m < 0 does not decay (branch B). So m > 0 is checked after training by the acceptance rule ([step 49](49-acceptance-rule.md)).
- The formula divides by 1 + 0.5A_L³, which is zero only if θ(20) ≈ 2.82; training starts at θ(20) ≈ 0.

## In the code
- [pinn/constraints.py:94 to 99](../../../src/bvp_pinn/pinn/constraints.py#L94): `fL` from the built f, then `far_conditions` ([reference/equations.py:176](../../../src/bvp_pinn/reference/equations.py#L176)) evaluated with zero slopes returns minus the target slopes, and the misses follow.

## Ready?
Yes.
