# Step 29 · B5, then B2

[Index](README.md) · page 3, the PINN model · group 9, exact conditions · [← step 28](28-wall-temperature.md) · [step 30 →](30-build-f.md)

**In one line:** with θ′(0) known, B5 gives φ′(0), and B2 then gives f(0).

## The equations
```text
φ′(0) target = −3 θ′(0)        (B5, 3 = N_t/N_b)
f(0)  target = φ′(0)/44        (B2, 1/44 = f_w/(Pr Le))
```
**Worked example.** θ′(0) = −1.25944 gives φ′(0) = 3.77831 and f(0) = 0.08587, both the reference values.

φ(0) is kept as the network gives it (its own row, miss 0), the same as θ(0).

## In the code
- [pinn/constraints.py:80 to 85](../../../src/bvp_pinn/pinn/constraints.py#L80): `ph1_0 = −Nt/Nb × th1_0`, then the f(0) miss (B2) and the φ′(0) miss (B5).

## Ready?
Yes.
