# Step 7 · Mesh and start

[Index](README.md) · page 2, the classical reference · group 5, solve the reference · [← step 6](06-first-order-form.md) · [step 8 →](08-branch-scan.md)

**In one line:** a mesh with small steps near the wall, and a smooth first guess that already meets five of the six wall conditions.

## What it does
- **Mesh:** 400 nodes on [0, L], spaced exponentially so the first step is 10⁻³. The wall layer gets most of the nodes.
- **Start:** every field is c + d e^(−rη), a decaying curve. The constants are chosen so B1, B2, B3, B5, B6 hold exactly at η = 0. B4 is left to the solver because it depends on the unknown wall temperature.

## The equations
```text
mesh:   η_i = L (e^(β i/(n−1)) − 1)/(e^β − 1),  β chosen so η_1 = 10⁻³
start:  a = 1/(1 + 0.1 r + 0.5)              (from B1)
        f = f0 + (a/r)(1 − e^(−rη)),  f0 = 3θ0 r/44   (B2 and B5)
        g = 0.3 a r e^(−rη)                   (B3)
        θ = θ0 e^(−rη),  φ = −3θ0 e^(−rη),  χ = e^(−rη)   (B5, B6)
```
r (the rate) and θ0 (the start wall temperature) are free; the branch scan tries many of them.

## In the code
- `wall_mesh` ([reference/solver.py:14](../../../src/bvp_pinn/reference/solver.py#L14)) finds β with a root finder (`brentq`).
- `smooth_guess` ([reference/solver.py:25](../../../src/bvp_pinn/reference/solver.py#L25)) returns all 12 states, each the true derivative of the one before.
- Settings `initial_nodes` 400 and `first_step` 0.001 in [configs/reference.json](../../../configs/reference.json).

## Checked by
`test_smooth_guess_satisfies_wall_conditions_except_b4` in [tests/test_equations_algebra.py](../../../tests/test_equations_algebra.py).

## Ready?
Yes. The PINN's start profile ([step 26](26-start-profile.md)) is this same guess with r = 1, θ0 = 0.3.
