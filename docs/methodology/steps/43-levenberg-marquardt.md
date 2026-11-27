# Step 43 · Levenberg-Marquardt

[Index](README.md) · page 4, training one run · group 13, refine · [← step 42](42-lbfgs.md) · [step 44 →](44-damping.md)

**In one line:** the final stage treats the loss as a sum of squares and takes Gauss-Newton steps with a safety term. It does almost all the accuracy work.

## The equations
```text
r = vector of all scaled residual rows × sqrt(1/N)  (5 per point; B1 adds the weighted jumps)
J = r · r                                            (the same loss as before, on the balanced scale)
K = ∂r/∂w                                            (Jacobian: one row per residual, one column per weight)
step h:   (KᵀK + μ I) h = −Kᵀ r
```
- KᵀK is the Gauss-Newton curvature: exact for the linear part of the problem, cheap to build from K.
- μ ≥ 0 is the damping ([step 44](44-damping.md)).
- 260 steps on the balanced scale.

## How the code makes it affordable
- K has 2 000 × 5 = 10 000 rows and 5 205 columns. It is built point by point (`vmap` of `jacrev`, 100 points at a time).
- A residual row depends only on the network that owns its point, so KᵀK is block diagonal with one 5 205 × 5 205 block per network. Each block is solved by a Cholesky factor. B1's few interface rows are added with the Woodbury identity, so B1 costs four block solves, not one four times larger.
- Measured cost, one CPU thread, 2 000 points: 21 to 24 s per step for B0, 32 s for B1. About 70 % of a run's time. In the B0 pilots it lowered the monitor loss from 2.3 × 10⁻² to 6.8 × 10⁻⁷.

## In the code
- `minimize` ([pinn/lm.py:83](../../../src/bvp_pinn/pinn/lm.py#L83)); per-point Jacobian `block_jacobian` ([line 111](../../../src/bvp_pinn/pinn/lm.py#L111)); `normal_equations` ([line 122](../../../src/bvp_pinn/pinn/lm.py#L122)); block solve with Cholesky and Woodbury `solve` ([line 142](../../../src/bvp_pinn/pinn/lm.py#L142), Woodbury at [line 161](../../../src/bvp_pinn/pinn/lm.py#L161)).
- Called from `refine_once` at [pinn/train.py:218](../../../src/bvp_pinn/pinn/train.py#L218).

## Checked by
`test_levenberg_marquardt_lowers_the_exact_mode_loss`, `test_levenberg_marquardt_works_on_the_training_objective` in [tests/test_pinn.py](../../../tests/test_pinn.py): it starts from exactly the training loss and lowers it, for every arm shape.

## Ready?
Code yes. 260 steps is a pilot value; the loss was still falling at 260 in the pilots.
