# Step 44 · Damping

[Index](README.md) · page 4, training one run · group 13, refine · [← step 43](43-levenberg-marquardt.md) · [step 45 →](45-self-check.md)

**In one line:** μ decides how bold a Levenberg-Marquardt step is; it shrinks after a good step and grows after a bad one.

## The equations
```text
start:     μ = 10⁻³ × largest diagonal entry of KᵀK
predicted decrease = hᵀ(μ h − Kᵀ r)
gain       ρ = (J − J_new)/predicted decrease
ρ > 0:     accept the step;  μ ← μ · max(1/3, 1 − (2ρ − 1)³);  ν ← 2
ρ ≤ 0:     reject it;        μ ← ν μ;  ν ← 2ν;  solve again
stop:      after 260 steps, or μ > 10²⁰, or largest gradient entry < 10⁻¹⁴, or J < 10⁻²⁸
```
- Small μ: a full Gauss-Newton step, fast near the answer.
- Large μ: a short step along the downhill direction, safe far from it.
- ρ compares what the loss really did with what the quadratic model promised. This is Nielsen's rule (Madsen, Nielsen and Tingleff, 2004).

**Rollback (diamond "Monitor loss lower?").** After each refine stage (L-BFGS, then LM) the monitor loss is compared with its value before that stage, on that stage's scale. If it is higher or not a number, the stage is undone and its start weights restored.

## In the code
- μ start [pinn/lm.py:189](../../../src/bvp_pinn/pinn/lm.py#L189); gain [lines 198 to 199](../../../src/bvp_pinn/pinn/lm.py#L198); updates [lines 203 to 208](../../../src/bvp_pinn/pinn/lm.py#L203).
- Rollback: [pinn/train.py:223 to 225](../../../src/bvp_pinn/pinn/train.py#L223).

## Ready?
Yes.
