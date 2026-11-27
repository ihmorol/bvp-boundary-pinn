# Step 39 · Loss weights

[Index](README.md) · page 4, training one run · group 12, Adam · [← step 38](38-adam.md) · [step 40 →](40-log-and-monitor.md)

**In one line:** when the loss has several terms, give each a weight so every term pulls on the network equally hard.

## The equations
```text
G_i = ‖∂J_i/∂w‖                         (gradient size of term i)
λ̃_i = (Σ_j G_j)/G_i
λ_i ← 0.9 λ_i + 0.1 λ̃_i,   at most 1 000,   every 1 000 steps
J_total = Σ_i λ_i J_i
```
- In the run settings B0, B2 and B3 have one weighted term (the equations), so λ = 1 always. B1 has two: equations and interface jumps.
- The weights are frozen before the refine stages.
- Source: Wang, Sankaran, Wang and Perdikaris, "An expert's guide to training physics-informed neural networks" (2023). The cap of 1 000 came from pilots where weights reached 10⁹ to 10¹⁰.

## In the code
- `GradNormWeights` ([pinn/losses.py:87](../../../src/bvp_pinn/pinn/losses.py#L87)); `update` ([line 106](../../../src/bvp_pinn/pinn/losses.py#L106)); called at [pinn/train.py:130](../../../src/bvp_pinn/pinn/train.py#L130).

## Checked by
`test_gradient_norm_weights_follow_wang_et_al_2023`, `test_weight_cap_bounds_the_weights` in [tests/test_pinn.py](../../../tests/test_pinn.py).

## Ready?
Yes.
