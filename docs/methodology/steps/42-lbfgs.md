# Step 42 · L-BFGS

[Index](README.md) · page 4, training one run · group 13, refine · [← step 41](41-stage-checkpoint.md) · [step 43 →](43-levenberg-marquardt.md)

**In one line:** after Adam, L-BFGS refines the weights using curvature built from recent gradients, on the whole problem.

## What it does
- Builds an estimate of the loss curvature from the last 50 steps' gradients and positions, and steps toward the minimum of that model.
- **Strong Wolfe line search:** each step length must lower the loss enough and make the slope along the step flat enough.
- Settings: up to 5 000 iterations (at most 1.25 loss evaluations each), log scale, weights λ frozen, all points and windows (B2, B3 drop the replay term: the refine objective is the same for every arm).

## In the code
- `refine_once` ([pinn/train.py:185](../../../src/bvp_pinn/pinn/train.py#L185)) with `torch.optim.LBFGS` at [line 205](../../../src/bvp_pinn/pinn/train.py#L205) (`lr` 1, `history_size` 50, `tolerance_grad` 10⁻¹², `tolerance_change` 10⁻¹⁶, `line_search_fn` "strong_wolfe").
- `refine_phase` ([line 154](../../../src/bvp_pinn/pinn/train.py#L154)) runs the chain "lbfgs:log", then "lm:balanced".

## Ready?
Code yes. 5 000 iterations is a pilot value.
