# Step 33 · Taylor jets

[Index](README.md) · page 3, the PINN model · group 10, residuals and loss · [← step 32](32-g-and-chi.md) · [step 34 →](34-residual-rows.md)

**In one line:** compute every derivative the equations need (up to f⁽⁴⁾) in one forward pass, instead of 12 nested gradient calls.

## What it does
The network has one input, so each layer can pass along the value and its first four η-derivatives together.

```text
linear layer:  y_0 = W a_0 + b,   y_k = W a_k  (k = 1 to 4)
tanh layer, t = tanh(a_0), D = 1 − t²:
  (tanh a)′  = D a′
  (tanh a)″  = −2tD a′² + D a″
  (tanh a)‴  = D(6t² − 2) a′³ − 6tD a′a″ + D a‴
  (tanh a)⁗ = 8tD(2 − 3t²) a′⁴ + 6D(6t² − 2) a′²a″ − 2tD(4a′a‴ + 3a″²) + D a⁗
```
(the chain rule to fourth order, also called Faà di Bruno's formula). The input map and the start profile have exact derivatives; the correction shapes too (dⁿe^(−d/0.5) differentiated by the product rule).

## Why
With PyTorch's usual nested autograd, each extra derivative order costs about 4 times the one before: 1 682 ms against 77 ms at order 4, 2 000 points (measured 2026-11-22). The weight gradient for training is still one normal backward pass.

## In the code
- `tanh_jet` ([pinn/jets.py:25](../../../src/bvp_pinn/pinn/jets.py#L25)), `input_jet` ([line 55](../../../src/bvp_pinn/pinn/jets.py#L55)), `mlp_jets` ([line 69](../../../src/bvp_pinn/pinn/jets.py#L69)), `_support_jet` ([line 85](../../../src/bvp_pinn/pinn/jets.py#L85)), `constrained_jets` ([line 100](../../../src/bvp_pinn/pinn/jets.py#L100)).
- `states` ([line 126](../../../src/bvp_pinn/pinn/jets.py#L126)) returns z (12 states) and H (5 highest derivatives) in the reference's order.

## Checked by
`test_forward_jets_equal_nested_autograd` and `test_derivatives_up_to_order_four` in [tests/test_pinn.py](../../../tests/test_pinn.py): same values and same weight gradients to 10⁻¹¹.

## Ready?
Yes.
