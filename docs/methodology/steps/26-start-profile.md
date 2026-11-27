# Step 26 · Start profile

[Index](README.md) · page 3, the PINN model · group 8, network · [← step 25](25-network.md) · [step 27 →](27-correction-rule.md)

**In one line:** a fixed smooth profile is added under the network, and the last layer starts at zero, so training starts from a sensible decaying shape.

## The equation
```text
N(η) = network(x(η)) + start(η)
start:  f = 0.0205 + 0.625(1 − e^(−η)),   g = 0.1875 e^(−η)
        θ = 0.3 e^(−η),   φ = −0.9 e^(−η),   χ = e^(−η)
```
- This is the reference's smooth guess ([step 7](07-mesh-and-start.md)) with rate 1 and θ(0) = 0.3. It uses only the locked values, never the reference solution.
- Because the last layer is zero at the start, the untrained network outputs exactly this profile.

## Why
From a random start the early pilots fell into a "cold wall" state (θ near 0 at the wall) and stayed there. Starting warm and decaying avoided it in every later run.

## In the code
- `SmoothStart` ([pinn/start.py:18](../../../src/bvp_pinn/pinn/start.py#L18)) holds c and d; `jets` gives the profile and its derivatives.
- Added in `MLP.forward` ([pinn/model.py:38](../../../src/bvp_pinn/pinn/model.py#L38)) and in `mlp_jets` ([pinn/jets.py:69](../../../src/bvp_pinn/pinn/jets.py#L69)). Zero last layer: `zero_last` in `MLP.__init__`.

## Checked by
`test_start_profile_matches_the_reference_smooth_guess` and `test_start_profile_with_zero_last_layer` in [tests/test_pinn.py](../../../tests/test_pinn.py).

## Ready?
Yes.
