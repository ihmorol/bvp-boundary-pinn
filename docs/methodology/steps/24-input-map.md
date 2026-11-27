# Step 24 · Input map

[Index](README.md) · page 3, the PINN model · group 8, network · [← step 23](23-points.md) · [step 25 →](25-network.md)

**In one line:** turn η into the network input x in [−1, 1] with a logarithm, so the thin wall layer gets a large share of the input.

## The equation
```text
x = 2 ln(1 + η/0.1)/ln(1 + 20/0.1) − 1 = 2 ln(1 + η/0.1)/ln 201 − 1
```
- x = −1 at the wall, +1 at η = 20.
- The wall layer [0, 0.5] maps to [−1, −0.32], about 34 % of the input range. With a plain linear map it would get 2.5 %.
- The derivatives of x are known exactly: dᵏx/dηᵏ = (2/ln 201)(−1)^(k−1)(k−1)!/(0.1 + η)ᵏ. The jets use them ([step 33](33-taylor-jets.md)).

## In the code
- `MLP.forward` ([pinn/model.py:38](../../../src/bvp_pinn/pinn/model.py#L38)) applies the map; `input_jet` ([pinn/jets.py:55](../../../src/bvp_pinn/pinn/jets.py#L55)) gives x and its first four derivatives.
- B1: every window network maps its own interval [lo, hi] the same way.
- Settings `input_map` "log", `map_scale` 0.1.

## Ready?
Yes.
