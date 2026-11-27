# Step 25 · Network

[Index](README.md) · page 3, the PINN model · group 8, network · [← step 24](24-input-map.md) · [step 26 →](26-start-profile.md)

**In one line:** a small tanh network maps x to five raw outputs, one per field.

## What it does
```text
x → Linear(1→40) → tanh → [Linear(40→40) → tanh] × 3 → Linear(40→5) → N_f, N_g, N_θ, N_φ, N_χ
weights: (40 + 40) + 3 × (1 600 + 40) + (200 + 5) = 5 205
```
- float64 throughout (about 16 digits), because L-BFGS and Levenberg-Marquardt need it.
- The hidden weights start random from the run's seed (`torch.manual_seed`), with PyTorch's standard start. The last layer starts at zero ([step 26](26-start-profile.md)).
- B1 has one such network per window: 4 × 5 205 = 20 820 weights.

## Words used here
| Term | Meaning |
|---|---|
| Linear(a→b) | b weighted sums of a inputs plus b offsets (bias) |
| tanh | S-shaped function between −1 and 1, smooth to any order, so f⁽⁴⁾ exists |

## In the code
- `MLP` ([pinn/model.py:16](../../../src/bvp_pinn/pinn/model.py#L16)); `build` ([line 48](../../../src/bvp_pinn/pinn/model.py#L48)) makes one network or one per window and wraps each in the boundary layer ([step 27](27-correction-rule.md)).
- Seed and precision: [pinn/run.py:57 to 60](../../../src/bvp_pinn/pinn/run.py#L57).
- Settings `width` 40, `depth` 4.

## Ready?
Yes. Size is a pilot value but has been used in every run so far.
