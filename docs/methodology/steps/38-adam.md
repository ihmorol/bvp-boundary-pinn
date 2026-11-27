# Step 38 · Adam

[Index](README.md) · page 4, training one run · group 12, Adam · [← step 37](37-stage-plan.md) · [step 39 →](39-loss-weights.md)

**In one line:** Adam moves the weights downhill on the loss, with a step size that shrinks from 10⁻³ to 10⁻⁴.

## The equations
```text
gradient g = ∂J_total/∂w
m ← 0.9 m + 0.1 g,   v ← 0.999 v + 0.001 g²         (running averages)
w ← w − lr · m̂/(√v̂ + 10⁻⁸)                           (m̂, v̂: bias-corrected)
lr after step k = 10⁻³ × 0.1^(k/steps)                  (exponential decay)
```
- 20 000 steps for B0 and B1; 5 000 per stage for B2 and B3, each stage with a fresh optimizer and its own decay.
- Adam is good at getting from a rough start into the right region; it is slow at the last digits. That is the refine chain's job.

## In the code
- `adam_stage` ([pinn/train.py:117](../../../src/bvp_pinn/pinn/train.py#L117)): optimizer [line 119](../../../src/bvp_pinn/pinn/train.py#L119), decay [line 120](../../../src/bvp_pinn/pinn/train.py#L120), then per step: loss, weight update every 1 000 steps, non-finite check, backward, step.
- Settings `adam_steps` 20 000, `lr` 0.001, `lr_end` 0.0001.

## Ready?
Code yes. 20 000 steps is a pilot value; the good B0 runs used 8 000.
