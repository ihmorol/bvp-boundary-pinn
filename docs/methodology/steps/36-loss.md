# Step 36 · Loss

[Index](README.md) · page 3, the PINN model · group 10, residuals and loss · [← step 35](35-residual-scale.md) · [step 37 →](37-stage-plan.md)

**In one line:** the one number training makes small: the mean over the points of the five scaled, squared residual rows. No boundary term and no reference data.

## The equation
```text
J = (1/N) Σ_points Σ_(k=14..18) ( s_k(η_j) · row_k(η_j) )²
```
- N training points, five rows each (steps 34 and 35). In the run settings J has no boundary term: all 12 conditions are already exact (steps 27 to 32).
- This is the whole loss for B0 (weight 1, [step 39](39-loss-weights.md)).

**Extra terms for the other arms ([step 37](37-stage-plan.md)):**
```text
B1:  λ_0 J + λ_I · mean over window edges of (jump of the 12 states)²
B2:  J on the new window + 1.0 × J on the earlier windows           (physics replay)
B3:  J on the new window + 1.0 × mean (u − u_frozen)² on earlier windows   (self replay)
```
u_frozen are the network's own five outputs at the end of the previous stage.

## In the code
- `equation_mse` ([pinn/losses.py:41](../../../src/bvp_pinn/pinn/losses.py#L41)): states via jets, rows, scale, squared, summed, divided by N.
- `loss_parts` ([pinn/train.py:57](../../../src/bvp_pinn/pinn/train.py#L57)) collects the terms; `weighted` ([line 79](../../../src/bvp_pinn/pinn/train.py#L79)) adds them with their weights.
- B1: `interface_jumps` ([pinn/losses.py:67](../../../src/bvp_pinn/pinn/losses.py#L67)). B3: `self_replay_mse` ([line 82](../../../src/bvp_pinn/pinn/losses.py#L82)).

## Ready?
Yes.
