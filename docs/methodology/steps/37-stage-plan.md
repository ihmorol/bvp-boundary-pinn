# Step 37 · Stage plan and the four arms

[Index](README.md) · page 4, training one run · group 12, Adam · [← step 36](36-loss.md) · [step 38 →](38-adam.md)

**In one line:** the four arms differ only in how the network is split and in what order the domain is trained; everything else is shared.

## The four arms
| Arm | Networks | Stages | Extra loss term |
|---|---|---|---|
| B0 | one on [0, 20] | one, all points | none |
| B1 | one per window [0, 0.5], [0.5, 2], [2, 6], [6, 20] | one, all windows together | jumps of the 12 states at the window edges |
| B2 | one | four: window 1, then 2, 3, 4 | residual on the earlier windows (physics replay) |
| B3 | one | four, as B2 | the network's own earlier outputs on the earlier windows (self replay) |

- All arms use the same points, the same seeds 0 to 4 (paired), the same total of 20 000 Adam steps, and the same refine chain on the full problem ([steps 42 to 44](42-lbfgs.md)).
- B1: the wall conditions belong to the first window network, the far conditions to the last; inner networks carry no conditions.
- Why no plain marching window by window: both ends carry conditions, and a wall error grows by about 6 × 10⁵ by η = 6 if marched outward (`tools/marching_growth.py`).

## In the code
- `plan_stages` ([pinn/train.py:39](../../../src/bvp_pinn/pinn/train.py#L39)) returns the stages and their point groups; `full_problem` ([line 50](../../../src/bvp_pinn/pinn/train.py#L50)) is the all-windows problem used by refine and monitor.
- Arm settings: `arms` block of [configs/pinn.json](../../../configs/pinn.json).

## Ready?
Code yes, tested end to end for every arm. B1 to B3 have never trained at a full budget (only dry runs). The window edges and the replay weight 1.0 are pilot values.
