# Step 23 · Points

[Index](README.md) · page 3, the PINN model · group 8, network · [← step 22](22-reference-for-checking.md) · [step 24 →](24-input-map.md)

**In one line:** the η values where the equations are enforced: 2 000 training points packed toward the wall, plus 1 000 monitor points.

## What it does
- Cut [0, 1] into n equal cells and put one random point s in each (stratified sampling: random, but never two points crowded together).
- Stretch with the formula below, so small s gives η close to 0. About 31 % of the points (618 of 2 000) fall in the wall layer [0, 0.5].
- A second set of 1 000 monitor points is drawn the same way. They never enter a gradient; they only measure the loss on points training did not use.
- Both sets come from the run's seed and stay fixed for the whole run.

## The equation
```text
η = 20 (e^(5s) − 1)/(e^5 − 1),   s_i = (i + u_i)/n,   u_i random in [0, 1)
```
The 5 is `stretch`: a larger value packs more points at the wall.

## In the code
- `wall_points` ([pinn/points.py:8](../../../src/bvp_pinn/pinn/points.py#L8)) and `make_points` ([line 15](../../../src/bvp_pinn/pinn/points.py#L15)), which uses `numpy.random.default_rng(seed)`.
- `in_range` ([line 30](../../../src/bvp_pinn/pinn/points.py#L30)) splits points into windows for B1 to B3.
- Settings `n_points` 2 000, `n_monitor` 1 000, `stretch` 5 in [configs/pinn.json](../../../configs/pinn.json).

## Ready?
Code yes. The point count 2 000 is a pilot value: the two good B0 runs used 1 000 (open decision 1 in [MAIN_RUN_READINESS](../../reviews/MAIN_RUN_READINESS_2026-11-26.md)).
