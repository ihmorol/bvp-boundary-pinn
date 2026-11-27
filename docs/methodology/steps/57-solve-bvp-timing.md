# Step 57 · solve_bvp timing

[Index](README.md) · page 5, acceptance and evaluation · group 15, test · [← step 56](56-stages.md) · [step 58 →](58-tables.md)

**In one line:** how long the classical solver takes to reach a given accuracy, measured the same way as the PINN, for the cost comparison.

## What it does
- solve_bvp on the locked case at L = 20, from the same smooth start (rate 1, θ(0) = 0.3), at tolerances 10⁻³, 10⁻⁴, …, 10⁻⁸.
- Each solve is timed 5 times on one CPU thread; the median counts.
- Each answer's error uses the PINN's measure: largest relative wall error against reference A.
- `time_to_reach(error)`: the cheapest solve_bvp time whose error is at or below a given error. Table TX1 uses it with each arm's median error.

## In the code
- `time_solves` ([evaluation/classical.py:37](../../../src/bvp_pinn/evaluation/classical.py#L37)) and `time_to_reach` ([line 65](../../../src/bvp_pinn/evaluation/classical.py#L65)). Writes `classical_timing.json` in the campaign's results folder.

## Ready?
Yes. Times depend on the machine, so this runs on the same machine type as the campaign.
