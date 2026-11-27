# Step 3 · Unknowns and domain

[Index](README.md) · page 1, the fixed problem · group 1, locked case · [← step 2](02-substitutions.md) · [step 4 →](04-one-transcription.md)

**In one line:** five unknown functions of η on 0 ≤ η ≤ 20; the far end stands in for η → ∞.

## What it does
- The unknowns are f, g, θ, φ, χ. The input is η alone; the 22 values are fixed numbers, never inputs.
- The source puts six conditions at η → ∞. No solver can reach infinity, so every solver stops at L = 20 and applies the far conditions there ([conditions.md](conditions.md)).
- L = 20 is our choice (kept by the user on 2026-11-26). The PINN and the reference both solve the L = 20 problem, so comparing them is fair. How far L = 20 is from infinity is measured separately ([step 21](21-l-study.md)): 1.6 to 3.3 × 10⁻⁴ relative for most wall values, 1.8 × 10⁻³ for φ(0).

## In the code
- `L_reference` in [configs/reference.json](../../../configs/reference.json) is 20.
- `validate` in [pinn/runs.py:55](../../../src/bvp_pinn/pinn/runs.py#L55) refuses a PINN setting with another L ([line 70](../../../src/bvp_pinn/pinn/runs.py#L70)) and window edges that do not run from 0 to L ([line 73](../../../src/bvp_pinn/pinn/runs.py#L73)).
- Evaluation refuses a run whose L differs from the reference's (`evaluate_run`, [evaluation/evaluate.py:96](../../../src/bvp_pinn/evaluation/evaluate.py#L96)).

## Ready?
Yes. The L error is a reported number, not a fault (decision recorded in [L_STUDY.md](../../plans/L_STUDY.md)).
