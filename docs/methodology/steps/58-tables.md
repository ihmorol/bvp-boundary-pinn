# Step 58 · Tables

[Index](README.md) · page 5, acceptance and evaluation · group 16, report · [← step 57](57-solve-bvp-timing.md) · [step 59 →](59-figures.md)

**In one line:** the result tables, all rebuilt from the run folders of one campaign.

## What they hold
| File | Content |
|---|---|
| `summary.csv` | one row per run: identity, verdict and reasons, score, nearest branch, every accuracy number, cost |
| `failures.md` | every run that failed, was rejected, is undecided, landed off branch A, or is missing |
| `wall_values.md` (TB2) | each wall value per arm: median and range over the seeds, the reference, then the relative errors |
| `accuracy.md` (TB3) | per arm: largest wall error and largest field relative L2, after Adam and final, median and range; then the paired-seed counts |
| `fields.md` (TB4) | per arm and field (f, f′, f″, g, θ, φ, χ): relative L2 and largest absolute error, median and range over seeds |
| `nodes.md` (TB5) | absolute error of f′, θ, φ at η = 0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20: mean and range over the seeds of each arm |
| `cost.md` (TX1) | per arm: device, wall time, Adam and refine time, LM steps, parameters, and the solve_bvp time to the same error |
| `settings.json` | the PINN settings, reference settings and fingerprints, case fingerprint, commit, versions, run list |

**Rule G6 (diamond "All 5 paired seeds?").** "Arm X beats arm Y" may be claimed only if X has the smaller error in every one of the five paired seeds. TB3 counts, for each pair of arms, in how many paired seeds the row arm wins. A failed run never wins. Anything short of 5 of 5 is reported as no consistent effect.

## In the code
- Tables: `wall_table` ([evaluation/plots.py:396](../../../src/bvp_pinn/evaluation/plots.py#L396)), `accuracy_table` ([line 355](../../../src/bvp_pinn/evaluation/plots.py#L355)), `field_table` ([line 428](../../../src/bvp_pinn/evaluation/plots.py#L428)), `node_table` ([line 450](../../../src/bvp_pinn/evaluation/plots.py#L450)), `cost_table` ([line 496](../../../src/bvp_pinn/evaluation/plots.py#L496)). Which paper each table follows: [RESULT_FIGURES.md](../../pinn/RESULT_FIGURES.md).
- `summary.csv`, `failures.md`, `settings.json`: `summary_row` ([evaluation/evaluate.py:173](../../../src/bvp_pinn/evaluation/evaluate.py#L173)), `failure_lines` ([line 208](../../../src/bvp_pinn/evaluation/evaluate.py#L208)), `settings_record` ([line 243](../../../src/bvp_pinn/evaluation/evaluate.py#L243)).

## Checked by
`test_paired_seed_table_counts_wins_and_failed_runs` in [tests/test_evaluation.py](../../../tests/test_evaluation.py).

## Ready?
Yes. Built from the two accepted B0 pilot runs and the dry runs; `test_campaign_figures_and_tables_are_built` builds every table.
