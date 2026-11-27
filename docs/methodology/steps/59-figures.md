# Step 59 · Figures

[Index](README.md) · page 5, acceptance and evaluation · group 16, report · [← step 58](58-tables.md) · [step 60 →](60-settings.md)

**In one line:** eleven result figures, each drawn the way one of the closest papers in our literature review draws the same kind of result.

## The figures
| Figure | Shows | Form taken from |
|---|---|---|
| FB1, FB1b | f′, f″, g, θ, φ, χ: reference A as a line, each arm's median seed as open markers; whole domain, then 0 ≤ η ≤ 1 | squeezing flow Fig. 14, stretching cylinder Fig. 3, Falkner-Skan Fig. 1 |
| FB2 | absolute error of the same fields along η, log scale, one line per arm | squeezing flow Fig. 15, Falkner-Skan Fig. 3a |
| FB3 | final mean square of each R and size of each B, every seed, medians | Falkner-Skan Fig. 5b |
| FB4 | seed 0 of each arm: loss, monitor loss, R14 to R18 against wall time, phases marked | micropolar Fig. 5, stretching cylinder Fig. 4 |
| FB5 | largest error per window after each stage, B2 and B3 | ours (retention) |
| FB6 | each residual row along η on the 4 001-point grid | Falkner-Skan Fig. 3b, overfitting Fig. 3 |
| FB7 | every seed of every arm for four metrics, paired seeds joined, rejected runs hollow | squeezing flow Figs. 20 to 22, PINNacle |
| FB8 | histogram of the training points, window edges | micropolar Fig. 4 |
| FX1 | largest wall error against wall time, PINN runs and solve_bvp | PINN against FEM Figs. 2, 4 |
| FX2 | median minutes per run split into Adam, L-BFGS, LM | micropolar Fig. 6 |

- **Median seed:** the run with the median field error, so a figure shows a typical run, not the best. A failed run sorts last.
- **Report points:** each arm draws its markers on its own subset of about 40 points, dense at the wall, so markers of different arms never sit on top of each other.
- Not drawn on purpose: parameter-effect figures, which fill most of those papers. The values are locked, so there is no sweep.
- Every figure is saved as PNG (300 dpi) and PDF in the colour-blind-safe palette of `figstyle.py`; `figures/captions.md` lists each figure with its source.

## In the code
- `fig_profiles` ([plots.py:90](../../../src/bvp_pinn/evaluation/plots.py#L90)), `fig_errors` ([line 117](../../../src/bvp_pinn/evaluation/plots.py#L117)), `fig_residuals` ([line 142](../../../src/bvp_pinn/evaluation/plots.py#L142)), `fig_training` ([line 166](../../../src/bvp_pinn/evaluation/plots.py#L166)), `fig_windows` ([line 206](../../../src/bvp_pinn/evaluation/plots.py#L206)), `fig_residual_profile` ([line 224](../../../src/bvp_pinn/evaluation/plots.py#L224)), `fig_seed_spread` ([line 248](../../../src/bvp_pinn/evaluation/plots.py#L248)), `fig_points` ([line 281](../../../src/bvp_pinn/evaluation/plots.py#L281)), `fig_time_split` ([line 305](../../../src/bvp_pinn/evaluation/plots.py#L305)), `fig_cost` ([line 328](../../../src/bvp_pinn/evaluation/plots.py#L328)).
- FB6 needs the residual rows on the grid: `evaluate_run` saves them as `R` in `pred_eval.npz` ([evaluation/evaluate.py:126](../../../src/bvp_pinn/evaluation/evaluate.py#L126)).
- The full mapping to the papers, with what was left out and why: [RESULT_FIGURES.md](../../pinn/RESULT_FIGURES.md).

## Checked by
`test_campaign_figures_and_tables_are_built` in [tests/test_evaluation.py](../../../tests/test_evaluation.py) builds every figure from two tiny runs. On 2026-11-27 the set was also built from the two accepted B0 pilot runs and the 20 dry runs, and every figure was looked at.

## Ready?
Yes.
