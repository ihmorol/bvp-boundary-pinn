# Result figures: how the closest papers show their results, and what our runs now make

Date: 2026-11-27. The figures and tables that `python -m bvp_pinn.evaluation.plots` builds for a campaign now follow the figures of the closest papers in our literature review, figure by figure. The main run builds them in its evaluate step ([tools/main_run.py](../../tools/main_run.py)), so the campaign produces them with no extra step. Checked on the two accepted B0 pilot runs and on 20 dry runs of all four arms (section 5).

## 1. Which papers, and why these

| Paper (local analysis) | Why it is close | What we take from its figures |
|---|---|---|
| Micropolar nanofluid PINN, Modelling 2026 ([analysis](../../sources/literature/micropolar-2026/analysis.md)) | Closest physics: a coupled similarity BVP from a stretching surface, Adam then L-BFGS | Phase-marked convergence history, point histogram, cost profile, wall value table |
| Squeezing flow PINN, 2026 ([analysis](../../sources/literature/squeezing-2026/analysis.md)) | Fourth-order momentum like our f; compares optimiser variants; 100 independent runs | Reference line with network markers, pointwise error per variant on a log axis, metrics per run, error at report nodes over runs |
| Hard-constrained Falkner-Skan PINN, 2026 ([analysis](../../sources/literature/falkner-skan-hard-bc-2026/analysis.md)) | Closest structure: hard boundary conditions, truncated domain, Adam then L-BFGS with rollback, a variant ablation | Velocity and shear profiles, pointwise errors and pointwise residual, final-metric comparison of variants, per-case error table |
| Stretching cylinder PINN, Discover Nano 2026 ([analysis](../../sources/literature/cylinder-nano-2026/analysis.md)) | Same problem class, decay conditions at the far end | Network dashed over the numerical curve, loss history, error per field |
| PINNacle, NeurIPS 2024 ([analysis](../../sources/literature/pinnacle-benchmark-2024/analysis.md)) | Method comparison over seeds | Mean and spread over seeds, failed runs kept |
| PINN against FEM, 2024 ([analysis](../../sources/literature/pinn-vs-fem-2024/analysis.md)) | Fair cost comparison with a classical solver | Accuracy against time, log-log |
| Failure as overfitting, 2026 ([analysis](../../sources/literature/overfitting-failure-2026/analysis.md)) | Residual between training points; train against test loss | Monitor loss beside training loss; residual on a grid that is not the training grid |

## 2. What their figures show and how they draw them

Read from the papers' pages on 2026-11-27 (figure captions and the rendered pages).

| Paper, figure | Content | How it is drawn |
|---|---|---|
| Squeezing, Fig. 14 | Reference (bvp4c) against the network, f(η) | Reference as a solid line, network as open circles at 21 report nodes |
| Squeezing, Fig. 15 (also 17, 19) | Absolute error for three optimiser variants | One line with markers per variant, log y axis, η on x |
| Squeezing, Figs. 20 to 22 | Fitness, Theil coefficient and MSE of each of 100 runs | Run index on x, log y, one point per run |
| Squeezing, Tables 4 to 7 | Absolute error at report nodes over the 100 runs | Min, max, mean and standard deviation per node |
| Falkner-Skan, Fig. 1 | PINN and reference velocity f′ and shear f″ | Overlaid lines, one panel each |
| Falkner-Skan, Fig. 3 | Pointwise profile errors and pointwise equation residual | Log y, η on x |
| Falkner-Skan, Fig. 4 | Fixed-grid residual monitor during Adam, adaptive weights | Log y against Adam step |
| Falkner-Skan, Fig. 5 | Three formulations compared: monitor convergence and final metrics | (a) curves on one log axis, (b) one bar per formulation for each metric |
| Falkner-Skan, Table 2 | Wall shear, wall error, residual MSE and maximum, relative L2 of f′ and f″, largest absolute error | One column per case |
| Micropolar, Fig. 4 | Points added by adaptive refinement | Histogram over η, most points near the wall |
| Micropolar, Fig. 5 | MSE, RMSE, MAE and largest error of the equations, the conditions and both | Log y against iteration, a dashed line where Adam ends and L-BFGS begins, final values written on the plot |
| Micropolar, Fig. 6 | Training time, inference time, memory | Bar charts |
| Micropolar, Table 1 | Wall shear against a finite element reference | Present value beside reference value |
| Cylinder, Fig. 3 | PINN against the numerical solution | Numerical solid, PINN dashed |
| Cylinder, Fig. 4 | Total, physics and boundary loss | Log y against epoch |
| Cylinder, Tables 2 to 6 | L2 error of f′, θ, φ per parameter value | One table per parameter |
| PINN against FEM, Figs. 2 and 4 | Error against time for many network sizes and meshes | Log-log, one point per configuration |
| Overfitting, Fig. 1d and Fig. 3 | Train loss against test loss; residual between collocation points | Log y; residual along a line with the points marked |

## 3. The figures and tables our runs make now

All are written to `results/<campaign>/` and listed with their sources in `figures/captions.md`. FB means a BVP figure, TB a table, FX and TX the cost comparison with solve_bvp.

| ID | File | Shows | Form taken from | Code |
|---|---|---|---|---|
| FB1 | `figures/FB1_profiles` | f′, f″, g, θ, φ, χ on [0, 20]: reference A as a black line, each arm's median-seed PINN as open markers on report points (each arm on its own points, so markers never overlap) | Squeezing Fig. 14; cylinder Fig. 3; Falkner-Skan Fig. 1 (with the shear f″) | `fig_profiles`, [plots.py:90](../../src/bvp_pinn/evaluation/plots.py#L90) |
| FB1b | `figures/FB1b_profiles_wall` | The same on the wall layer 0 ≤ η ≤ 1 | As FB1 | same, with `xmax=1` |
| FB2 | `figures/FB2_errors` | Absolute error of the six fields along η, log scale, one line per arm, six decades shown | Squeezing Fig. 15; Falkner-Skan Fig. 3a | `fig_errors`, [line 117](../../src/bvp_pinn/evaluation/plots.py#L117) |
| FB3 | `figures/FB3_residuals` | Final mean square of each equation and size of each condition, every seed, medians | Falkner-Skan Fig. 5b; micropolar Fig. 5 (equations and conditions apart) | `fig_residuals`, [line 142](../../src/bvp_pinn/evaluation/plots.py#L142) |
| FB4 | `figures/FB4_training` | Seed 0 of each arm: loss, monitor loss, R14 to R18 against wall time, with Adam, L-BFGS and LM marked; the largest logged boundary error in the title | Micropolar Fig. 5; cylinder Fig. 4; overfitting Fig. 1d | `fig_training`, [line 166](../../src/bvp_pinn/evaluation/plots.py#L166) |
| FB5 | `figures/FB5_windows` | Largest error per window after each stage, B2 and B3 | Ours (retention, PINN_PLAN section 7 question 3) | `fig_windows`, [line 206](../../src/bvp_pinn/evaluation/plots.py#L206) |
| FB6 | `figures/FB6_pointwise_residual` | Absolute value of each residual row along η on the 4 001-point grid, and the largest of the five | Falkner-Skan Fig. 3b; overfitting Fig. 3 | `fig_residual_profile`, [line 224](../../src/bvp_pinn/evaluation/plots.py#L224) |
| FB7 | `figures/FB7_seed_spread` | Every seed of every arm for wall error, field relative L2, residual MSE and wall time; paired seeds joined; medians; hollow marker for a rejected run | Squeezing Figs. 20 to 22; PINNacle; Falkner-Skan Fig. 5b | `fig_seed_spread`, [line 248](../../src/bvp_pinn/evaluation/plots.py#L248) |
| FB8 | `figures/FB8_points` | Histogram of the training points over η, whole domain and wall layer, window edges | Micropolar Fig. 4 | `fig_points`, [line 281](../../src/bvp_pinn/evaluation/plots.py#L281) |
| FX1 | `figures/FX1_error_vs_time` | Largest wall error against wall time: every PINN run and solve_bvp at six tolerances | PINN against FEM Figs. 2 and 4 | `fig_cost`, [line 328](../../src/bvp_pinn/evaluation/plots.py#L328) |
| FX2 | `figures/FX2_time_split` | Median minutes per run of each arm, split into Adam, L-BFGS and LM | Micropolar Fig. 6 | `fig_time_split`, [line 305](../../src/bvp_pinn/evaluation/plots.py#L305) |
| TB2 | `wall_values.md` | Wall values per arm (median and range) beside reference A, then the relative errors | Micropolar Table 1; cylinder Table 7 | `wall_table`, [line 396](../../src/bvp_pinn/evaluation/plots.py#L396) |
| TB3 | `accuracy.md` | Accuracy after Adam and final, and the paired-seed counts for rule G6 | PINNacle (mean and spread, failures kept) | `accuracy_table`, [line 355](../../src/bvp_pinn/evaluation/plots.py#L355) |
| TB4 | `fields.md` | Relative L2 and largest absolute error of f, f′, f″, g, θ, φ, χ per arm | Cylinder Tables 2 to 6; Falkner-Skan Table 2 | `field_table`, [line 428](../../src/bvp_pinn/evaluation/plots.py#L428) |
| TB5 | `nodes.md` | Absolute error of f′, θ, φ at η = 0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20: mean and range over seeds | Squeezing Tables 4 to 7 | `node_table`, [line 450](../../src/bvp_pinn/evaluation/plots.py#L450) |
| TX1 | `cost.md` | Cost per arm and the solve_bvp time to reach the same error | PINN against FEM | `cost_table`, [line 496](../../src/bvp_pinn/evaluation/plots.py#L496) |

What changed in the code for this: `evaluate.py` now saves the five residual rows on the grid (`R` in `pred_eval.npz`, [evaluate.py:126](../../src/bvp_pinn/evaluation/evaluate.py#L126)) and the L-BFGS and LM seconds of each run in `summary.csv`. The test `test_campaign_figures_and_tables_are_built` in [tests/test_evaluation.py](../../tests/test_evaluation.py) builds the whole set from two tiny runs.

## 4. What we did not take, and why

| Their figures | Why not |
|---|---|
| Profiles and surfaces for changing parameters (most figures of all four application papers), and skin friction, Nusselt and Sherwood numbers against parameters | The 22 values are locked and there is no parameter sweep (supervisor, 2026-11-19; AGENTS.md section 3). Our wall values are reported at the locked case only (TB2) |
| Regression, error histogram and training-state plots of a supervised surrogate (squeezing Figs. 23, 24) | We train no surrogate on reference data; the reference never feeds training |
| Adaptive weight trajectory (Falkner-Skan Fig. 4b) | In exact mode one network has one loss term with weight 1. B1's two weights are in `history.csv` (columns w0, w1) if needed later |
| Network and residual schematics (micropolar Figs. 2, 3; cylinder Fig. 2) | They belong to the methodology diagrams, which already exist |
| Memory bars (micropolar Fig. 6) | Memory is not recorded per run; time is (FX2, TX1) |
| Error tables over 100 runs | Our rule is five paired seeds (gate G6); FB7 and TB5 show all five |

## 5. Checked on real runs

Built on 2026-11-27 from copies of the two accepted B0 pilot runs (`bvp-B0-s0-3a276995`, `bvp-B0-s1-3a276995`; 1 000 points, 8 000 Adam steps) and the 20 dry runs. Every figure and table was built and looked at. From the two pilot runs (pilot numbers, not campaign results):

- TB4: relative L2 error 1.4 × 10⁻⁴ (χ) to 3.0 × 10⁻³ (f), median of the two seeds.
- TB2: relative wall error of θ(0) 1.2 × 10⁻⁴ and of φ(0) 3.3 × 10⁻³, median.
- FX2: 93 minutes per run on one CPU thread, of which Adam 14, L-BFGS 13 and LM 66 minutes.
- FB6: the largest residual sits in the wall layer: up to 0.37 (seed 0) and 1.4 (seed 1) for η < 1, against at most 5.5 × 10⁻⁴ and 1.2 × 10⁻³ for η ≥ 1.

The dry runs check that the four-arm figures (FB4, FB5, FB7) lay out correctly; their numbers mean nothing.

## 6. Rebuild

```bash
uv run python -m bvp_pinn.evaluation.plots --campaign P4
```
