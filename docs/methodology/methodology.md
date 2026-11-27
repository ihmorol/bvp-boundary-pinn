# Methodology

**Status (2026-11-24):** the diagrams show the method as the code in `src/bvp_pinn/` runs it. The reference is built and audited. The PINN code is complete and tested, and B0 reached the reference in two seeds (IMPLEMENTATION_NOTES section 7.3). The pilots P2b and P3 and the 20-run campaign have not run, so the diagrams show no results.

Two one-page diagrams show the method, a one-page compact methodology gives the reduced equations and the whole run in 13 steps, and a six-page detailed methodology shows every step (below). All are A4 landscape pages in the style of [methodology-flowchart-a4.pdf](figures/methodology-flowchart-a4.pdf).

| Diagram | What it shows | Files |
|---|---|---|
| Methodology | The whole study in four gated stages: case and code, reference, training, test and report | [PDF](figures/bvp-methodology.pdf), [SVG](figures/bvp-methodology.svg), [HTML](figures/bvp-methodology.html) |
| Architecture | One training step (boxes 7 to 9 of the methodology), and the checks that stay outside the loss | [PDF](figures/bvp-pinn-architecture.pdf), [SVG](figures/bvp-pinn-architecture.svg), [HTML](figures/bvp-pinn-architecture.html) |

Both are drawn by [tools/build_diagrams.py](../../tools/build_diagrams.py). Edit that file, then rebuild:

```bash
uv run --no-project --with fonttools --with pymupdf python -I tools/build_diagrams.py
```

The build stops if text does not fit its box, boxes overlap, an arrow runs through a box or a label sits on a border. It also checks the writing rule. Mermaid text versions are in [flowcharts.md](flowcharts.md).

## Methodology diagram, box by box

The four stages are the gates of [strategy.md](../plans/strategy.md). A later stage starts only when the one before it is done.

| Stage | Box | Step | Where the detail is |
|---|---|---|---|
| 1 Case and code | 1 | Locked case: Eqs. 14 to 19 for f, g, theta, phi, chi, 22 supervisor values. delta_star = 0, so the equations are steady. | [parameters.md](../problem/parameters.md), [equations.md](../problem/equations.md) |
| | 2 | Wall coefficient sizes. With q = h = eta + 0.2, the wall coefficients reach 250 at eta = 0. The fixed residual scales are built from eta and the locked values only. | [parameters.md](../problem/parameters.md#coefficient-sizes-for-the-locked-target), `pinn/losses.py` |
| | 3, diamond | `uv run pytest`: equations against SymPy, derivatives to order 4, built-in conditions, optimizers, an end-to-end run of every arm, the no-leakage check. Any failure goes back to the code. | [IMPLEMENTATION_NOTES.md](../pinn/IMPLEMENTATION_NOTES.md) section 6 |
| 2 Reference | 4, 5 | The same problem as 12 first-order states, solved with SciPy solve_bvp, checked with tolerances 1e-5 to 1e-8, Runge-Kutta shooting and a C++ box scheme, L from 20 to 10 240, and a branch search. | [SPEC](../reference-solver/SPEC_reference_solutions.md) |
| | diamond | Converged: the reference curves are kept for checking only. Not converged: the reference is marked unverified and no accuracy claim is made. | [SPEC](../reference-solver/SPEC_reference_solutions.md) section 8 |
| 3 Train 4 arms, 5 seeds | 6 | 2000 training points stretched toward the wall; a separate monitor set that never enters a gradient. | `pinn/points.py` |
| | 7 | Network eta to 4 tanh layers of 40 to 5 fields, with a log input map. B1 to B12 are built into the output; the loss holds R14 to R18. | [BC_ENFORCEMENT_REVIEW.md](../pinn/BC_ENFORCEMENT_REVIEW.md), [PINN_GUIDE.md](../pinn/PINN_GUIDE.md) |
| | 8, 9 | Adam from a smooth start profile on the log scale, then L-BFGS on the log scale and Levenberg-Marquardt on the balanced scale, weights frozen. Each refine stage is kept only if its monitor loss improves. | [IMPLEMENTATION_NOTES.md](../pinn/IMPLEMENTATION_NOTES.md) sections 2 and 7 |
| | diamond, 10 | A non-finite value stops the run and saves the first bad term. A finished run is accepted or rejected by the rule in `configs/pinn.json`: a decaying far tail and a small Newton-step error estimate, both made without the reference. Rejected runs stay in every table. | [PINN_PLAN.md](../pinn/PINN_PLAN.md) section 3 |
| 4 Test and report | 11 | Every run against the reference: fields, wall values, every R and B; and the solve_bvp time to reach the same error. | [PINN_PLAN.md](../pinn/PINN_PLAN.md) sections 4 and 6 |
| | diamond | A claim that one arm beats another needs all 5 paired seeds (gate G6). | [PINN_PLAN.md](../pinn/PINN_PLAN.md) section 5 |
| | 12 | B1 to B3 against B0, paired by seed; retention per window for B2 and B3; cost. | [PINN_PLAN.md](../pinn/PINN_PLAN.md) section 7 |
| | 13 | Report fields, wall values, errors and cost, or the failure with its cause. | [PINN_PLAN.md](../pinn/PINN_PLAN.md) section 7 |

## Architecture diagram, box by box

| Box | Step |
|---|---|
| 2 to 4 | Training points go through one shared network (log input map, 4 tanh layers of 40, 5 outputs). Taylor-mode jets give f to f'''' and the first and second derivatives of g, theta, phi, chi in one forward pass. |
| 6, 7 | The boundary layer reads the network's end values and adds fixed correction shapes, so B1 to B12 hold exactly for any weights. The locked coefficients set B4, B10 and B11. |
| 5 | The locked coefficients turn the fields into R14 to R18 at the training points. |
| 8 | One scalar loss: the mean over points of the scaled squared residuals. The scale depends on eta and the locked values only. No reference data. |
| Diamond, 9 | If every value is finite, backpropagate and update the weights (Adam, then L-BFGS and Levenberg-Marquardt). If not, stop and save the first bad term. |
| 10 to 13 | Monitor points decide whether a refine stage is kept. The acceptance rule judges the finished run without the reference. Then the run is compared with the classical reference. |
| Bottom row | What changed from the Lorenz PINN: position instead of time, twelve conditions built in instead of one initial value, derivatives to fourth order, five outputs. |

## Compact methodology, one page

[bvp-methodology-compact.pdf](figures/bvp-methodology-compact.pdf) ([SVG](figures/bvp-methodology-compact.svg), [HTML](figures/bvp-methodology-compact.html)): the six detailed pages on one page. Top: the reduced Eqs. 14 to 18 and B1 to B12 with the locked values put in (the same checked forms as page 1 of the detailed version). Then 13 steps: reference (1 to 3), PINN with the four arms, exact conditions, loss, Adam, refine and the acceptance rule (4 to 10), test, rule G6, report and the Colab main run (11 to 13). Rebuild only this page with `--only compact`.

## Detailed methodology, six pages

**Step by step:** [steps/](steps/README.md) has one file per step, 1 to 72: what it does, the words and equations it uses, where the code does it, and whether it is ready. Start with its [glossary](steps/glossary.md).

Every step the code takes, numbered 1 to 72, at the same page size and in the same style. The equations are the reduced forms with the locked values put in ([simplified-equations.md](../problem/simplified-equations.md)); on 2026-11-26 each was checked against `src/bvp_pinn/reference/equations.py` at random states (largest difference 3e-16). One PDF: [bvp-methodology-detailed.pdf](figures/bvp-methodology-detailed.pdf); single pages (PDF, SVG, HTML, PNG) are in [figures/detailed/](figures/detailed/).

| Page | Steps | What it shows | Code |
|---|---|---|---|
| 1 The fixed problem | 1 to 5 | Locked values, substitutions, the reduced Eqs. 14 to 18, B1 to B12, which conditions are linear, one transcription and its guards | `config.py`, `reference/equations.py`, `pinn/runs.py` |
| 2 The classical reference | 6 to 22 | First-order form, mesh and start, branch scan, tolerance and domain ladders, far decay, checks V7 to V12, gate G3, export, the C++ box scheme, branch search, error bars, L study | `reference/`, `engine/`, `tools/reference_error_bars.py`, `tools/l_choice.py` |
| 3 The PINN model | 23 to 36 | Points, input map, network, start profile, the exact boundary layer in build order, Taylor jets, residual rows, residual scales, loss | `pinn/points.py`, `model.py`, `start.py`, `constraints.py`, `jets.py`, `physics.py`, `losses.py` |
| 4 Training one run | 37 to 46 | The four arms, Adam stages with gradient-norm weights, stage checkpoints, L-BFGS and Levenberg-Marquardt with rollback, self check, run folder | `pinn/train.py`, `lm.py`, `selfcheck.py`, `runs.py` |
| 5 Acceptance and evaluation | 47 to 59 | Newton-step estimate, score and rule; the test against reference A; rule G6; tables and figures | `pinn/estimate.py`, `evaluation/` |
| 6 The main run | 60 to 72 | The Colab notebook, pins, device, the resumable steps, failures and restarts | `tools/colab_main_run.ipynb`, `tools/main_run.py`, `pinn/campaign.py` |

Rebuild only these pages:

```bash
uv run --no-project --with fonttools --with pymupdf python -I tools/build_diagrams.py --only detailed
```

Not drawn: the options that are in the code but off in the run settings (full BFGS refine, best checkpoint, resampling, the soft and linear boundary modes), and the analysis tools that do not feed the run (`tools/marching_growth.py`, `tools/time_pinn_step.py` outside the timing step).

## What the diagrams leave out on purpose

- Full equations. They are in [equations.md](../problem/equations.md) and [simplified-equations.md](../problem/simplified-equations.md).
- The arms in detail and the run rules. They are in [PINN_PLAN.md](../pinn/PINN_PLAN.md).
- The settings that pilots P2b and P3 still have to freeze (point count, budgets, window edges, replay weight).
