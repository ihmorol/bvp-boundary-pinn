# Methodology

**Status:** parameters locked, validation pending. Nothing in this file has run yet. It is the plan, not a result.

Two diagrams show the method. Both are one A4 landscape page and follow the style of [methodology-flowchart-a4.pdf](../methodology-flowchart-a4.pdf).

| Diagram | What it shows | Files |
|---|---|---|
| Methodology | The whole study in four gated stages: case and code, reference, training, test and report | [PDF](../bvp-methodology.pdf), [SVG](../bvp-methodology.svg), [HTML](../bvp-methodology.html) |
| Architecture | One training step (box 7 of the methodology), and the checks that stay outside the loss | [PDF](../bvp-pinn-architecture.pdf), [SVG](../bvp-pinn-architecture.svg), [HTML](../bvp-pinn-architecture.html) |

Both are drawn by [tools/build_diagrams.py](../tools/build_diagrams.py). Edit that file, then rebuild:

```bash
uv run --no-project --with fonttools --with pymupdf python -I tools/build_diagrams.py
```

The build stops if text does not fit its box, boxes overlap, an arrow runs through a box or a label sits on a border. It also checks the writing rule.

The detail lives in [strategy.md](strategy.md), [equations.md](equations.md) and [parameters.md](parameters.md). The diagrams only point at it. The earlier fifty-task version is kept in [the archive](archive/fifty-task-methodology-2026-11-16.md).

## Methodology diagram, box by box

The figure has four stages, in the gate order of [strategy.md](strategy.md). A later stage starts only when the one before it is done.

| Stage | Box | Step | Where the detail is |
|---|---|---|---|
| 1 Case and code | 1 | Locked case: Eqs. 14 to 19 for f, g, theta, phi, chi, 22 supervisor values. delta_star = 0, so the equations are steady. | [parameters.md](parameters.md), [equations.md](equations.md) |
| | 2 | Wall coefficient sizes. With q = h = eta + 0.2, the wall coefficients reach 250 at eta = 0. The fixed loss scales s, t are chosen from these sizes. | [parameters.md](parameters.md#coefficient-sizes-for-the-locked-target), [strategy.md section 5](strategy.md#5-boundary-enforcement-and-loss) |
| | 3, diamond | Thirteen software tests on synthetic cases. Any failure goes back to the code. | [strategy.md section 8](strategy.md#8-testing-before-any-long-run) |
| 2 Reference | 4, 5 | The same problem as 12 first-order states, solved with SciPy solve_bvp, then checked with tolerance 1e-5 to 1e-7, a finer mesh, L = 10, 15, 20 and other guesses. | [strategy.md section 6](strategy.md#6-classical-reference-simulation) |
| | diamond | Converged: the reference curves are kept for checking only. Not converged: the reference is marked unverified. The PINN may still run, but no accuracy claim is made. | [strategy.md section 6](strategy.md#6-classical-reference-simulation) |
| 3 Train, seeds 0, 1, 2 | 6 | 1024 interior points, half near the wall. Monitor and test points are separate sets. | [strategy.md section 7](strategy.md#7-point-sets-and-near-wall-coverage) |
| | 7 | Network eta to 4 tanh layers of 64 to 5 fields. Loss from scaled R14 to R18 and B1 to B12. The [architecture diagram](../bvp-pinn-architecture.pdf) shows this step in full. | [strategy.md sections 4 and 5](strategy.md#4-network-and-derivatives) |
| | 8, 9 | Adam with trial rate 1e-3 to 1e-4 over about 20000 steps, then L-BFGS with points, scales and weights frozen. | [strategy.md sections 9 and 10](strategy.md#9-adam-training-flow) |
| | diamond, 10 | Any non-finite value stops the run and saves the first bad term, keeping the last good checkpoint. Otherwise the checkpoint is picked on monitor points, never on reference or test error. | [strategy.md section 15](strategy.md#15-failure-handling-and-implementation-gates) |
| 4 Test and report | 11 | The picked model runs once on fresh points: fields, f', every R and B, compared with the reference. Targets 1e-4 and 1e-3 are proposals, not supervisor values. | [strategy.md section 12](strategy.md#12-final-evaluation-and-results), [training-proposal.json](../configs/training-proposal.json) |
| | diamond | Pass needs the targets met and the reference error far below the PINN error. | [strategy.md section 12](strategy.md#12-final-evaluation-and-results) |
| | 12 | Sensitivity to L, point count and seed, then variants B1 to B6 against baseline B0, one change at a time. | [strategy.md section 13](strategy.md#13-controlled-research-comparisons) |
| | 13 | Report fields, wall quantities, errors, time and memory, or the failure with its cause. Which wall quantities to report is still a supervisor question. | [questions-for-supervisor.md](questions-for-supervisor.md) |

## Architecture diagram, box by box

| Box | Step |
|---|---|
| 2 to 4 | N interior points and the two ends go through one shared network. Autograd gives f to f'''' and the first and second derivatives of g, theta, phi, chi. |
| 5 to 7 | The locked coefficients turn these into R14 to R18 (interior) and B1 to B12 (ends). The coefficients are never trained. |
| 8 | One scalar loss: weighted means of the scaled squared equation errors plus weighted scaled squared boundary errors. No reference data. |
| Diamond, 9 | If every value is finite, backpropagate and update the weights (Adam, then L-BFGS on frozen points). If not, stop and save the first bad term. |
| 10 to 13 | Saved weights are ranked on monitor points, the best is locked, then tested on fresh points against the classical reference. |
| Bottom row | What changes from the Lorenz PINN. The checked comparison is in [lorenz-to-bvp-methodology.md](lorenz-to-bvp-methodology.md). |

## What the diagrams leave out on purpose

- Full equations. They are in [equations.md](equations.md) and [simplified-equations.md](simplified-equations.md).
- Optional method changes B1 to B6 in detail. They are in [strategy.md section 11](strategy.md#11-optional-improvements-in-a-controlled-order).
- The eight unset training settings. They stay null in [training-proposal.json](../configs/training-proposal.json) until pilots decide them.
