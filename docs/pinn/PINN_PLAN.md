# PINN plan for the BVP

Date: 2026-11-19, updated 2026-11-24. Status: code complete and tested for all four arms (`src/bvp_pinn/pinn/` for training and acceptance, `src/bvp_pinn/evaluation/` for evaluation). B0 reached reference A in seeds 0 and 1 ([IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) section 7.3). Pilots P2b and P3 and the campaign have not run. Scope: this repository only.

Related files:

- [PRD.md](../plans/PRD.md): problem and goals.
- [SPEC_reference_solutions.md](../reference-solver/SPEC_reference_solutions.md): the classical reference.
- [TODO.md](../plans/TODO.md): ordered tasks.
- [CODING_RULES.md](../team/CODING_RULES.md): code rules.

The Lorenz-1960 work and the joint journal plan are in [../journal/](../journal/README.md). Gate numbers here match that plan; gate G2 belongs to the Lorenz work and is not used here.

## 1. Decisions this plan follows

From the supervisor, recorded in [questions-for-supervisor.md](../problem/questions-for-supervisor.md):

- **Values:** use them exactly as given. The physical meaning of symbols is not needed.
- **Task:** compute a classical reference, train a PINN, and test the PINN against the reference.
- **Network input:** eta only. Parameters are fixed values, never inputs.
- **No physical parameter sweep.**
- **Structures:** try both the basic structure and knowledge retention.

## 2. Arms

All arms solve the locked case on [0, 20] (branch A), with the same boundary conditions as the reference: asymptotic far conditions for theta and phi at eta = 20 (`equations.far_conditions`), and zero values there for f1, f2, g and chi ([SPEC](../reference-solver/SPEC_reference_solutions.md) section 8). Common settings:

- tanh networks, float64
- the five residuals and twelve boundary errors from [equations.md](../problem/equations.md)
- all 12 boundary conditions built into the output (exact mode, chosen by pilot P2a on 2026-11-22, [BC_ENFORCEMENT_REVIEW.md](BC_ENFORCEMENT_REVIEW.md)), so the loss holds only the equations
- a smooth start profile under the network with a zero last layer, a log input map, and a fixed residual scale built from eta and the locked values
- gradient-norm loss weights (Wang et al. 2023) for the terms that remain, frozen before the refine stages. With one network in exact mode there is one term, so the weight stays 1; B1 has a second term, the interface
- Adam on the log scale, then the same refine chain for every arm: L-BFGS on the log scale, then Levenberg-Marquardt on the balanced scale, on the full problem. Each refine stage is kept only if its monitor loss improves. B1 uses the block form of Levenberg-Marquardt, one block per window network (decided 2026-11-24, [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) section 5)
- matched total step budgets across arms

**Every arm imposes all twelve boundary conditions at every training step.**

Windows: K = 4 windows in eta, smaller near the wall. Example edges: 0, 0.5, 2, 6, L. The edges are fixed after the pilots (TODO phase P3).

| Arm | Structure | Training |
|---|---|---|
| B0 | One network on the whole [0, L] | All residuals and all boundary terms at once. The basic PINN |
| B1 | K separate networks, one per window | Trained together. Interface conditions make f, f1, f2, f3, g, g1, theta, th1, phi, ph1, chi, ch1 continuous between neighbours |
| B2 | One network | Windows added from the wall outward. Stage k trains the residual on [0, eta_k+1], keeping it on all windows already reached (physics replay). The last stage is the full problem, then L-BFGS |
| B3 | One network | As B2, but earlier windows are held to the network's own frozen outputs (self-replay), and the residual covers only the new window |

Why no marching in eta: both ends carry conditions. A window near the wall cannot be solved before the far field is known.

Note for reading results: after its last stage, B2 has the same objective as B0. So any difference between them comes from the training path, not from different equations.

No fallback through other parameter values (rule of 2026-11-23): Eqs. 14 to 18, B1 to B12 and the locked values are fixed. Only the way they are imposed and solved may change. If every arm fails at the locked case, the failure is reported with its logs.

## 3. Run rules

- **Seeds:** 0 to 4 for every arm, paired across arms.
- **Acceptance (declared 2026-11-24, before the campaign; our choice):** one run per arm and seed, never re-run to get a better result. After training, `bvp_pinn.pinn.estimate` judges each run without the reference, by the rule in `configs/pinn.json` (`selection`): training finished; the far field decays (m > 0 and mp > 0); theta has no constant tail (tail share at most 0.05); every boundary error at most 1e-8; the Newton-step error-estimate score at most 1e-3. A run's loss alone never decides, because a small loss can hide a wrong tail (run E8). Rejected runs stay in every table. The reference is read only afterwards, by `bvp_pinn.evaluation`.
- **Retention weight (B2, B3):** chosen from 0.1, 1, 10 by training residual only, in the pilot.
- **Run ID:** `bvp-{arm}-s{seed}-{confighash8}`.
- **Run folder:** `runs/<run_id>/` with:
  - `config.json`, including `campaign` for campaign runs
  - `history.csv`: every loss term per step
  - `model.pt` and `stage<k>.pt` (one per stage), `points.npz`
  - `meta.json`: git hash, versions, hardware, threads, parameter count, wall time, refine stages
  - `error_estimate.json` and `selection.json`: the acceptance rule's inputs and verdict
  - after evaluation: `metrics.json` and `pred_eval.npz` (outputs and derivatives on the evaluation grid)
- **Commands:** `python -m bvp_pinn.pinn.campaign` runs every arm and seed (resumable, `--jobs` for parallel runs); `python -m bvp_pinn.evaluation.evaluate`, `.classical` and `.plots` evaluate afterwards ([IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) section 2).
- **Threads:** `torch.set_num_threads(1)`. Up to 3 runs at once per laptop, if the pilot shows they do not slow each other much.
- **Summaries:** `results/<campaign>/summary.csv` is rebuilt by `bvp_pinn.evaluation.evaluate` from the run folders tagged with that campaign, never from pilots. Figures read only `summary.csv`, `classical_timing.json` and `.npz` files.
- **Failures:** failed, rejected, missing or diverged runs stay in every table and get one line in `results/<campaign>/failures.md`.

## 4. Metrics

Computed against the reference on its evaluation grid ([SPEC_reference_solutions.md](../reference-solver/SPEC_reference_solutions.md) section 6):

- **Field errors:** relative L2 and max absolute error for f, f1, f2, f3, f4, g, g1, g2, theta, th1, th2, phi, ph1, ph2, chi, ch1, ch2.
- **Wall values:** f2(0), theta(0), th1(0), phi(0), ph1(0), g1(0), ch1(0) against the reference, absolute and relative, and the largest relative one. These are plain numerical checks, not physical outputs.
- **Equation residuals:** mean square of R14 to R18, each separately, on the evaluation grid.
- **Boundary errors:** B1 to B12, each separately.
- **Retention (B2, B3):** error on each window after every stage.
- **Cost:** wall time, Adam and refine seconds, Levenberg-Marquardt steps, parameter count, threads; and the solve_bvp time to reach each arm's median wall error, on the same machine (`bvp_pinn.evaluation.classical`).
- **Acceptance:** each run's verdict and reasons, and its error-estimate score next to its true wall error.

## 5. Gates

| Gate | Pass rule |
|---|---|
| G1 code | Derivative tests up to order 4, residuals against SymPy, R(H) - R(0) = M*H, B4 stable form, NumPy against PyTorch evaluators all pass |
| G3 reference | SPEC V4 to V8, V10 and V12 pass at the locked case, and L is chosen |
| G4 retention | Error per window after each stage reported for B2 and B3. No pass bar is set for the BVP |
| G5 no leakage | Training and acceptance code (`bvp_pinn.pinn`) never imports the reference data or `bvp_pinn.evaluation`; `test_training_code_never_reads_the_reference` checks every file, and a review confirms it |
| G6 claims | "Arm X beats arm Y" only if it holds in all 5 paired seeds; otherwise "no consistent effect" |

## 6. Figures and tables

| ID | Content |
|---|---|
Since 2026-11-27 every figure follows a figure of the closest papers in our literature review; the mapping, figure by figure, is in [RESULT_FIGURES.md](RESULT_FIGURES.md).

| ID | Content |
|---|---|
| TB1 | Reference study: wall values against tolerance and L (from gate G3) |
| FB1, FB1b | f′, f″, g, θ, φ, χ: reference as a line, each arm's median seed as markers; whole domain and wall layer |
| FB2 | Pointwise absolute error of the same fields, log scale, one line per arm |
| FB3 | Final residual per equation and error per boundary condition, per arm |
| FB4 | Convergence history of seed 0 per arm against wall time, Adam, L-BFGS and LM marked |
| FB5 | Error per window after each stage, B2 and B3 |
| FB6 | Pointwise residual of each equation on the evaluation grid |
| FB7 | Every seed of every arm for four metrics, paired seeds joined (rule G6) |
| FB8 | Histogram of the training points |
| TB2 | Wall values, all arms against reference, median and range over seeds, with relative errors |
| TB3 | Accuracy after Adam and final, paired-seed counts |
| TB4 | Relative L2 and largest absolute error per field |
| TB5 | Absolute error at report points over the seeds |
| FX1 (BVP part) | Error against wall time: PINN arms and solve_bvp |
| FX2 | Time per run split into Adam, L-BFGS and LM |
| TX1 (BVP part) | Cost table |

## 7. Questions the results must answer

Each answer gives the finding, its evidence (figure and run IDs), and whether all 5 seeds agree.

1. Does B0 reach the reference wall values? Which equation or boundary term limits it?
2. Do separate window networks (B1) help near the wall compared with B0?
3. Do B2 or B3 do better than B0, and which retention form does better?
4. Where is the error largest, and does the residual point to the same place?
5. How far is each arm from solve_bvp at equal wall time?

## 8. Work order and target dates

Dates follow the joint journal timeline ([../journal/implementation-workflow.md](../journal/implementation-workflow.md)). Task details are in [TODO.md](../plans/TODO.md).

| TODO phases | Content | Target end |
|---|---|---|
| R0 to R2 | Environment, first-order system, boundary function | 14 Dec 2026 |
| R3 to R5 | Locked-case solve, convergence and branch study (gate G3), reference export | 31 Dec 2026 |
| P1 to P3 | PINN building blocks (gate G1), B0 pilot, B1 to B3 pilots, configs frozen | 14 Jan 2027 |
| P4 | Campaign: 4 arms x 5 seeds | 1 Feb 2027 |
| P5 | Evaluation, figures, answers to section 7 | 10 Feb 2027 |

Review rule: nobody merges their own code. The other member reviews each phase before it counts as done.

## 9. Risks

| Risk | Response |
|---|---|
| A solver does not converge at the locked case | More starting guesses, a longer L, the deflation search. Never another parameter value. If it still fails, report the failure |
| 4th-order derivatives are slow | Taylor-mode jets ([RUNTIME_STRATEGY.md](RUNTIME_STRATEGY.md)) and a per-point Levenberg-Marquardt Jacobian. Measured 2026-11-24 at 2000 points, one thread: 24 s per LM step for B0, 32 s for B1. Fewer points in pilots only, never in final runs |
| PINN stalls near the wall | Declared fixes in order: loss scales, wall clustering, longer budget. Then freeze and report |
| B1 interface terms make training unstable | Report it. B1 is a comparison arm, not the main claim |
| A result goes against expectations | Report it. The section 7 questions are answered either way |
