# PINN plan for the BVP

Date: 2026-11-19, updated 2026-11-21. Status: code written (`src/bvp_pinn/pinn/`), pilots running. Scope: this repository only.

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
- boundary conditions built into the output or as loss terms, one mode for all arms, chosen by pilot P2a ([BC_ENFORCEMENT_REVIEW.md](BC_ENFORCEMENT_REVIEW.md))
- gradient-norm loss weights (Wang et al. 2023), frozen before L-BFGS (decided 2026-11-21, [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) section 5)
- Adam, then L-BFGS on the full problem, kept only if the monitor loss improves
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
- **Selection:** by training residual only. The reference is read once, after selection, by evaluation code.
- **Retention weight (B2, B3):** chosen from 0.1, 1, 10 by training residual only, in the pilot.
- **Run ID:** `bvp-{arm}-s{seed}-{confighash8}`.
- **Run folder:** `runs/<run_id>/` with:
  - `config.json`
  - `history.csv`: every loss term per step
  - `metrics.json`
  - `pred_eval.npz`: outputs and derivatives on the evaluation grid
  - the final checkpoint
  - `meta.json`: git hash, versions, hardware, threads, wall time
- **Threads:** `torch.set_num_threads(1)`. Up to 3 runs at once per laptop, if the pilot shows they do not slow each other much.
- **Summaries:** `results/summary.csv` is rebuilt by one script from the run folders. Figures read only `summary.csv` and `.npz` files.
- **Failures:** failed or diverged runs stay in every table and get one line in `results/failures.md`.

## 4. Metrics

Computed against the reference on its evaluation grid ([SPEC_reference_solutions.md](../reference-solver/SPEC_reference_solutions.md) section 6):

- **Field errors:** relative L2 and max absolute error for f, f1, f2, f3, f4, g, g1, g2, theta, th1, th2, phi, ph1, ph2, chi, ch1, ch2.
- **Wall values:** f2(0), g1(0), th1(0), ph1(0), ch1(0) against the reference, absolute and relative. These are plain numerical checks, not physical outputs.
- **Equation residuals:** mean square of R14 to R18, each separately, on the evaluation grid.
- **Boundary errors:** B1 to B12, each separately.
- **Retention (B2, B3):** error on each window after every stage.
- **Cost:** wall time, Adam steps, L-BFGS calls, parameter count, threads.

## 5. Gates

| Gate | Pass rule |
|---|---|
| G1 code | Derivative tests up to order 4, residuals against SymPy, R(H) - R(0) = M*H, B4 stable form, NumPy against PyTorch evaluators all pass |
| G3 reference | SPEC V3 to V8 pass, and L is chosen |
| G4 retention | Error per window after each stage reported for B2 and B3. No pass bar is set for the BVP |
| G5 no leakage | Review confirms training and selection code never reads the reference |
| G6 claims | "Arm X beats arm Y" only if it holds in all 5 paired seeds; otherwise "no consistent effect" |

## 6. Figures and tables

| ID | Content |
|---|---|
| TB1 | Reference study: wall values against tolerance and L (from gate G3) |
| FB1 | Five profiles from B0 and the best retention arm, against the reference |
| FB2 | Pointwise error against eta on a log scale, with a near-wall inset |
| FB3 | Final residual per equation and error per boundary condition, per arm |
| FB4 | Training curves of each loss term, seed 0, per arm |
| FB5 | Error per window after each stage, B2 and B3 |
| TB2 | Wall values, all arms against reference, median and range over seeds |
| FX1 (BVP part) | Error against wall time: PINN arms and solve_bvp |
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
| R0 to R2 | Environment, first-order system, boundary function, manufactured-solution check | 14 Dec 2026 |
| R3 to R5 | Locked-case solve, convergence and branch study (gate G3), reference export | 31 Dec 2026 |
| P1 to P3 | PINN building blocks (gate G1), B0 pilot, B1 to B3 pilots, configs frozen | 14 Jan 2027 |
| P4 | Campaign: 4 arms x 5 seeds | 1 Feb 2027 |
| P5 | Evaluation, figures, answers to section 7 | 10 Feb 2027 |

Review rule: nobody merges their own code. The other member reviews each phase before it counts as done.

## 9. Risks

| Risk | Response |
|---|---|
| solve_bvp does not converge at the locked case | Continuation in (kappa, b, Da) inside the reference only, more guesses, longer L. If it still fails, report to the supervisor; no value is changed |
| 4th-order autograd is slow | Fewer points in pilots only, never in final runs |
| PINN stalls near the wall | Declared fixes in order: loss scales, wall clustering, longer budget. Then freeze and report |
| B1 interface terms make training unstable | Report it. B1 is a comparison arm, not the main claim |
| A result goes against expectations | Report it. The section 7 questions are answered either way |
