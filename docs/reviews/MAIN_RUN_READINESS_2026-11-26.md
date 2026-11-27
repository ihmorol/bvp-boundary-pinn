# Main run readiness: code review, fixes and the Colab run

Date: 2026-11-26. Scope: the whole repository on branch `impl/campaign-readiness` (524c2ae plus the uncommitted work of the same day), read as a fresh start: nothing counted as right because it ran before. Result: the code solves the fixed problem correctly; nine problems were fixed (section 4); four decisions are open for the user (section 5). The main run itself has not been started.

## 1. What each part does

```text
configs/physical-case-pending.json  22 locked values (SHA-256 pinned)       --> config.load_case (read-only)
configs/reference.json              reference settings (L = 20, tolerances) --> reference/, pinn/runs.validate
configs/pinn.json                   PINN settings, acceptance rule, campaign --> pinn/runs.resolve

reference/equations.py   Eqs. 14 to 18 (residual_terms) and B1 to B12 (boundary_terms), one copy, used by all
      |                                         |
reference/run.py  solve_bvp: branch scan,       pinn/physics.py  the same functions on tensors
  tolerance and L ladders, RK shooting (V10),   pinn/constraints.py  all 12 conditions built into the output
  core-then-chi split (V8), slope check (V12)   pinn/jets.py  derivatives to order 4 in one pass
  --> reference/locked-branchA-L20-tol1e-08/    pinn/model.py, start.py, points.py, losses.py
                                                pinn/train.py  Adam, then L-BFGS (log scale), then LM (balanced scale)
engine/bvp_engine.cpp  second transcription,    pinn/lm.py, bfgs.py  refine optimizers
  box scheme; Newton-step error estimate  -->   pinn/selfcheck.py, estimate.py  acceptance without the reference
                                                pinn/run.py, campaign.py  one run; all arms and seeds, resumable
                                                      |
                     evaluation/evaluate.py  metrics against reference A, summary.csv, failures.md, settings.json
                     evaluation/classical.py solve_bvp timing;  evaluation/plots.py  FB1 to FB5, FX1, TB2, TB3, TX1

tools/main_run.py           the whole main run, step by step, resumable (new)
tools/colab_main_run.ipynb  the Colab entry point that installs and calls it (new)
```

The training package `bvp_pinn.pinn` never reads reference data (gate G5, tested). Only `bvp_pinn.evaluation` compares with the reference, after training and acceptance.

## 2. The equations and conditions as the code writes them

State order everywhere: z = [f, f1, f2, f3, g, g1, theta, th1, phi, ph1, chi, ch1], highest derivatives H = [f4, g2, th2, ph2, ch2]. With q = eta + kappa, h = eta + b, c = theta_w - 1, A = 1 + c theta (`reference/equations.py`, `residual_terms`):

```text
R14 = (1+Delta)(f4 + 2f3/q - f2/q^2 + f1/q^3) + kappa/q (f f3 - f1 f2) + kappa/q^2 (f f2 - f1^2) - kappa/q^3 f f1
      - 0.5 delta* ((eta f3 + 3 f2) + (eta/2 f2 + f1)/q) - (f2 + f1/q)/Da - Delta (g2 + g1/q)
      - 2 beta_m/h^4 ((1/q - 4/h) theta + th1)
R15 = (1+Delta/2)(g2 + g1/q) - Delta B (f2 + f1/q + 2g) - kappa/q (f1 g - f g1) - delta*/2 (eta g1 + 3g)
R16 = (th2 + th1/q)/Pr + Nr/Pr (A^3 th1/q + 3c A^2 th1^2 + A^3 th2) + kappa/q f th1 + Nt/Pr th1^2
      + Nb/Pr th1 ph1 + 2 beta_m lambda_m (theta - epsilon)/(Pr h^3) (kappa f/q (1 - 2/h^2) - 2 f1/h) - delta* eta/2 th1
R17 = (ph2 + ph1/q)/(Le Pr) + Nt/(Le Pr Nb)(th2 + th1/q) + kappa/q f ph1 - delta* eta/2 ph1
R18 = (ch2 + ch1/q)/Lb + kappa/q f phi + Pe/Pr (chi ph2 + ch1 ph1) - delta* eta/2 ph1

B1 = f1(0) - 1 - N1 (f2(0) - f1(0)/kappa)          B7  = f1(L)
B2 = f(0) - fw/(Pr Le) ph1(0)                      B8  = f2(L)
B3 = g(0) + ell f2(0)                              B9  = g(L)
B4 = -th1(0) - Bi (1 - theta(0)) - Rd Qw           B10 = th1(L) + m theta(L)/q_L            (asymptotic form of theta -> 0)
B5 = ph1(0) + Nt/Nb th1(0)                         B11 = ph1(L) + (mp phi(L) - Nt/Nb m theta(L))/q_L
B6 = chi(0) - 1                                    B12 = chi(L)
Qw = (1 - theta(0))(theta_w + Aw)(theta_w^2 + Aw^2), Aw = 1 + c theta(0)  (equal to Rd/c (theta_w^4 - Aw^4), tested)
m = Pr kappa f(L)/(1 + Nr A(L)^3),  mp = Le Pr kappa f(L),  q_L = L + kappa,  L = 20
```

**Checked against the source today.** Eqs. 14 to 19 were rendered from `sources/project-draft/project3Nov_2026.docx` (images 44 to 49, LibreOffice, 200 dpi) and compared term by term with the code above: every sign, constant, bracket and variable matches, including R18's printed kappa/q f phi and -delta* eta/2 phi' terms and B4's radiation bracket. The C++ engine has its own transcription; `test_engine_solves_the_same_problem` checks it against a fresh solve.

**Boundary conditions: hard or soft.** The decision recorded in [BOUNDARY_CONDITIONS_2026-11-26.md](BOUNDARY_CONDITIONS_2026-11-26.md) is: B4 is nonlinear as printed, B10 and B11 are nonlinear in the asymptotic form applied at L, the rest are linear, and all 12 stay exact (hard). The code matches it: `configs/pinn.json` has `bc_mode = exact`, so `SOFT_BCS["exact"]` is empty, the loss holds only the five equations, and the constraint layer meets all 12 conditions to about 1e-15 for any weights (`test_built_in_conditions_hold_for_random_weights`). No change was made to how the conditions are imposed.

## 3. Settings of the main run

| Part | Setting | Value | Source |
|---|---|---|---|
| Problem | Eqs. 14 to 18, B1 to B12, 22 values | fixed | supervisor, locked 2026-11-16 |
| Domain | L (far end) | 20, asymptotic far conditions for theta and phi | configs/reference.json, user decision 2026-11-26 |
| Reference | solver, tolerance | solve_bvp, tol and bc_tol 1e-8 (ladder 1e-5 to 1e-8), up to 100 000 nodes | configs/reference.json |
| Reference | branch scan | 48 smooth guesses at L = 10 (rates 0.5 to 4, theta(0) 0.1 to 1.2) | configs/reference.json |
| Reference | checks | tolerance (V4), L ladder 10 to 160 (V5), core then chi (V8), RK shooting 40 segments rtol 1e-11 (V10), slope check (V12) | reference/run.py |
| Reference | evaluation grid | 4 001 wall-clustered points, first step 5e-4 | configs/reference.json |
| PINN | network | tanh, 4 hidden layers of 40, float64; 5 205 weights (B1: 4 networks, 20 820) | configs/pinn.json |
| PINN | input and start | log input map (scale 0.1); smooth start profile (rate 1, theta(0) 0.3) under a zero last layer | configs/pinn.json |
| PINN | points | 2 000 training (stretch 5; 618, 485, 421, 476 per window), 1 000 monitor, fixed per seed | configs/pinn.json |
| PINN | conditions | all 12 exact, correction-shape decay 0.5 | configs/pinn.json |
| PINN | Adam | 20 000 steps in total, lr 1e-3 to 1e-4 (exponential), residual scale (eta + b)^order | configs/pinn.json |
| PINN | loss weights | gradient-norm rule, alpha 0.9, every 1 000 steps, cap 1 000 (one network in exact mode: one term, weight 1) | configs/pinn.json |
| PINN | refine | L-BFGS 5 000 iterations (history 50) on the log scale, then 260 Levenberg-Marquardt steps on the balanced scale; each kept only if the monitor loss falls | configs/pinn.json |
| PINN | arms | B0 one network; B1 window networks [0, 0.5, 2, 6, 20]; B2 staged with physics replay; B3 staged with self replay; replay weight 1 | configs/pinn.json |
| PINN | seeds, threads | 0 to 4 for every arm (paired); 1 CPU thread per run | configs/pinn.json |
| Acceptance | rule, before the reference is read | finished; m > 0 and mp > 0; tail share at most 0.05; every boundary error at most 1e-8; error-estimate score at most 1e-3 | configs/pinn.json |
| Evaluation | measures | per field relative L2 (trapezoid weights) and max error; wall values; residual mean squares; boundary errors; per-window errors after each stage; after-Adam accuracy; cost | evaluation/evaluate.py |

## 4. Problems found and fixed

Each fix keeps the method as it was. None touches the equations, the conditions, the values, the loss or the optimizers.

| # | Part | File | What was wrong | Why it matters | Change |
|---|---|---|---|---|---|
| 1 | Campaign | `pinn/campaign.py`, `pinn/estimate.py` | A run whose session stopped after training but before the acceptance rule kept meta.json without selection.json. The campaign skipped it as finished, so it was never judged | On Colab this happens on any disconnect in that minute; the run would show "acceptance not decided" in every table | The campaign now judges such runs on its next start (never retrains them). Test added |
| 2 | Campaign, evaluation | `pinn/runs.py`, `pinn/estimate.py`, `evaluation/evaluate.py` | meta.json, selection.json, error_estimate.json and metrics.json were written in place | An interrupted write leaves half a file that the resumed campaign trusts and evaluation cannot read | Written through a temporary file and a rename (`runs.write_json`) |
| 3 | Evaluation | `evaluation/evaluate.py` | Reference folders were hard-coded names, and evaluation did not check that the target is valid | A rebuilt reference whose branch A failed its checks, or a run at another L, would be compared without warning | Folders are named from configs/reference.json; branch A must sit at the reference L, pass G3 at fixed L and decay (m > 0), and each run's L must match. Tests added |
| 4 | Evaluation | `evaluation/evaluate.py`, `classical.py`, `plots.py` | "Largest error" used Python's max(), which depends on order when a value is NaN; the median run was picked by sorting values that may be NaN | A diverged run could show a finite largest wall error, and a failed run could be drawn as the "median seed" | NaN-aware maximum (`evaluate.worst`); failed runs sort last |
| 5 | Evaluation | `evaluation/plots.py` | FX1 labelled PINN times "one thread"; TX1 had no device | On a GPU the label is false, and GPU against one-thread CPU times would be compared silently | FX1 names the device (and GPU model) of the runs; TX1 has a Device column; run meta records the GPU model |
| 6 | Evaluation, output | `evaluation/evaluate.py` | The results folder kept no record of the settings and versions that made it | Results must be traceable to their settings | `results/<campaign>/settings.json`: PINN config, reference settings and hashes, case hash, git commit, versions, run list |
| 7 | Evaluation, output | `evaluation/evaluate.py`, `plots.py` | Arms were reported only after the shared refine chain, and no table applied rule G6 (claims only if all paired seeds agree) | The shared finisher may hide arm differences (TRAINING_WALKTHROUGH section 7); without a paired count, G6 has to be done by hand | After-Adam accuracy per run (from the last `stage<k>.pt`), and table TB3 (`accuracy.md`): per arm after Adam and final, plus paired-seed win counts |
| 8 | Reproducibility | `pinn/runs.py`, `evaluation/evaluate.py` | Run folders and results could only live inside the clone | On Colab the clone is on the machine's disk and is lost on disconnect | `BVP_RUNS_DIR` and `BVP_RESULTS_DIR` move them (to Drive on Colab); unset, nothing changes |
| 9 | Records | `configs/reference.json` | `L_reference_note` said every f, g, theta, phi wall value agrees with the next L to 1e-4 | False for phi(0): 3.9e-4 from L = 20 to 40 (V5 fails its bar); the paper must not repeat it | Note rewritten with the measured numbers. L itself is unchanged |

Also added, for the run itself: progress lines during training (every 2 000 Adam steps, 1 000 L-BFGS evaluations, 20 LM steps), a `--device`, `--lm` and `--json` option in `tools/time_pinn_step.py`, `tools/main_run.py` and `tools/colab_main_run.ipynb` (section 7). Tests: 64 pass (5 new).

Checked and found right (no change): equations and conditions against the source (section 2); one shared transcription for reference, PINN and evaluation; parameter values only from the pinned case file; the same L, state order and wall indices in all three parts; the exact boundary layer (all misses, targets and build order); Taylor-mode derivatives; the residual scales; the LM normal equations and gain ratio; the acceptance rule; seeds (NumPy points and torch weights from the run seed); same evaluation grid (4 001 points) and trapezoid-weighted relative L2; every device-dependent tensor in the PINN code follows the default device, so `--device cuda` works without code changes (read, not run: there is no GPU here; the smoke step checks it on Colab before training).

## 5. Open decisions for the user

1. **Training budgets and pilots.** `n_points` 2 000, `adam_steps` 20 000, `lbfgs_iters` 5 000, `lm_iters` 260, the window edges and replay weight 1 are still pilot values: P2b and P3 have not run. Only B0 has been trained with the whole method, and at smaller settings (1 000 points, 8 000 Adam steps); B1 to B3 have only run as dry runs. The notebook runs the committed values unless they are changed in its settings cell.
2. **phi(0) in the acceptance score** (open since 2026-11-24). Adding it would have rejected the seed 1 pilot.
3. **L = 20 and "the answer must not change with a longer L".** It does change: phi(0) by 3.9e-4 from L = 20 to 40, and the L = 20 values are 1.6e-4 to 3.3e-4 (phi(0) 1.8e-3) from the eta -> infinity limit; chi'(0) has no limit. The user kept L = 20 on 2026-11-26. PINN and reference solve the same L = 20 problem, so the comparison stays valid; the L error has to be reported beside it (`reference/l_choice.md`, rebuilt by the main run).
4. **The GitHub repository is public** (`gh repo view` on 2026-11-26), while README says it must stay private while sources/ holds the supervisor documents and papers.

## 6. Review passes

Each pass read the code as if someone else wrote it. Severity: high (wrong results), medium (wrong or lost record, or a run that stops), low (cosmetic or slow).

| Pass | What was read or run | Findings | Severity, confidence | Outcome |
|---|---|---|---|---|
| 1 | Every changed file and the new pipeline | "Largest error" with NaN depends on order (fix 4); median run picked with NaN (fix 4); an unused constant | Medium, sure; low, sure; low, sure | Fixed |
| 2 | The whole pipeline run here with tiny settings, then restarted with a lost verdict, a cut-off run and a deleted reference | The code-version check counted `engine/out/engine_convergence.csv`, which the reference step rewrites, as a code change, so every restart would have stopped; once the campaign step was marked done, seeds or arms added later would never run; interrupting the notebook cell would leave training running in the background | Medium, sure (seen in the test); medium, sure; medium, likely | Fixed: the check reads code paths only; the campaign step always restarts and skips finished runs; the cell stops the whole process group |
| 3 | The notebook cells in a simulated Colab (fake Drive, local git remote, `/content` in a scratch folder, no GPU), then the pipeline and notebook read again | The GPU check crashed on runtimes without `nvidia-smi`; after a driver fallback the install marker never matched, so every rerun reinstalled; a GPU that no PyTorch build can use would have trained on CPU without a clear message | Medium, sure (seen); low, sure; low, sure | Fixed and rerun: all six cells and all seven steps pass; the second run of the install cell skips |
| 4 | Final read of the full diff, the tests (64 pass) and the record check (passes) | Nothing that changes a result. Left as is: a run that crashes inside training (for example out of memory) is recorded as failed and never retried, by the project's rule that a failed run is a result; the GPU path is read, not run, because this machine has no GPU | n/a | Stop |

What the test runs showed: the reference rebuilt from scratch is bit-identical to the committed one (largest difference 0.0; the L error table matches to every printed digit), both branches are found again (scan and deflation), branch A passes G3 at fixed L with m = 0.2390, and V5 fails its 1e-4 bar at 3.9e-4 (phi(0)), as before. Install from the lock gives Python 3.12.13, numpy 2.5.5, scipy 1.18.1, matplotlib 3.11.2, sympy 1.14.0, pytest 9.2.0, torch 2.15.0+cpu; the CUDA build of torch 2.15.0 resolves against the same pins and only adds CUDA libraries (dry run).

## 7. The Colab run

`tools/colab_main_run.ipynb`, cells in order: settings (every choice in one place), GPU check and Drive, clone at a fixed commit, install, main run, first look at the results.

- **Packages.** A separate environment at `/content/venv` with Python 3.12.13 (fetched by uv 0.10.9) and every package at its `uv.lock` version (numpy 2.5.5, scipy 1.18.1, matplotlib 3.11.2, sympy 1.14.0, pytest 9.2.0, and their dependencies). torch 2.15.0: the PyPI build with CUDA 13.2 on a GPU, `2.15.0+cpu` otherwise. CUDA 13.2 needs a recent NVIDIA driver; if Colab's is older, the notebook falls back to the PyTorch build Colab ships and records it. The CPU path was installed and checked here; the GPU build was checked to resolve against the pins.
- **Steps** (`tools/main_run.py`): tests and the record check; the reference from scratch, the engine grid study, the deflation branch search and the L error bars, then a check that branch A is a usable target; a dry run of all four arms on the device with its evaluation; measured step times and a run-time estimate; the campaign; evaluation, solve_bvp timing and figures; packaging.
- **Restarts.** Each finished step leaves a marker in Drive; run folders are in Drive; a restart restores the reference from Drive and skips finished steps and runs. The first start pins the commit and the training settings; a later start with others stops with a message.
- **Outputs in Drive:** `runs/` (models, histories, verdicts, metrics), `results/<campaign>/` (summary.csv, failures.md, settings.json, TB2, TB3, TX1, figures), `reference/`, `engine_out/`, `logs/`, `configs/`, `environment-*.json`.

## 8. Run-time estimate

Measured on this laptop (4 cores, float64, one thread, 2 000 points): one Adam step 0.19 s for B0 and 0.20 s for B1 (B2's first stage 0.075 s); one Levenberg-Marquardt step 21 s for B0 (2026-11-26) and 32 s for B1 (2026-11-24). Whole runs so far: the seed 0 pilot at 1 000 points and 8 000 Adam steps took 71 minutes (2026-11-26). Per run, L-BFGS is counted at its limit of 1.25 loss evaluations per iteration, so the estimates are upper bounds there.

| Step | This laptop, one thread per run | Colab CPU runtime (2 cores, 2 runs at once) | Colab T4 GPU, one run at a time |
|---|---|---|---|
| Install (each new session) | n/a | 2 to 4 min | 3 to 6 min (the CUDA build is about 3 GB) |
| Tests, record check | 1 min | 1 to 2 min | 1 to 2 min |
| Reference, engine, branch search, L error bars | 6 min (measured) | 6 to 10 min | 6 to 10 min (CPU work) |
| Dry run and timing | 2 min | 2 to 3 min | 2 to 4 min |
| B0, one run | 3.0 h | 3.6 to 4.5 h | 0.45 to 1.2 h |
| B1, one run | 3.8 h | 4.6 to 5.7 h | 1.35 to 2.9 h |
| B2 or B3, one run | 2.6 h | 3.1 to 3.9 h | 0.4 to 1.0 h |
| Campaign, 20 runs | 60 h of runs | 36 to 45 h | 13 to 30 h, most likely about 20 h |
| Evaluation, solve_bvp timing, figures | 5 min | 5 to 10 min | 5 to 10 min |

How the GPU column was worked out (estimate, not measured: there is no GPU here). Levenberg-Marquardt is dense float64 algebra, about 3e11 operations per step (the 10 000 by 5 205 normal matrix and its Cholesky factor), and the T4 does about 0.25e12 float64 operations per second at best, so a step should take 2 to 6 s instead of 21 s (B1: 3 to 8 s, four factors). Adam steps are thousands of very small GPU operations and are limited by launch overhead, not arithmetic: about 0.04 to 0.1 s per step for one network and 0.15 to 0.3 s for B1's four (B1 already takes 0.16 s per step here at tiny settings, so its overhead is large). The CPU column assumes a Colab core is 1.2 to 1.5 times slower than this laptop's.

The notebook's timing step measures both step times on the session's own device before the campaign and prints the estimate for that machine, so the first session replaces this table with real numbers. Free Colab sessions end after at most about 12 hours, so the campaign needs two or more sessions; each restart loses at most the run in progress.
