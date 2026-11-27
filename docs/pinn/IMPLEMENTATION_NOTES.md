# PINN implementation notes

Date: 2026-11-21, updated 2026-11-24. Scope: the PINN stage (TODO phases P1 to P5) for Eqs. 14 to 19. This file says what the code does, where, and why: what we took from papers and their code, which known failures each part guards against, the decisions and who made them, and the trade-offs we measured.

Related: [PINN_PLAN.md](PINN_PLAN.md) (arms, gates, metrics), [BC_ENFORCEMENT_REVIEW.md](BC_ENFORCEMENT_REVIEW.md) (how the boundary conditions are imposed), [literature-synthesis.md](../literature/literature-synthesis.md) (the 25 papers), [CODING_RULES.md](../team/CODING_RULES.md).

## 1. Where things are

Two packages, one file per job. `src/bvp_pinn/pinn/` trains and judges runs and never reads the reference. `src/bvp_pinn/evaluation/` compares finished runs with the reference.

| File | Job | Lines |
|---|---|---:|
| `pinn/model.py` | The tanh network (eta in, five fields out), window networks for B1, derivatives (jets, autograd as a fallback) | about 110 |
| `pinn/jets.py` | All derivatives up to order 4 in one forward pass (Taylor mode); same numbers as autograd | about 135 |
| `pinn/start.py` | Optional fixed start profile under the network (`start = smooth`), the analytic guess of the reference solvers | about 35 |
| `pinn/constraints.py` | Boundary conditions built into the output (soft, linear or exact mode) | about 160 |
| `pinn/physics.py` | Eqs. 14 to 18 and B1 to B12 on tensors, by calling `reference/equations.py`; nothing can add a term | about 30 |
| `pinn/points.py` | Training and monitor points, denser near the wall | about 35 |
| `pinn/losses.py` | Equation, boundary, interface and replay losses; fixed residual scale; gradient-norm weights with an optional cap | about 115 |
| `pinn/train.py` | The four arms: stages, Adam (best checkpoint, resampling), refine chain (L-BFGS, BFGS, Levenberg-Marquardt) with rollback | about 235 |
| `pinn/bfgs.py` | Full BFGS with a dense inverse Hessian and a strong Wolfe line search | about 85 |
| `pinn/lm.py` | Levenberg-Marquardt on the frozen-weight training objective, for every arm: one Jacobian block per network, interface and soft boundary rows through the Woodbury identity | about 225 |
| `pinn/selfcheck.py` | Checks without the reference: wall values, far decay powers m and mp, tail fit, failure share | about 65 |
| `pinn/estimate.py` | Newton-step error estimate with the C++ engine (built on demand) and the acceptance rule | about 120 |
| `pinn/runs.py` | Configs and their checks, run IDs, run folders, `load_run` | about 135 |
| `pinn/run.py` | Command line: train one arm and seed, then judge it | about 70 |
| `pinn/campaign.py` | Command line: every arm and seed of the campaign, tagged, resumable, several at once | about 85 |
| `evaluation/evaluate.py` | Metrics per run; `results/<campaign>/summary.csv` and `failures.md` | about 225 |
| `evaluation/classical.py` | solve_bvp timing at tolerances 1e-3 to 1e-8, for the time-to-accuracy comparison | about 85 |
| `evaluation/plots.py` | Figures FB1 to FB5 and FX1, tables TB2 and TX1 | about 230 |
| `figstyle.py` | One figure style for the reference and PINN figures | about 35 |

Settings: [configs/pinn.json](../../configs/pinn.json): shared settings, one block per arm, the acceptance rule (`selection`) and the campaign (`campaign`). Tests: `tests/test_pinn.py` (gates G1 and G5, the refine chain for every arm, the acceptance rule, an end-to-end run of each arm).

How data flows:

```text
configs/pinn.json --> pinn.campaign / pinn.run --> train.py --> runs/<run_id>/ --> estimate.py --> selection.json
                                   (bvp_pinn.pinn never reads the reference)            |
reference/locked-branch*/ --> evaluation.evaluate --> metrics.json, results/<campaign>/summary.csv, failures.md
                              evaluation.classical --> results/<campaign>/classical_timing.json
                              evaluation.plots     --> results/<campaign>/figures/, wall_values.md, cost.md
```

## 2. Commands

```bash
uv run pytest                                                    # all tests, including gates G1 and G5
uv run python -m bvp_pinn.pinn.run --arm B0 --seed 0 --dry-run   # whole pipeline of one arm in under a minute, results meaningless
uv run python -m bvp_pinn.pinn.run --arm B2 --seed 3             # one real run, then its acceptance verdict
uv run python -m bvp_pinn.pinn.run --arm B0 --seed 0 --set adam_steps=3000   # a pilot with one setting changed
uv run python -m bvp_pinn.pinn.campaign --list                   # campaign: what is done, what is left
uv run python -m bvp_pinn.pinn.campaign --jobs 3                 # campaign: run what is left, three at once
uv run python -m bvp_pinn.evaluation.evaluate                    # after the campaign: metrics, summary, failures
OMP_NUM_THREADS=1 uv run python -m bvp_pinn.evaluation.classical # solve_bvp timing, on the machine the runs used
uv run python -m bvp_pinn.evaluation.plots                       # figures FB1 to FB5, FX1; tables TB2, TX1
uv run python -m bvp_pinn.evaluation.evaluate --runs runs/<run_id>   # a pilot: metrics.json only
```

Checks outside training (tools/):

```bash
uv run python tools/pinn_error_estimate.py runs/<run_id>                       # estimate and verdict for an older run folder
uv run python tools/reference_error_bars.py                                    # grid, L and far-field error bars of the reference
```

### Training options added on 2026-11-23

Each option defaults to the old behaviour, so older run folders rebuild the same networks. None changes an equation, a boundary condition or a parameter value.

| Key | Values | What it does | Gap |
|---|---|---|---|
| `start` | none, smooth | Adds the analytic smooth guess (rate `start_rate`, wall temperature `start_theta0`) under the network | P5 |
| `zero_last` | false, true | Zero last layer, so training starts exactly on the start profile | P5 |
| `residual_scale` | none, log, balanced | log multiplies R_k by (eta + b)^order_k: the equations written in s = log(eta + b). balanced caps the far-field growth of the R14 and R15 factors at their own far balance (section 7.2). Reported residuals stay unscaled | P3 |
| `weight_cap` | null or a number | Upper bound on the gradient-norm weights (soft and linear modes) | P2 |
| `best_checkpoint` | false, true | Restores the Adam state with the lowest monitor loss (one-stage arms) | P9 |
| `resample_every` | 0 or steps | New stratified training points during Adam (B0 only); the refine stage uses the fixed points | P7 |
| `refine` | lbfgs, bfgs, lm, or a chain such as `lbfgs:log+lm:balanced` | Refine stages after Adam, in order, each with an optional scale after the colon. `lbfgs_iters` for L-BFGS and BFGS, `lm_iters` for Levenberg-Marquardt. Since 2026-11-24 the chain `lbfgs:log+lm:balanced` is shared by all arms | P1 |

Colab: see section 8.

## 3. What we took, from where

### 3.1 From papers

| Source | What we took | Where |
|---|---|---|
| Raissi, Perdikaris, Karniadakis 2019 (JCP 378) | Residual plus boundary loss on collocation points; Adam then L-BFGS | `train.py` |
| Wang, Sankaran, Wang, Perdikaris 2023, "An expert's guide to training PINNs", arXiv:2308.08468, Algorithm 1 steps (c) and (d), eqs. 2.12 to 2.15 | Gradient-norm global weights: lambda_hat_i = sum_j norm(grad L_j) / norm(grad L_i); lambda <- 0.9 lambda + 0.1 lambda_hat; update every 1000 steps; weights not trained | `losses.GradNormWeights` |
| Wang, Teng, Perdikaris 2021 (SIAM J. Sci. Comput. 43), Algorithm 1 | Tried first, as chosen. Replaced after it failed in our pilot (section 5.2) | not used |
| Mattey, Ghosh 2022 (CMAME 390), sections 3.1 and 4.3 | Self-replay: residual only on the new window, earlier windows held to the network's own stored outputs | B3 in `train.py`, `losses.self_replay_mse` |
| Mortari 2017 (Mathematics 5, 57); Johnston 2021 thesis (arXiv:2105.08034) | Theory of functional connections: output = network + fixed shapes times the miss of each linear condition, including derivative and coupled conditions | `constraints.py` |
| Sukumar, Srivastava 2022 (CMAME, doi 10.1016/j.cma.2021.114333) | Evidence that exact enforcement can beat untuned soft penalties in 1D Robin and fourth-order problems | reason for the pilot in section 4 |
| Kiyani et al. 2025 (CMAME 446) and synthesis items 12 and 13 | float64 throughout; L-BFGS with strong Wolfe line search | `run.py`, `train.py` |
| Overfitting study, sources/literature/overfitting-failure-2026 (synthesis item 14) | Monitor points never used for gradients, to catch overfitting between training points | `points.py`, `train.monitor_loss` |
| Our synthesis items 11 and 12 | Keep the earlier state if a refine stage makes the monitor loss worse or gives NaN | `train.refine_once` |
| Madsen, Nielsen and Tingleff 2004, section 3.2 | Levenberg-Marquardt damping update (Nielsen's rule) | `lm.py` |
| Woodbury identity (standard linear algebra) | Adds the few interface and soft-boundary rows to the per-network blocks of J'J without forming one large matrix | `lm.solve` inside `lm.minimize` |

Full citations are in [RESEARCH_NOTES.md](../literature/RESEARCH_NOTES.md) and [BC_ENFORCEMENT_REVIEW.md](BC_ENFORCEMENT_REVIEW.md).

### 3.2 From code repositories (read on 2026-11-21, not run)

| Repository | What we saw | What we did with it |
|---|---|---|
| a1k12/characterizing-pinns-failure-modes (PyTorch) | A plain `nn.Sequential` network class, a separate optimizer chooser, L-BFGS with `history_size=50`, residuals by chained `autograd.grad`, float32 | Same simple structure. We keep float64 (synthesis item 13) |
| TamaraGrossmann/FEM-vs-PINNs (JAX) | Times Adam, L-BFGS and evaluation separately and averages over repeats; reports relative L2 | We log Adam seconds, L-BFGS seconds and wall time separately in `meta.json` |
| EliKiani/Optimizing_the_Optimizer_PINNs | The self-scaled BFGS variants work only by overwriting SciPy's own `_optimize.py` | Not used: patching SciPy is fragile and hard to explain. Plain L-BFGS stays |
| pnnl/ET-PINN | A large general framework (geometry, datasets, 33 kB model file) | Not used; far more than one 1D problem needs |
| Mattey and Ghosh (bc-PINN) | No public repository found on GitHub on 2026-11-21 | B3 written from the paper |

Our own Lorenz code (`prior-work/lorenz-pinn`) set the house style: small modules, `nn.Sequential` with tanh, an Adam wrapper and an L-BFGS wrapper with strong Wolfe and `tolerance_change=1e-16` in float64.

## 4. Known PINN failures and what guards against each

| Failure (source) | Guard in our code | Status |
|---|---|---|
| Unbalanced loss terms, boundary gradients starved (Wang 2021; synthesis item 4) | Gradient-norm weights give every weighted term the same gradient norm | Done |
| The 2021 annealing weights explode once a term is nearly met | Replaced by the 2023 gradient-norm rule (section 5.2) | Done, measured |
| Soft conditions never met exactly (synthesis item 9) | TFC layer: all 12 conditions exact | Done: exact mode chosen by pilot P2a (2026-11-22) |
| Equations of very different size (synthesis items 6, 7) | Each equation divided by the coefficient of its highest derivative, so all five are in highest-derivative units. A rescaling only; the solutions are unchanged | Done |
| float32 underflow in fourth-order autograd (synthesis item 13) | float64 everywhere | Done |
| Thin wall layer under-resolved (reference figures: f'''' = -1728 at the wall) | Points stretched toward the wall (`stretch`); optional log input map; B1 to B3 give the wall its own window | Settings fixed in pilots |
| Overfitting between collocation points (synthesis item 14) | Separate monitor points; L-BFGS kept only if the monitor loss improves | Done |
| L-BFGS NaN or blow-up (synthesis item 11) | Each refine stage is rolled back on its own if its monitor loss rises or is not finite | Done |
| A small loss can hide a wrong solution (run E8, section 7.1) | Acceptance rule: decaying far field, no constant tail, Newton-step error estimate; never the loss alone, never the reference | Done, tested; on the 13 pilot runs it accepts exactly the three that match the reference (section 7.4) |
| Arms differ by optimizer, not by structure | One refine chain for every arm; B1 gets the block form of Levenberg-Marquardt | Done, tested (end-to-end test of every arm) |
| Training reads the answer (gate G5) | Only `bvp_pinn.evaluation` loads the reference; no file of `bvp_pinn.pinn` imports it or the evaluation package, and a test parses every import | Done, tested |
| Pilot runs mixed into campaign tables | Campaign runs carry a `campaign` tag; evaluation reads only tagged runs | Done |
| A PINN can land on the other solution branch | `evaluation/evaluate.py` reports which reference branch each run is nearest to | Done |
| Truncated domain (Pantokratoras 2009) | Same L = 20 and same asymptotic far conditions as the reference; the config check refuses any other L | Done, tested |
| The problem drifts: another parameter value, a changed case file, a changed equation (AGENTS.md section 3) | `config.load_case` pins the case file's SHA-256 and returns read-only values; `train.train` and the reference study call `config.check_locked`; `runs.validate` refuses settings that name a physical value or are not in `configs/pinn.json`; no code adds a term or source to the equations; the C++ engine's transcription is checked against solve_bvp | Done, tested (`tests/test_fixed_problem.py`, 2026-11-24) |

## 5. Decisions and trade-offs

### 5.1 Decisions

| Decision | By | Date |
|---|---|---|
| Arms B0 to B3 only; literature fixes are shared settings, not extra arms | User | 2026-11-21 |
| Adaptive loss weights (then the gradient-norm variant, after the pilot in 5.2) | User | 2026-11-21 |
| Boundary mode chosen by a B0 pilot of exact, linear and soft modes with the rule below | User, on the subagent review | 2026-11-21 |
| Pilots on this CPU, final runs on Google Colab | User | 2026-11-21 |
| Weighted terms: the summed equation loss is one term; each soft boundary condition is its own term; B1's interface is one term | Us (the paper weights one term per loss type) | 2026-11-21 |
| Replay weight of B2 and B3 is fixed, chosen from 0.1, 1, 10 by training residual in the pilot (PINN_PLAN section 3) | Plan | 2026-11-19 |
| The refine stages in every arm run on the full problem; B2 and B3 drop the replay term there | Us, so all arms end on the same objective | 2026-11-21 |
| Each Adam stage restarts the optimizer and its learning-rate decay | Us, as bc-PINN trains each segment afresh | 2026-11-21 |
| B0 refine chain `lbfgs:log+lm:balanced` | Us, from runs X3 and X4 (section 7.2) | 2026-11-23 |
| The same refine chain for all four arms; B1 gets a block Levenberg-Marquardt (one block per window network) instead of staying on L-BFGS | Us, so an arm difference is not an optimizer difference ([gap report](../reviews/IMPLEMENTATION_GAP_2026-11-24.md) C1, C2) | 2026-11-24 |
| Acceptance rule and one run per arm and seed (PINN_PLAN section 3; values in `configs/pinn.json`) | Us, declared before the campaign | 2026-11-24 |
| Evaluation in its own package; campaign runs tagged; results per campaign | Us (gate G5; gap report C5, C8) | 2026-11-24 |
| The problem is enforced in code: pinned case file, read-only values, `check_locked`, config refuses physical values and other L | User's instruction, built by us | 2026-11-24 |
| Remove every check that solves other parameter values or a changed equation: published-limit benchmarks (V11), manufactured solutions (V3, the PINN tool and its forcing hook), the SymPy test at altered values | User | 2026-11-24 |

Boundary-mode pilot rule, declared before running (from BC_ENFORCEMENT_REVIEW.md section 8): B0 with exact, linear and soft modes, seeds 0 to 2. Keep exact unless it fails to converge in any seed, gives NaN, or ends with a higher mean equation residual on the monitor points than linear in all 3 seeds. Soft runs count only if every boundary error ends below 1e-6. Only training-side numbers are used. Then the mode is frozen for all arms.

### 5.2 Trade-offs we measured on this laptop

| Question | Measured | Choice |
|---|---|---|
| Same equations code for NumPy and PyTorch, or a second transcription? | One function now serves both; the SymPy test already checks it independently | Reuse (`residual_terms`, `boundary_terms`). A second copy would add about 60 lines and a new place for typos |
| Forward-mode (`torch.func.jvp`) or reverse-mode derivatives? | 1.63 s against 0.36 s per step, 2000 points, 1 thread; same values to 3e-20 | Reverse mode |
| Wang 2021 annealing as chosen? | In a 300-step B0 run the B10 weight reached 2.8e14 and the total loss rose from 1.5e7 at step 50 to 2.75e7 at step 299 | Replaced by the 2023 gradient-norm rule, with the user's approval |
| Points: 2000 or 1000? | 0.42 s against 0.20 s per Adam step (1 thread); 2 threads help only at 2000 points | 1000 for pilots; the final count is set from the pilots |
| Cost of a full run | About 0.2 to 0.4 s per Adam step on 1 CPU thread, so 20 000 steps take 1 to 2 hours | Final runs on Colab; see section 8 |
| TFC layer cost | One extra network evaluation with two derivatives at each end, per forward pass | Small; measured in the pilot |
| Levenberg-Marquardt cost at 2000 points (2026-11-24, 1 thread, 2 steps each) | B0 (5 205 weights): 24 s per step, peak 2.1 GB. B1 (20 820 weights, 4 blocks): 32 s per step, peak 1.6 GB. At 1000 points B0 took about 12 s per step | 260 LM steps take about 1.7 h (B0) to 2.3 h (B1). One dense 20 820 x 20 820 matrix for B1 would need 3.5 GB; the block solve avoids it |

## 6. Gate status

| Gate | Check | Status |
|---|---|---|
| G1 | Derivatives to order 4 on known functions; PyTorch equations equal NumPy; gradient-norm weights equal the paper's formula; replay and interface losses; TFC conditions hold for random weights (both modes, one network and window networks); Levenberg-Marquardt starts from exactly the training loss and lowers it for B1 and soft modes; config checks; the acceptance rule; every arm end to end | Pass, 60 tests in the whole suite (`uv run pytest`, 2026-11-24) |
| G5 | No module of `bvp_pinn.pinn` imports the reference data or `bvp_pinn.evaluation` | Pass (test parses every import) |
| G4 | Per-window error after each stage, B2 and B3 | Computed in `evaluate.py`, figure FB5 |
| G6 | Claims only if all 5 paired seeds agree | Applied when writing results |

## 7. Pilots

| Pilot | Question | Status |
|---|---|---|
| P2a | Boundary mode (rule in 5.1) | Done 2026-11-22: exact, frozen (issue #3) |
| P2b | Network size, point count, stretch, input map, Adam, L-BFGS and LM budgets for B0 | Partly done: sections 7.1 to 7.3 set the method at 1000 points. The 2000-point count and the budgets are still pilot values; LM cost at 2000 points is measured (section 7.4) |
| P3 | Window edges, replay weight from 0.1, 1, 10 (B2, B3), B1 interface | Not started; the code runs for all arms (section 7.4) |

Results are added here with run IDs when each pilot ends.

### 7.1 B0 fix ladder, 2026-11-23 (seed 0, one seed only)

Setup: B0, exact mode (all 12 conditions built in), tanh 40 x 4, float64, log input map, 1 000 training and 500 monitor points, one thread per run, three runs in parallel on this laptop. Each change sits on top of the row it names. The reference was read only after each run ended (`evaluate.py`). Reference A at L = 20: theta(0) 0.2502, f''(0) -4.2033, theta'(0) -1.2594, phi'(0) 3.7783.

| Run | Change | Budget | theta(0) | f''(0) | theta'(0) | phi'(0) | m (decay power) | Nearest branch | Scaled monitor loss |
|---|---|---|---:|---:|---:|---:|---:|---|---:|
| Pilots of 2026-11-21 | soft and linear modes, linear map | 3 000 Adam, 500 L-BFGS | -0.031, 0.0004 | 0.074, 1.034 | -0.018, 0.058 | 0.055, -0.173 | | | |
| E1 `ff56fa23` | exact mode, log map | 8 000 Adam, 2 000 L-BFGS | 0.070 | -3.540 | -1.919 | 5.756 | -0.245 (no decay) | B | 0.100 (plain) |
| E2 `8f63690c` | E1 + smooth start profile, zero last layer | same | 0.072 | -3.538 | -1.911 | 5.734 | -0.274 (no decay) | B | 0.089 (plain) |
| E3 `3612dc7e` | E2 + residual scale (eta + b)^order | same | 0.175 | -3.365 | -1.507 | 4.520 | 0.265 | B by theta L2, but decaying | 0.163 |
| E4 `d6175769` | E3, then full BFGS | + 3 000 BFGS (1 152 s) | 0.189 | -3.531 | -1.456 | 4.369 | 0.294 | A | 0.030 |
| E5 `8a08d410` | E3, then L-BFGS (control for E4) | + 3 000 L-BFGS (783 s) | 0.206 | -3.717 | -1.400 | 4.200 | 0.280 | A | 0.033 |
| E7 `9509fb7b` | E5, then more L-BFGS, same scale | + 10 000 L-BFGS (1 649 s); stalled at 0.0106 | 0.197 | -3.652 | -1.430 | 4.291 | 0.270 | decaying | 0.0106 |
| E8 `ef1aa5d1` | E5, then L-BFGS on the plain (unscaled) residual | + 10 000 L-BFGS (1 580 s) | 0.091 | -2.621 | -1.829 | 5.487 | 0.122 | decaying, but far from A | plain 2.3e-4 |

What it shows (one seed, so a signal, not a claim under rule G6):

- **The cold wall is gone.** Every new run has f''(0) with the right sign and theta(0) well above 0. The exact mode and the log map did most of that (E1).
- **The residual scale decided the far field.** Without it (E1, E2) the network ends in a state whose f(L) gives m < 0: theta and phi cannot decay, so it is not the solution B10 and B11 ask for. With it (E3 onward) m > 0 and the state moves toward branch A.
- **Full BFGS was no better than L-BFGS** per iteration here (E4 against E5), and 1.5 times slower. L-BFGS stays the default.
- **No overfitting.** E5 training loss 0.0336, monitor loss 0.0333. More points would not help yet; the limit is the optimizer.
- **A small loss is not a solution (E8).** Continued on the plain residual, the equation MSE fell from 2.3e2 to 2.3e-4 (about 5e-5 per equation), yet theta(0) fell to 0.091 and f''(0) to -2.62. The plain loss gives the far field almost no weight, because its terms are tiny there, so the tail and through it the wall values drift. The reference-free checks caught it: the Newton-step estimate gave wall errors of order 1 (f''(0) 3.1), and the failure share at 1e-3 was 1.0. So no run is accepted on its loss alone.
- **The log scale has the opposite weakness (E7).** It stalled at 0.0106 with the wall residual R14 still large (plain MSE 63). The two weightings fail at opposite ends of the domain. The next step is a balance between them: bounded weights per region (gap P2, ARRB style), which targets exactly this wall-against-outer imbalance.
- **Final run of this session by the training-side rule** (lowest monitor loss of its own objective, chosen without the reference): E7. Against the reference it is close to E5: f''(0) 13 % off, theta(0) 21 % off. Its Newton-step estimate of the f''(0) error is 0.51, against a true 0.55.
- **Not converged.** All runs except E7 were still improving when the budget ran out; E7 stalled. f''(0) is still 12 % off and theta(0) 18 % off. The error is largest at the wall, where the scale (eta + b)^4 gives R14 a small weight.

Checks outside training:

- **Error estimate without the reference** (`tools/pinn_error_estimate.py`, one Newton step of the box scheme). For E3 it matched the true wall errors within 10 to 20 % (f''(0): estimate 0.92, true 0.84; theta'(0): -0.29 against -0.25). For E1 and E2, which are far from the solution, it overshot by 2 to 5 times, as a one-step linear estimate should. So it is reliable once a run is near the solution.
- **Manufactured solution** (tool removed on 2026-11-24, see section 7.4; E3 settings, 4 000 Adam and 1 500 L-BFGS, 1 142 s): **failed** to recover the known answer (theta(0) 0.72 against 0.50; theta relative L2 1.28). Its forcing R(u*) is largest at the wall, where the dipole coefficient of Eq. 14 is about 250, and the forced R14 monitor MSE was still 1.5e4 when the budget ran out. So this check fails for the same reason as the real case (wall layer, budget). It does not yet separate a code fault from slow training; a longer run of this check is the next test.
- **Levenberg-Marquardt** (`refine = lm`, `pinn/lm.py`): built and unit-tested, but one step on the full network did not finish in about 10 minutes on 4 threads here (the Jacobian through the jets is the cost). Not used for results on this laptop.

### 7.2 Balanced scale and Levenberg-Marquardt, 2026-11-23 (seed 0, one seed only)

The changes come from the two reviews: [REFERENCE_AUDIT_2026-11-23.md](../reviews/REFERENCE_AUDIT_2026-11-23.md) (the reference is right; the PINN gap is about 700 times the reference error) and [PINN_METHOD_SYNTHESIS_2026-11-23.md](../reviews/PINN_METHOD_SYNTHESIS_2026-11-23.md) (changes P-A to P-C). Built:

- `residual_scale = balanced` (P-A): R14 times q^4/(1 + c14 q^2), R15 times q^2/(1 + c15 q^2), R16 to R18 times q^2, with c14 = (1/Da)/(1 + Delta) and c15 = 2 Delta B/(1 + Delta/2). From eta and the locked coefficients only. The log scale had weighted the far-field f and g rows up to 1.7e5.
- Levenberg-Marquardt with a per-point Jacobian (P-B): `vmap` over points of a per-point `jacrev`. Equal to the full Jacobian to 1.4e-14. One step on the 40 x 4 network with 1 000 points now takes about 12 s on one thread. The old code took over 10 minutes, partly because it rebuilt the whole Jacobian once per parameter tensor.
- Selection by the Newton-step error estimate (P-C): `tools/pinn_error_estimate.py` writes a score, the largest estimated relative error of f''(0), theta(0), theta'(0), phi'(0). On E3 to E8 it ranks the runs exactly as the true error does. `selfcheck.py` also fits the far tail (theta = C1 + C2 q^-m) and reports the non-decaying part C1; on the reference its share is 0.003, on E8 1.8.
- `refine` can chain stages (`lbfgs+lm`), and `lm_iters` sets the LM steps.

Runs, each continued from the one before it (the reference read only after each run):

| Run | From | Change | Wall time | Balanced loss | theta(0) | f''(0) | theta'(0) | phi'(0) | Estimate score |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| E5 `8a08d410` | E3 | (section 7.1) | | 2.0e-2 | 0.2058 | -3.7174 | -1.4001 | 4.2002 | 0.19 |
| X1 `7d9f7cc1` | E5 | balanced scale, 5 000 L-BFGS | 785 s | 2.8e-4 | 0.2638 | -4.3966 | -1.2186 | 3.6558 | 0.063 |
| X2 `8ee900a5` | E5 | balanced scale, 60 LM steps | 866 s | 1.7e-4 | 0.2583 | -4.3281 | -1.2348 | 3.7044 | 0.036 |
| X3 `c0b70da5` | X2 | 200 more LM steps (3 threads) | 2 415 s | **2.3e-6** | **0.25022** | **-4.20445** | **-1.25932** | **3.77796** | 2e-4 |
| Reference A | | | | | 0.25019 | -4.20330 | -1.25944 | 3.77831 | |

X3 against the reference: wall values within 5e-5 to 2.7e-4 relative (theta(0) 1.2e-4, f''(0) 2.7e-4, theta'(0) 1.0e-4, phi'(0) 9e-5, g'(0) 1.8e-4, chi'(0) 5e-5). Relative L2 over [0, 20]: f 3.8e-3, g 1.4e-3, theta 2.0e-3, phi 2.4e-3, chi 1.1e-4. Far decay power m 0.238 (reference 0.239), tail share 0.0013, theta(L) 0.0685 (reference 0.068). The Newton-step estimate, made without the reference, gave the f''(0) error as -8.4e-4 (true -1.15e-3) and the theta(0) error as 2.2e-5 (true 3e-5). The loss was still falling when the 200 steps ended (3.2e-6 to 2.3e-6 over the last 25).

What it shows:

- **The scale was the blocker, then the optimizer.** From the same start, the balanced scale with L-BFGS got 70 times lower in loss than the log scale did with twice the iterations (X1 against E7). Levenberg-Marquardt then went 100 times lower again, where L-BFGS stalls.
- **Wall values at the level of the reference's own error.** The reference is within about 2e-4 relative of the eta -> infinity values (SPEC section 8.6), so X3 is as close to the true problem as the L = 20 reference is.
- **Honest limits.** One seed. The path E3, E5, X2, X3 was built step by step in one session, and the reference was read between steps, although each choice also had a reference-free reason (lower loss, lower estimate score). Total compute about 1.6 hours on this laptop.

From scratch with the balanced scale everywhere (X4 `2f46f204`: 8 000 Adam, 3 000 L-BFGS, 100 LM, all balanced, 3 379 s): **failed**. Balanced loss 2.0e-4, but theta(0) 0.295, f''(0) -4.80, m 0.013, tail share 0.61, theta(L) 0.29: the theta tail is almost flat, a state the balanced loss barely sees. The reference-free checks caught it before the reference was read (estimate score 0.43 against 2e-4 for X3). So the scale has to change between phases: the log scale first, which brings the tail into the decaying state (E3 onward), and the balanced scale only for the final Levenberg-Marquardt stage. `refine` now takes a scale per stage, and the B0 default is `lbfgs:log+lm:balanced` after Adam on the log scale. The from-scratch test of that pipeline, seeds 0 and 1, is in section 7.3.

### 7.3 The full pipeline from scratch, seeds 0 and 1, 2026-11-23

Settings (now the B0 default in configs/pinn.json, except the pilot point counts): tanh 40 x 4, float64, log input map, smooth start profile with a zero last layer, exact mode for all 12 conditions, 1 000 training and 500 monitor points. Adam 8 000 steps on the log scale, then `refine = lbfgs:log+lm:balanced`: 5 000 L-BFGS iterations on the log scale, then 260 Levenberg-Marquardt steps on the balanced scale. One thread per run, the two seeds in parallel on this laptop, about 93 minutes each (Adam 861 s, L-BFGS about 780 s, LM about 3 940 s). The reference was read only after both runs ended.

| | Seed 0 `bvp-B0-s0-3a276995` | Seed 1 `bvp-B0-s1-3a276995` | Reference A |
|---|---:|---:|---:|
| Balanced loss (monitor) | 6.8e-7 | 1.6e-6 | |
| theta(0) | 0.2501906 (2.1e-5) | 0.2502378 (2.1e-4) | 0.2501853 |
| f''(0) | -4.20353 (5.5e-5) | -4.20496 (3.9e-4) | -4.20330 |
| theta'(0) | -1.25942 (1.6e-5) | -1.25928 (1.3e-4) | -1.25944 |
| phi'(0) | 3.77826 (1.3e-5) | 3.77783 (1.3e-4) | 3.77831 |
| g'(0) | -2.87141 (3.8e-5) | -2.87213 (2.9e-4) | -2.87130 |
| chi'(0) | -2.00882 (4.5e-5) | -2.00858 (7.5e-5) | -2.00873 |
| phi(0), added 2026-11-24 | -0.1202070 (4.3e-4) | -0.1194119 (6.2e-3) | -0.1201550 |
| Relative L2, f / g / theta / phi / chi | 9.1e-4 / 6.2e-4 / 2.6e-4 / 1.5e-4 / 1.3e-4 | 5.1e-3 / 2.1e-3 / 3.0e-3 / 2.9e-3 / 1.5e-4 | |
| m (decay power), tail share, theta(L) | 0.2388, 0.0011, 0.0684 | 0.2374, 0.0043, 0.0687 | 0.2390, 0.0028, 0.068 |
| Error-estimate score (no reference) | 4.1e-5 | 3.2e-4 | |

Numbers in brackets are relative errors against reference A at L = 20. What it shows:

- **The pipeline reaches the reference from scratch, in both seeds.** Every wall value except phi(0) is within 4e-4 relative, and seed 0 within 6e-5. phi(0), added to the table on 2026-11-24, is 4.3e-4 off in seed 0 and 6.2e-3 in seed 1 (section 7.4). That is at or below the reference's own distance from the eta -> infinity values (about 2e-4, SPEC section 8.6). Both runs are on branch A with a decaying tail.
- **The reference-free score ranks the seeds correctly** (seed 0 better than seed 1) and is of the right size. So a run can be accepted or rejected without the reference.
- **The loss target is met.** The balanced equation loss is below 2e-6 in both runs, so 1e-3 to 1e-4 is passed by two to three orders of magnitude. The plain (unscaled) MSE is 8e-4 (seed 0) and 7e-3 (seed 1), almost all of it from R14 in the wall layer, where the terms are of size 1e3.
- **Limits.** Two seeds, not the five of the campaign rule (G6). 1 000 points, not the 2 000 of the default. One laptop, one thread per run. The LM loss was still falling at step 260, so more steps would likely help.

### 7.4 Campaign readiness, 2026-11-24

What changed in the code is listed in the [gap report](../reviews/IMPLEMENTATION_GAP_2026-11-24.md) and section 5.1. What was measured:

- **Dry campaign** (`python -m bvp_pinn.pinn.campaign --dry-run --jobs 4`): all 20 runs (4 arms x 5 seeds, tiny settings) trained, ran the full refine chain, and were judged, in 72 s on this laptop. The acceptance rule rejected all 20 with reasons, as it should for 40 Adam steps. `evaluate`, `classical` (1 repeat) and `plots` then built `summary.csv`, `failures.md`, TB2, TX1 and figures FB1 to FB5 and FX1 for that campaign (`results/P4-dry/`, not committed).
- **Acceptance rule on the 13 real pilot runs** (E1 to E8, X1 to X4, and the two runs of section 7.3; run before evaluation): accepted X3 `c0b70da5` (score 2e-4), seed 0 `bvp-B0-s0-3a276995` (4.1e-5) and seed 1 `bvp-B0-s1-3a276995` (3.2e-4); rejected the other ten (E1, E2 for no far decay; X1, X2, X4 for the tail; all for the score). These are the three runs that match the reference, so the code applies the rule as intended. The bars were set from these same runs, so this is a check of the code, not an independent test of the bars.
- **phi(0) is the largest wall error.** Evaluation of the two section 7.3 runs, with phi(0) now in the wall table: phi(0) is 4.3e-4 relative off in seed 0 and 6.2e-3 in seed 1; every other wall value is within 4e-4. The Newton-step estimate sees it: estimated phi(0) error -4.7e-5 against a true -5.2e-5 (seed 0), 7.52e-4 against 7.43e-4 (seed 1). The declared score does not include phi(0). Adding it would reject seed 1 and X3. This is a choice for the team before the campaign, recorded in the gap report.
- **LM cost at 2000 points** is in section 5.2.
- **The estimate's own floor.** On a fresh solve_bvp solution of the locked case, the engine's Newton step on the 3 201-node estimate mesh is 4e-6 to 7.5e-5 relative at the wall: the box scheme's own error there. So scores near 1e-4 or below, such as seed 0's 4.1e-5, are at the estimator's floor and do not rank runs further. The acceptance bar of 1e-3 is ten times above it.
- **Not run yet:** pilots P2b and P3, and the campaign.
- **Removed later the same day** (user's decision; AGENTS.md section 3): the manufactured-solution checks (`tools/pinn_manufactured.py`, the forcing hook in `pinn/physics.py`, reference check V3) and the published-limit benchmarks (V11). They solved a forced equation or other parameter values. Every check now uses the locked case only.

## 8. Colab

- **The main run has one entry point (2026-11-26):** [tools/colab_main_run.ipynb](../../tools/colab_main_run.ipynb). It mounts Drive, installs the exact versions of `uv.lock` in their own environment (Python 3.12.13; torch 2.15.0, the CUDA 13.2 build on a GPU), checks out the branch tip (the main run stops if its training code differs from the first session's, since 2026-11-28), and runs `tools/main_run.py`: tests, the reference from scratch, a dry run on the device, measured step times with a run-time estimate, the campaign, and the evaluation. Run folders and results go to Drive (`BVP_RUNS_DIR`, `BVP_RESULTS_DIR`), so a disconnect loses at most the run in progress. Details and the run-time estimate: [MAIN_RUN_READINESS_2026-11-26.md](../reviews/MAIN_RUN_READINESS_2026-11-26.md).
- Colab gives float64 on its CPUs and GPUs, but consumer GPUs run float64 far slower than float32. The notebook's timing step measures the Adam and Levenberg-Marquardt step on the session's device before the campaign starts. **[judgment]**
- `pyproject.toml` pins Python 3.12. The notebook installs Python 3.12.13 with uv, whatever Python the Colab image has.
- The campaign is 4 arms x 5 seeds: `python -m bvp_pinn.pinn.campaign`. It skips finished runs, so it can be split across sessions (`--arms`, `--seeds`) and merged by copying `runs/` back before `evaluate`. `--jobs N` runs N at once; budget about 2.1 GB per B0 run and 1.6 GB per B1 run at 2000 points. Since 2026-11-28 each run trains in its own process, `JOBS` auto is 2 on a GPU, a run out of GPU memory beside others trains again alone, and a session limit keeps runs from being cut off ([PARALLEL_RUNS_2026-11-28.md](../reviews/PARALLEL_RUNS_2026-11-28.md)).
- The acceptance rule needs `engine/bvp_engine`. It is built with g++ on first use (`bvp_pinn.pinn.estimate.ensure_engine`); Colab images include g++.
- Run `bvp_pinn.evaluation.classical` on the same machine type as the runs, so the times in FX1 and TX1 compare.
