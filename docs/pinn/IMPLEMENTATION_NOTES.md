# PINN implementation notes

Date: 2026-11-21. Scope: the PINN stage (TODO phases P1 to P5) for Eqs. 14 to 19. This file says what the code does, where, and why: what we took from papers and their code, which known failures each part guards against, the decisions and who made them, and the trade-offs we measured.

Related: [PINN_PLAN.md](PINN_PLAN.md) (arms, gates, metrics), [BC_ENFORCEMENT_REVIEW.md](BC_ENFORCEMENT_REVIEW.md) (how the boundary conditions are imposed), [literature-synthesis.md](../literature/literature-synthesis.md) (the 25 papers), [CODING_RULES.md](../team/CODING_RULES.md).

## 1. Where things are

All PINN code is in `src/bvp_pinn/pinn/`, one file per job:

| File | Job | Lines |
|---|---|---:|
| `model.py` | The tanh network (eta in, five fields out), window networks for B1, derivatives by autograd | about 110 |
| `jets.py` | All derivatives up to order 4 in one forward pass (Taylor mode); same numbers as autograd | about 135 |
| `start.py` | Optional fixed start profile under the network (`start = smooth`), the analytic guess of the reference solvers | about 45 |
| `constraints.py` | Boundary conditions built into the output (soft, linear or exact mode) | about 160 |
| `physics.py` | Eqs. 14 to 18 and B1 to B12 on tensors, by calling `reference/equations.py` | about 30 |
| `points.py` | Training and monitor points, denser near the wall | about 35 |
| `losses.py` | Equation, boundary, interface and replay losses; fixed residual scale; gradient-norm weights with an optional cap | about 100 |
| `train.py` | The four arms: stages, Adam (best checkpoint, resampling), refine stage (L-BFGS or full BFGS) with rollback | about 200 |
| `bfgs.py` | Full BFGS with a dense inverse Hessian and a strong Wolfe line search | about 90 |
| `lm.py` | Levenberg-Marquardt for the exact-mode sum of squares, with a per-point Jacobian (vmap of jacrev) | about 120 |
| `selfcheck.py` | Checks without the reference: wall values, far decay powers m and mp, failure share | about 50 |
| `runs.py` | Configs, run IDs, run folders | about 65 |
| `run.py` | Command line: train one arm and seed | about 65 |
| `evaluate.py` | Compare runs with the reference; `results/summary.csv` | about 140 |
| `plots.py` | Figures FB1 to FB5 and table TB2 | about 170 |

Settings: [configs/pinn.json](../../configs/pinn.json) (shared settings, then one block per arm). Tests: `tests/test_pinn.py` (gate G1 and G5 checks).

How data flows:

```text
configs/pinn.json --> run.py --> train.py --> runs/<run_id>/   (never reads the reference)
                                                   |
reference/locked-branch*/ --> evaluate.py --> metrics.json, results/summary.csv --> plots.py
```

## 2. Commands

```bash
uv run pytest                                                    # all tests, including gate G1
uv run python -m bvp_pinn.pinn.run --arm B0 --seed 0 --dry-run   # whole pipeline in seconds, results meaningless
uv run python -m bvp_pinn.pinn.run --arm B2 --seed 3             # one real run
uv run python -m bvp_pinn.pinn.run --arm B0 --seed 0 --set bc_mode=soft adam_steps=3000
uv run python -m bvp_pinn.pinn.evaluate                          # after training: metrics and summary
uv run python -m bvp_pinn.pinn.plots                             # after evaluate: figures and TB2
```

Checks outside training (tools/):

```bash
uv run python tools/pinn_manufactured.py --set input_map=log start=smooth zero_last=1   # known-answer check of the training code
g++ -O2 -std=c++17 -o engine/bvp_engine engine/bvp_engine.cpp
uv run python tools/pinn_error_estimate.py runs/<run_id>                               # error estimate without the reference
uv run python tools/reference_error_bars.py                                            # grid and L error bars of the reference
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
| `refine` | lbfgs, bfgs, lm, or a chain such as `lbfgs:log+lm:balanced` | Refine stages after Adam, in order, each with an optional scale after the colon. `lbfgs_iters` for L-BFGS and BFGS, `lm_iters` for Levenberg-Marquardt | P1 |

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
| Our synthesis items 11 and 12 | Keep the Adam result if L-BFGS makes the monitor loss worse or gives NaN | `train.lbfgs_phase` |

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
| Soft conditions never met exactly (synthesis item 9) | TFC layer: up to all 12 conditions exact; the mode is chosen by a declared pilot | Pilot running |
| Equations of very different size (synthesis items 6, 7) | Each equation divided by the coefficient of its highest derivative, so all five are in highest-derivative units. A rescaling only; the solutions are unchanged | Done |
| float32 underflow in fourth-order autograd (synthesis item 13) | float64 everywhere | Done |
| Thin wall layer under-resolved (reference figures: f'''' = -1728 at the wall) | Points stretched toward the wall (`stretch`); optional log input map; B1 to B3 give the wall its own window | Settings fixed in pilots |
| Overfitting between collocation points (synthesis item 14) | Separate monitor points; L-BFGS kept only if the monitor loss improves | Done |
| L-BFGS NaN or blow-up (synthesis item 11) | Rollback to the Adam state | Done |
| Training reads the answer (gate G5) | Only `evaluate.py` and `plots.py` load the reference; a test fails otherwise | Done, tested |
| A PINN can land on the other solution branch | `evaluate.py` reports which reference branch each run is nearest to | Done |
| Truncated domain (Pantokratoras 2009) | Same L = 20 and same asymptotic far conditions as the reference | Done |

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
| Final L-BFGS in every arm runs on the full problem; B2 and B3 drop the replay term there | Us, so all arms end on the same objective | 2026-11-21 |
| Each Adam stage restarts the optimizer and its learning-rate decay | Us, as bc-PINN trains each segment afresh | 2026-11-21 |

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

## 6. Gate status

| Gate | Check | Status |
|---|---|---|
| G1 | Derivatives to order 4 on known functions; PyTorch equations equal NumPy; gradient-norm weights equal the paper's formula; replay and interface losses; TFC conditions hold for random weights (both modes, one network and window networks) | Pass (`tests/test_pinn.py`) |
| G5 | No training module loads the reference | Pass (test) |
| G4 | Per-window error after each stage, B2 and B3 | Computed in `evaluate.py`, figure FB5 |
| G6 | Claims only if all 5 paired seeds agree | Applied when writing results |

## 7. Pilots

| Pilot | Question | Status |
|---|---|---|
| P2a | Boundary mode (rule in 5.1) | Running |
| P2b | Network size, point count, stretch, input map, Adam and L-BFGS budgets for B0 | After P2a |
| P3 | Window edges, replay weight from 0.1, 1, 10 (B2, B3), B1 interface | After P2b |

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
- **Manufactured solution** (`tools/pinn_manufactured.py`, E3 settings, 4 000 Adam and 1 500 L-BFGS, 1 142 s): **failed** to recover the known answer (theta(0) 0.72 against 0.50; theta relative L2 1.28). Its forcing R(u*) is largest at the wall, where the dipole coefficient of Eq. 14 is about 250, and the forced R14 monitor MSE was still 1.5e4 when the budget ran out. So this check fails for the same reason as the real case (wall layer, budget). It does not yet separate a code fault from slow training; a longer run of this check is the next test.
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
| Relative L2, f / g / theta / phi / chi | 9.1e-4 / 6.2e-4 / 2.6e-4 / 1.5e-4 / 1.3e-4 | 5.1e-3 / 2.1e-3 / 3.0e-3 / 2.9e-3 / 1.5e-4 | |
| m (decay power), tail share, theta(L) | 0.2388, 0.0011, 0.0684 | 0.2374, 0.0043, 0.0687 | 0.2390, 0.0028, 0.068 |
| Error-estimate score (no reference) | 4.1e-5 | 3.2e-4 | |

Numbers in brackets are relative errors against reference A at L = 20. What it shows:

- **The pipeline reaches the reference from scratch, in both seeds.** Every wall value is within 4e-4 relative, and seed 0 within 6e-5. That is at or below the reference's own distance from the eta -> infinity values (about 2e-4, SPEC section 8.6). Both runs are on branch A with a decaying tail.
- **The reference-free score ranks the seeds correctly** (seed 0 better than seed 1) and is of the right size. So a run can be accepted or rejected without the reference.
- **The loss target is met.** The balanced equation loss is below 2e-6 in both runs, so 1e-3 to 1e-4 is passed by two to three orders of magnitude. The plain (unscaled) MSE is 8e-4 (seed 0) and 7e-3 (seed 1), almost all of it from R14 in the wall layer, where the terms are of size 1e3.
- **Limits.** Two seeds, not the five of the campaign rule (G6). 1 000 points, not the 2 000 of the default. One laptop, one thread per run. The LM loss was still falling at step 260, so more steps would likely help.

## 8. Colab

- Colab gives float64 on its CPUs and GPUs, but consumer GPUs run float64 far slower than float32. Measure one short run on each before the campaign. **[judgment]**
- `pyproject.toml` pins Python 3.12. If the Colab image moves to a newer Python, install with `pip install --no-deps -e .` and the listed packages, or relax the pin with a dated note.
- The campaign is 4 arms x 5 seeds. Each run writes its own folder, so runs can be split across sessions and merged by copying `runs/` back before `evaluate`.
