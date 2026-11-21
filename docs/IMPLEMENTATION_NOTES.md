# PINN implementation notes

Date: 2026-11-21. Scope: the PINN stage (TODO phases P1 to P5) for Eqs. 14 to 19. This file says what the code does, where, and why: what we took from papers and their code, which known failures each part guards against, the decisions and who made them, and the trade-offs we measured.

Related: [PINN_PLAN.md](PINN_PLAN.md) (arms, gates, metrics), [BC_ENFORCEMENT_REVIEW.md](BC_ENFORCEMENT_REVIEW.md) (how the boundary conditions are imposed), [literature-synthesis.md](literature-synthesis.md) (the 25 papers), [CODING_RULES.md](CODING_RULES.md).

## 1. Where things are

All PINN code is in `src/bvp_pinn/pinn/`, one file per job:

| File | Job | Lines |
|---|---|---:|
| `model.py` | The tanh network (eta in, five fields out), window networks for B1, derivatives by autograd | about 90 |
| `constraints.py` | Boundary conditions built into the output (soft, linear or exact mode) | about 170 |
| `physics.py` | Eqs. 14 to 18 and B1 to B12 on tensors, by calling `reference/equations.py` | about 30 |
| `points.py` | Training and monitor points, denser near the wall | about 35 |
| `losses.py` | Equation, boundary, interface and replay losses; gradient-norm weights | about 70 |
| `train.py` | The four arms: stages, Adam, L-BFGS with rollback | about 160 |
| `runs.py` | Configs, run IDs, run folders | about 65 |
| `run.py` | Command line: train one arm and seed | about 65 |
| `evaluate.py` | Compare runs with the reference; `results/summary.csv` | about 140 |
| `plots.py` | Figures FB1 to FB5 and table TB2 | about 170 |

Settings: [configs/pinn.json](../configs/pinn.json) (shared settings, then one block per arm). Tests: `tests/test_pinn.py` (gate G1 and G5 checks).

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

Full citations are in [RESEARCH_NOTES.md](RESEARCH_NOTES.md) and [BC_ENFORCEMENT_REVIEW.md](BC_ENFORCEMENT_REVIEW.md).

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

## 8. Colab

- Colab gives float64 on its CPUs and GPUs, but consumer GPUs run float64 far slower than float32. Measure one short run on each before the campaign. **[judgment]**
- `pyproject.toml` pins Python 3.12. If the Colab image moves to a newer Python, install with `pip install --no-deps -e .` and the listed packages, or relax the pin with a dated note.
- The campaign is 4 arms x 5 seeds. Each run writes its own folder, so runs can be split across sessions and merged by copying `runs/` back before `evaluate`.
