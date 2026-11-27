# What runs when we train, and what decides the result

Date: 2026-11-26. Audience: the team. This is a code-level walk through one training run: what runs, in which order, in which file, and which parts decide whether the PINN reaches the reference. It does not explain every line. For the ideas behind each part, read [PINN_GUIDE.md](PINN_GUIDE.md); for every file, [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) section 1.

## 1. Short answer: why the dry runs are all rejected

The dry campaign (`results/P4-dry/`) is a plumbing test. It runs every line of the pipeline with a budget 100 to 1 000 times smaller than a real run, depending on the stage (table below; about 370 times less wall time), so that all 20 runs finish in about a minute. Its results are not meant to be good, and the acceptance rule rejecting all 20 is the rule working. The dry settings are in `runs.DRY_RUN` ([runs.py:28](../../src/bvp_pinn/pinn/runs.py)).

| | Dry run | Accepted pilot `bvp-B0-s0-3a276995` | Campaign default (configs/pinn.json) |
|---|---:|---:|---:|
| Network | 16 x 2, 389 weights | 40 x 4, 5 205 weights | 40 x 4, 5 205 weights |
| Training points | 120 | 1 000 | 2 000 |
| Adam steps | 40 | 8 000 | 20 000 |
| L-BFGS iterations | 5 | 5 000 | 5 000 |
| Levenberg-Marquardt steps | 3 | 260 | 260 |
| Wall time, one thread | 4 to 17 s | 5 570 s | not run yet |
| Error-estimate score | 0.28 to 1.07 | 4.1e-5 (accepted) | |

The score is the estimated relative error of the wall values f''(0), theta(0), theta'(0), phi'(0) (section 5). Near the solution the score is the size of the worst wall error (relative). Far from it, as in the dry runs and after Adam, it is only a size signal: the estimate overshot the true error 2 to 5 times for runs E1 and E2 (IMPLEMENTATION_NOTES 7.1). It is reliable once it is small, about 1e-2 or below.

**The stronger evidence: even the accepted run was far off after Adam.** Each B0 run saves its state after Adam (`stage0.pt`) as well as its final state (`model.pt`). Scored with the same reference-free estimate:

| Run | After Adam only | After L-BFGS and Levenberg-Marquardt |
|---|---|---|
| Seed 0 `bvp-B0-s0-3a276995` | score 4.2; theta(0) 0.047, f''(0) -0.79 | score 4.1e-5; theta(0) 0.25019, f''(0) -4.2035 |
| Seed 1 `bvp-B0-s1-3a276995` | score 5.7; theta(0) 0.038, f''(0) -1.02 | score 3.2e-4; theta(0) 0.25024, f''(0) -4.2050 |
| Reference A | | theta(0) 0.25019, f''(0) -4.2033 |

After 8 000 Adam steps, the run that was later accepted scored worse than every dry run. Almost all of the accuracy comes from the two refine stages at the end. A dry run gives them 5 and 3 iterations, so it cannot be accepted. **Nothing in the dry results says the method or the plan is wrong.** What they show is that the pipeline runs end to end, every run gets a verdict with reasons, and evaluation builds every table and figure.

Check of the current code: no full-budget run had used the code of PR #12, so the seed 0 pilot settings were rerun on it. The run is accepted and matches the accepted pilot to about six digits (section 8).

## 2. What runs, in order

Command: `uv run python -m bvp_pinn.pinn.run --arm B0 --seed 0`. Importance: **K** = decides the result, **M** = matters, **P** = plumbing and safety, needed but not tuned.

```text
run.main / run.run ............................................. run.py:26, :39
  1  resolve + validate the config ............................. runs.py:42, :52            P
  2  float64, 1 thread, seed; load the 22 locked values ........ run.py, config.load_case    P (float64: K)
  3  build the network ......................................... model.py:48                K
       MLP: log input map, 4 tanh layers of 40, 5 outputs ...... model.py:16
       + SmoothStart under it, last layer zero ................. start.py:18
       + Constrained: all 12 conditions built in ............... constraints.py:23
  4  training and monitor points, stretched to the wall ........ points.py:15               M
  5  train ..................................................... train.py:83
       check_locked(p): the values are the supervisor's ........ config.py                  P
       plan_stages: one stage (B0), four (B2, B3) .............. train.py:39                M (for B2, B3)
       adam_stage: the Adam loop ............................... train.py:117               M
       refine_phase: L-BFGS (log), then LM (balanced) .......... train.py:154, :185         K
       self_check: decay powers, tail, failure share ........... selfcheck.py:26            M
  6  save the run folder ....................................... runs.py:122                P
  7  assess_run: error estimate + acceptance rule .............. estimate.py:112            K (for the verdict)
```

## 3. Inside one training step (Adam)

What happens each Adam step, in `adam_stage` ([train.py:117](../../src/bvp_pinn/pinn/train.py)):

1. **Derivatives** ([jets.py:126](../../src/bvp_pinn/pinn/jets.py), `states`). One forward pass through the network carries every derivative order at once (Taylor mode). It returns the 12 states z (f to f''', and g, theta, phi, chi with their first derivatives) and the 5 highest derivatives H (f'''', g'', theta'', phi'', chi'') at every training point. Before jets, nested autograd for these derivatives was about 85 % of a step and four times slower ([RUNTIME_STRATEGY.md](RUNTIME_STRATEGY.md)).
2. **Boundary layer** (`constrained_jets`, [jets.py:100](../../src/bvp_pinn/pinn/jets.py), with `Constrained.misses`, [constraints.py:63](../../src/bvp_pinn/pinn/constraints.py)). It reads the raw network's values at eta = 0 and eta = 20, computes how far each of B1 to B12 is missed, and adds fixed correction shapes that cancel the miss. So every network the optimizer ever sees meets all 12 conditions exactly. **K**: with the conditions as loss terms instead, the network settled at theta(0) near 0 (pilot P2a).
3. **Residuals** ([physics.py](../../src/bvp_pinn/pinn/physics.py), `residuals`). The same functions as the classical solver (`reference/equations.py`) turn z and H into R14 to R18 at every point, each divided by the coefficient of its highest derivative.
4. **Residual scale** ([losses.py:13](../../src/bvp_pinn/pinn/losses.py), `residual_scale`). Each residual is multiplied by a fixed positive factor that depends only on eta and the locked values. **K**, see section 4. The zero set of the equations is unchanged.
5. **Loss** ([losses.py:41](../../src/bvp_pinn/pinn/losses.py), `equation_mse`; [train.py:57](../../src/bvp_pinn/pinn/train.py), `loss_parts`). The mean over points of the scaled squared residuals. For B0 in exact mode this is the only term, so the gradient-norm weight stays at 1 and has no effect.
6. **Backward and update.** One backward pass to the 5 205 weights, one Adam step, learning rate falling from 1e-3 to 1e-4.
7. **Every `log_every` steps:** the full-problem loss on the separate monitor points (never in a gradient) and one history row. For B2 and B3 this differs from the stage loss, which covers only the stage window plus replay.

The B1 to B3 arms change only which points and which networks are in step 5: B1 sums over four window networks and adds an interface term; B2 and B3 train window by window with a replay term ([train.py:39](../../src/bvp_pinn/pinn/train.py), `plan_stages`).

## 4. What decides the result, ranked

Each line is backed by a run. Run IDs and numbers are in IMPLEMENTATION_NOTES section 7.

| Rank | Part | Where | Evidence | Leave alone? |
|---|---|---|---|---|
| 1 | **Levenberg-Marquardt refine** on the balanced scale | `lm.py:83`, `train.py:185` | Seed 0: monitor loss 0.023 to 6.8e-7 in 260 steps; score 4.2 after Adam to 4.1e-5 at the end. Half the gain comes in the first 30 to 60 steps; the loss is still falling at step 260 | The step count is a P2b choice |
| 2 | **The order of the residual scales**: log scale for Adam and L-BFGS, balanced scale for LM | `losses.py:13`, `refine` in configs/pinn.json | Balanced from the start left theta's tail flat (X4, score 0.43). Without any scale the far field did not decay (E1, E2) | Yes |
| 3 | **Exact boundary layer** for all 12 conditions | `constraints.py` | Soft and linear modes ended at theta(0) near 0, B4 missed by about 2.3 (P2a) | Yes (frozen) |
| 4 | **Log input map** | `model.py:16` | Together with exact mode it removed the cold wall (E1, without the start profile). It gives the thin wall layer about a fifth to a quarter of the input range (eta below 0.2 to 0.3) | Yes |
| 5 | **L-BFGS** on the log scale | `train.py:185` | Monitor loss 1.7 to 0.084 (20 times). No run isolates its effect on the far tail; the decaying tail comes from the log scale (rank 2) | Yes |
| 6 | **float64** | `run.py` | Fourth-order derivatives need it: float32 underflow in fourth-order autograd is a known failure (literature synthesis item 13); not tested here in float32 | Yes |
| 7 | **Adam** | `train.py:117` | 9.1 to 1.7 over 8 000 steps (5 times). It gives L-BFGS a reasonable start, nothing more | The step count is a P2b choice |
| 8 | Points and wall stretch | `points.py` | Training loss 0.0336 against monitor 0.0333 (E5): no overfitting at 1 000 points | P2b checks 2 000 |

What does not matter for B0 at present: the smooth start profile (no measurable effect, E2 against E1), the gradient-norm weights (one term in exact mode, weight stays 1), full BFGS instead of L-BFGS (no better per iteration, 1.5 times slower, E4), `best_checkpoint` and `resample_every` (off, not compared yet).

## 5. After training: the verdict, without the reference

`assess_run` ([estimate.py:112](../../src/bvp_pinn/pinn/estimate.py)) runs right after the run folder is saved:

1. `estimate` ([estimate.py:75](../../src/bvp_pinn/pinn/estimate.py)) puts the PINN's 12 states on a 3 201-node mesh and calls `engine/bvp_engine --estimate`, which takes one Newton step of the classical box scheme from that state. Near the solution the step equals the PINN's error. The **score** is the largest estimated relative error of f''(0), theta(0), theta'(0) and phi'(0). Its floor on this mesh is about 1e-4 (the box scheme's own error), so scores below that do not rank runs further.
2. `self_check` ([selfcheck.py:26](../../src/bvp_pinn/pinn/selfcheck.py)) gives the far decay powers m and mp (they must be positive for theta and phi to reach zero) and the **tail share**, the constant part of theta on [10, 20] (it must be near 0).
3. `judge` ([estimate.py:87](../../src/bvp_pinn/pinn/estimate.py)) accepts a run only if it finished, m and mp are positive, the tail share is at most 0.05, every boundary error is at most 1e-8 and the score is at most 1e-3 (values in `configs/pinn.json`, `selection`).

Reading a dry-run verdict such as "tail share 0.449 > 0.05; estimate score 0.779 > 0.001": theta still has a large constant part on [10, 20], and the wall values are about 78 % off. Both are what a network looks like after 40 Adam steps. B1's dry scores spread less (0.66 to 0.78) than the other arms' (0.28 to 1.07); the cause was not tested.

## 6. Where the time goes

| Phase | Seed 0 pilot, 1 000 points | Per unit | At 2 000 points (campaign default) |
|---|---:|---|---|
| Adam, 8 000 steps | 861 s (15 %) | 0.11 s per step | about 0.2 s per step **[estimate]**; 20 000 steps about 70 minutes |
| L-BFGS, 5 000 iterations | 769 s (14 %) | about 0.12 s per evaluation | about twice as long **[estimate]** |
| Levenberg-Marquardt, 260 steps | 3 943 s (71 %) | 15 s per step | 24 s per step (measured 2026-11-24); 260 steps about 1.7 hours |
| Error estimate and verdict | a few seconds | | |

So a run with the campaign defaults takes about 3 to 3.5 hours on one thread **[estimate]**, and the 20 runs about 70 thread-hours. The phase that does most of the work, Levenberg-Marquardt, is also most of the cost.

## 7. What this means for the plan

- **No change of strategy is needed because of the dry run.** It tested the plumbing, and the plumbing works.
- **What is not known yet, and is the real risk:** only B0 has ever been trained at a full budget. B1, B2 and B3 have only run as dry runs. Whether they reach the reference is what pilot P3 answers. B1 runs Levenberg-Marquardt on four networks at once (32 s per step at 2 000 points).
- **The shared finisher may hide differences between the arms.** All four arms end with the same L-BFGS and Levenberg-Marquardt stages on the full problem, and those stages do almost all of the work in B0. So after refinement the arms may end in the same place, and the retention effect of B2 and B3 may show only before refinement. The run folders already keep the state after Adam (`stage<k>.pt`). The evaluation should report each arm both after Adam and after refinement, so PINN_PLAN question 3 can be answered either way. **[judgment]**
- **The budget split is a P2b question.** In B0, Adam gave a 5-fold gain in 8 000 steps; Levenberg-Marquardt gave 30 000-fold in 260 steps and was still improving. The campaign default of 20 000 Adam steps may spend an hour for little. P2b should compare fewer Adam steps against more LM steps, judged by training-side numbers and the reference-free score only.

## 8. Check of the current code

Run `bvp-B0-s0-d3abaccf`, 2026-11-26: the seed 0 pilot settings (1 000 training and 500 monitor points, 8 000 Adam steps, 5 000 L-BFGS, 260 LM) on commit 524c2ae, the head of PR #12. The working tree differed from that commit only in docs. It is compared here with the accepted pilot `bvp-B0-s0-3a276995`, which ran on a8cf833, before the PR.

**Result: the PR's code reproduces the accepted run.** The training side accepted it. After that, the evaluation against reference A gives the same numbers as the pilot to about six digits.

| | Pilot `3a276995` | Check `d3abaccf` |
|---|---:|---:|
| Verdict, training side | accepted | accepted |
| Error-estimate score | 4.108e-5 | 4.108e-5 |
| Tail share | 0.00113 | 0.00113 |
| Largest boundary error | 1.1e-16 | 1.1e-16 |
| LM loss after 260 steps (balanced scale) | 6.78798e-7 | 6.78798e-7 |
| theta(0), reference 0.2501853 | 0.2501906 | 0.2501906 |
| f''(0), reference -4.2032976 | -4.2035277 | -4.2035277 |
| Largest wall error, relative (phi(0)) | 4.32e-4 | 4.32e-4 |
| Relative L2, f, g, theta, phi, chi | 9.1e-4, 6.2e-4, 2.6e-4, 1.5e-4, 1.3e-4 | the same |
| Nearest branch | A | A |
| Wall time | 5 573 s | 4 245 s |

What the logs show:

- **Adam and L-BFGS are bit-identical.** Every history row the two runs share (73 rows; the check logged every 100 steps, the pilot every 200, and that is the only setting that differs) has the same loss to the last digit. The monitor loss enters LM at 0.025711942884581666 in both.
- **LM differs only at rounding level.** The losses differ by 4e-15 relative at step 1 and the gap grows to 7e-7 relative by step 260. Every field metric is within 2.1e-6 relative of the pilot's. The PR rewrote LM as a block solver (one Jacobian block per network), so B0 now goes through a different sequence of floating-point operations. Which operation causes the gap was not traced; at this size it changes no reported number.
- **The wall time is not a code speed-up.** L-BFGS ran the same arithmetic as in the pilot and still took 528 s instead of 769 s. So the shorter time is most likely machine load, and LM's 2 901 s against 3 943 s cannot be credited to the new LM. **[judgment]**
- **One meta field changed meaning.** The top-level `monitor_before_refine` is now measured on the final stage's scale (balanced): 0.181 here. The pilot recorded it on the log scale, 1.69. The per-stage values in `refine_stages` mean the same in both runs. Compare those when reading old and new run folders.

What this does not check: B1, B2 and B3 at a full budget (pilot P3), and the campaign default of 2 000 points.
