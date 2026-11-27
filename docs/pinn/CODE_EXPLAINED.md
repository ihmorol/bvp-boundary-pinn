# The hard parts of the code, the four arms and the experiments

Date: 2026-11-26. Line links point at the code of PR #12 (commit 524c2ae).

This is for a reader who knows what a PINN is and wants the parts of our code that are hard to follow. [PINN_GUIDE.md](PINN_GUIDE.md) covers the basics, and [TRAINING_WALKTHROUGH.md](TRAINING_WALKTHROUGH.md) covers the call order and what decides a run. This doc does not repeat them.

## 1. Three facts about the problem that explain most of the code

### 1.1 Six conditions at each end

The five equations have orders 4, 2, 2, 2 and 2. So the state has 12 entries, `z = [f, f1, f2, f3, g, g1, theta, th1, phi, ph1, chi, ch1]` ([model.py:62](../../src/bvp_pinn/pinn/model.py)), and the problem needs 12 conditions. Six sit at each end ([constraints.py:33](../../src/bvp_pinn/pinn/constraints.py)):

| End | Conditions | What they fix |
|---|---|---|
| Wall, eta = 0 | B1 to B6 | f(0) from phi'(0) (B2); a slip mix of f'(0) and f''(0) (B1); g(0) from f''(0) (B3); theta'(0) from theta(0) (B4); phi'(0) from theta'(0) (B5); chi(0) = 1 (B6) |
| Far, eta = L = 20 | B7 to B12 | f'(L) = 0, f''(L) = 0 (B7, B8); g(L) = 0 (B9); the asymptotic conditions for theta and phi (B10, B11); chi(L) = 0 (B12) |

So at the wall six starting values are not given: f''(0), f'''(0), g'(0), theta(0), phi(0) and chi'(0). They are part of the answer, and the far conditions decide them. The wall values we report (f''(0), theta(0), theta'(0), phi(0), phi'(0), g'(0), chi'(0)) all follow from these six.

### 1.2 Small errors grow in both directions

Measured on reference A with `uv run python tools/marching_growth.py` (2026-11-26). It linearizes the 12-state system along the reference and multiplies the step matrices across the mesh:

| March from the wall to | Largest growth of an error made at the wall |
|---:|---:|
| eta = 0.5 | 50 |
| eta = 2 | 540 |
| eta = 6 | 6.0e5 |
| eta = 10 | 7.9e8 |
| eta = 20 | 5.9e16 |

Four modes grow when marching outward (growth rate up to about 1.8 per unit eta, and 8.3 at the wall itself). Seven to eight grow when marching inward (rates down to -1.9, and -19 at the wall). Marching from eta = 2 back to the wall already multiplies an error by 4.4e3.

So any method that marches from one end and hands its state on multiplies every hand-off error by these factors. This is why the classical reference never marches: solve_bvp, the box scheme and multiple shooting all solve [0, L] at once. Plain single shooting is not used, because of the same growth ([SPEC](../reference-solver/SPEC_reference_solutions.md) section 3, `reference/shooting.py`). It is also why no PINN arm marches ([PINN_PLAN.md](PINN_PLAN.md) section 2).

### 1.3 The far tail is slow, and one mode is almost invisible to the loss

Far out, theta falls like C q^-m with q = eta + kappa and m about 0.239. At eta = 20 theta is still 0.068. A constant C1 added to theta there has almost no residual in Eq. 16. Only B10 holds it at zero ([selfcheck.py:11](../../src/bvp_pinn/pinn/selfcheck.py)). So a run can have a small loss and a wrong tail (run E8). This is why the acceptance rule checks the tail, and why the residual scale changes between phases (section 3.2).

## 2. Hard terms, and where they live

| Term | What it means here | Code |
|---|---|---|
| State z, top derivatives H | z holds the 12 values above. H = [f4, g2, th2, ph2, ch2] holds the highest derivative of each field | [model.py:62](../../src/bvp_pinn/pinn/model.py) |
| Residual R14 to R18 | Each equation's left side for the current network, divided by the coefficient of its highest derivative. Zero when exact | [physics.py:15](../../src/bvp_pinn/pinn/physics.py) |
| Residual scale | A fixed positive factor on each residual at each point. It changes the weighting, never the exact solution | [losses.py:13](../../src/bvp_pinn/pinn/losses.py) |
| Log scale | R_k times (eta + b)^order_k. It is each equation written in s = log(eta + b), where the thin wall layer and the slow tail both have size of order one | same |
| Balanced scale | Like the log scale near the wall, but the R14 and R15 factors stop growing where the far-field balance takes over, so far-out f and g rows do not dominate | same |
| Log input map | The network sees log(1 + eta/0.1), scaled to [-1, 1], not eta. The thin wall layer gets a larger part of the input range | [model.py:38](../../src/bvp_pinn/pinn/model.py) |
| Start profile, zero last layer | A fixed profile c + d exp(-eta) is added to the output, and the last layer starts at zero. So the untrained network is a warm, decaying guess, not noise | [start.py:18](../../src/bvp_pinn/pinn/start.py) |
| Exact mode, TFC | All 12 conditions are built into the output (theory of functional connections). They hold for any weights | [constraints.py:23](../../src/bvp_pinn/pinn/constraints.py) |
| Miss | Target minus what the raw network gives, for one condition | [constraints.py:63](../../src/bvp_pinn/pinn/constraints.py) |
| Correction shape | A fixed curve d^n exp(-d/0.5), d the distance to the end, that carries a miss into the output. `tfc_decay` = 0.5 sets its width | [constraints.py:121](../../src/bvp_pinn/pinn/constraints.py) |
| Far conditions | B10 and B11 at eta = 20 keep only the decaying solutions theta = C q^-m and phi = E q^-mp + P q^-m (Lentini and Keller 1980) | `reference/equations.py`, `far_conditions` |
| Taylor-mode jets | All derivatives up to order 4 in one forward pass, layer by layer by the chain rule, instead of 12 nested autograd calls | [jets.py:24](../../src/bvp_pinn/pinn/jets.py) |
| Training points | 2 000 eta values (pilots: 1 000), one random point per equal cell of s, then stretched toward the wall (`stretch` = 5) | [points.py:8](../../src/bvp_pinn/pinn/points.py) |
| Monitor points, monitor loss | A second, separate set of points. They never enter a gradient. The full-problem loss on them decides whether a refine stage is kept | [train.py:148](../../src/bvp_pinn/pinn/train.py) |
| Gradient-norm weights | Weights that give every loss term the same gradient size (Wang et al. 2023), updated every 1 000 Adam steps, capped at 1 000. In exact mode a one-network arm has one term, so its weight stays 1. Only B1 has a second term, the interface | [losses.py:87](../../src/bvp_pinn/pinn/losses.py) |
| Refine chain | The optimizers after Adam, in order: `lbfgs:log+lm:balanced` is L-BFGS on the log scale, then Levenberg-Marquardt on the balanced scale. Each stage is rolled back if its monitor loss gets worse | [train.py:154](../../src/bvp_pinn/pinn/train.py) |
| Levenberg-Marquardt (LM) | A Gauss-Newton method with a damping value mu. Section 3.3 | [lm.py:83](../../src/bvp_pinn/pinn/lm.py) |
| Newton-step error estimate | The C++ engine takes one Newton step of a classical solver from the PINN's curves. Near the solution that step is about the PINN's error | [estimate.py:75](../../src/bvp_pinn/pinn/estimate.py) |
| Score | The largest relative estimated error of f''(0), theta(0), theta'(0) and phi'(0). phi(0) is not in it | [estimate.py:42](../../src/bvp_pinn/pinn/estimate.py) |
| m, mp | Far decay powers of theta and phi, from f(L) and theta(L). Both must be positive, or theta and phi cannot decay | [selfcheck.py:26](../../src/bvp_pinn/pinn/selfcheck.py) |
| Tail share | theta on [10, 20] fitted to C1 + C2 q^-m; tail share = abs(C1) / max abs(theta) there. Reference A: 0.003 | [selfcheck.py:51](../../src/bvp_pinn/pinn/selfcheck.py) |
| Failure share | Share of points where the largest plain residual is above 1e-3. Logged, not used by the rule | [selfcheck.py:44](../../src/bvp_pinn/pinn/selfcheck.py) |
| Branch A, B | The two solutions the reference found at the locked values. A is the one we solve for; B fails m > 0 | `reference/` |
| Acceptance rule | Five checks made without the reference: finished, m > 0 and mp > 0, tail share at most 0.05, boundary error at most 1e-8, score at most 1e-3 | [estimate.py:87](../../src/bvp_pinn/pinn/estimate.py) |
| Arm, stage, window, replay | Arm: one training set-up (B0 to B3). Window: a piece of [0, 20]. Stage: one block of Adam steps on given points. Replay: a loss term that keeps earlier windows right while a new one trains | [train.py:39](../../src/bvp_pinn/pinn/train.py) |
| Run ID | `bvp-{arm}-s{seed}-{hash8}`; hash8 is the first 8 hex digits of the SHA-256 of the run's full config | [runs.py:99](../../src/bvp_pinn/pinn/runs.py) |
| Dry run | The whole pipeline on a tiny budget (`runs.DRY_RUN`), as a plumbing test. Its results are not meant to be good | [runs.py:28](../../src/bvp_pinn/pinn/runs.py) |
| Gate G5, G6 | G5: training code never reads the reference (tested). G6: "arm X beats arm Y" only if all 5 paired seeds agree | [PINN_PLAN.md](PINN_PLAN.md) section 5 |

## 3. Hard logic, piece by piece

### 3.1 How the output meets all 12 conditions for any weights

`Constrained.forward` ([constraints.py:55](../../src/bvp_pinn/pinn/constraints.py)) does four things:

1. Run the raw network N(eta), start profile included.
2. At each end this network owns, read N and its first two derivatives there (`end_values`).
3. For every condition, compute the miss: target minus what N gives.
4. Add, for each field, the sum of miss_j times shape_j(eta). Shape_j is built so that it meets its own condition with value 1 and every other condition of that field with 0 (`Field.inv`, the inverse of a small matrix of shape values at the ends). So the sum fixes every miss and disturbs no other condition.

Three parts are hard to see in the code:

- **The order matters.** f is built first. B3 needs the built f''(0) to set g(0), and B10 and B11 need the built f(L) to get m and mp ([constraints.py:87](../../src/bvp_pinn/pinn/constraints.py)).
- **The nonlinear conditions fix a slope from a value.** B4 has a radiation term in theta(0). The layer leaves theta(0) free (a "keep" row with zero miss) and solves B4 for theta'(0) ([constraints.py:72](../../src/bvp_pinn/pinn/constraints.py)). B10 and B11 work the same way at eta = 20: theta(L) and phi(L) stay free, and the slopes are set from them (line 94).
- **The far conditions act only near eta = 20.** The shapes are 0.5 wide, so a far miss is corrected inside about [19, 20]. In the accepted runs the largest boundary error is 1.1e-16.

### 3.2 Why the residual scale changes between phases

All three choices have the same exact solution. They differ in which points the optimizer listens to:

| Scale | What goes wrong alone | Run |
|---|---|---|
| None (plain) | Wall terms are of size 1e3 and far terms tiny, so the far field is ignored. The loss fell to 2.3e-4, but the tail drifted and theta(0) fell to 0.091 (reference 0.250) | E8 |
| Log | Brings the tail into the decaying state (m went from -0.27 to +0.27), but weights far-out f and g rows up to 1.7e5 and stalls with the wall residual large | E3, E7 |
| Balanced | From the start it left the theta tail almost flat (m 0.013, tail share 0.61) at a small loss | X4 |

So Adam and L-BFGS run on the log scale, which gets the tail right, and LM runs on the balanced scale, which then finishes near the wall (X3, and seeds 0 and 1). The scale of the last stage is the one a finished run is judged on (`final_scale`, [train.py:180](../../src/bvp_pinn/pinn/train.py)).

### 3.3 Levenberg-Marquardt, at code level

`lm.minimize` ([lm.py:83](../../src/bvp_pinn/pinn/lm.py)) writes the loss as a sum of squares of rows: five rows per point, each sqrt(w0/N) times a scaled residual, plus a few global rows (B1's 36 interface jumps). Then, per step:

1. Build the Jacobian J of every row against every weight, one point at a time (`vmap` of `jacrev`), in chunks of 100 points.
2. Solve (J'J + mu I) h = -J'r by a Cholesky factorization ([lm.py:142](../../src/bvp_pinn/pinn/lm.py)).
3. Try the step. If the loss falls, keep it and lower mu by Nielsen's rule, mu times max(1/3, 1 - (2 gain - 1)^3) ([lm.py:203](../../src/bvp_pinn/pinn/lm.py)). If not, raise mu (times 2, then 4, 8, ...) and solve again. The first mu is 1e-3 times the largest diagonal entry of J'J.

Why it works where L-BFGS stalls: J'J uses the curvature of every row at once, so each step is close to a Newton step on the residual. In the seed 0 pilot, L-BFGS took the log-scale monitor loss from 1.7 to 0.083 in 5 000 iterations. LM then took the balanced-scale monitor loss from 0.026 to 6.8e-7 in 260 steps, and it was still falling.

**The block form.** A point row depends only on the network that owns the point. So for B1, J'J is one block per window network plus a low-rank part C'C from the 36 interface rows. Each block is factored on its own, and C'C is added by the Woodbury identity ([lm.py:161](../../src/bvp_pinn/pinn/lm.py)). For one network there is one block and no C.

**Cost.** One block is 5 205 x 5 205 in float64, about 220 MB. One step takes about 15 s at 1 000 points and 24 s at 2 000 points for B0, and 32 s for B1, on one thread.

### 3.4 Judging a run without the reference

After training, `estimate.assess_run` ([estimate.py:112](../../src/bvp_pinn/pinn/estimate.py)) uses two things:

- **The self check**, saved in `meta.json` by `train`: m and mp from f(L) and theta(L), the tail fit on [10, 20], and the largest boundary error.
- **The Newton-step estimate**: the PINN's 12 curves go to `engine/bvp_engine --estimate` on 3 201 nodes. The engine takes one Newton step of its box scheme. Near the solution that step is about minus the PINN's error, so its size estimates the wall errors.

Two limits to keep in mind. Far from the solution the one-step estimate overshoots, by 2 to 5 times in E1 and E2. And the box scheme has its own error, 4e-6 to 7.5e-5 relative at the wall, so scores near 1e-4 or below do not rank runs further. The bar of 1e-3 is ten times above that floor.

### 3.5 How windows and stages are built

`plan_stages` ([train.py:39](../../src/bvp_pinn/pinn/train.py)) turns an arm into a list of stages. Each stage says which points go to which network:

- **single** (B0): one stage, all points to network 0.
- **windows** (B1): one stage. `in_range` sends each window's points to that window's network, and the interface term is on. Window 0 owns the wall end (B1 to B6 exact), window 3 owns the far end (B7 to B12 exact), and the two middle windows own no condition. At each of the 3 joints the 12 states must match: 36 jumps, as one soft term with a gradient-norm weight.
- **staged** (B2, B3): one stage per window, from the wall outward. The stage's points are the new window; `old` holds the points of the windows already done. B2 adds the equation loss on `old` (physics replay). B3 adds the distance from the network's own outputs on `old`, frozen at the start of the stage ([train.py:95](../../src/bvp_pinn/pinn/train.py)).

Adam gets `adam_steps // number of stages` steps per stage, so B2 and B3 get 4 x 5 000 at the default 20 000 ([train.py:91](../../src/bvp_pinn/pinn/train.py)). After Adam, the refine chain runs on `full_problem` ([train.py:50](../../src/bvp_pinn/pinn/train.py)): all points on one network for B0, B2 and B3, with no replay; all windows plus the interface for B1. So B2 and B3 end on exactly B0's objective.

## 4. The four arms

All four arms share every setting in `configs/pinn.json` `common`: tanh 40 x 4, float64, log input map, start profile, exact mode for all 12 conditions at every step, Adam on the log scale, then `lbfgs:log+lm:balanced`. They differ only in the structure:

| Arm | Networks | Adam stages | Adam loss | Refine | Question it answers ([PINN_PLAN](PINN_PLAN.md) section 7) |
|---|---|---|---|---|---|
| B0 | 1 on [0, 20] | 1 | Equation loss on all points | On all points | 1. Does a plain PINN reach the reference? |
| B1 | 4, one per window | 1 | Each window's equation loss on its own network, plus the weighted interface term | Block LM over the 4 networks, interface rows included | 2. Do separate window networks help near the wall? |
| B2 | 1 on [0, 20] | 4, wall outward | Equation loss on the new window, plus 1.0 times the equation loss on earlier windows | Same as B0 | 3. Does staged training with physics replay beat B0? |
| B3 | 1 on [0, 20] | 4, wall outward | Equation loss on the new window, plus 1.0 times the distance from the frozen outputs on earlier windows | Same as B0 | 3. Which form of retention does better? |

**The window edges are not random.** They are 0, 0.5, 2, 6 and 20, small near the wall where the fields change fast. With the stretched points they hold 31, 24, 21 and 24 % of the training points. They are also close to four equal windows in the network's own log input, whose edges would be 0, 0.28, 1.32, 5.24 and 20. Equal windows in eta, as the Lorenz work used in time (0, 5, 10, 15, 20), would put 73 % of the points and the whole wall layer in the first window. The edges are still pilot values until pilot P3 fixes them.

**Where each arm stands (2026-11-26).** B0 has three accepted full-budget runs (seeds 0 and 1, and the check of the PR code). B1, B2 and B3 have run only as dry runs. Whether they reach the reference is pilot P3's question.

**One caution for reading the results.** All arms end with the same L-BFGS and LM stages on the full problem, and in B0 those stages do almost all of the work. So arms may differ after Adam and end in the same place after refinement. The run folders keep the state after each Adam stage (`stage<k>.pt`), and FB5 reports the error per window after each stage ([TRAINING_WALKTHROUGH.md](TRAINING_WALKTHROUGH.md) section 7).

## 5. The experiments so far

Every run is B0 at seed 0 unless noted. Details and run IDs: [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) section 7.

| When | Runs | What changed | What it showed |
|---|---|---|---|
| 2026-11-21 | First pilots | Soft and linear boundary modes, plain input | Cold wall: theta(0) -0.031 and 0.0004 against 0.250 ([PINN_REVIEW_2026-11-21.md](../reviews/PINN_REVIEW_2026-11-21.md)) |
| 2026-11-22 | P2a | Boundary mode | Exact mode chosen and frozen ([BC_ENFORCEMENT_REVIEW.md](BC_ENFORCEMENT_REVIEW.md)) |
| 2026-11-23 | E1 | Exact mode, log input map | Cold wall gone, but m < 0 (no decay), nearest to branch B |
| | E2 | E1 plus start profile | No change from E1 |
| | E3 | E2 plus log residual scale | m > 0: the tail decays, heading to branch A |
| | E4, E5 | E3, then full BFGS or L-BFGS | BFGS no better than L-BFGS and 1.5 times slower |
| | E7 | E5, then 10 000 more L-BFGS | Stalled at 0.0106, wall residual still large |
| | E8 | E5, then L-BFGS on the plain residual | Small loss, wrong solution (theta(0) 0.091). The reason a loss alone never accepts a run |
| | X1 to X3 | From E5: balanced scale, then LM | X3 within 5e-5 to 2.7e-4 of reference A at the wall |
| | X4 | Balanced scale from scratch | Failed: flat tail at a small loss. Hence the scale per phase |
| | Seeds 0 and 1 | The full pipeline from scratch | Both reach reference A; every wall value but phi(0) within 4e-4 |
| 2026-11-24 | Rule check | Acceptance rule on the 13 pilot runs | Accepts exactly the 3 that match the reference |
| | P4-dry | 20 dry runs, all arms, 72 s | All rejected, as a plumbing test should be |
| 2026-11-26 | `d3abaccf` | Seed 0 settings on the PR #12 code | Same result as the accepted pilot to about six digits |

Not run yet: P2b (budgets and point count for B0), P3 (window edges, replay weight, B1 to B3 at a full budget), the P4 campaign (4 arms x 5 seeds, about 70 thread-hours **[estimate]**) and P5 (evaluation).

## 6. Our windows and the Lorenz windows

The Lorenz PINN (repository `lorenz1960-pinn`, `src/pinn/train.py`, `train_windows`) solves an initial value problem. Its windowed run uses 27 equal windows in time, one network per window, trained one after another. Each window's start state is set exactly to the previous window's end state (the hand-off), its weights start from the previous window's (warm start), and inside each window causal weights order the training in time. The weight of point i is exp(-eps times the summed loss of all earlier points), with eps raised from 0.01 to 10. A stage ends when every weight is above 0.99, or after 4 000 Adam steps (`functions/losses.py`).

| | Lorenz windowed run | Our arms |
|---|---|---|
| Conditions | All 3 at t = 0 | 6 at each end (section 1.1) |
| Windows | 27, equal in t | 4, small near the wall (section 4) |
| Order | One after another, each finished before the next | B1: all at once; B2, B3: wall outward, but one network over [0, 20] |
| Joining | Exact hand-off of the end state | B1: soft continuity of all 12 states; B2, B3: no joints, one network |
| Far conditions | None | Exact from the first step in every arm |
| Causal weights | Yes | No |

Why the difference, with the evidence:

- **A window at the wall cannot be finished first.** It has 6 of its 12 conditions. The other 6 are set by the far end (section 1.1).
- **A hand-off error would grow by up to 5.9e16** across [0, 20], and by 6e5 already to eta = 6 (section 1.2). The joints that work for a two-point problem are the ones solved together, as in multiple shooting. B1 is the PINN form of that: the reference's multiple shooting solves 40 segments together, and B1 trains 4 networks together.
- **Causal weights order the loss from one end.** The problem has no end that comes first: errors grow in both directions (section 1.2). This was decided before the code was written ([PINN_PLAN.md](PINN_PLAN.md) section 2, [literature-synthesis.md](../literature/literature-synthesis.md), [PINN_METHOD_SYNTHESIS_2026-11-23.md](../reviews/PINN_METHOD_SYNTHESIS_2026-11-23.md) section 6).
- **In the Lorenz work the effect of causal weights alone was not measured.** Its batch run and its windowed run differ in windows, hand-off, warm start, causal weights and point sampling at once. The journal review lists this as issue I19 ([journal-strategy.md](../journal/journal-strategy.md)).
