# Which L to use: strategy and first results

Date: 2026-11-26. Stage 1 (classical) has run. Stage 2 (PINN) is planned for later, not run. Decision (user, 2026-11-26): keep L = 20 for now, and run the PINN at L = 15 and 40, or similar, later. L is our numerical choice, not a supervisor value. The equations, B1 to B12 and the 22 locked values stay the same at every L (AGENTS.md section 3).

## 1. The question, in plain words

The source puts the far conditions B7 to B12 at eta = infinity. No computer can solve on an infinite domain, so every solver stops at some eta = L and imposes the far conditions there. That cut adds its own error, the **L error**. A longer L makes the L error smaller. The question is how much smaller, and what it costs.

Three numbers decide it:

| Number | What it is | How we measure it |
|---|---|---|
| L error | How far the L problem's wall values are from the eta -> infinity values | Classical solver on a ladder of L (Stage 1, this doc) |
| PINN error | How far the PINN is from the reference at the same L | PINN runs at each L (Stage 2) |
| Cost | Time and memory to reach that accuracy | Solve times, LM step times, point counts |

The answer we report should be the PINN error plus the L error, against its cost.

## 2. Stage 1: the classical L study (done)

Script: `uv run python tools/l_choice.py`, about 15 s. Output: [reference/l_choice.md](../../reference/l_choice.md) and `reference/l_choice.json`. It solves branch A at L = 8, 10, 12, 15, 20, 25, 30, 40, 60, 80, 120, 160, in both forms of the far condition. The eta -> infinity values come from the solves at L = 2 560, 5 120 and 10 240 that were already in `reference/error_bars.json`. The L = 20 solve matches the saved reference to 2.1e-10.

### What it found

**1. The classical solver does not care about L.** Every solve took 0.52 to 0.60 s, from L = 8 to L = 160, and the earlier ladder to L = 10 240 took the same.

**2. Above L = 20 the L error falls very slowly.** It falls like L^-0.41. Doubling L removes only about a quarter of it.

| L | L error, largest of the other settled values (f(L) included) | L error of phi(0) |
|---:|---:|---:|
| 20 | 3.3e-4 (f(L)); f''(0) 1.7e-4, theta(0) 3.0e-4 | 1.8e-3 |
| 40 | 2.7e-4 | 1.4e-3 |
| 80 | 2.1e-4 | 1.1e-3 |
| 160 | 1.6e-4 | 8.3e-4 |

To bring every L error below 1e-3 needs L of about 84 (because of phi(0)). Below 1e-4 needs L of about 370 without phi(0), and about 22 000 with it.

**3. Below L = 20 the values swing through their limits.** Between L = 10 and 20 several wall values cross their eta -> infinity value, not all at the same L. So at L = 12, f''(0) happens to be within 3.4e-5, while phi(0) and f(L) are 2.5e-3 and 2.6e-3 off. At L = 8 and 10, phi(0) is 3.9 % and 1.2 % off. A short L can look right for one value by luck. Short L is not a safe way to save cost.

**4. "Agrees with the next L" is a weak test here.** From L = 20 on, the largest change of any wall value between neighbouring L is only 1e-4 to 2e-4. Yet phi(0) is still 1.8e-3 from its limit at L = 20. When the values creep so slowly, neighbours agree long before the limit is reached. This was the rule that chose L = 20 (`configs/reference.json`, `L_reference_note`).

**5. chi'(0) has no proper L.** It goes from -2.32 at L = 8 to -2.01 at L = 20 and -1.08 at L = 160, and it keeps changing (SPEC section 8.2: Eq. 18 as given has no decaying chi solution). So any chi result belongs to the L it was solved at. This is supervisor question 9, still open.

**6. A longer L makes the PINN's job harder at the same settings.** Without training, from the settings in `configs/pinn.json` (2 000 points, stretch 5):

| L | Training points in the wall layer [0, 0.5] | Points needed for the L = 20 wall density | (L + b)^4, the log-scale factor on R14 at eta = L | LM step **[estimate]** |
|---:|---:|---:|---:|---:|
| 15 | 35.5 % | 1 738 | 5.3e4 | 22 s |
| 20 | 30.9 % | 2 001 | 1.7e5 | 24 s |
| 40 | 20.9 % | 2 957 | 2.6e6 | 32 s |
| 80 | 13.1 % | 4 731 | 4.1e7 | 48 s |
| 160 | 7.6 % | 8 153 | 6.6e8 | 78 s |

The wall layer holds the largest residual, so it needs its points. The log-scale factor grows like L^4, and at L = 20 it already over-weights the far rows (run E7, IMPLEMENTATION_NOTES 7.1). The step times assume LM time grows in line with the points, fitted to the two measured times (15.2 s at 1 000 points, 24 s at 2 000). That is an estimate, not a measurement.

### What Stage 1 means

- For the task the supervisor set, a PINN tested against a classical reference, any L works if both use the same L. At L = 20 the PINN already matches the reference to 1e-5 to 4e-4 (seed 0).
- For closeness to eta -> infinity, the L error at L = 20 (1.6e-4 to 3.3e-4, phi(0) 1.8e-3) is larger than the PINN's own error for every wall value in seed 0, and for three of six in seed 1. But a longer L buys little: L = 80 cuts it by a factor of 1.6, and it costs the PINN about 2.4 times the points.
- The cheap way to state the distance from eta -> infinity is the classical L error itself. It costs 0.5 s per L, and the PINN does not need to be retrained for it. **[judgment]**

## 3. Stage 2: the PINN at other L (planned, not run)

What it would answer: does the PINN error grow, shrink or stay the same when L changes, and what does it cost?

| Item | Plan |
|---|---|
| Runs | B0, seeds 0 and 1, the pilot settings of IMPLEMENTATION_NOTES 7.3, at L = 15 and 40. L = 20 exists (`3a276995`, `d3abaccf`, seed 1) |
| Points | Two versions at L = 40: 1 000 points, and the count that keeps the L = 20 wall density (about 1 480) |
| Measured | Acceptance verdict, reference-free score, Adam, L-BFGS and LM times, LM steps, memory. After that, the PINN error against the reference at the same L, and the total error against the eta -> infinity values |
| Cost | About 1.2 to 2 hours per run on one thread, 6 runs **[estimate]** |
| Before it can run | (a) Reference folders at L = 15 and 40, built by the reference code at those L. (b) A change to a guard: `runs.validate` refuses any L other than `L_reference` ([runs.py:66](../../src/bvp_pinn/pinn/runs.py)). AGENTS.md section 3 says never to weaken the guards, so this needs the team's decision. One option that keeps the guard's purpose: accept an L only when a checked reference folder exists for it, and evaluate each run against that folder |

**Decision rule, declared before any Stage 2 run.** Pick the L with the smallest estimated total error: the classical L error plus the PINN's reference-free score, among the L where both seeds are accepted. If two L are within a factor of 2 of each other, keep the smaller L, which costs less and keeps the chi definition. The PINN's error against the reference is read only after the choice, so the reference does not pick the L (the same rule as gate G5). chi'(0) is always reported with its L.

## 4. Where this stands

- **L stays 20** (user's decision, 2026-11-26). Stage 2, the PINN at L = 15 and 40 or similar, is planned for later and is not part of the campaign ([TODO.md](TODO.md), phase P5). Stage 1 gives no reason to move. A longer L buys little accuracy for more PINN cost, a shorter one adds L error, and both change chi'(0).
- **To report:** each wall value with its L error at L = 20 from `reference/l_choice.md`, next to its PINN error. The SPEC section 8.6 error bars now point here.
- **Found on the way:** SPEC section 8.6 said the L = 20 values are within about 2e-4 relative of the limit for every wall value except chi'(0). That holds for four of the six (f''(0), theta'(0), phi'(0), g'(0)). theta(0) is 3.0e-4 off and phi(0) 1.8e-3. SPEC 8.6 is corrected. The `L_reference_note` in `configs/reference.json` still says L = 20 meets 1e-4 agreement with the next L for f, g, theta and phi. phi(0) changes by 3.9e-4 from L = 20 to 40. The config is unchanged; whether to edit a dated record note is the team's call.
