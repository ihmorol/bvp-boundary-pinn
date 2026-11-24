# One network with knowledge retention: feasibility report

Date: 2026-11-19. Status: advice only, nothing run. This updates the framing in [journal-strategy.md](journal-strategy.md) section 5. The audit, journal policies, compliance checklist and timeline there still apply.

**Update, 2026-11-19 (later):** the supervisor fixed the BVP input as eta only, with no parameter sweep. The two-input u(eta, lambda) design in sections 1, 3, 6 and 9 is therefore dropped. [implementation-workflow.md](implementation-workflow.md) section 3.2 replaces it with retention over windows in eta, keeping all twelve boundary conditions at every stage. The literature review (section 2) and the Lorenz design still hold.

Tags: **[V]** checked in the linked source (some through search excerpts, marked "excerpt"), **[J]** my judgment, **[A]** assumption.

## 1. Verdict

**Yes for Lorenz. Yes for the BVP, but only in one specific form.**

- **Lorenz: realistic and well-founded.** One network marched over the time windows, with a loss term that keeps its own earlier solution, is exactly the published bc-PINN method (Mattey and Ghosh, CMAME 2022) [V]. It also answers the weakest point of our conference paper. There, the shared-network run reused one network over the windows *without* retention and reached RMSE 0.350 to 0.648, against 1.75e-4 to 5.16e-4 for separate networks. The paper itself says that run "does not include that backward-compatibility constraint" [V: conference section II-B and Table II]. Retention is the natural next step.
- **BVP: realistic only as retention across continuation stages, with the stage parameter as a second network input.** The BVP has no time direction, so "earlier windows" do not exist in eta. The phases that do exist are the continuation stages already in our plan (kappa = b = Da going 1, 0.5, 0.2). Different stages have different solutions on the same eta, and one network can only keep all of them if it also takes the stage parameter as input. So the BVP network becomes u(eta, lambda), not u(eta).
- **Spatial windows in eta with retention: not recommended.** Information in a BVP flows from both ends, so a window near the wall cannot be solved before the far field is known. Marching outward from the wall turns into a shooting problem with unknown wall values [J].

With these two forms, one narrative holds: **one network trained along a path, keeping what it learned on earlier parts of the path.** The path is time for the IVP and the continuation parameter for the BVP.

## 2. What the literature already does

| Work | What it does | What it means for us |
|---|---|---|
| bc-PINN, Mattey and Ghosh, CMAME 390 (2022) [V: [arXiv 2106.07606](https://arxiv.org/abs/2106.07606), sections 3.1, 4.3] | One network retrained over successive time segments. For segment n ≥ 2, the loss adds a term that matches the network on all earlier segments to its own stored earlier predictions; no reference data. Residual points come only from the current segment. Authors report that error accumulates over time, as in time integrators | Our main precedent. We apply it to an ODE IVP and test it against separate-window networks at matched budgets |
| Adaptive time sampling, Wight and Zhao (2020), as described by Roy and Castonguay [V, excerpt] | One network trained on a small time interval, then on a growing interval | The "physics replay" form of retention (residual on old points) |
| HCS-PINN, Roy and Castonguay, CMAME (2024) [V: [arXiv 2403.03223](https://arxiv.org/abs/2403.03223), sections 2.2, 3] | One network *per* window with exact continuity. Its review says soft continuity terms give only approximate continuity and compete with the residual | A separate-network method, like our conference paper. Their warning about soft-term competition applies to retention losses too |
| iPINNs, Dekhovich et al., Engineering with Computers (2024/25) [V: [arXiv 2304.04854](https://arxiv.org/abs/2304.04854)] | One network learns several PDEs in sequence using overlapping subnetworks; earlier tasks help later ones | Closest to retention across BVP stages, but uses subnetwork masks. We use a parameter input instead, which is simpler [J] |
| Multifidelity continual learning, Howard, Fu, Stinis (2024) [V: [arXiv 2304.03894](https://arxiv.org/abs/2304.03894)] | Continual learning for physical systems; can combine with replay and memory aware synapses | Shows that weight-importance penalties (the EWC family) are an option. We keep them optional |
| Penwarden et al., JCP 493 (2023) [V: [arXiv 2302.14227](https://arxiv.org/abs/2302.14227)] | Unified framework for causal sweeping and time decomposition; transfer learning between subdomains | Already cited by us. Positions single-network marching within the family |

**Gap [J, based on these searches only]:** I found no study that:

- compares retention mechanisms for one PINN against separate-window networks on an ODE IVP at matched parameter and step budgets, or
- uses retention across continuation stages to reach a stiff high-order BVP.

Both are modest but real contributions. A proper literature check in Phase 1 must confirm this gap.

## 3. How it works, in plain terms

**The weights.** There is one set of weights w. After window k, save a frozen copy w_k* and the predictions it makes on a fixed set of anchor points inside the earlier windows [0, t_k]. While training window k+1, the weights change freely, but the loss punishes any change in the network's *outputs* on the old anchor points. The weights move; the function on old windows stays almost the same. That is the retention.

**Lorenz, stage k (k = 0 to 26):**

```text
u(t) = u0 + t * N(t; w)                      one network, hard initial state, whole [0, T]

J_k(w) = mean over new window W_k of |r(t)|^2 / 3          physics on the new window
       + beta  * mean over anchors in [0, t_k] of |u(t) - u_k*(t)|^2 / 3   self-replay (bc-PINN)
       + gamma * mean over a subsample of [0, t_k] of |r(t)|^2 / 3         physics replay (optional)
```

**Read:** fit the equations on the new window, keep the old outputs, and optionally keep the equations satisfied on a sample of old points. beta and gamma are fixed positive weights declared in the config.

Two benefits come for free with one network:

- **No handoff step.** The hard initial state applies once at t = 0, and the curve is continuous everywhere.
- **Smooth derivatives.** The conference paper notes that derivatives can jump at window boundaries [V: section III-B]. With one smooth network they cannot.

Optional final step: run L-BFGS on the full-interval residual, starting from the marched network. A full-interval run from random weights fell into a wrong solution (RMSE 2.237). Starting from a correct marched solution may avoid that trap. This is a hypothesis to test [J].

**BVP, stage s (lambda_1 = 1, then 0.5, then 0.2 for kappa = b = Da):**

```text
u(eta, lambda) = network with inputs (eta, lambda), five outputs

J_s(w) = J_phys(lambda_s)                                     Eqs. 14 to 19 at the new stage
       + gamma * sum over earlier stages j of J_phys(lambda_j)    physics replay of old stages
       + beta  * sum over j of mean |u(eta, lambda_j) - u_s-1*(eta, lambda_j)|^2   self-replay (optional)
```

**Read:** solve the new stage, and keep the old stages solved. Physics replay suits the BVP better than self-replay. Checking an old stage's residual is cheap, and it does not freeze old errors in place [J].

Only the target stage (the locked supervisor case) is reported as the result. The other stages are researcher-chosen numerical steps, labeled as such, as [strategy.md](../plans/strategy.md) section 2 already requires. Physical sweeps (Pr, Nb and others) need supervisor approval first.

## 4. Honest risks

| Risk | How likely [J] | What we do |
|---|---|---|
| One network is less accurate than 27 separate ones (8.13e-5 median) | Likely | Fine if reported honestly. The trade is about 27x fewer parameters and smooth derivatives. Show accuracy against parameter count and steps |
| Retention vs new learning trade-off: too much beta blocks the new window, too little forgets | Certain | Fixed beta grid from a pilot (for example 0.1, 1, 10); choose by training residual only, never by reference |
| Error accumulates over windows (reported by bc-PINN) | Likely | Measure error against time. Test the global L-BFGS polish |
| Cost grows with the number of old anchors | Medium | Fixed-size anchor subsample per stage |
| BVP two-input network is harder: 4th derivatives in eta plus a new input | Medium-high | Fallback: one network u(eta) with warm-start continuation and no retention. Retention is then a Lorenz-only claim |
| Continuation crosses into a different solution branch | Medium | Check every stage against solve_bvp from several initial guesses |
| Reviewer: "bc-PINN already exists" | Certain | Do not claim the method. Claim the matched comparison, the ODE and BVP evidence, and the stage-retention form |

## 5. Unified narrative and contributions

Working title: *One network, one path: sequential physics-informed training with knowledge retention for nonlinear ODE initial and boundary value problems.*

Contribution bullets:

- We replace the 27 separate window networks of our earlier windowed PINN with a single network marched over the windows. Retention keeps the solution on earlier windows, and we compare self-replay, physics replay and no retention at matched parameter and step budgets over at least five seeds.
- We carry the same idea to a coupled five-field boundary value problem. One network takes the continuation parameter as an input and keeps earlier continuation stages solved while it moves to the target case.
- We compare all variants with DOP853 and a collocation BVP solver at equal accuracy and equal time, and we report every run, including failures.
- We show where retention helps and where it fails: error growth over windows, the retention trade-off, and near-wall accuracy in the BVP.

**What happens to the error estimator from journal-strategy.md.** It becomes optional. Doing both well in 13 to 17 weeks is risky. Keep the cheap Lorenz version (a linear solve on saved networks) if Phase 1 goes fast [J].

**Target journal.** With retention as the core and no estimator, the paper is a training-method study. Neurocomputing fits best as primary. The Journal of Scientific Computing returns as an option only if the estimator is included. Mathematics and Computers in Simulation stays as a backup [J]. The over-30% new-content estimate still holds: the single-network method, the BVP class, the matched comparison and the classical baselines are all new.

## 6. Experiments (must-have tier, fits the 7.2 timeline)

| ID | Problem | Arm | Purpose |
|---|---|---|---|
| L0 | Lorenz | 27 separate networks (conference method), re-run at 5 seeds | Upper reference among PINNs |
| L1 | Lorenz | One network, full interval, no marching | Known failure (RMSE 2.237 in conference) |
| L2 | Lorenz | One network, marched, no retention | Forgetting baseline (conference "shared network") |
| L3 | Lorenz | One network, marched, self-replay (bc-PINN) | Main arm |
| L4 | Lorenz | One network, marched, physics replay | Second retention form |
| L5 | Lorenz | L3 plus final global L-BFGS | Tests the polish hypothesis |
| B0 | BVP | u(eta) at the target directly | No-path baseline |
| B1 | BVP | u(eta) with warm-start continuation, no retention | Path without retention |
| B2 | BVP | u(eta, lambda), continuation with physics replay | Main arm |
| B3 | BVP | u(eta, lambda), all stages trained together | Joint-training control |
| C | Both | DOP853 and solve_bvp at equal accuracy and equal time | Classical baseline |

Rules:

- 5 seeds per arm.
- Single-network widths 60 and 100, reported with parameter counts.
- One orbit and the conference initial state first. Other initial states and 3 orbits after that, as the optional tier in [journal-strategy.md](journal-strategy.md) 7.2.1.
- Use the P3 published benchmark if Q1 (ownership) blocks the supervisor's case.

## 7. Verification gates and pass or fail criteria

These come from the benchmark-and-MMS planner (risk high, reference type benchmark), applied to our problems. The planner's refinement rule applies to the *classical references*, since a PINN has no formal order.

| Gate | Test | Pass when |
|---|---|---|
| G1 Code | Manufactured smooth functions through the derivative code (orders 1 to 4 in eta; order 1 in t) and the residual code | Matches analytic derivatives to near float64 rounding; residual of an exact manufactured solution with its added source term is near zero |
| G2 Lorenz reference | DOP853 at 4 tolerance levels (ratio 10 in rtol) | Reported errors stable to 3 significant figures at the two tightest levels; both invariants drift below 1e-9 |
| G3 BVP reference | solve_bvp at 4 mesh or tolerance levels (ratio 2), L in {10, 15, 20, 30}, at least 5 initial guesses, every continuation stage | Wall values plateau; observed order at least 3.6 for the 4th-order collocation in the asymptotic range (screening heuristic, not a certified bound); all guesses reach the same branch, or the branches are reported |
| G4 Retention works | Error on window 0 measured after every later stage (L3, L4) | Window-0 error grows by less than 10x over the 26 later stages. L2 is expected to fail this, and that is the point |
| G5 No leakage | Audit of selection code | Reference used only after model selection |
| G6 Result claim | Each arm against its reference at 5 seeds | Medians and spreads reported for all runs. "Retention helps" only if L3 or L4 beats L2 in all 5 paired seeds, or the paper says it does not |
| G7 Conservation | Lorenz invariants on PINN output | Drift reported for every arm; no threshold claimed, it is a diagnostic |

The G4 factor of 10 is my proposed bar, to be fixed before the runs and not changed after [J]. Per the planner, G2 and G3 must pass before any training result is believed.

## 8. Effort and timeline fit

- **Lorenz retention:** about 1 to 2 weeks for one person. The Lorenz code already has windows, warm start and the trial function; retention adds an anchor store and one loss term [J]. Do it in Phase 1 (December), in parallel with the BVP reference gate.
- **BVP two-input network:** about 2 weeks on top of the BVP baseline, so Phase 2 (late December to 10 January) [J].
- **Compute:** single-network runs cost about the same as one window network over the full point set, which is cheaper than 27 networks. Measure this in the pilot before fixing the grid.

## 9. Questions for the supervisor

1. Do you accept "one network with knowledge retention" as the paper's method, with bc-PINN cited as the precedent?
2. For the BVP, may the network take the continuation parameter as input, with only the locked case reported as the result?
3. Would you like a physical parameter sweep (for example Pr or Nb) using the same two-input network? It needs your approval of the swept range.
4. Ownership of Eqs. 14 to 19 (question 1 in journal-strategy.md) still blocks the BVP part.

## 10. Check against the request

| Request | Where |
|---|---|
| Is it realistic? | Section 1, with the BVP limit stated |
| Previous research on single-network retention | Section 2, sources linked |
| How the weights work per window | Section 3 |
| Same idea for the BVP | Sections 1 and 3, with the required two-input form |
| One narrative | Section 5 |
| Accuracy and verification | Sections 4, 6 and 7 |

Limits:

- bc-PINN's loss details were read from the arXiv HTML version, not the CMAME text.
- Adaptive time sampling is cited second-hand through Roy and Castonguay.
- The gap claim rests on targeted searches only.
