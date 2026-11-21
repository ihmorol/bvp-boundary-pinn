# PINN and reference review

Date: 2026-11-21. Reviewer: Claude (AI agent), asked to act as researcher, professor and reviewer. Status: review only. No code was changed.

## 1. What was reviewed

| Item | Where | State on 2026-11-21 |
|---|---|---|
| Source draft, Eqs. 14 to 19 | `sources/project-draft/project3Nov_2026.docx`, page 6 to 7 | Rendered at 220 dpi and read term by term |
| Equation excerpt | `sources/equation-excerpt/problem_3Nov_2026.docx` | Its five equation images are byte-identical to draft images 45 to 49 |
| Parameter sheet | `sources/parameter-sheet/parameter-values.pdf` | Read against `configs/physical-case-pending.json` |
| Transcription | `docs/equations.md`, `docs/first-order-form.md` | Read |
| Reference code | `src/bvp_pinn/reference/` (committed, `main` at `b88bb8a`) | Read every file; read `reference/study.md` |
| PINN code | `src/bvp_pinn/pinn/` in the worktree `.claude/worktrees/fydp-journal-strategy-84b3e1` (branch `bvp/pinn`) | **Not committed.** All PINN files are untracked; read as they were at 16:04 to 16:47 |
| PINN docs | `docs/PINN_PLAN.md`, `docs/IMPLEMENTATION_NOTES.md`, `docs/BC_ENFORCEMENT_REVIEW.md`, `configs/pinn.json` (same worktree) | Read |
| Pilot runs | `runs/bvp-B0-s0-*` (same worktree) | Two finished, one still running at 16:52 |
| GitHub | `ihmorol/bvp-boundary-pinn` | One merged PR (#1, the reference). No PINN code on GitHub yet |

### What I ran

| Check | Result |
|---|---|
| Full test suite in the PINN worktree (`pytest`) | 23 passed |
| PINN residual function on the saved reference solution (branch A) | Largest residual 4.5e-13; largest boundary error 4.4e-16. The PINN equations agree with the reference |
| Autograd derivatives through the exact-mode constraint layer, against finite differences | Agree to 7e-7 (f'), 4e-6 (theta''), 9e-6 (f'''), which is the finite-difference error for step 1e-3 |
| Exact mode, random weights, width 40, depth 4, linear and log input maps | All 12 boundary errors below 3e-16 |
| Finished pilots against the reference | See finding H1 |

All my check scripts were run from a scratch folder. Nothing was written into either worktree.

## 2. Short verdict

- **Equations: correct.** Eqs. 14 to 19 in the code match the source page term by term. I found no copying error, no missing term and no invented term.
- **Reference solver: sound.** Two independent methods (solve_bvp and Runge-Kutta multiple shooting) agree to about 1e-10. The checks are honest, including the ones that fail.
- **PINN code: correct as written.** The derivative code, the residuals, the boundary layer (TFC) and the losses do what the docs say, with the exceptions listed in section 4. I found no AI-style made-up functions or APIs.
- **PINN training: not working yet.** Both finished pilots end far from the reference. The wall temperature sits near zero and the wall heat condition B4 is missed by about 2.3. These were short pilots, so this does not prove the method fails. It does mean there is no evidence yet that it works.
- **Biggest risks:** loss weights that blow up (H2), a very thin wall layer the network does not resolve (H1), and the code living only in an uncommitted worktree (M1).

## 3. Strengths

1. **The transcription was checked against the source.** I re-rendered page 6 and 7 of the draft and compared every term. `docs/equations.md` and `reference/equations.py` match it, including the tricky groupings in Eq. 14 (the `-0.5*delta_star` bracket and the `(1/q - 4/h)` factor) and the Nr bracket in Eq. 16.
2. **One equation file for both solvers.** `residual_terms` and `boundary_terms` serve NumPy and PyTorch. A second, separate SymPy transcription tests them. So a typo would have to be made twice in the same way.
3. **The reference is checked many ways.** Manufactured solution (order 3.4 to 3.8), tolerance ladder, L ladder, core-then-chi split, Runge-Kutta shooting, slope check, and known limits (curved sheet, flat sheet, slip plus Darcy).
4. **Failures are written down.** `G3_far_field_pass = false`, branch B flagged as invalid, the published Abbas table found to carry a domain-length error. This is good research practice.
5. **No leakage.** Training code never imports the reference loader, and a test enforces it.
6. **The TFC layer is a clean idea.** B4, B10 and B11 are nonlinear, but each fixes one slope from a value the layer leaves alone. So they can be made exact. My checks confirm it holds for any weights.

## 4. Findings, most serious first

Severity: **High** can change the main result. **Medium** affects how results are read or kept. **Low** is worth fixing but does not change conclusions.

### H1. Training has not reached the solution yet (High)

**Evidence.** Pilot runs, B0, seed 0, 1000 points, 3000 Adam steps, 500 L-BFGS iterations. My read-only comparison with reference branch A:

| Quantity | Reference A | Pilot, soft mode (`dabb2750`) | Pilot, linear mode (`cb4a1bd2`) |
|---|---:|---:|---:|
| theta(0) | 0.250 | -0.031 | 0.0004 |
| theta'(0) | -1.259 | -0.018 | 0.058 |
| f''(0) | -4.203 | 0.074 | 1.034 |
| phi'(0) | 3.778 | 0.055 | -0.173 |
| B4 error | 0 | 2.39 | 2.30 |
| Relative L2 error of theta | | 1.84 | 0.92 |
| Monitor equation MSE, R16 | | 6.81 | 6.61 |

f''(0) even has the wrong sign. With theta(0) = 0 and theta'(0) = 0, B4 equals -2.248. Both pilots are close to that value. So the network has settled near a "cold wall" state that ignores the heat condition.

**Likely cause (not yet tested).** The wall layer is very thin and very strong:

- The term `2*beta_m/(eta+b)**4` in Eq. 14 is 250 at the wall (b = 0.2).
- The reference f''''(0) is -1728. At eta = 1 it is -0.088. That is a drop of about 20 000 times inside one unit of eta.
- With the linear input map on [0, 20], the wall layer uses about 1 % of the network's input range.
- In the linear pilot the B4 weight stayed at 1.8, while B10 and B11 (already met) got weights of 2e6 and 3e9 (see H2).

**What it means.** No arm result can be trusted until B0 reaches the reference wall values. The plan's question 1 ("Does B0 reach the reference wall values?") is currently answered "no" at pilot scale.

**Remedy.** Before the boundary-mode decision, run one diagnostic B0 with: `input_map=log`, more Adam steps, and the gradient weights capped. Log B4 and R16 near the wall separately. If B0 still sits at theta(0) near 0, try a warm start only with supervisor approval, as the plan already allows.

### H2. The gradient-norm weights also blow up (High)

**Evidence.** `meta.json` of the pilots:

| Run | Final weights |
|---|---|
| linear | eq 1.34, B4 1.83, B10 2.2e6, B11 2.8e9 |
| soft | from 1.1 up to 7.1e10 (13 terms) |

`IMPLEMENTATION_NOTES.md` section 4 says the 2021 annealing rule "explodes once a term is nearly met" and was "replaced by the 2023 gradient-norm rule". The 2023 rule explodes in the same way here.

**Why.** The weight of term i is (sum of all gradient norms) divided by (gradient norm of term i). For a squared boundary error, the gradient is 2·B·(gradient of B). When B is already near zero, its gradient is near zero, so its weight goes to infinity. This is built into the formula when a term can reach zero.

**Consequence.** L-BFGS then runs with these weights frozen, on a loss whose parts differ in scale by 1e10. That is a badly scaled problem for a quasi-Newton method. The monitor loss used for the L-BFGS rollback also uses these weights.

**Note.** In the default config (`bc_mode = exact`) only one weighted term is left for B0, B2 and B3 (the equation loss). The weighting then does nothing. It only matters for B1 (interface term) and for the soft and linear modes.

**Remedy.** Cap the weights (for example at 1e3 times the equation weight), or stop updating a weight once its term is below a set level. Record the choice with a date. Update section 4 of the notes so it does not claim the explosion is solved.

### M1. The PINN code exists only as uncommitted files (Medium)

All files under `src/bvp_pinn/pinn/`, `tests/test_pinn.py`, `configs/pinn.json` and the two new docs are untracked in one worktree. The shared file `reference/equations.py` was also changed there (split into `residual_terms` and `boundary_terms`) and not committed. One `git clean` or a deleted worktree loses it all. The TODO says "Review of every PINN file by the other member" is still open.

The refactor of `equations.py` does not change results. I checked: the reference still satisfies the PINN residuals to 1e-13.

**Remedy.** Commit on `bvp/pinn` and open a PR so the other member can review it, as the plan requires.

### M2. The B2 objective is not what the plan says (Medium)

`PINN_PLAN.md` section 2 says stage k "trains the residual on [0, eta_k+1]" and "after its last stage, B2 has the same objective as B0".

The code (`train.plan_stages`, `loss_parts`) does this instead: the mean residual on the **new** window only, plus `replay_weight` times the mean residual on the **old** windows. These are two separate means. With the default stretch, about 76 % of the points lie in [0, 6). So in the last stage each outer point counts about three times more than in B0. The replay term is also outside the gradient-norm weights.

The final L-BFGS uses the full problem for every arm, so the end objective does match. Only the sentence in the plan is wrong.

**Remedy.** Fix the sentence in the plan, or change the code to use one mean over [0, eta_k+1].

### M3. The far-field set-up is our choice, not the source's (Medium)

The source puts the far conditions at infinity: theta, phi and chi go to 0. The code (reference and PINN) uses:

- asymptotic conditions for theta and phi at eta = 20 (Lentini and Keller 1980),
- chi(20) = 0.

This is a reasonable numerical choice and is recorded in SPEC section 8 and supervisor question 9. But readers must know three things:

1. At eta = 20 the reference still has theta = 0.068 and phi = 0.18. The tails are slow, so "zero at L" would be wrong.
2. chi has no limit. chi'(0) moves from -2.01 (L = 20) to -1.08 (L = 160). So any chi error we report is about the L = 20 problem only.
3. Eq. 18 as printed has `kappa/q*f*phi` and `-delta_star*eta/2*phi'`, where a chi term would be expected. The supervisor confirmed it as written, and delta_star = 0 removes the second one. The first one is the reason chi has no limit. Question 9 to the supervisor is still open.

**Remedy.** Keep this caveat next to every chi number in results.

### M4. A stale note in `configs/reference.json` (Medium)

`L_reference_note` says L = 20 is the smallest L whose f, g, theta and phi wall values agree with the next L to 1e-4. The study says otherwise: phi(0) changes by 3.9e-4 from L = 20 to L = 40, and `V5_core_L_independent = false`. SPEC section 8 already admits this. Only the config note is out of date.

**Remedy.** Change the note to "about 4e-4 for phi(0), 7e-5 for the others".

### M5. Exact mode is much slower (Medium)

The three pilots ran at the same time, one thread each:

| Mode | Wall time |
|---|---|
| soft | 1223 s |
| linear | 2566 s |
| exact | still running after 48 minutes |

`IMPLEMENTATION_NOTES.md` section 5.2 says the TFC cost is "small". The pilots so far show it is about twice the soft cost or more. The likely reason: every forward pass also evaluates the network and its first two derivatives at both ends, inside the fourth-order autograd graph.

**Remedy.** Record the measured cost. If exact mode wins the pilot, budget the Colab time for it.

### L1. Constraint tensors are not registered buffers (Low)

`constraints.Field` stores `inv` and `ends` as plain tensors, not as module buffers. So `model.to("cuda")` will not move them. It works now only because `run.py` sets the default device before building the model. On Colab, building on CPU and then moving to GPU would fail.

**Remedy.** Make `Field` an `nn.Module` and use `register_buffer`.

### L2. Self-replay holds values only (Low)

B3 holds the five outputs of earlier windows to their frozen values, but not their derivatives. For a fourth-order f, the network can keep f close and still change f'' and f''' in old windows. This follows the bc-PINN paper, so it is a design point, not a bug. Worth one line in the results.

### L3. Evaluation always compares with branch A (Low)

`evaluate.py` reports all errors against branch A. It also reports the nearest branch, but only by theta. If a run lands on branch B, the summary shows large errors with no other flag. Branch B is not a valid infinite-domain solution (m < 0), so A is the right target. Just read `nearest_branch` before reading errors.

### L4. Log noise (Low)

Each run log fills with "Converting a tensor with requires_grad=True to a scalar". It comes from `float(...)` on tensors that still need grad in `train.row`. Use `.detach().item()`.

### L5. The published curved-sheet table matches only at L = 10 (note, not a bug)

The benchmark reproduces Abbas et al. to the fifth decimal at L = 10, but at L = 80 the values are up to 0.6 % lower. The team found and recorded this (status log, SPEC lines 196 to 202). The check is fair: it tests our code on the paper's own domain.

## 5. Source, parameters and transcription

| Check | Result |
|---|---|
| Eq. 14, all terms and brackets | Match |
| Eq. 15 | Match |
| Eq. 16, Nr bracket and beta_m-lambda_m term | Match |
| Eq. 17 (`eta + k` printed, read as kappa) | Match, alias recorded |
| Eq. 18 (Pe/Pr bracket printed in red) | Match as printed. Physics concern in M3 |
| Eq. 19, B1 to B12 | Match. B5 printed as `Nt Nb^-1`, read as Nt/Nb |
| Parameter sheet against locked case | Pr 6.8 on the sheet is replaced by the supervisor's 8.8 (recorded). Le, delta_star, ell, fw are inside the sheet's ranges |
| theta_w = 0.2 | Below 1, so c = theta_w - 1 = -0.8. Unusual for a temperature ratio, already flagged in the case file |

The November 10 file with the wrong Eq. 14 reading is not used anywhere in code.

## 6. Citations and "made-up" code

I looked for invented APIs, wrong library calls and invented references.

- **APIs:** every PyTorch, NumPy and SciPy call used exists and is used correctly (`torch.bucketize`, `torch.set_default_device`, `LBFGS(line_search_fn="strong_wolfe")`, `np.trapezoid`, `solve_bvp`, `solve_ivp(method="DOP853")`).
- **References I recognise and that match their use:** Wang, Sankaran, Wang, Perdikaris 2023 (arXiv 2308.08468); Lentini and Keller 1980 (SIAM J. Numer. Anal., doi 10.1137/0717049); Mortari 2017 (Mathematics 5(4):57); Sukumar and Srivastava 2022 (CMAME 389:114333); Mattey and Ghosh 2022 (CMAME 390).
- **Not verified by me:** the exact equation numbers "2.12 to 2.15" in Wang et al., the journal volume given for Kiyani et al. 2025, and the four GitHub repository names in `IMPLEMENTATION_NOTES.md` section 3.2. They look plausible. Someone should open each once.
- **Formula checks:** I re-derived the asymptotic far conditions in `equations.far_conditions` from Eqs. 16 and 17. Both are correct, and B11 needs m different from mp (here m = 0.239, mp = 0.340).

## 7. Recommendations, in order

1. Commit the PINN work and open a PR for review (M1).
2. Cap the gradient-norm weights and fix the claim in the notes (H2).
3. Run one B0 diagnostic aimed at the wall layer before the boundary-mode pilot is decided (H1).
4. Fix the B2 sentence in the plan or the code (M2) and the reference config note (M4).
5. Register the constraint tensors as buffers before the Colab runs (L1).
6. Keep the far-field caveat with every chi result until the supervisor answers question 9 (M3).

## 8. What this review does not cover

- I did not run full-length training (20 000 Adam steps). All training evidence here is from the team's short pilots.
- The exact-mode pilot had not finished.
- I did not review the Lorenz code in `prior-work/` in detail. It is not imported by the BVP code (checked with grep).
- I did not open the literature PDFs.
