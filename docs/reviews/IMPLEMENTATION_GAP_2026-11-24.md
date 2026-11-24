# What is left before the implementation is complete, 2026-11-24

Scope: everything on `main` after PR #11 (commit 9473c70), read line by line: `src/bvp_pinn/pinn/`, `tools/`, `configs/pinn.json`, the tests, the open issues #2, #4, #5, #6, and the plan docs (PINN_PLAN, TODO, IMPLEMENTATION_NOTES, strategy, the two 2026-11-23 reviews, the gap plan). Question: what still has to be built or fixed before the 20-run campaign (TODO P4) can start, so that the results can then be verified.

## 1. Short answer

The B0 pipeline works. Seeds 0 and 1 reach reference A from scratch (IMPLEMENTATION_NOTES 7.3). The other three arms run end to end, but only in dry runs, and they do not yet train the same way B0 does. So the implementation is not complete. Ten code items remain (section 3), three of them block a fair campaign: B2 and B3 still end with plain L-BFGS, B1 has no Levenberg-Marquardt at all, and the manufactured-solution tool cannot test the current pipeline. Then come the pilots P2b and P3 (section 4), a declared selection rule, and record fixes.

**Later the same day:** the code items and the reference defects are done; see section 9 for the state of every item.

## 2. Checked today

| Check | Result |
|---|---|
| `uv run pytest` | 40 pass, 30 s |
| `python3 -I tools/check_records.py` | Pass |
| Dry run of B0, B1, B2, B3, seed 0 | All four finish. B0 49.6 s (refine `lbfgs:log+lm:balanced`), B1 23.1 s, B2 16.8 s, B3 17.7 s (all `lbfgs`) |
| B3 dry run with `refine=lbfgs:log+lm:balanced lm_iters=3` | Finishes in 9.8 s. So LM already works on the staged arms |
| B1 dry run with the same refine | Fails after Adam: `ValueError: refine=lm needs one network in exact mode` (`lm.py:186`). No run folder is written |
| `runs/` on this laptop | 12 B0 seed-0 runs (pilots E1 to E8, X1 to X4, and `3a276995`) and one B0 seed-1 run. No B1, B2 or B3 run beyond dry runs |

The dry-run folders made for this review were deleted afterwards.

## 3. Code gaps

| ID | Where | Gap | Why it matters | Fix |
|---|---|---|---|---|
| C1 | `configs/pinn.json`, arms block | B0 alone sets `refine = lbfgs:log+lm:balanced`. B2 and B3 inherit `refine = lbfgs` from `common` | After their last stage B2 and B3 have the same objective as B0 (PINN_PLAN section 2), but a weaker optimizer. Any B0 against B2/B3 difference would come from the optimizer, not from retention. 7.1 and 7.2 showed L-BFGS alone stalls far from the reference | Move the chain into `common`. The B3 test above shows it already runs |
| C2 | `lm.py:186`, `train.refine_once` | No LM for B1 (window networks). The TODO note says "B1 uses L-BFGS" | Same confound as C1, for question 2 of PINN_PLAN section 7. The run also dies after Adam if anyone sets LM for B1 | Either extend LM to window networks (in exact mode the loss is still a sum of squares: stack the per-window residual rows and the interface jumps into one r, Jacobian over all window weights), or declare B1 as L-BFGS only before the campaign and say in every B1 result that it had a weaker optimizer. Check the refine names when the config is resolved, so a bad setting fails before training |
| C3 | `tools/pinn_manufactured.py:60` | The tool swaps in the forced residual only at `losses.residuals`. `lm.py` imports `residuals` straight from `physics`, so the LM stage of the current B0 default trains on the real equations, not the forced ones | TODO P2 asks for "manufactured-solution check with the new pipeline". As written, that run would mix two problems and its result would mean nothing | Swap `physics.residuals` (or pass the residual function in) so Adam, L-BFGS, LM and the monitor all see the forced equations. Add a test that LM on a forced problem lowers the forced loss |
| C4 | `run.py`, PINN_PLAN section 3 | No declared selection rule in code. The status line of `configs/pinn.json` says runs are chosen by the Newton-step error estimate, PINN_PLAN still says "by training residual only", and `run.py` never calls the estimate | Gate G5 and the honesty rules need the rule written down before the campaign. "Chosen" is also unclear: there is one run per arm and seed | Decide and write: one run per arm and seed, never re-run a seed; after training, run `tools/pinn_error_estimate.py`; mark a run accepted if its score is below a declared bar and the tail is decaying; rejected runs stay in every table. Then call the estimate from the campaign script |
| C5 | `evaluate.main`, `plots.median_run` | `evaluate` globs every `runs/bvp-*` folder. The 12 pilot B0 seed-0 runs would enter `summary.csv`, and `median_run` and `wall_table` would mix them with campaign runs | The campaign tables would include hand-built pilot paths (E1 to X4, some with `init_from`) | Give campaign runs a tag (for example `campaign: "P4"` in the config) or a list file, and have `evaluate` and `plots` read only those. Leave pilots in their own summary |
| C6 | `evaluate.py`, `plots.py` | Outputs the plan asks for are missing: `results/failures.md`; FX1 and TX1 (error against wall time, cost table); the solve_bvp time to reach each arm's median error (TODO P5); theta(0) in the wall table (`WALL` has only the slopes, yet theta(0) is the main number in 7.3); the error-estimate score, self-check, parameter count, refine seconds and LM steps in `summary.csv` | P5 cannot answer PINN_PLAN section 7 question 5 or fill TB2 with theta(0) | Add the columns, a `failures.md` writer, and a small timing script for solve_bvp (5 repeats, one thread) |
| C7 | `train.py:168-169` | `monitor_before_refine` comes from the first refine stage (log scale) and `monitor_after_refine` from the last (balanced scale). The two numbers are on different objectives | The logged before/after pair looks like a gain or loss that did not happen. Each stage's own pair in `refine_stages` is right | Report per stage only, or compute both ends on the last scale |
| C8 | `tools/pinn_error_estimate.py:41` | The selection tool imports `evaluate.load_run`, and `evaluate` imports the reference loader. No reference is read, but selection code now imports the evaluation module | Gate G5 says selection never touches the reference. The G5 test scans only `src/bvp_pinn/pinn/`, so it does not see this | Move `load_run` to `runs.py`, and extend the G5 test to the selection tool |
| C9 | `runs.DRY_RUN` | The dry run keeps `lm_iters = 260`, so the B0 dry run takes about three times the others | Small. The "whole pipeline in seconds" note is not true for B0 | Add `lm_iters: 3` to `DRY_RUN` |
| C10 | `constraints.py:138`, `configs/pinn.json` windows | The far end is found by `net.hi == L`, and the last window edge is fixed at 20.0. `--set L=30` would leave B1 with no far conditions and no error | Small, only if someone changes L | Assert that the last window edge equals L when the config is resolved |

## 4. Runs that are part of the implementation, not the campaign

These set the frozen settings. Each needs only training-side numbers.

| Item | State | What is left |
|---|---|---|
| P2a boundary mode (#3) | Done, exact | Tick it in TODO and IMPLEMENTATION_NOTES section 7 (both still say pending or running) |
| P2b B0 sizes and budgets (#4) | Partly done by the 7.1 to 7.3 ladder at 1 000 points | The config says 2 000 training and 1 000 monitor points, but no LM run has used them. LM cost grows with the point count (J is about 400 MB at 2 000 points). One B0 run at 2 000 points, timed, then freeze the count with a dated note. The LM loss was still falling at step 260 in both seeds, so the LM step budget also needs a dated choice |
| Manufactured-solution check | Failed on 2026-11-23 with the old pipeline | After C3, run it with the B0 default and record pass or fail |
| P3 windows and replay (#5) | Not started | Pilots of B1, B2, B3 at seed 0 with C1 and C2 settled: window edges, replay weight from 0.1, 1, 10 by training residual, B1 interface behaviour. Then freeze `configs/pinn.json` |
| Colab timing (#6, first two boxes) | Not started | One short run on Colab CPU and GPU in float64; check the image's Python against the 3.12 pin |

## 5. Reference side

The audit found the reference right. Its record and test defects are still open: D1 (state the f-bounded far-field selection in SPEC 8.2 and 8.6 and add the q f' check), D2 (far-field error bar from both forms of the condition), D3 (the circular far-condition test), D4 (what the SymPy test checks), D6 (L study to 10 240 to test the fit), D7 (branch completeness by continuation or deflation). None changes Eqs. 14 to 18, B1 to B12 or the locked values.

## 6. Records that are out of date

| File | Stale text |
|---|---|
| `docs/pinn/PINN_PLAN.md` | Header "pilots running"; section 2 "Adam, then L-BFGS"; section 3 selection "by training residual only" |
| `docs/pinn/IMPLEMENTATION_NOTES.md` section 7 | P2a "Running", P2b "After P2a" |
| `docs/plans/TODO.md` P2 | P2a unticked; "Seeds 2 to 4" and "Seeds 1 and 2 with the frozen B0 settings" say different things |
| `tools/check_records.py:147` | Prints "SOFTWARE VALIDATION PENDING: no solver, reference simulation or training test was run" |
| `src/bvp_pinn/README.md` | Ends with "PINN modules come next" |
| Issue #2 | Second-member review not done; issue still open |

## 7. People

- Supervisor: questions 7 to 9 (target branch A, far-field treatment at L = 20, chi defined by chi(20) = 0), plus audit D1 as a confirmation. The campaign can run without the answers, but the paper cannot claim the target without them.
- Second member: review of every PINN file (issue #2) and of every term in `equations.py` (TODO R1). Nobody merges their own code.

## 8. Order to reach "implementation complete"

1. C3, C1, C9, C10, C7: small fixes, each with a test.
2. C2: build LM for window networks, or record the decision that B1 stays on L-BFGS.
3. C4: write the selection rule in PINN_PLAN section 3 before any campaign run.
4. C5, C6, C8: campaign tagging, missing outputs, G5 cleanup.
5. P2b at 2 000 points, the manufactured-solution check, then P3 pilots; freeze `configs/pinn.json`.
6. Colab timing; then the campaign (P4) and evaluation (P5).

Done means: all four arms train with the same final objective and optimizer (or a recorded reason why not), the frozen config has dated notes, the manufactured check has a recorded result, the selection rule is written before the campaign, and `evaluate` plus `plots` produce every table and figure in PINN_PLAN section 6 from campaign runs only.

## 9. Follow-up, later on 2026-11-24

Branch `impl/campaign-readiness`. 48 tests pass; the record check passes.

| Item | State | What was done |
|---|---|---|
| C1 | Done | `refine = lbfgs:log+lm:balanced` is in `common`, so every arm uses it; a test checks all four arms resolve to it |
| C2 | Done | `lm.py` works on the frozen-weight training objective for any arm: one Jacobian block per network, interface and soft-boundary rows added through the Woodbury identity. B1 at 2000 points: 32 s per step, peak 1.6 GB. Tests: the LM loss equals the training loss for B1 and soft modes, and falls |
| C3 | Done, then removed | At first `physics.forced()` set the source for every residual call; later on 2026-11-24 the user removed the manufactured-solution checks altogether, with the hook (no check may change the equations). Before that: so Adam, L-BFGS, LM, the monitor and the self check all see the forced equations. The tool has a declared pass bar (1e-3). Not run yet |
| C4 | Done | The acceptance rule is in `configs/pinn.json` and PINN_PLAN section 3; `pinn/estimate.py` applies it after every run. On the 13 pilot runs it accepts the three that match the reference |
| C5 | Done | `pinn/campaign.py` tags runs with `campaign`; evaluation reads only tagged runs and writes `results/<campaign>/` |
| C6 | Done | `failures.md`, FX1, TX1, solve_bvp timing (`evaluation/classical.py`), theta(0) and phi(0) in TB2, acceptance and cost columns in `summary.csv`. Tested on a 20-run dry campaign |
| C7 | Done | Both monitor values of the refine phase are now on the last stage's scale |
| C8 | Done | Evaluation moved to `src/bvp_pinn/evaluation/`; `load_run` moved to `pinn/runs.py`; the G5 test parses every import of `bvp_pinn.pinn` |
| C9 | Done | `lm_iters` is 3 in a dry run |
| C10 | Done | `runs.validate` refuses window edges that do not end at L, unknown refine stages and scales, before training |
| Reference D1 to D7 | Done | [Audit follow-up](REFERENCE_AUDIT_2026-11-23.md); D7 is a deflation search, still finite |
| Stale records | Done | PINN_PLAN, IMPLEMENTATION_NOTES, TODO, PINN_GUIDE, SPEC, README files, the check tool's message. The 2026-11-16 strategy, training proposal and Lorenz comparison moved to docs/archive; strategy.md is rewritten around the gates; the two diagrams and the flowcharts show the current method |
| Found while doing it | Fixed | `load_run` did not set float64, so judging an older run outside `run.py` failed; now tested from a float32 start |
| Found while doing it | Open | phi(0) is the largest wall error of the two section 7.3 runs (4.3e-4 and 6.2e-3 relative) and is not in the acceptance score. A team choice before the campaign (IMPLEMENTATION_NOTES section 7.4) |
| P2b, P3, Colab, campaign | Not run | Runs of one to several hours each; see TODO |
| People | Open | Supervisor questions 7 to 10; second-member review (issue #2, TODO R1) |

