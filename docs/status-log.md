# Status log

Newest first. One entry per working session. Keep it short and factual.

## 2026-11-23, later (session: reviews, balanced scale, Levenberg-Marquardt)

- Installed the `research-synthesis` skill (user and ZCode skill folders) and ran two review sessions. [Reference audit](reviews/REFERENCE_AUDIT_2026-11-23.md): the reference is right (an independent fourth solver agrees to 1e-11; the PINN gap was about 700 times the reference error); seven record and test defects listed (D1 to D7), D5 fixed in engine/README. [PINN synthesis](reviews/PINN_METHOD_SYNTHESIS_2026-11-23.md): the log scale over-weights far-field f and g rows, the plain scale starves the far field, and the LM Jacobian was built row by row.
- Built its top three changes: the balanced residual scale, a per-point LM Jacobian (12 s a step instead of over 10 minutes), and selection by the Newton-step error estimate plus a far-tail fit. `refine` can chain stages (`lbfgs+lm`). 3 new tests.
- Result, B0 seed 0, continued from E5 (IMPLEMENTATION_NOTES 7.2): run X3 `c0b70da5` reaches a balanced loss of 2.3e-6 and matches reference A at the wall to 5e-5 to 2.7e-4 relative (theta(0) 0.25022 against 0.25019, f''(0) -4.20445 against -4.20330), relative L2 1e-4 to 4e-3, at the size of the reference's own far-field error. The balanced scale from the start failed (X4: small loss, flat theta tail, caught by the checks), so the scale now changes by stage: Adam and L-BFGS on the log scale, Levenberg-Marquardt on the balanced scale (`refine = lbfgs:log+lm:balanced`, now the B0 default).
- From scratch with that pipeline (IMPLEMENTATION_NOTES 7.3), seeds 0 and 1, about 93 minutes each on one thread: balanced loss 6.8e-7 and 1.6e-6; every wall value within 4e-4 relative of reference A (seed 0 within 6e-5: theta(0) 0.2501906 against 0.2501853, f''(0) -4.20353 against -4.20330); relative L2 1e-4 to 5e-3; branch A, decaying tail; the reference-free score ranks the seeds correctly. 40 tests pass. Nothing committed.

## 2026-11-23 (session: PINN training fixes, checks, and docs layout)

- Rule made explicit in AGENTS.md: Eqs. 14 to 18, B1 to B12 and the locked values are fixed; only how they are imposed and solved may change. Removed the proposals that broke it from the gap plan (chi model options, parameter continuation, re-derivation, curriculum) and the warm-start fallback from PINN_PLAN.
- PINN code: new options in configs/pinn.json, all defaulting to the old behaviour: smooth start profile with zero last layer (`pinn/start.py`), residual scale (eta + b)^order, weight cap, best checkpoint, resampling, full BFGS (`pinn/bfgs.py`), Levenberg-Marquardt (`pinn/lm.py`), `init_from`, and a reference-free self check (`pinn/selfcheck.py`: wall values, decay powers m and mp, failure share). 7 new tests; 38 pass.
- B0 fix ladder at seed 0 (IMPLEMENTATION_NOTES 7.1): the cold wall is gone in every run. The residual scale decided the far field: without it the network ends with m < 0 (no decay); with it, it heads to branch A. Full BFGS was no better than L-BFGS. Final run by the training-side rule, E7 (`9509fb7b`): theta(0) 0.197 against 0.250, f''(0) -3.65 against -4.20. Not converged: the log scale stalls with the wall residual large, while the plain residual (E8) reaches MSE 2e-4 but drifts away from the solution. The next step is bounded per-region weights to balance the two
- Checks: engine/bvp_engine gained `--estimate` (one Newton step of the box scheme from a PINN state); within 10 to 20 % of the true wall errors once a run is near the solution. Manufactured-solution tool (`tools/pinn_manufactured.py`) failed at the budget tried. Levenberg-Marquardt too slow on this laptop (one step over 10 minutes).
- Reference error bars (`tools/reference_error_bars.py`, SPEC 8.6): mesh error about 1e-7 relative; L = 20 against the L-extrapolated limit 8e-5 to 8e-4 per wall value; chi'(0) has no limit.
- Layout: all docs moved into folders under docs/ (problem, plans, reference-solver, pinn, reviews, literature, methodology with figures, team, journal, archive) with an index in docs/README.md. Root diagrams, the strategy guide and TODO.md moved into docs/. Links, tools, AGENTS.md and README updated; the record check now covers the subfolders. Deleted build and test caches (.cache, .pytest_cache, __pycache__). The two git worktrees under .claude/worktrees hold uncommitted work, so they were left alone. Nothing committed.

## 2026-11-22 (session: literature against our method)

- Read all 25 papers in sources/literature/ in full text and compared their reference and PINN methods with ours. Checked seven recent published works online (abstracts only): Gauss-Newton natural gradient, NysNewton-CG, multi-stage networks, SOAP, and three benchmark candidates. Report: [LITERATURE_GAP_STRATEGY.md](plans/LITERATURE_GAP_STRATEGY.md).
- Measured on reference A: the terms the far-field condition drops from Eq. 16 are 2.5 % (Nr 3cA^2 theta'^2) and 0.7 % (Nt theta'^2) of the kept terms at eta = 20, falling only slowly. Likely cause of the phi(0) drift with L.
- Plan: a one-change-at-a-time B0 ladder (base profile from the smooth guess, scale functions, integral weights, bounded weights, Levenberg-Marquardt or full BFGS refine), plus reference work (L extrapolation, corrected far condition, GCI, an error estimator from one Newton step). Five supervisor questions listed. No training or reference run. Nothing committed.

## 2026-11-22 (session: runtime, equations and a C++ check solver)

- Profiled one PINN step on `main`. Nested autograd for the derivatives was about 85 % of the step; its cost grows 3.5 to 4.3 times per derivative order. Added `pinn/jets.py` (forward Taylor-mode derivatives); `model.states` now uses it. Same values and weight gradients to 1e-14; 200 Adam steps give the same losses to 11 digits, 118.7 s against 30.1 s. Report: [RUNTIME_STRATEGY.md](pinn/RUNTIME_STRATEGY.md).
- Checked every term of Eqs. 14 to 18 on reference A: only the delta_star terms are zero; nothing is dropped. Proved an exact rewrite of Eq. 14 with a pressure-like unknown that needs f''' instead of f'''' (`equations.momentum_flux`, `reduced_momentum`, SymPy test). Not used in training; needs supervisor agreement.
- Added engine/: a C++ box-scheme solver with Newton, banded LU and Richardson. Order 2.00; wall values match the locked reference to 2.7e-10; 0.48 s. Plot script compares a PINN run with it. Added tools/time_pinn_step.py. All tests pass. Nothing committed.

## 2026-11-21 (session: PINN review and reading guide)

- Reviewed the uncommitted PINN code (worktree `fydp-journal-strategy-84b3e1`, branch `bvp/pinn`), the reference code, and Eqs. 14 to 19 against the draft page rendered at 220 dpi. Equations match; 23 tests pass; PINN residuals on reference A are 4.5e-13.
- The two finished B0 pilots (soft, linear; 3000 Adam steps) do not reach the reference: theta(0) near 0, B4 missed by about 2.3. Gradient-norm weights reached 3e9 to 7e10. Report: [PINN_REVIEW_2026-11-21.md](reviews/PINN_REVIEW_2026-11-21.md).
- Added a plain-language guide to the PINN: [PINN_GUIDE.md](pinn/PINN_GUIDE.md). Nothing committed.

## 2026-11-19 (session: evidence and standard figures)

- Rebuilt the reference figures after review: profiles on a linear eta axis (full domain and wall layer), with Runge-Kutta circles and wall values in each title; supplementary figures for branch B, far-field tails and the verification study. Old log-axis figures removed.
- Added check V11 (`benchmark.py`): the same code on limits of Eqs. 14 and 16 with known answers. Reproduces the published curved-sheet skin friction of Abbas et al. (2013, via the reprint in Shafiq et al. 2019) to the fifth decimal at L = 10; at L = 80 the values are up to 0.6 % lower, so the published table has a domain-length error. Matches the exact flat-sheet f', theta (Pr 0.72 to 10) and the slip plus Darcy solution at the locked Da and N1 to 1e-7; the distance to the flat limit falls as 1/kappa.
- Added check V12: the slope of every saved curve matches the next saved curve (and the equation-implied highest derivative) to 3.5e-9. Two new tests; 13 pass. The benchmark limits do not exercise the micropolar, dipole, unsteady, radiation, nanoparticle and chi terms (SPEC 8.5). Nothing committed.

## 2026-11-19 (session: first-order form explained)

- Wrote docs/reference-solver/first-order-form.md: step-by-step derivation of M*H = -r0 and the first-order system from Eqs. 14 to 19, with the reason the step is needed (solve_bvp takes first-order systems only; the PINN does not use it). Linked from SPEC section 2. No equation, value or code changed.

## 2026-11-19 (session: quotes and figures)

- Checked every direct quotation in the docs against its source. Fixed three: the Elsevier policy quote (now verbatim from the page), the ponytail skill quote (case), and the Kierzenka and Shampine title (published as "Maltab", marked [sic]). Checked 24 numbers in SPEC section 8 against the saved study files: all match.
- Read figure guidance (Rougier et al. 2014, Wilke ch. 20, the scientific-visualization skill). Rewrote plots.py: seven figures with one message each, Okabe-Ito colours fixed per field, second cues, log (eta + kappa) axes, legends off the data, panel letters and generated captions. The run now also saves the old zero-at-L L ladder, per-L profiles (reference/l_study.npz) and per-interval residuals.
- Inspected every figure. Fixed overlapping legends, crowded ticks, a clipped label, and an unreadable raw-difference plot (now a labelled running maximum). Wrote the comparison analysis in SPEC section 8.5. New observations: thin wall layer (f''''(0) = -1728), phi stays near 0.15 to 0.19 far from the wall, and starting guesses with theta(0) of 0.7 or more mostly reach branch B. 11 tests pass; nothing committed.

## 2026-11-19 (session: Runge-Kutta cross-check)

- Added src/bvp_pinn/reference/shooting.py: multiple shooting with SciPy's DOP853 Runge-Kutta on 40 segments, with Newton on the joins and boundary conditions, started from the smooth guess only. A new test recovers the manufactured solution to 1e-8.
- Added check V10 to the reference study and the RK solution (`z_rk`) to each reference.npz. Branch A: RK and solve_bvp agree to 2.8e-10 on every state and 1.9e-10 on wall values; RK's own error estimate is 5.1e-10. Branch B agrees to 8.7e-9. Gate G3 (fixed L) still passes, now including V10.
- V10 confirms the discretization, not the equations or the far-field choices; those have their own checks (SPEC section 8). 11 tests pass. The study now takes about seven minutes. Nothing committed.

## 2026-11-19 (session: far-field treatment)

- Checked how the selected papers set eta at infinity: micropolar PINN 7, Falkner-Skan PINN 12, stretching cylinder unstated (plots to 8), Kumar et al. 2023 RKF45 with no value in the text (Europe PMC full text). None tests that results stop changing with eta_inf.
- Derived the far-field behaviour: f1, f2, g decay exponentially; theta and phi only as powers of (eta+kappa); chi, as Eq. 18 is printed, has no decaying solution. Numerical slopes confirm the theta exponent.
- Implemented asymptotic far conditions for theta and phi (Lentini and Keller 1980, verified via Crossref) in equations.far_conditions, with a test that the decaying modes satisfy them exactly. Set L = 20 with an L study to 160. The old L = 10 reference was off by about 3% in f''(0) and 5% in theta(0).
- Branch A is the valid eta -> infinity solution; branch B has f_inf < 0 and is flagged. Wall slopes settle to about 7e-5 per doubling of L; phi(0) to about 4e-4; chi has no limit. Supervisor questions 7 to 9 updated. 10 tests pass. Nothing committed.

## 2026-11-19 (session: source verification and code review)

- Rendered Eqs. 14 to 19 from the supervisor's docx (MathType previews image44 to image49) and checked every term against equations.md and equations.py: all match. Eq. 17 prints eta+k in two denominators; k=kappa=0.2 per the receipt. Recorded in equations.md.
- Added tests/test_config.py, which parses the received parameter block verbatim and checks all 24 symbols against the locked config: pass.
- Trimmed unused options and redundant code with the ponytail skill, then reviewed with ponytail-review. Two confirmed findings fixed: meta.json was written before the V9 and G3 results existed, and the loader's physical-case check was untested. 9 tests pass; the reference study gives the same results.

## 2026-11-19 (session: classical reference)

- Updated GitHub issue ihmorol/lorenz1960-pinn#43 so its plan links point to docs/journal/.
- Added pyproject.toml and uv.lock (Python 3.12.13, NumPy 2.5.5, SciPy 1.18.1, Matplotlib 3.11.2, SymPy 1.14.0, pytest 9.2.0), src/bvp_pinn/config.py, the src/bvp_pinn/reference/ package, configs/reference.json and 10 tests. All tests pass.
- Ran the reference study. Manufactured-solution order 3.38, 3.83, 3.63 (pass). All 48 scan guesses converged to two branches: A (theta(0)=0.2382) and B (theta(0)=0.7423). Tolerance, boundary, core-then-chi and coefficient checks pass for both.
- Far-field check fails by the nature of the equations: theta, phi and chi decay close to logarithmically, so wall values change by about 5 to 7% between L=20 and L=30. Fixed L=10 as our choice; the reference is the exact solution of the truncated problem there. Exported both branches with figures to reference/. Added supervisor questions 7 and 8 (target branch, L).
- No physical value or equation changed. No PINN code yet. Nothing committed.

## 2026-11-19 (session: journal strategy)

- Read the submitted ICEEICT 2027 paper (lorenz1960-pinn, writings/iceeict2027-final), the supervisor's draft and the BVP records. Found that the draft is an unpublished manuscript by M.J. Uddin and M.I. Hasan, so ownership of Eqs. 14 to 19 is now an open supervisor question.
- Searched journal policies on extended conference papers, ICEEICT dates (notification 20 Nov 2026) and editor guidance. Several publisher pages blocked direct reading; those points are marked as excerpts in the report.
- Wrote docs/journal-strategy.md: journal shortlist, 22-item audit of the conference paper against the BVP plan, recommended framing (residual vs error, reference-free estimator), revised method and timeline, paper outline, compliance checklist and supervisor questions. Advice only; no physical value, equation or config changed.
- Revised the timeline after the team reported that FYDP-III ends in mid-February 2027, with a possible extension of up to a month. Plan now targets journal submission by mid-March, splits experiments into must-have and optional tiers, and asks the supervisor to approve the extension.
- Assessed the team's idea of one network with knowledge retention. Wrote docs/single-network-retention.md: realistic for Lorenz (bc-PINN precedent); for the BVP only as retention across continuation stages with the stage parameter as a second input; spatial windows in eta not recommended. Ran the benchmark-and-MMS planner and turned its output into seven verification gates. Linked the report from journal-strategy.md and README.
- Recorded the supervisor decisions (both structures on both problems; BVP input eta only; no sweep; problem from Uddin and Hasan as collaborators) in docs/problem/questions-for-supervisor.md. Wrote docs/implementation-workflow.md: Lorenz arms L0 to L4, BVP arms B0 to B3 with retention over eta windows and all 12 boundary conditions at every stage, six phases for two members with sync points, gates G1 to G6, figure list and result questions. Marked the two-input BVP design in single-network-retention.md as dropped.
- Recorded the supervisor's answer that physical meanings are not needed: values are used as given, the task is reference solution, PINN training and testing against the reference. Closed definition questions 1 to 6 in questions-for-supervisor.md and aligned parameters.md, README, the record checker message, journal-strategy.md and implementation-workflow.md. No value or equation changed.
- Created GitHub issue ihmorol/lorenz1960-pinn#43 for the Lorenz journal extension (no code changed there). Wrote docs/plans/PRD.md, docs/reference-solver/SPEC_reference_solutions.md, docs/team/CODING_RULES.md, docs/team/SKILLS_AND_TOOLS.md, docs/literature/RESEARCH_NOTES.md and TODO.md. Checked every DOI through Crossref, doc links by HTTP status, and solve_bvp facts against the SciPy v1.18.0 manual. Left uncommitted for review.
- Separated the two works: moved journal-strategy.md, single-network-retention.md and implementation-workflow.md into a new docs/journal/ folder with a README, and fixed their links. Wrote docs/pinn/PINN_PLAN.md as the BVP-only plan (arms B0 to B3, run rules, metrics, gates, figures, dates) and repointed PRD, SPEC, CODING_RULES and TODO to it. tools/check_records.py now also scans TODO.md and docs/journal/.
- tools/check_records.py: the new files pass. The run still fails on pre-existing links in docs/methodology/lorenz-to-bvp-methodology.md, which point to sibling repos and do not resolve from this worktree. No solver, training or evaluation run.

## 2026-11-17 (session: redrawn one-page diagrams)

- The user found the earlier methodology and architecture diagrams hard to read: too much text, poor layout. Redrew both as one A4 landscape page each, in the style of methodology-flowchart-a4.pdf: numbered boxes, dashed green groups, yellow decision diamonds, short labelled arrows.
- Methodology diagram: 13 boxes. Three training seeds on the left, software tests and the classical reference set up once on the right, then the accuracy check, sensitivity, method comparison, wall quantities and the result or a recorded failure.
- Architecture diagram: one training step (points, shared network, derivatives, R14 to R18, B1 to B12, locked case, one loss, finite check, update), the checks outside the loss, and a bottom row with the four changes from the Lorenz PINN.
- Added tools/build_diagrams.py. It draws both figures and stops if text overflows a box, boxes overlap, an arrow crosses a box or tab, a label sits on a group border, or the writing rule breaks. Checked that the new checks catch the two layout faults fixed during this session. Exports are one-page A4 PDFs with embedded STIX fonts, outlined SVGs and PNG previews, about 80 to 90 KB per PDF.
- Replaced docs/methodology/methodology.md with a short box-by-box guide. Moved the fifty-task version to docs/archive/ with a note. Removed the old builders and checkers. Updated README, flowcharts.md, the Lorenz comparison links and the guide link; rebuilt strategy-guide.html.
- Ran tools/check_records.py and git diff --check: passed. No solver, reference, training or evaluation ran. No commit.
- Second pass, same day: the user said the methodology copied the reference figure's diffusion-problem flow. Kept the visual style and redrew the flow for this BVP in four gated stages: case and code (locked case, wall coefficient sizes, software tests), reference for checking only (with an unverified branch), training per seed (near-wall points, Adam, L-BFGS, finite check, monitor-based checkpoint), then test and report. Updated docs/methodology/methodology.md and the Mermaid overview in docs/methodology/flowcharts.md to the new numbering. The architecture diagram is unchanged.

## 2026-11-16 (session: substituted reading sheet and commit of the day's work)

- Wrote docs/problem/simplified-equations.md: Eqs. 14 to 19 and the Eq. 19 boundary errors restated with the locked values substituted, as a reading aid. equations.md stays authoritative and links the sheet. delta_star=0 removes all unsteady terms; kappa=b=0.2 merges h into q; Eqs. 16 and 17 are shown multiplied by 8.8, which does not change their zero sets.
- Corrected the wall radiation constant during the term-by-term check: B4 gives 0.998 = 1 - 0.002 from 1.25*0.0016, not the 0.9984 printed in the earlier chat draft. The sheet records the fix.
- Linked the sheet from README and equations.md and rebuilt strategy-guide.html.
- Analyzed every previously uncommitted change and matched it to the three recorded sessions of this day: the supervisor parameter lock, the fifty-task methodology diagram and the Lorenz-style architecture diagram. No unexplained edits were found. The unreferenced methodology-flowchart-a4.pdf export was included with the methodology group.
- Ran tools/check_records.py, tools/check_methodology.py, tools/check_pinn_architecture.py and git diff --check. Committed the day's work in four groups with plain messages. No push. No solver, reference, training or evaluation run.

## 2026-11-16 (session: Lorenz-style PINN architecture)

- Inspected the live lorenz1960-pinn code and paper architecture source at revision 0c5f9c86c597bd27897632fb917d6d17548eeffc. Its tracked worktree was unchanged; untracked notes were not used as implementation evidence. Wrote docs/methodology/lorenz-to-bvp-methodology.md with source links, retained patterns and required BVP replacements.
- Recorded two old figure differences: the implemented Lorenz MLP has 11283 parameters including biases, not 11163, and its residual loss averages over all three components. The Lorenz trainer returns its terminal model and logs reference errors during training; these details are not copied as BVP selection rules.
- Added a new architecture-centered diagram with one shared five-output MLP, nested eta derivatives, separate interior and endpoint residual branches, one scaled loss, Adam/L-BFGS feedback, independent monitoring and a check-only reference. Retained the existing fifty-task diagram unchanged. No causal time windows, hard initial-value lift, endpoint handoff or reference-data loss is added to the BVP.
- Exported vector PDF and SVG, editable HTML and an image preview. Reused the canonical equation and boundary parser. Content checks confirm 21 distinct graph nodes, all 17 original residual expressions, valid MathML, embedded fonts and no missing-glyph markers or out-of-page text. Inspected the rendered sections directly because the visual-review agent was unavailable. Repaired arrow routing and made gradient checks occur after backpropagation, before weight updates.
- Updated links and the guide freshness note; rebuilt strategy-guide.html. HTML preflight has no errors; its white-background advisory matches the user's theme. PDF metadata, fonts and boundary checks pass; six advisory punctuation flags concern multiplication dots in math. Record and whitespace checks pass. No physics, solver tests, training, commit, push or publication.

## 2026-11-16 (session: complete methodology diagram)

- Built the BVP methodology from the locked parameter record and unchanged Eqs. 14 to 19. Split it into 50 small tasks across 10 phases, with full equation, boundary, reference-matrix, training, error and acceptance formulas in separate panels. Physical and training configs were not changed.
- Matched the user's reference theme: pale task boxes, numbered badges, green dashed groups and tabs, yellow decision diamonds, dark arrows and serif text. Saved the requested theme preference for later diagrams. Added vector PDF and SVG, editable HTML and Markdown, and a preview image. The original diffusion-example PDF is unchanged.
- Added the methodology builder and content/PDF checker. Reused the canonical equation parser instead of copying residual formulas into the method. Checks confirm 22 displayed locked values, 17 canonical residual expressions, 50 ordered tasks, valid MathML, embedded PDF fonts and no out-of-page text or missing-glyph markers. SVG contains vector paths, not raster images.
- A read-only math review found a circular acceptance gate. Corrected it: phase 8 records final-test pass with sensitivity pending; phase 9 decides final acceptance. Repaired multiline mathematical rendering and checked every rendered section for overlap and clipping. The visual-review agent was unavailable, so the rendered sections were inspected directly instead.
- HTML preflight passed with no errors or warnings. PDF checks passed metadata, fonts and page-boundary checks; six advisory punctuation flags refer to multiplication dots in mathematical expressions, not clipped prose. No solver, reference, training or physical accuracy test ran.
- Updated navigation links and rebuilt strategy-guide.html with the complete locked-case status. Ran the record and whitespace checks. No commit, push or publication.

## 2026-11-16 (session: supervisor parameter lock)

- Recorded the user's message with values directly from the supervisor. Saved all 24 received names and values, the date, explicit mappings and approved notation aliases in configs/physical-case-pending.json. The 22 equation coefficients are now complete and locked. Pr=8.8 supersedes the old sheet value; Da=0.2 is derived from the supplied M=5 and M=1/Da definition.
- Kept H=1 and Prm=0.1 in the receipt, outside the equation coefficients. Kept the temperature, Lb, B and fw definition questions open. These questions do not authorize changing the locked numbers or Eqs. 14 to 19.
- Added checks for the received-value checksum, mappings, case lock and pending validation status. The locked baseline passed, and all 24 in-memory altered-record tests were rejected as expected. No test changed repository data.
- Checked the parameter values against the message and verified the wall-coefficient and conditional temperature/Lb arithmetic. The training config is byte-for-byte unchanged, including its eight null fields and proposed domain lengths. No solver validation, reference simulation, training or evaluation was run.
- Updated the parameter record, supervisor questions, README and strategy. Left the generated guide unchanged and marked it stale. Hash checks confirmed that the existing flowchart and methodology work, equations, training config and generated guides were preserved.
- Ran `python3 -I tools/check_records.py` and `git diff --check`: passed. The record checker reports complete physical parameters and pending software validation. No commit or push.

## 2026-11-14

- Committed the 2026-11-13 parameter audit entry that was left uncommitted in a linked worktree, placed it under the per-paper analysis entry for the same day, and labeled it "session: parameter audit". The worktree change was a status log edit only; no file differed in content from what is now committed.
- Checked the rest of the repo for pending work: all 26 PDFs in sources/literature/ are tracked, no other branch, stash or untracked file exists, and only tools/__pycache__/ is ignored. No runs; tools/check_records.py passed.

## 2026-11-13 (session: per-paper deep analysis)

- Ran one analysis agent per downloaded paper, two at a time, over all 25 papers with local PDFs in sources/literature/. Each agent read the full PDF and wrote analysis.md inside that paper's folder, with a fixed section order: record, problem, method, author-reported results, adopt, do not import, quotes, relevance.
- All 25 analysis files pass the writing rules: no banned words, no em dashes, page or section reference on each claim, author numbers framed as author-reported.
- Created docs/literature/paper-analysis-tracker.md with the status and one-line takeaway per paper. Created docs/literature/literature-synthesis.md as the cross-paper guide: must-know facts, decisions per strategy section for arms B3 to B6 and the AL arm, implementation checklist, writing guide, do-not-import list, internal paper errors, and open choices to log.
- Key cross-paper findings recorded in the synthesis: unbalanced boundary gradients are the core failure mode; fixed scales must come before optimizer choice; small loss does not imply small error (Acta Numerica); plain L-BFGS was the worst quasi-Newton in its only comparison, so a full-memory BFGS arm is planned; no verified paper covers our five-field (4,2,2,2,2) system.
- Logged internal paper errors found during analysis (micropolar dissipation term inconsistency, cylinder paper solver naming and missing definitions, squeezing best-run headline, swapped figure citations in the overfitting preprint, PINNacle cell disagreements). No parameter values were taken from any paper; the seven missing coefficients stay null.
- Ran a verification pass over the synthesis: all 32 repeated numbers match their analysis files, and the load-bearing ones (optimizer table, PINNacle reweighting cells, annealing rule eq. 40 to 42, NTK weight rule eqs. 7.13 to 7.15, FI-PINN defaults, AL penalty grid, BC comparison rates, beams failure case, Falkner-Skan wall error, overfitting residual figure) were re-checked against the PDFs by text extraction and page images. One count was corrected: PINNacle Table 3 shows NaN on 7 of 22 tasks for L-BFGS, not 8; fixed in the analysis file, the tracker and the synthesis.
- Added short TL;DR records: every analysis.md now opens with a 3 to 4 line TL;DR (what it is, gives us, avoid, relevance). Added docs/literature/papers-tldr.md, the two-minute overview of all 25 papers, and docs/literature/reading-shortlist.md, the tiered list of which papers deserve a full human read.
- No solver code and no training has run. tools/check_records.py passes.

## 2026-11-13 (session: parameter audit)

- Checked configs/physical-case-pending.json against docs/problem/parameters.md: 22 physical coefficients, 15 entered values and 7 null values. Of the entered values, Pr = 6.8 is supplied as fixed; the other 14 are proposed choices.
- Reviewed the recorded limits and the pending B, fw, epsilon and Lb definition questions. No value is established as best by a solver or training run. Left both configs unchanged.
- Counted 8 unset numerical or training fields in configs/training-proposal.json. These are separate from the missing physical values. No solver, reference simulation or training was run.
- Ran `python3 -I tools/check_records.py`: passed the record checks and reported the physical case as blocked by the 7 missing values.
- Mapped all 22 parameters to Eqs. 14 to 19 and the 12 boundary conditions. Every parameter is used at least once; no extra physical parameter is needed. R16, R17, R18 and B5 cannot be evaluated while any of the 7 missing values is null.

## 2026-11-12 (second session: literature pass 2)

- Ran a second literature pass with four parallel searches: closest physical precedents, training methodology, boundary and sampling methods, benchmarks and surveys.
- Re-verified every candidate in-session against the arXiv API or the Crossref DOI registry before it entered docs/literature/reading-list.md. Entries that could not be verified were dropped, not cited.
- Added entries R12 to R48 to docs/literature/reading-list.md with a role map that ties each paper to a project phase or comparison arm (introduction, loss and weights, boundary and sampling choices, honest benchmarking).
- Corrected three records: the often-cited Shin-Darbon-Karniadakis "RAR paper" is actually their convergence paper (arXiv:2004.01806); the Grossmann et al. comparison is against the finite element method, not finite differences; PINNacle is NeurIPS 2024 Datasets and Benchmarks, not 2023.
- Downloaded 21 open-access PDFs (arXiv versions plus publisher open-access copies) into sources/literature/, one directory per paper, with SHA-256 hashes recorded in sources/manifest.json. Entries without a readable full text are marked metadata-only in the reading list.
- Recorded access limits: ScienceDirect, AIP, Nature and PMC PDF endpoints returned bot walls or CAPTCHAs to scripted download attempts. No paywalled content was circumvented.
- No physical parameter values were taken from any paper. The seven missing coefficients stay null and the first case stays blocked. No solver code and no training has run.
- Extended the protected-title list in tools/check_records.py with the PINNacle paper title so the checker does not flag a real title.

## 2026-11-12

- Created this repository locally. Copied the supervisor sheet, the equation draft and the project draft into sources/, with SHA-256 hashes in sources/manifest.json.
- Recorded the 22 parameter names from Eqs. 14 to 19. Chose rounded values where the sheet gives a range. Kept Pr = 6.8 as supplied. Marked lambda_m, epsilon, Nr, Nt, Nb, Lb and Pe as pending. The user said: wait for the supervisor, do not use paper values.
- Corrected the Eq. 14 transcription in docs/problem/equations.md after re-checking the enlarged source. The convective bracket is f*f3 - f1*f2, and the unsteady part sits inside one bracket.
- Checked the page 7 definitions. Two conditional links came out: epsilon = -1/(theta_w-1) if DeltaT = Tf - T_infinity; and Lb = Pr*Le = 6.8 only if Lb is defined with DB. Both need supervisor confirmation.
- Found a conflict: the sheet allows fw = -0.5, but the printed fw = Cw/(1-Cw) would give a negative wall concentration. fw stays provisional.
- Found a second conflict: with delta_star = 0, the printed relations on page 7 give B = 1, not our chosen 0.1. Needs a supervisor decision on whether we follow the relations or pick each value inside its range.
- Wrote the strategy, the test plan and the flowcharts. Set up the repo rules, the status log and the supervisor questions.
- Copied the FYDP paper context (CONTEXT.md, its agent rules, chapter 2 text and the architecture figure) and the previous Lorenz PINN code into prior-work/ as read-only reference. Added the two remaining saved papers to sources/literature/, with hashes.
- Moved the superseded 2026-11-10 methodology into docs/archive/ with a note about its Eq. 14 transcription issue.
- No solver code, no training and no reference simulation has run yet.

## Next

1. Resolve the definition, wall-output and accuracy questions in docs/problem/questions-for-supervisor.md without changing the locked values automatically.
2. Build and test the parameter loader, original residuals and boundary calculations using the confirmed case and separate synthetic fixtures. Numerical settings still need choices and checks.
3. Then build the reference, train and evaluate behind the gates in docs/plans/strategy.md. Refresh the generated guides in a separate update.
