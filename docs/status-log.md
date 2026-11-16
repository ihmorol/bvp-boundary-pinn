# Status log

Newest first. One entry per working session. Keep it short and factual.

## 2026-11-16 (session: substituted reading sheet and commit of the day's work)

- Wrote docs/simplified-equations.md: Eqs. 14 to 19 and the Eq. 19 boundary errors restated with the locked values substituted, as a reading aid. equations.md stays authoritative and links the sheet. delta_star=0 removes all unsteady terms; kappa=b=0.2 merges h into q; Eqs. 16 and 17 are shown multiplied by 8.8, which does not change their zero sets.
- Corrected the wall radiation constant during the term-by-term check: B4 gives 0.998 = 1 - 0.002 from 1.25*0.0016, not the 0.9984 printed in the earlier chat draft. The sheet records the fix.
- Linked the sheet from README and equations.md and rebuilt strategy-guide.html.
- Analyzed every previously uncommitted change and matched it to the three recorded sessions of this day: the supervisor parameter lock, the fifty-task methodology diagram and the Lorenz-style architecture diagram. No unexplained edits were found. The unreferenced methodology-flowchart-a4.pdf export was included with the methodology group.
- Ran tools/check_records.py, tools/check_methodology.py, tools/check_pinn_architecture.py and git diff --check. Committed the day's work in four groups with plain messages. No push. No solver, reference, training or evaluation run.

## 2026-11-16 (session: Lorenz-style PINN architecture)

- Inspected the live lorenz1960-pinn code and paper architecture source at revision 0c5f9c86c597bd27897632fb917d6d17548eeffc. Its tracked worktree was unchanged; untracked notes were not used as implementation evidence. Wrote docs/lorenz-to-bvp-methodology.md with source links, retained patterns and required BVP replacements.
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
- Created docs/paper-analysis-tracker.md with the status and one-line takeaway per paper. Created docs/literature-synthesis.md as the cross-paper guide: must-know facts, decisions per strategy section for arms B3 to B6 and the AL arm, implementation checklist, writing guide, do-not-import list, internal paper errors, and open choices to log.
- Key cross-paper findings recorded in the synthesis: unbalanced boundary gradients are the core failure mode; fixed scales must come before optimizer choice; small loss does not imply small error (Acta Numerica); plain L-BFGS was the worst quasi-Newton in its only comparison, so a full-memory BFGS arm is planned; no verified paper covers our five-field (4,2,2,2,2) system.
- Logged internal paper errors found during analysis (micropolar dissipation term inconsistency, cylinder paper solver naming and missing definitions, squeezing best-run headline, swapped figure citations in the overfitting preprint, PINNacle cell disagreements). No parameter values were taken from any paper; the seven missing coefficients stay null.
- Ran a verification pass over the synthesis: all 32 repeated numbers match their analysis files, and the load-bearing ones (optimizer table, PINNacle reweighting cells, annealing rule eq. 40 to 42, NTK weight rule eqs. 7.13 to 7.15, FI-PINN defaults, AL penalty grid, BC comparison rates, beams failure case, Falkner-Skan wall error, overfitting residual figure) were re-checked against the PDFs by text extraction and page images. One count was corrected: PINNacle Table 3 shows NaN on 7 of 22 tasks for L-BFGS, not 8; fixed in the analysis file, the tracker and the synthesis.
- Added short TL;DR records: every analysis.md now opens with a 3 to 4 line TL;DR (what it is, gives us, avoid, relevance). Added docs/papers-tldr.md, the two-minute overview of all 25 papers, and docs/reading-shortlist.md, the tiered list of which papers deserve a full human read.
- No solver code and no training has run. tools/check_records.py passes.

## 2026-11-13 (session: parameter audit)

- Checked configs/physical-case-pending.json against docs/parameters.md: 22 physical coefficients, 15 entered values and 7 null values. Of the entered values, Pr = 6.8 is supplied as fixed; the other 14 are proposed choices.
- Reviewed the recorded limits and the pending B, fw, epsilon and Lb definition questions. No value is established as best by a solver or training run. Left both configs unchanged.
- Counted 8 unset numerical or training fields in configs/training-proposal.json. These are separate from the missing physical values. No solver, reference simulation or training was run.
- Ran `python3 -I tools/check_records.py`: passed the record checks and reported the physical case as blocked by the 7 missing values.
- Mapped all 22 parameters to Eqs. 14 to 19 and the 12 boundary conditions. Every parameter is used at least once; no extra physical parameter is needed. R16, R17, R18 and B5 cannot be evaluated while any of the 7 missing values is null.

## 2026-11-12 (second session: literature pass 2)

- Ran a second literature pass with four parallel searches: closest physical precedents, training methodology, boundary and sampling methods, benchmarks and surveys.
- Re-verified every candidate in-session against the arXiv API or the Crossref DOI registry before it entered docs/reading-list.md. Entries that could not be verified were dropped, not cited.
- Added entries R12 to R48 to docs/reading-list.md with a role map that ties each paper to a project phase or comparison arm (introduction, loss and weights, boundary and sampling choices, honest benchmarking).
- Corrected three records: the often-cited Shin-Darbon-Karniadakis "RAR paper" is actually their convergence paper (arXiv:2004.01806); the Grossmann et al. comparison is against the finite element method, not finite differences; PINNacle is NeurIPS 2024 Datasets and Benchmarks, not 2023.
- Downloaded 21 open-access PDFs (arXiv versions plus publisher open-access copies) into sources/literature/, one directory per paper, with SHA-256 hashes recorded in sources/manifest.json. Entries without a readable full text are marked metadata-only in the reading list.
- Recorded access limits: ScienceDirect, AIP, Nature and PMC PDF endpoints returned bot walls or CAPTCHAs to scripted download attempts. No paywalled content was circumvented.
- No physical parameter values were taken from any paper. The seven missing coefficients stay null and the first case stays blocked. No solver code and no training has run.
- Extended the protected-title list in tools/check_records.py with the PINNacle paper title so the checker does not flag a real title.

## 2026-11-12

- Created this repository locally. Copied the supervisor sheet, the equation draft and the project draft into sources/, with SHA-256 hashes in sources/manifest.json.
- Recorded the 22 parameter names from Eqs. 14 to 19. Chose rounded values where the sheet gives a range. Kept Pr = 6.8 as supplied. Marked lambda_m, epsilon, Nr, Nt, Nb, Lb and Pe as pending. The user said: wait for the supervisor, do not use paper values.
- Corrected the Eq. 14 transcription in docs/equations.md after re-checking the enlarged source. The convective bracket is f*f3 - f1*f2, and the unsteady part sits inside one bracket.
- Checked the page 7 definitions. Two conditional links came out: epsilon = -1/(theta_w-1) if DeltaT = Tf - T_infinity; and Lb = Pr*Le = 6.8 only if Lb is defined with DB. Both need supervisor confirmation.
- Found a conflict: the sheet allows fw = -0.5, but the printed fw = Cw/(1-Cw) would give a negative wall concentration. fw stays provisional.
- Found a second conflict: with delta_star = 0, the printed relations on page 7 give B = 1, not our chosen 0.1. Needs a supervisor decision on whether we follow the relations or pick each value inside its range.
- Wrote the strategy, the test plan and the flowcharts. Set up the repo rules, the status log and the supervisor questions.
- Copied the FYDP paper context (CONTEXT.md, its agent rules, chapter 2 text and the architecture figure) and the previous Lorenz PINN code into prior-work/ as read-only reference. Added the two remaining saved papers to sources/literature/, with hashes.
- Moved the superseded 2026-11-10 methodology into docs/archive/ with a note about its Eq. 14 transcription issue.
- No solver code, no training and no reference simulation has run yet.

## Next

1. Resolve the definition, wall-output and accuracy questions in docs/questions-for-supervisor.md without changing the locked values automatically.
2. Build and test the parameter loader, original residuals and boundary calculations using the confirmed case and separate synthetic fixtures. Numerical settings still need choices and checks.
3. Then build the reference, train and evaluate behind the gates in docs/strategy.md. Refresh the generated guides in a separate update.
