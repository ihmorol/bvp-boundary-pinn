# Status log

Newest first. One entry per working session. Keep it short and factual.

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

1. Send the questions in docs/questions-for-supervisor.md.
2. When values arrive, update configs/physical-case-pending.json and run tools/check_records.py.
3. Then build and test the reference solver and the PINN step by step, as written in docs/strategy.md.
