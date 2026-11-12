# Status log

Newest first. One entry per working session. Keep it short and factual.

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
