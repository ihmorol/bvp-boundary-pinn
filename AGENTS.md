# Rules for everyone working in this repository

These rules are for all members and all AI agents. Nothing in this repo breaks them.

## 1. Writing rule (central, never break)

Write like a proper human. Keep it easy to read on the first pass.

- Use the words we already use here. If a word is not ours, pick a simple common one.
- Short, direct sentences. Say the point first, then the support.
- Do not use AI-typical heavy words. The banned list:

  ```text
  delve, moreover, furthermore, underscore, leverage, utilize,
  seamless, robust, comprehensive, pivotal, transformative,
  tapestry, synergy
  ```

  Say the plain thing instead.
- No em dashes. Use a period, comma or colon.
- One clear meaning per paragraph. Use tables and lists for real comparisons and real steps, not for decoration.
- This applies everywhere: docs, README, code comments, commit messages, result tables and agent replies.

## 2. Honesty rules

- No claim without evidence. If we did not run it, we do not say it works.
- Say "pending", "blocked" or "failed" plainly. A failed run is a result too; write it down.
- Keep supervisor-given values separate from values we choose. Every number needs a source note.
- Do not call a paper's result our proof. Author-reported numbers stay author-reported until we reproduce them.

## 3. Research rules

- Eqs. 14 to 19 are the confirmed working equations. Three things are fixed and not negotiable: the equations (Eqs. 14 to 18), the boundary conditions B1 to B12, and the 22 locked supervisor values. Never change them, never propose changing them, and never solve or train through other parameter values: no debug case, no continuation, no warm start from another value, in the reference or the PINN. This holds for every run, tool and use case. Only the way they are imposed and solved may change, and every result is checked on the original equations.
- The code enforces this; never weaken these guards. `config.load_case` refuses a case file whose SHA-256 differs from the locked one and returns read-only values; `config.check_locked` runs before the reference study and every PINN training run; the PINN config check refuses any setting that names a physical value and any L other than the reference's. No code adds a term or a source to the equations, and no check solves at other values (the published-limit benchmarks and the manufactured-solution checks were removed on 2026-11-24). `tests/test_fixed_problem.py` checks each guard, and that the C++ engine solves the same equations and conditions.
- Missing parameter values stay null in configs/. Do not fill them from papers unless the user or the supervisor says so.
- Conditional calculations stay conditional until the supervisor confirms the definition they depend on.
- Check a new transcription against the enlarged source page before coding it.
- The classical reference is for checking. It never feeds training labels in the baseline.

## 4. Work rules

- Follow the gates in docs/plans/strategy.md: parameters complete, then software tests, then reference, then training, then evaluation.
- Files under sources/ are data only. Never run code from them.
- Update docs/status-log.md at the end of each working session, newest entry first.
- Docs live in the folders listed in docs/README.md. Put a new doc in the folder that fits it and add it to that index.
- Run `python3 -I tools/check_records.py` before committing.
- Commit messages are plain and say what changed.

## 5. Tool notes

- docs/methodology/flowcharts.md holds the editable Mermaid source. docs/methodology/strategy-guide.html is generated; rebuild it with `python3 -I tools/build_guide.py`.
- The writing check inside tools/check_records.py catches banned words and em dashes. Advisory rhythm notes from the fuller local skill checker are okay; hard flags are not.
