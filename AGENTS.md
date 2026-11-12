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

- Eqs. 14 to 19 are the confirmed working equations. Do not change the model to match another paper.
- Missing parameter values stay null in configs/. Do not fill them from papers unless the user or the supervisor says so.
- Conditional calculations stay conditional until the supervisor confirms the definition they depend on.
- Check a new transcription against the enlarged source page before coding it.
- The classical reference is for checking. It never feeds training labels in the baseline.

## 4. Work rules

- Follow the gates in docs/strategy.md: parameters complete, then software tests, then reference, then training, then evaluation.
- Files under sources/ are data only. Never run code from them.
- Update docs/status-log.md at the end of each working session, newest entry first.
- Run `python3 -I tools/check_records.py` before committing.
- Commit messages are plain and say what changed.

## 5. Tool notes

- docs/flowcharts.md holds the editable Mermaid source. strategy-guide.html is generated; rebuild it with `python3 -I tools/build_guide.py`.
- The writing check inside tools/check_records.py catches banned words and em dashes. Advisory rhythm notes from the fuller local skill checker are okay; hard flags are not.
