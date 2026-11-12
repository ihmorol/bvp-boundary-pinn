# Five-variable boundary PINN

Local research repository for the FYDP five-function boundary problem: solve Eqs. 14 to 19 with a physics-informed neural network. Separate from the old Lorenz project.

**Status:** planning and setup are done. No solver code, no training and no reference simulation yet. Seven physical values are pending, so the first case is blocked until the supervisor answers.

**Read AGENTS.md first.** Its writing rule and honesty rules apply to every file here and to every agent that works here.

## Where things are

| Path | What it holds |
|---|---|
| AGENTS.md | Central rules: writing style, honesty, research and work rules |
| docs/parameters.md | The 22 parameters, sheet limits, our chosen values, pending items |
| docs/equations.md | Corrected transcription of Eqs. 14 to 19 and the first-order reference setup |
| docs/strategy.md | Architecture, training flow, reference plan, tests, outputs, failure handling |
| docs/reading-list.md | Papers we checked, with what each one does and does not establish |
| docs/flowcharts.md | Editable Mermaid diagrams of the full flow |
| docs/status-log.md | What we did, newest first |
| docs/questions-for-supervisor.md | Open questions to ask the supervisor |
| configs/physical-case-pending.json | Physical case; pending values stay null |
| configs/training-proposal.json | Proposed training settings, not validated choices |
| tools/ | Record checks and the guide builder |
| src/bvp_pinn/ | Code will live here; nothing implemented yet |
| sources/ | Supervisor sheet, drafts and checked papers, with SHA-256 hashes |
| strategy-guide.html | Generated readable guide; rebuild with tools/build_guide.py |

## How we work

1. Pick a task from docs/strategy.md or the status log.
2. Work behind the right gate: parameters complete, then software tests, then reference, then training, then evaluation.
3. Write what happened in docs/status-log.md. Keep it short and factual.
4. Run `python3 -I tools/check_records.py` before committing. It checks names, values, links, source hashes and the writing rule.
5. Commit with a plain message that says what changed.

## GitHub

Private remote: `ihmorol/bvp-boundary-pinn`. The supervisor documents and papers under sources/ are uploaded to that private repository only. Do not make the repository public while sources/ holds them.

## Next input needed

Values or ranges for Nr, Nt, Nb, Lb and Pe, plus the answers in docs/questions-for-supervisor.md. Then the case unblocks and implementation follows docs/strategy.md.
