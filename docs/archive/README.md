# docs/archive

Superseded documents kept for history. Do not use them as the working specification. Links inside them may point at files that moved.

## bvp-pinn-methodology-2026-11-10.md

The first full methodology, written in the Lorenz repository on 2026-11-10. It is superseded by docs/plans/strategy.md and docs/problem/equations.md in this repository.

Known issue: its Eq. 14 transcription misplaced part of the unsteady term and guessed the convective bracket reading. The corrected transcription is in docs/problem/equations.md, checked against the enlarged source page on 2026-11-12.

## fifty-task-methodology-2026-11-16.md

The fifty-task methodology written on 2026-11-16. Replaced the next day by docs/methodology/methodology.md and the redrawn one-page diagrams, because the old figures held too much text to read. Its formulas were not found wrong, but the working sources are docs/plans/strategy.md and docs/problem/equations.md. Some links inside it point at files that moved or were removed.

## strategy-2026-11-16.md

The pre-implementation PINN strategy (formerly docs/plans/strategy.md). Archived on 2026-11-24 because it no longer matched the code or the rules: a 64-wide network, soft boundary penalties, arms B0 to B6, a debug case with other parameter values (against AGENTS.md section 3), and "nothing has run". The current gates are in docs/plans/strategy.md; the method is in docs/pinn/PINN_PLAN.md and docs/pinn/IMPLEMENTATION_NOTES.md.

## training-proposal-2026-11-16.json

The numerical proposal that went with that strategy (formerly configs/training-proposal.json), with eight null settings. Replaced by configs/pinn.json, which holds the settings the code runs, the acceptance rule and the campaign. tools/check_records.py now checks configs/pinn.json instead.

## lorenz-to-bvp-methodology-2026-11-16.md

The source comparison of the Lorenz PINN code with the planned BVP design (formerly docs/methodology/). Archived on 2026-11-24 because the planned design it compared (soft conditions, nested autograd, a 12 933-weight network) changed before any run. What the BVP code took from the Lorenz code is in docs/pinn/IMPLEMENTATION_NOTES.md section 3; the changes from the Lorenz loop are the bottom row of the architecture diagram.
