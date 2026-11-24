# Journal alignment

This folder holds the plans that join two separate works into one journal paper. It sets no BVP tasks. The BVP work for this repository is in [../plans/PRD.md](../plans/PRD.md) and [../plans/TODO.md](../plans/TODO.md).

**Note, 2026-11-23:** parameter continuation or warm starts through other kappa, b, Da values are ruled out for the BVP PINN. The equations, boundary conditions and locked values are fixed as given. Read the continuation parts of these plans as history.

**Note, 2026-11-24:** these plans were written on 2026-11-19, before the BVP code existed. For the BVP side, [PINN_PLAN.md](../pinn/PINN_PLAN.md) and [IMPLEMENTATION_NOTES.md](../pinn/IMPLEMENTATION_NOTES.md) are current and win where they differ: all 12 conditions are built into the output (not soft), every arm ends with L-BFGS and Levenberg-Marquardt, runs are accepted by a rule without the reference (not picked by training residual), and results live in `results/<campaign>/`. Links here to the 2026-11-16 strategy and training proposal point to their archived copies.

## The two works

| Work | Where it is done | Its plan |
|---|---|---|
| Lorenz-1960 PINN (conference paper extension) | Separate repository: ihmorol/lorenz1960-pinn | [Issue 43](https://github.com/ihmorol/lorenz1960-pinn/issues/43) |
| Five-field BVP (reference and PINN) | This repository | [PRD.md](../plans/PRD.md), [PINN_PLAN.md](../pinn/PINN_PLAN.md), [TODO.md](../plans/TODO.md) |

Both works test the same two structures:

- **Basic structure:** separate networks per window.
- **Knowledge retention:** one network trained window by window that keeps what it learned on earlier windows.

Both also share seeds, selection rule, metric names and result format. That is what lets their results go into one paper.

## Files here

| File | What it holds |
|---|---|
| [journal-strategy.md](journal-strategy.md) | Target journals and their policies on extended conference papers; editor and reviewer lessons; audit of the conference paper against the BVP plan; paper outline; new-content estimate; compliance checklist |
| [single-network-retention.md](single-network-retention.md) | Literature on one network with knowledge retention and the reasoning for the chosen design. Its BVP two-input design is dropped, as marked inside |
| [implementation-workflow.md](implementation-workflow.md) | Joint two-member timeline across both works, sync points, gates G1 to G6, and the shared figure list |

## How to keep the two apart

- **BVP tasks go in `docs/` and `TODO.md`,** never here.
- **Lorenz tasks go in the Lorenz repository's issue,** never in this repository.
- **Change shared rules (arms logic, seeds, metric names, result format) here first,** then update both works.
