# Journal alignment

This folder holds the plans that join two separate works into one journal paper. It sets no BVP tasks. The BVP work for this repository is in [../docs/](../docs/PRD.md) and [../TODO.md](../TODO.md).

## The two works

| Work | Where it is done | Its plan |
|---|---|---|
| Lorenz-1960 PINN (conference paper extension) | Separate repository: ihmorol/lorenz1960-pinn | [Issue 43](https://github.com/ihmorol/lorenz1960-pinn/issues/43) |
| Five-field BVP (reference and PINN) | This repository | [../docs/PRD.md](../docs/PRD.md), [../docs/PINN_PLAN.md](../docs/PINN_PLAN.md), [../TODO.md](../TODO.md) |

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
