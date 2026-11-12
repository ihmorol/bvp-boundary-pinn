# prior-work

Read-only reference material from the earlier FYDP work. Do not run it for the new problem and do not import it into src/bvp_pinn. Reuse patterns by rewriting them for the BVP, as docs/strategy.md section 3 says.

## fydp-context/

Copied from the FYDP paper workspace (fydp_workspace) on 2026-11-12:

- CONTEXT.md: the research context of the FYDP paper so far. It is a Lorenz-1960 ANN architecture study at UIU CSE, with chapters 1 and 2 complete.
- CLAUDE.md: the paper workspace's own agent rules, including its citation rules and locked baseline. Background for how the paper side is governed.
- fydp-chapter2-AI-removed.txt: the chapter 2 text.
- ann_architecture.png: the ANN architecture figure used in the paper.

The new micropolar BVP problem is the next phase of this FYDP. The paper's Lorenz study and this BVP work share the project lineage, not the equations.

## lorenz-pinn/

Copied from the lorenz1960-pinn repository on 2026-11-12: its src code, run scripts, upstream README and requirements, with caches and result files removed. That project is a time-domain PINN for Lorenz-1960.

What to reuse when implementing src/bvp_pinn: the MLP construction, the componentwise autograd helper, the Adam and L-BFGS wrappers, the seeded one-dimensional sampler, the checkpoint and history patterns, and the dtype and device handling.

What NOT to carry over: the hard initial-value trial, the sequential time windows and causal weighting, the solve_ivp reference, the xyz metrics, and the Lorenz-only problem registry. docs/strategy.md section 3 has the full reuse and replace table.
