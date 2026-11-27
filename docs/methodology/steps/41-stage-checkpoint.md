# Step 41 · Stage checkpoint

[Index](README.md) · page 4, training one run · group 12, Adam · [← step 40](40-log-and-monitor.md) · [step 42 →](42-lbfgs.md)

**In one line:** save the weights after each Adam stage; stop the run if the loss ever becomes infinite or NaN.

## What it does
- After each stage, a copy of the weights becomes `stage<k>.pt` (B0 and B1: one, `stage0.pt`, the state after Adam). Evaluation uses them for the after-Adam accuracy and the per-window retention of B2 and B3.
- B3 only: at the start of each stage, the network's outputs on the earlier windows are frozen as the self-replay target.
- **Loss finite?** If not, the run stops at once with result "non-finite loss in Adam stage k". It is saved and judged like any run (it fails the rule) and is never re-run.

## In the code
- Checkpoint: [pinn/train.py:99](../../../src/bvp_pinn/pinn/train.py#L99). Frozen targets: [line 96](../../../src/bvp_pinn/pinn/train.py#L96). Finite check: [line 132](../../../src/bvp_pinn/pinn/train.py#L132).

## Ready?
Yes.
