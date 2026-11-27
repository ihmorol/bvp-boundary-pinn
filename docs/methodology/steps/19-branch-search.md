# Step 19 · Branch search

[Index](README.md) · page 2, the classical reference · group 7, cross-checks · [← step 18](18-box-scheme.md) · [step 20 →](20-error-bars.md)

**In one line:** a stronger search for other solutions, which finds only A and B.

## What it does
- **Deflation:** once a solution r is known, Newton is run on a changed problem G(z) = M(z) R(z), with M(z) = Π_i (‖z − r_i‖⁻² + 1). M is huge near every known solution, so Newton cannot settle there again; it either finds a new solution or fails. No equation or value changes.
- 52 starts (4 rates × 13 start temperatures from −0.5 to 2.5), 105 Newton runs in all. Each new root is confirmed on a twice finer mesh.
- Result: only branches A and B.

## In the code
- `deflate` ([engine/bvp_engine.cpp:414](../../../engine/bvp_engine.cpp#L414)), `deflated_newton` ([line 334](../../../engine/bvp_engine.cpp#L334)), `confirm` ([line 384](../../../engine/bvp_engine.cpp#L384)). Run as `engine/bvp_engine --deflate`; writes `engine/out/engine_branches.csv`.

## Ready?
Yes.
