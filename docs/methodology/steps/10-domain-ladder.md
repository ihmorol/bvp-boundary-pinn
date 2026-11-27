# Step 10 · Domain ladder

[Index](README.md) · page 2, the classical reference · group 5, solve the reference · [← step 9](09-tolerance-ladder.md) · [step 11 →](11-far-decay.md)

**In one line:** solve at L = 10, 20, 40, 80, 160 to see how much the answer moves when the far end moves.

## What it does
- Each L starts from the previous answer, stretched to the new length (beyond the old end every state keeps its far-end value).
- **Check V5:** the change of the wall values from L = 20 to L = 40 should be below 10⁻⁴. It is 3.9 × 10⁻⁴, from φ(0), so V5 fails its bar. The user kept L = 20 (2026-11-26); the size of this effect is reported ([step 21](21-l-study.md)).
- A second ladder with the old zero-at-L far conditions (θ(L) = φ(L) = 0) is solved only to draw comparison figures.
- χ′(0) keeps changing with L (−2.01 at 20, −1.08 at 160): Eq. 18 as given has no decaying χ, so χ is defined by χ(20) = 0.

## In the code
- `ladder_L` ([reference/run.py:68](../../../src/bvp_pinn/reference/run.py#L68)) and `extend` ([reference/solver.py:114](../../../src/bvp_pinn/reference/solver.py#L114)).
- Both ladders run at [reference/run.py:133 to 135](../../../src/bvp_pinn/reference/run.py#L133); their profiles go to `reference/l_study.npz` ([line 191](../../../src/bvp_pinn/reference/run.py#L191)).

## Ready?
Yes, with the L error stated beside every result.
