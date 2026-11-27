# Step 11 · Far decay

[Index](README.md) · page 2, the classical reference · group 5, solve the reference · [← step 10](10-domain-ladder.md) · [step 12 →](12-conditions-check.md)

**In one line:** check that the answer's θ and φ really go to zero at infinity.

## What it does
- Far out, f is almost constant, f(L). Then Eq. 16 says θ behaves like C q^(−m) with the decay power below. θ goes to zero only if m > 0.
- Branch A: m = 0.239, decays. Branch B: m = −0.006, does not decay, so B cannot be the answer to "θ → 0 as η → ∞". A is the target.

## The equation
```text
m = Pr κ f(L)/(1 + N_r A_L³) = 1.76 f(L)/(1 + 0.5 A_L³),   A_L = 1 − 0.8 θ(L)
```

## In the code
- [reference/run.py:160](../../../src/bvp_pinn/reference/run.py#L160) computes m; `far_field_valid` at line 161 is m > 0.
- The same formula is the far condition B10 ([conditions.md](conditions.md)) and the PINN's self check ([step 45](45-self-check.md)).

## Ready?
Yes.
