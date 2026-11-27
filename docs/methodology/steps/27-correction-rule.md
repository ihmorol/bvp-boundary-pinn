# Step 27 · Correction rule

[Index](README.md) · page 3, the PINN model · group 9, exact conditions · [← step 26](26-start-profile.md) · [step 28 →](28-wall-temperature.md)

**In one line:** add fixed correction shapes to the network output, each times how much the network misses one condition. Then the condition holds for any weights.

## The idea with one condition
Say u(0) must be 1. Take the network N(η) and one shape ψ(η) = e^(−η/0.5), which is 1 at η = 0 and dies away from the wall. Build

```text
u(η) = N(η) + ψ(η) × (1 − N(0))
u(0) = N(0) + 1 × (1 − N(0)) = 1      for any network N
```
The bracket (1 − N(0)) is the **miss**: target minus what the network gives.

## Several conditions on one field
A field can carry up to four conditions (two at each end). Each needs its own shape, and the shapes must not spoil each other's conditions. So:

```text
u = N + Σ_j ψ_j(η) (t_j − C_j[N])          t_j: target,  C_j: the rule that reads condition j
choose ψ_j so that  C_i[ψ_j] = 1 if i = j, else 0
```
Then C_i[u] = C_i[N] + (t_i − C_i[N]) = t_i for every i at once.

How the ψ_j are made: start from simple supports s = dⁿ e^(−d/0.5), with d = η for the wall and d = 20 − η for the far end, and n = 0, 1, 2 for the first, second, third condition at that end. Fill the small matrix M_ij = C_i[s_j] and take ψ_j = Σ_i s_i (M⁻¹)_ij. The decay 0.5 makes a wall shape 6 × 10⁻⁶ at η = 6, so corrections stay near their own end.

**Kept values.** For θ and φ the layer adds a row "value at the end = the network's own value" (miss 0). That forces the other shapes to have zero value there, so θ(0), θ(20), φ(0), φ(20) stay whatever the network gives. The nonlinear conditions need that ([step 28](28-wall-temperature.md)).

## The rows of each field
| Field | At η = 0 | At η = 20 |
|---|---|---|
| f | B2: f(0); B1: 1.5f′(0) − 0.1f″(0) | B7: f′(20); B8: f″(20) |
| g | B3: g(0) | B9: g(20) |
| θ | keep θ(0); B4: θ′(0) | keep θ(20); B10: θ′(20) |
| φ | keep φ(0); B5: φ′(0) | keep φ(20); B11: φ′(20) |
| χ | B6: χ(0) | B12: χ(20) |

Nothing else is fixed: f″(0), f(20), f‴(20) and the kept values are free, as the source requires.

## In the code
1. `Constrained` ([pinn/constraints.py:23](../../../src/bvp_pinn/pinn/constraints.py#L23)) wraps a network. `functionals` ([line 33](../../../src/bvp_pinn/pinn/constraints.py#L33)) lists the rows above.
2. `Field.__init__` ([line 106](../../../src/bvp_pinn/pinn/constraints.py#L106)) builds the supports, the matrix M from their exact derivatives, its inverse ([line 116](../../../src/bvp_pinn/pinn/constraints.py#L116)), and the values and first two derivatives of each ψ_j at both ends ([line 118](../../../src/bvp_pinn/pinn/constraints.py#L118)). Done once, in float64.
3. Every forward pass: read the raw network at the two ends, compute the misses in build order (`misses`, [line 63](../../../src/bvp_pinn/pinn/constraints.py#L63); steps 28 to 32), add the shapes (`Field.shape`, [line 139](../../../src/bvp_pinn/pinn/constraints.py#L139), or the jets version `constrained_jets`, [pinn/jets.py:100](../../../src/bvp_pinn/pinn/jets.py#L100)).

## Checked by
`test_built_in_conditions_hold_for_random_weights` and `test_every_mode_imposes_all_twelve_conditions` in [tests](../../../tests/): all 12 conditions below 10⁻¹² for random weights, one network and window networks.

## Ready?
Yes. Design and sources: [BC_ENFORCEMENT_REVIEW.md](../../pinn/BC_ENFORCEMENT_REVIEW.md) section 7.
