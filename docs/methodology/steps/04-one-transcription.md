# Step 4 · One transcription

[Index](README.md) · page 1, the fixed problem · group 4, solvers · [← step 3](03-unknowns-and-domain.md) · [step 5 →](05-guards.md)

**In one line:** Eqs. 14 to 18 and B1 to B12 are written once, and both the reference and the PINN call that one copy.

## What it does
- One copy cannot disagree with itself, so the reference and the PINN are guaranteed to solve the same equations with the same values and the same state order.
- The functions use only plain arithmetic (+, −, ×, /, powers), which works the same on NumPy arrays (reference) and on PyTorch tensors (PINN).

## Words used here
| Term | Meaning |
|---|---|
| Transcription | Writing the printed equations as code |
| State order | z = [f, f′, f″, f‴, g, g′, θ, θ′, φ, φ′, χ, χ′], H = [f⁽⁴⁾, g″, θ″, φ″, χ″], the same everywhere |

## In the code
- `residual_terms` ([reference/equations.py:44](../../../src/bvp_pinn/reference/equations.py#L44)) returns R14 to R18. `boundary_terms` ([line 204](../../../src/bvp_pinn/reference/equations.py#L204)) returns B1 to B12.
- State order: `STATE_NAMES`, `HIGH_NAMES` at [line 18](../../../src/bvp_pinn/reference/equations.py#L18).
- The PINN side: `physics.residuals` ([pinn/physics.py:15](../../../src/bvp_pinn/pinn/physics.py#L15)) and `physics.boundary` ([line 27](../../../src/bvp_pinn/pinn/physics.py#L27)) call the same two functions.
- The C++ engine keeps its own copy on purpose, as an independent check ([step 18](18-box-scheme.md)).

## Checked by
- [tests/test_equations_sympy.py](../../../tests/test_equations_sympy.py): a second, separate transcription in SymPy gives the same numbers.
- `test_torch_equations_match_numpy` in [tests/test_pinn.py](../../../tests/test_pinn.py): NumPy and PyTorch give the same residuals.
- On 2026-11-26 the equations were also compared by eye with the source draft's equation images.

## Ready?
Yes.
