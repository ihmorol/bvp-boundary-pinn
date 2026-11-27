# Step 5 · Guards

[Index](README.md) · page 1, the fixed problem · group 4, solvers · [← step 4](04-one-transcription.md) · [step 6 →](06-first-order-form.md)

**In one line:** checks in the code that stop any run that would change the fixed problem.

## What the guards are
| Guard | Where | Stops |
|---|---|---|
| File fingerprint | `load_case`, [config.py:41](../../../src/bvp_pinn/config.py#L41) | an edited case file |
| Read-only values | same | code that writes into a value |
| `check_locked` | [config.py:63](../../../src/bvp_pinn/config.py#L63), called by the reference and every training run | other values passed in by hand |
| Settings check | `validate`, [pinn/runs.py:55](../../../src/bvp_pinn/pinn/runs.py#L55) | a training setting that names a physical value ([line 62](../../../src/bvp_pinn/pinn/runs.py#L62)), an unknown setting ([line 66](../../../src/bvp_pinn/pinn/runs.py#L66)), another L |
| SymPy test | [tests/test_equations_sympy.py](../../../tests/test_equations_sympy.py) | a typing error in the equations |
| Engine test | `test_engine_solves_the_same_problem`, [tests/test_fixed_problem.py:98](../../../tests/test_fixed_problem.py#L98) | the C++ copy drifting from the Python one |

## Ready?
Yes. All guards are tested in [tests/test_fixed_problem.py](../../../tests/test_fixed_problem.py).
