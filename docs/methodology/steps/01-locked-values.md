# Step 1 · Locked values

[Index](README.md) · page 1, the fixed problem · group 1, locked case · [step 2 →](02-substitutions.md)

**In one line:** the 22 supervisor numbers are read from one file, checked against a fingerprint, and handed out read-only.

## What it does
- Reads `configs/physical-case-pending.json` (the name is historical; the values are complete and locked).
- Refuses the file if a single byte changed. Every result therefore uses exactly the values received on 2026-11-16.
- Gives the values as a read-only mapping, so no code can change one by accident.

## Words used here
| Term | Meaning |
|---|---|
| SHA-256 | A 64-character fingerprint of a file. Change any byte and it changes |
| Read-only mapping | A dictionary that raises an error if code tries to write into it |

## In the code
1. `load_case` in [config.py:41](../../../src/bvp_pinn/config.py#L41) reads the file bytes and computes their SHA-256.
2. It compares the result with `LOCKED_CASE_SHA256` ([config.py:28](../../../src/bvp_pinn/config.py#L28)). A mismatch raises `ValueError` and stops the program.
3. It checks that all 22 names are present and returns them as a `MappingProxyType` (read-only), plus the fingerprint ([config.py:60](../../../src/bvp_pinn/config.py#L60)).
4. `check_locked` ([config.py:63](../../../src/bvp_pinn/config.py#L63)) raises unless a set of values equals the locked one exactly. The reference study calls it first ([reference/run.py:109](../../../src/bvp_pinn/reference/run.py#L109)), and so does every PINN training run ([pinn/train.py:87](../../../src/bvp_pinn/pinn/train.py#L87)).

## Checked by
- `test_case_file_is_pinned`, `test_values_are_read_only_and_checked` in [tests/test_fixed_problem.py](../../../tests/test_fixed_problem.py).
- `test_locked_case_equals_receipt` in [tests/test_config.py](../../../tests/test_config.py): the values equal the supervisor's message, symbol by symbol.

## Ready?
Yes.
