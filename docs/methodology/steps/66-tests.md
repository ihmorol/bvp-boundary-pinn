# Step 66 · Tests

[Index](README.md) · page 6, the main run on Colab · group 18, main run · [← step 65](65-device.md) · [step 67 →](67-reference-step.md)

**In one line:** run the whole test suite and the record check before anything else.

## What it does
- `pytest`: 64 tests, about one minute. They cover the equations against SymPy, derivatives to order 4, the exact conditions, the optimizers, the acceptance rule, evaluation, and a tiny end-to-end run of every arm.
- `tools/check_records.py`: names, values, the supervisor lock, links in the docs and the writing rule.
- Runs once per Drive folder.

## In the code
- `step_tests` ([tools/main_run.py:207](../../../tools/main_run.py#L207)); tests in [tests/](../../../tests/); [tools/check_records.py](../../../tools/check_records.py).

## Ready?
Yes. All 64 pass on the pushed commit.
