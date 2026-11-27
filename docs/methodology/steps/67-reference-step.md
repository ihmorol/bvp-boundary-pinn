# Step 67 · Reference

[Index](README.md) · page 6, the main run on Colab · group 18, main run · [← step 66](66-tests.md) · [step 68 →](68-dry-run.md)

**In one line:** build the classical reference from scratch on the Colab machine, check it, and keep a copy in Drive.

## What it does
1. `python -m bvp_pinn.reference.run`: steps 6 to 17 (scan, ladders, checks, export).
2. Build the C++ engine, run its mesh study and its branch search (steps 18 and 19).
3. `tools/reference_error_bars.py` and `tools/l_choice.py` (steps 20 and 21).
4. Stop unless branch A is a valid target (L = 20, gate G3, m > 0); print the branch counts and the L change.
5. Copy `reference/` and `engine/out/` into the Drive folder. On a later session the step is skipped and the copy is put back, so evaluation uses the same reference.

About 6 minutes here. On 2026-11-26 the rebuilt reference was identical to the committed one.

## In the code
- `step_reference` ([tools/main_run.py:214](../../../tools/main_run.py#L214)), `check_reference` ([line 228](../../../tools/main_run.py#L228)), `restore_reference` ([line 245](../../../tools/main_run.py#L245)).

## Ready?
Yes.
