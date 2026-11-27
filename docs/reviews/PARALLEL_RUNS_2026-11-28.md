# Parallel runs, the session limit and a second Colab account

Date: 2026-11-28. Why: the first Colab session of the main run (free T4 GPU) ended when Colab's daily GPU allowance ran out. B0 seeds 0 to 4 had finished, about 43 minutes each, and B1 seed 0 was 50 minutes into Adam. The GPU used only 2 to 3 GB of its 15 GB. The user asked to continue from a second Google account, to run several jobs at once, and to change nothing in how a run trains.

## 1. What changed, and what did not

Only scheduling, bookkeeping and the notebook changed. The training code, the acceptance rule and the settings are those of the first session: no commit since 2895060 changed a file that decides how a run trains or is judged (`git diff`, checked by `test_later_commits_changed_no_training_code`).

| Change | Where | Why |
|---|---|---|
| Up to `JOBS` runs at once, each in a new process with one thread and its own seed. Auto is 2 on a GPU | [pinn/campaign.py](../../src/bvp_pinn/pinn/campaign.py) `schedule`, `worker`; [tools/main_run.py](../../tools/main_run.py) `pick_device` | Use the idle GPU. A new process per run also frees the GPU memory after each run |
| A run that runs out of GPU memory beside other runs has no result: its failure record is removed and it trains again from its start, alone | `worker`, `forget_out_of_memory` | Without this, a full machine would be recorded as a failure of the method and, by our rule, never re-run |
| `SESSION_HOURS`: a run starts only if it is expected to end before the session limit; otherwise a shorter run that fits starts, or nothing | `schedule`, `run_time_priors`, `deadline` | A free session ends after a few hours. A run cut off at the end is lost work |
| `schedule.json` in every run folder: the most runs that trained beside it | `schedule`, `backfill_schedule`; shown in TX1 (`cost.md`), FX1 and FX2 | Times measured with 2 runs at once are longer than alone. B0 ran alone, so its times must not be compared with shared times unmarked |
| A later session may use a newer commit if the training code is the same as the first session's | `check_same_code`, `method_changes`, `METHOD_PATHS` | The first rule pinned the whole commit, so no fix of scheduling could reach a running campaign |
| The dry run repeats once per device, job count and commit | `step_smoke` | New scheduling code is tried on the GPU for a few minutes before hours of training |
| A usage line every ten minutes: GPU busy share, GPU memory, CPU load | `usage`, `monitor`; `logs/usage.log` | So we see whether the machine is used, instead of guessing |
| Notebook: `MUST_RESUME`, `SESSION_HOURS`, `DISCONNECT_AT_END`, the branch tip by default, a list of the finished runs found | [tools/colab_main_run.ipynb](../../tools/colab_main_run.ipynb) | Safe resume from another account; idle time after the run is not counted |

Evidence that a run trains the same way: `test_a_run_beside_another_trains_exactly_as_alone` trains two dry runs through the campaign with 2 at once and the same runs alone, and every weight is equal, bit for bit (CPU, float64). On the GPU the same code runs; there it was not tested here (no GPU on this machine).

## 2. What it can gain, honestly

The free memory does not mean the GPU is idle. Two other limits decide the speed:

- **CPUs.** A run sends the GPU thousands of very small operations per Adam step, and a CPU has to send each one. Free Colab gives 2 CPUs. Two runs can each have one; a third waits.
- **Float64 on a T4.** The T4 does float64 at about 1/32 of its float32 speed. The Levenberg-Marquardt steps are large float64 matrix work, so two of them at once share the same slow arithmetic and gain little.

So 2 at once should help in the Adam and L-BFGS phases and little in the Levenberg-Marquardt phase. Our guess is that 2 at once finish the work 1.2 to 1.8 times faster than one at a time. This is not measured: there is no GPU here. The first minutes of the next session measure it. Alone, B1 needed about 500 s per 2 000 Adam steps (the screenshot of the first session). With 2 at once, each B1 run printing about 500 s per 2 000 steps would mean a gain near 2; about 1 000 s would mean no gain. The usage lines show the GPU busy share and the CPU load (near 2.0 on 2 CPUs means the CPUs are the limit).

More than 2 at once is not the default: with 2 CPUs the runs would wait for each other. Memory would allow 4: one run needs about 1.6 to 1.9 GB at its heaviest step (measured on CPU, Levenberg-Marquardt step, B0 and B1), about 3 GB on the GPU with the CUDA context.

## 3. Time left

Measured on the T4 in the first session: B0 about 43 minutes per run; B1 Adam about 0.25 s per step.

| Runs left | One run alone on the T4 | All of them alone |
|---|---|---|
| B1, seeds 0 to 4 (seed 0 starts again from zero) | about 2.0 to 2.4 h (Adam 83 min, L-BFGS up to 26 min, Levenberg-Marquardt 13 to 35 min) | 10 to 12 h |
| B2 and B3, seeds 0 to 4 | about 0.5 to 0.7 h (same final problem as B0, earlier stages on fewer points) | 5 to 7 h |
| Total | | 15 to 19 h of GPU time |

With 2 at once and a gain of 1.2 to 1.8, that is about 8 to 16 hours of session time: about three free sessions of 4.5 hours, two if the gain is good, four if it is poor. Then the evaluation, about 15 to 30 minutes.

## 4. Options not taken

| Option | Why not now |
|---|---|
| Save a run mid-way and continue it in the next session | It changes the training code in the middle of the campaign; B0 ran without it. A cut-off run starts again instead, and `SESSION_HOURS` keeps that rare |
| `torch.compile` or CUDA graphs to cut the per-operation cost | Changes the kernels and their rounding, so B1 to B3 would not train the same way as B0 |
| The CPU runtime (the user can get about 70 h) | Possible, but B0 trained on the T4. Accuracy does not depend on the device beyond rounding, but run times do, so the time comparison (FX1, FX2, TX1) would mix devices. The main run warns and lists the device per arm. On Colab's 2 CPUs a B1 run takes about 5 h, so the 15 runs left need about 45 to 60 h. A research decision for the user |
| Two sessions at once on the same Drive folder | The main run assumes one session per folder. Two sessions with disjoint `ARMS` would mostly work, but their logs could clash in Drive. Not tested |

## 5. Checked, and not checked

Checked on this machine (CPU):

- The six tests of [tests/test_campaign.py](../../tests/test_campaign.py) pass: a run beside another equals the same run alone, bit for bit; the slots stay full in order; the session limit starts only runs that fit; a run out of memory beside others trains again alone and waits for an empty machine; the failure record is removed and marked; no commit since 2895060 changed training code.
- Every changed file compiles, every notebook cell compiles, the record check passes, and `git diff` against e04e00d shows no change in `METHOD_PATHS`.

Not checked here, on the user's request (run only on Colab): the whole test suite after the change, and a local rehearsal of the resume (old commit cut off during B1, then the new code with 2 at once). Both were started and stopped when the session ended. So on Colab:

- The dry run (step 68) runs again for this commit with 2 at once, on the GPU, before any real run. It is the first test of the new scheduling on a GPU and of the evaluation with `schedule.json`.
- The tests step does not run again (its marker exists in Drive). Set `RUN_TESTS` and delete `steps/tests.json` to run it; on Colab's 2 CPUs it costs session time.
- Watch the first campaign lines: B0 seeds 0 to 4 listed as finished, B1 to B3 as to run, two B1 runs started, and their Adam lines.
