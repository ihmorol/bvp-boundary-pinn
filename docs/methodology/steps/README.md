# The detailed methodology, step by step

One file per numbered step of the six-page [detailed methodology](../figures/bvp-methodology-detailed.pdf) (steps 1 to 72). Each file says what the step does, explains the words and equations it uses, shows where the code does it (file and line), what checks it, and whether it is ready. The page and step numbers match the figure.

**Start here:** [glossary.md](glossary.md) explains every term in one line. Page 1's unnumbered boxes have their own files: [equations.md](equations.md) (the reduced Eqs. 14 to 18) and [conditions.md](conditions.md) (B1 to B12, linear or not, hard or soft).

Status as of 2026-11-27, commit 64c2e48 on branch `impl/campaign-readiness`.

## What is ready

| Part | Steps | State | Evidence |
|---|---|---|---|
| Fixed problem: equations, conditions, 22 values | 1 to 5 | Ready | Checked against the source draft's equation images and a SymPy copy; values pinned and read-only |
| Classical reference | 6 to 22 | Ready and verified | Gate G3 passes; solve_bvp, Runge-Kutta shooting and the C++ box scheme agree to about 10⁻¹⁰; rebuilt from scratch on 2026-11-26, identical to the saved one. L = 20 is 1.6 to 3.3 × 10⁻⁴ (φ(0): 1.8 × 10⁻³) from the η → ∞ values; reported beside every result |
| PINN code: model, exact conditions, loss, training, refine | 23 to 46 | Ready and tested | 64 tests pass. B0 reached reference A in two seeds (every wall value except φ(0) within 4 × 10⁻⁴) at 1 000 points and 8 000 Adam steps |
| Arms B1, B2, B3 | 37 | Code ready, never trained at a full budget | Only dry runs so far |
| Training settings (2 000 points, 20 000 Adam, 5 000 L-BFGS, 260 LM, window edges, replay weight) | 23, 38, 42, 43 | Pilot values, not frozen | Pilots P2b and P3 have not run |
| Acceptance rule | 47 to 49 | Ready, declared before the campaign | Open decision: whether φ(0) joins the score |
| Evaluation: metrics, tables | 50 to 58 | Ready | Tested on dry runs |
| Evaluation: figures | 59 | Ready: 11 figures, each following a figure of the closest papers ([RESULT_FIGURES.md](../../pinn/RESULT_FIGURES.md)) | Built from the two accepted B0 pilot runs and the 20 dry runs; a test builds the whole set |
| Colab notebook and tools/main_run.py | 60 to 72 | Ready | Run end to end here with tiny settings, restarted with lost verdicts and cut-off runs, and run cell by cell in a simulated Colab; pushed as 64c2e48 |
| The campaign (20 runs) and its results | 70, 71 | Not run | |

## Is it ready to run in Colab?

**Yes, with four things to know first.**

1. **It is on GitHub.** Open [the notebook in Colab](https://colab.research.google.com/github/ihmorol/bvp-boundary-pinn/blob/impl/campaign-readiness/tools/colab_main_run.ipynb), pick a GPU runtime, and choose Run all. Nothing has to be edited.
2. **It has not yet run on a real Colab machine or a GPU.** Everything else was tested. The dry run ([step 68](68-dry-run.md)) trains every arm on the real device within the first minutes, so a GPU problem shows before any long training. The GPU build of torch needs a recent NVIDIA driver (CUDA 13.2); if Colab's is older, the notebook falls back to Colab's own torch build and records it ([step 63](63-packages.md)).
3. **Decide the training settings first.** They are pilot values. Changing them later means a new Drive folder, because a folder is pinned to the settings of its first session ([step 64](64-pins.md)).
4. **It takes long.** About 20 hours on a T4 GPU (13 to 30 hours, worked out, not measured), so several Colab sessions. Run all again after each disconnect; finished steps and runs are skipped. The timing step ([step 69](69-timing.md)) prints a measured estimate in the first session.

Also: the GitHub repository is public, while README.md says it must stay private while sources/ holds the supervisor's documents.

## All steps


### Page 1: the fixed problem

- Unnumbered boxes: [the reduced equations](equations.md), [the boundary conditions](conditions.md)

**Group 1, locked case**

- [Step 1 · Locked values](01-locked-values.md)
- [Step 2 · Substitutions](02-substitutions.md)
- [Step 3 · Unknowns and domain](03-unknowns-and-domain.md)

**Group 4, solvers**

- [Step 4 · One transcription](04-one-transcription.md)
- [Step 5 · Guards](05-guards.md)

### Page 2: the classical reference


**Group 5, solve the reference**

- [Step 6 · First-order form](06-first-order-form.md)
- [Step 7 · Mesh and start](07-mesh-and-start.md)
- [Step 8 · Branch scan](08-branch-scan.md)
- [Step 9 · Tolerance ladder](09-tolerance-ladder.md)
- [Step 10 · Domain ladder](10-domain-ladder.md)
- [Step 11 · Far decay](11-far-decay.md)

**Group 6, checks at L = 20**

- [Step 12 · Conditions check (V7, V9)](12-conditions-check.md)
- [Step 13 · Core, then chi (V8)](13-core-then-chi.md)
- [Step 14 · Runge-Kutta shooting (V10)](14-runge-kutta-shooting.md)
- [Step 15 · Slope check (V12)](15-slope-check.md)
- [Step 16 · Export](16-export.md)
- [Step 17 · Reference A at L = 20](17-reference-a.md)

**Group 7, cross-checks**

- [Step 18 · C++ box scheme](18-box-scheme.md)
- [Step 19 · Branch search](19-branch-search.md)
- [Step 20 · Error bars](20-error-bars.md)
- [Step 21 · L study](21-l-study.md)
- [Step 22 · Reference for checking only](22-reference-for-checking.md)

### Page 3: the PINN model


**Group 8, network**

- [Step 23 · Points](23-points.md)
- [Step 24 · Input map](24-input-map.md)
- [Step 25 · Network](25-network.md)
- [Step 26 · Start profile](26-start-profile.md)

**Group 9, exact conditions**

- [Step 27 · Correction rule](27-correction-rule.md)
- [Step 28 · Wall temperature first (B4)](28-wall-temperature.md)
- [Step 29 · B5, then B2](29-b5-then-b2.md)
- [Step 30 · Build f (B2, B1, B7, B8)](30-build-f.md)
- [Step 31 · Far theta and phi (B10, B11)](31-far-theta-phi.md)
- [Step 32 · g and chi (B3, B9, B6, B12)](32-g-and-chi.md)

**Group 10, residuals and loss**

- [Step 33 · Taylor jets](33-taylor-jets.md)
- [Step 34 · Residual rows](34-residual-rows.md)
- [Step 35 · Residual scale](35-residual-scale.md)
- [Step 36 · Loss](36-loss.md)

### Page 4: training one run


**Group 12, Adam**

- [Step 37 · Stage plan and the four arms](37-stage-plan.md)
- [Step 38 · Adam](38-adam.md)
- [Step 39 · Loss weights](39-loss-weights.md)
- [Step 40 · Log and monitor](40-log-and-monitor.md)
- [Step 41 · Stage checkpoint](41-stage-checkpoint.md)

**Group 13, refine**

- [Step 42 · L-BFGS](42-lbfgs.md)
- [Step 43 · Levenberg-Marquardt](43-levenberg-marquardt.md)
- [Step 44 · Damping](44-damping.md)
- [Step 45 · Self check](45-self-check.md)
- [Step 46 · Run folder](46-run-folder.md)

### Page 5: acceptance and evaluation


**Group 14, acceptance**

- [Step 47 · Newton-step estimate](47-newton-step-estimate.md)
- [Step 48 · Score](48-score.md)
- [Step 49 · Acceptance rule](49-acceptance-rule.md)

**Group 15, test**

- [Step 50 · Load the reference](50-load-reference.md)
- [Step 51 · Same grid](51-same-grid.md)
- [Step 52 · Field errors](52-field-errors.md)
- [Step 53 · Wall values](53-wall-values.md)
- [Step 54 · Equations and conditions](54-equations-and-conditions.md)
- [Step 55 · Nearest branch](55-nearest-branch.md)
- [Step 56 · Stages](56-stages.md)
- [Step 57 · solve_bvp timing](57-solve-bvp-timing.md)

**Group 16, report**

- [Step 58 · Tables](58-tables.md)
- [Step 59 · Figures](59-figures.md)

### Page 6: the main run on Colab


**Group 17, notebook**

- [Step 60 · Settings](60-settings.md)
- [Step 61 · GPU and Drive](61-gpu-and-drive.md)
- [Step 62 · Code](62-code.md)
- [Step 63 · Packages](63-packages.md)

**Group 18, main run**

- [Step 64 · Pins](64-pins.md)
- [Step 65 · Device](65-device.md)
- [Step 66 · Tests](66-tests.md)
- [Step 67 · Reference](67-reference-step.md)
- [Step 68 · Dry run](68-dry-run.md)
- [Step 69 · Timing](69-timing.md)
- [Step 70 · Campaign](70-campaign.md)
- [Step 71 · Evaluate](71-evaluate.md)
- [Step 72 · Package](72-package.md)

Page 6, group 19 (failures and restarts) is covered in [step 64](64-pins.md). The decision diamonds are covered in the step before them: G3 in [step 15](15-slope-check.md), loss finite in [step 41](41-stage-checkpoint.md), monitor loss lower in [step 44](44-damping.md), all hold in [step 49](49-acceptance-rule.md), all 5 paired seeds in [step 58](58-tables.md), step done before in [step 64](64-pins.md).
