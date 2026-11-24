# TODO: BVP implementation

Ordered task list for the BVP repository, starting from the first reference-solution step.

- Problem and goals: [docs/plans/PRD.md](PRD.md)
- Reference details: [docs/reference-solver/SPEC_reference_solutions.md](../reference-solver/SPEC_reference_solutions.md)
- Code rules: [docs/team/CODING_RULES.md](../team/CODING_RULES.md)
- Arms, gates, figures and dates: [docs/pinn/PINN_PLAN.md](../pinn/PINN_PLAN.md)
- Journal alignment with the Lorenz work (not tasks for this workspace): [docs/journal/README.md](../journal/README.md)

Work the phases in order. A phase is done only when its definition of done holds.

## R0. Environment and skeleton (done 2026-11-19)

- [x] `pyproject.toml` with Python 3.12, NumPy, SciPy, Matplotlib, SymPy and pytest; uv environment and `uv.lock` (PyTorch comes with P1)
- [x] `src/bvp_pinn/` and `tests/`
- [x] Config loader for [configs/physical-case-pending.json](../../configs/physical-case-pending.json) with its SHA-256

**Done when:** `pytest` runs, the loader returns the 22 locked values, and `python3 -I tools/check_records.py` shows no new failures. Met.

## R1. First-order system and boundary function (done 2026-11-19)

- [x] NumPy residuals R14 to R18 and boundary errors B1 to B12 (`src/bvp_pinn/reference/equations.py`)
- [x] Independent SymPy transcription; agreement at random states at the locked values (SPEC V1; altered values removed 2026-11-24)
- [x] Matrix M and the linear solve for H; R(H) - R(0) = M*H (SPEC V2)
- [x] B4 stable form equals the original form
- [x] `rhs` and `boundary` in solve_bvp's shapes
- [ ] Review of every term against equations.md by the other member

**Done when:** all tests pass and the other member has reviewed every term. Tests pass; review pending.

## R2. Manufactured-solution verification (done 2026-11-19, removed 2026-11-24)

Removed on 2026-11-24 with its code and test: it solved a forced version of the equations, and nothing may change Eqs. 14 to 18 (AGENTS.md section 3). The items below are history.

- [x] Manufactured states, forcing and shifted boundary function (SPEC V3)
- [x] Fixed meshes 81 to 641 with the mesh confirmed unchanged
- [x] Observed order: 3.38, 3.83, 3.63

**Done when:** the observed order is at least 3.6. Met.

## R3. First solve at the locked case (done 2026-11-19)

- [x] Clustered mesh, L = 10, tol = 1e-5, 48 scan guesses; all converged, no continuation needed
- [x] Status, boundary errors and smallest diagonal entry recorded

**Done when:** at least one solve has status 0 with all B1 to B12 at or below bc_tol. Met, with two solution branches.

## R4. Convergence and branch study (gate G3, done 2026-11-19)

- [x] Tolerance sequence 1e-5 to 1e-8 (SPEC V4): pass
- [x] L study 10 to 160 with asymptotic far conditions for theta and phi (SPEC V5): wall slopes settle to about 7e-5 per doubling from L = 20; phi(0) about 4e-4; chi has no limit
- [x] Branch scan (SPEC V6): two branches, both exported
- [x] Core plus chi cross-check (SPEC V8): pass
- [x] Second, independent reference by Runge-Kutta multiple shooting (SPEC V10): agrees with solve_bvp to 3e-10 (branch A)
- [x] Table TB1: [reference/study.md](../../reference/study.md)
- [x] Known-answer checks (old SPEC V11, removed 2026-11-24 because they solved other parameter values); saved-curve slope check (SPEC V12); figures redrawn on a linear eta axis ([reference/figures/captions.md](../../reference/figures/captions.md))
- [ ] Supervisor answers: questions 7 to 10 (target branch, far-field treatment, chi at infinity, f bounded at infinity)
- [x] Reference audit fixes D1 to D7 (2026-11-24): [REFERENCE_AUDIT](../reviews/REFERENCE_AUDIT_2026-11-23.md), follow-up section

**Done when:** SPEC V4 to V8 pass and L is fixed with a dated note. Met for the fixed-L problem (L = 20 with asymptotic far conditions, [configs/reference.json](../../configs/reference.json)). The strict far-field check fails because of phi(0) and chi, see SPEC section 8.

## R5. Reference data export (done 2026-11-19)

- [x] `reference/locked-branchA-L20-tol1e-08/` (target) and `reference/locked-branchB-L20-tol1e-08/` (flagged: no decay at infinity), each with `reference.npz`, `meta.json`, `SHA256SUMS` and figures
- [x] `load_reference` with hash and physical-case checks, and its tests
- [x] Session logged in [docs/status-log.md](../status-log.md)

**Done when:** a fresh process reloads the reference through the loader and the hash check fails on a tampered copy. Met (`tests/test_reference_verify.py`; the plots reload through the loader).

## P1. PINN building blocks (gate G1, done 2026-11-21)

- [x] PyTorch residuals and boundary errors, by calling the same `reference/equations.py` functions; test against NumPy at identical states
- [x] Derivatives up to order 4 for f and order 2 for the others (Taylor-mode jets since 2026-11-22, equal to nested autograd); test on known functions
- [x] Network: input eta only (scaled inside the model), five outputs, tanh, float64
- [x] Boundary layer (TFC) with soft, linear and exact modes ([docs/pinn/BC_ENFORCEMENT_REVIEW.md](../pinn/BC_ENFORCEMENT_REVIEW.md)); test that its conditions hold for random weights
- [x] Gradient-norm loss weights (Wang et al. 2023); each of the 5 equation terms and 12 boundary errors logged separately
- [ ] Review of every PINN file by the other member

Details: [docs/pinn/IMPLEMENTATION_NOTES.md](../pinn/IMPLEMENTATION_NOTES.md). Issue #2.

**Done when:** all G1 tests pass, and a gate G5 review confirms the training code does not import the reference loader.

## P2. Arm B0 pilot (basic single network)

- [x] Wall-clustered collocation points; Adam, then refine stages with rollback
- [x] P2a (2026-11-22): boundary mode pilot by the rule in IMPLEMENTATION_NOTES section 5.1; exact mode frozen (issue #3)
- [x] B0 fix ladder at seed 0 (2026-11-23): exact mode, log map, start profile, residual scale, BFGS and L-BFGS refine. [IMPLEMENTATION_NOTES](../pinn/IMPLEMENTATION_NOTES.md) section 7.1
- [x] Checks without the reference: self check (decay powers m, mp, tail fit), Newton-step error estimate
- [x] B0 to the reference at seed 0 (2026-11-23): balanced scale and per-point Levenberg-Marquardt; run X3. Section 7.2
- [x] Full pipeline from scratch, seeds 0 and 1 (2026-11-23), 1 000 points: every wall value except phi(0) within 4e-4 relative; phi(0) 4.3e-4 and 6.2e-3. Sections 7.3 and 7.4
- [x] Acceptance rule declared and tested (2026-11-24); on the 13 pilot runs it accepts the three that match the reference. Section 7.4
- [ ] Decide before the campaign whether phi(0) joins the acceptance score (section 7.4; it would reject seed 1 and X3)
- [ ] P2b: one B0 run at 2 000 points with the campaign settings; freeze `n_points`, `adam_steps`, `lbfgs_iters` and `lm_iters` with dated notes (issue #4). LM cost is measured: 24 s per step at 2 000 points
- [ ] Optional, after P2b: compare `best_checkpoint` and `resample_every` (both off now)

**Done when:** B0 trains end to end and writes a complete run folder (CODING_RULES section 7), and the settings are frozen. Met for the first part; the freeze is pending.

## P3. Arms B1 to B3 and pilots

- [x] B1, B2, B3 in code (2026-11-21), with the same refine chain as B0 since 2026-11-24 (block Levenberg-Marquardt for B1's window networks); unit tests for the interface, replay and LM losses; every arm runs end to end in a test
- [ ] Pilots at seed 0 (issue #5): window edges; replay weight from 0.1, 1, 10 by training residual only; B1 interface behaviour and per-window errors (FB5)
- [ ] Freeze the window edges, budgets and replay weight in configs/pinn.json with dated notes

**Done when:** all four arms run end to end (met), and the window edges, budgets and weights are frozen in configs.

## P4. Campaign

- [x] Campaign runner (2026-11-24): `python -m bvp_pinn.pinn.campaign`, tagged runs, resumable, `--jobs`; a dry campaign of all 20 runs passed
- [ ] Colab: time one short run on CPU and on GPU in float64; check the image's Python against the pin (issue #6)
- [ ] B0 to B3 with seeds 0 to 4 (20 runs), one thread each
- [ ] Failed and rejected runs kept in every table, one line each in `results/P4/failures.md` (written by `evaluate`)

**Done when:** all 20 run folders are complete and `results/P4/summary.csv` rebuilds from them.

## P5. Evaluation, figures and understanding

- [x] Code (2026-11-24): metrics in PINN_PLAN section 4; FB1 to FB5, FX1, TB2, TX1; failures.md; solve_bvp timing. Tested on the dry campaign
- [ ] On the campaign: `bvp_pinn.evaluation.evaluate`, then `.classical` (5 repeats, one thread, the machine the runs used), then `.plots`
- [ ] Answer the questions in PINN_PLAN section 7, each with figure and run IDs

**Done when:** the results memo is written and every number traces to a run ID.
