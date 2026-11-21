# TODO: BVP implementation

Ordered task list for the BVP repository, starting from the first reference-solution step.

- Problem and goals: [docs/PRD.md](docs/PRD.md)
- Reference details: [docs/SPEC_reference_solutions.md](docs/SPEC_reference_solutions.md)
- Code rules: [docs/CODING_RULES.md](docs/CODING_RULES.md)
- Arms, gates, figures and dates: [docs/PINN_PLAN.md](docs/PINN_PLAN.md)
- Journal alignment with the Lorenz work (not tasks for this workspace): [journal/README.md](journal/README.md)

Work the phases in order. A phase is done only when its definition of done holds.

## R0. Environment and skeleton (done 2026-11-19)

- [x] `pyproject.toml` with Python 3.12, NumPy, SciPy, Matplotlib, SymPy and pytest; uv environment and `uv.lock` (PyTorch comes with P1)
- [x] `src/bvp_pinn/` and `tests/`
- [x] Config loader for [configs/physical-case-pending.json](configs/physical-case-pending.json) with its SHA-256

**Done when:** `pytest` runs, the loader returns the 22 locked values, and `python3 -I tools/check_records.py` shows no new failures. Met.

## R1. First-order system and boundary function (done 2026-11-19)

- [x] NumPy residuals R14 to R18 and boundary errors B1 to B12 (`src/bvp_pinn/reference/equations.py`)
- [x] Independent SymPy transcription; agreement at random states with locked and altered parameters (SPEC V1)
- [x] Matrix M and the linear solve for H; R(H) - R(0) = M*H (SPEC V2)
- [x] B4 stable form equals the original form
- [x] `rhs` and `boundary` in solve_bvp's shapes
- [ ] Review of every term against equations.md by the other member

**Done when:** all tests pass and the other member has reviewed every term. Tests pass; review pending.

## R2. Manufactured-solution verification (done 2026-11-19)

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
- [x] Table TB1: [reference/study.md](reference/study.md)
- [x] Known-answer checks (SPEC V11): published curved-sheet table and exact flat-sheet solutions; saved-curve slope check (SPEC V12); figures redrawn on a linear eta axis ([reference/figures/captions.md](reference/figures/captions.md))
- [ ] Supervisor answers: questions 7 to 9 (target branch, far-field treatment, chi at infinity)

**Done when:** SPEC V4 to V8 pass and L is fixed with a dated note. Met for the fixed-L problem (L = 20 with asymptotic far conditions, [configs/reference.json](configs/reference.json)). The strict far-field check fails because of phi(0) and chi, see SPEC section 8.

## R5. Reference data export (done 2026-11-19)

- [x] `reference/locked-branchA-L20-tol1e-08/` (target) and `reference/locked-branchB-L20-tol1e-08/` (flagged: no decay at infinity), each with `reference.npz`, `meta.json`, `SHA256SUMS` and figures
- [x] `load_reference` with hash and physical-case checks, and its tests
- [x] Session logged in [docs/status-log.md](docs/status-log.md)

**Done when:** a fresh process reloads the reference through the loader and the hash check fails on a tampered copy. Met (`tests/test_reference_verify.py`; the plots reload through the loader).

## P1. PINN building blocks (gate G1, done 2026-11-21)

- [x] PyTorch residuals and boundary errors, by calling the same `reference/equations.py` functions; test against NumPy at identical states
- [x] Nested autograd derivatives up to order 4 for f and order 2 for the others; test on known functions
- [x] Network: input eta only (scaled inside the model), five outputs, tanh, float64
- [x] Boundary layer (TFC) with soft, linear and exact modes ([docs/BC_ENFORCEMENT_REVIEW.md](docs/BC_ENFORCEMENT_REVIEW.md)); test that its conditions hold for random weights
- [x] Gradient-norm loss weights (Wang et al. 2023); each of the 5 equation terms and 12 boundary errors logged separately
- [ ] Review of every PINN file by the other member

Details: [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md). Issue #2.

**Done when:** all G1 tests pass, and a gate G5 review confirms the training code does not import the reference loader.

## P2. Arm B0 pilot (basic single network)

- [x] Wall-clustered collocation points; Adam, then L-BFGS with rollback
- [ ] P2a: boundary mode pilot, rule declared in IMPLEMENTATION_NOTES section 5.1 (issue #3)
- [ ] P2b: sizes and budgets in configs/pinn.json, each with a dated note (issue #4)
- [ ] Pilot at seed 0; evaluate against the reference only after the run ends

**Done when:** B0 trains end to end and writes a complete run folder (CODING_RULES section 7). The pilot errors are recorded, whatever they are.

## P3. Arms B1 to B3 and pilots

- [ ] B1: K separate window networks trained together with interface continuity (PINN_PLAN section 2)
- [ ] B2: one network, windows added from the wall outward, physics replay, all 12 boundary conditions every stage
- [ ] B3: as B2 with self-replay of frozen outputs
- [ ] Unit tests for interface and replay losses; pilots at seed 0; replay weight chosen from 0.1, 1, 10 by training residual only

**Done when:** all four arms run end to end, and the window edges, budgets and weights are frozen in configs.

## P4. Campaign

- [ ] B0 to B3 with seeds 0 to 4 (20 runs), `torch.set_num_threads(1)`
- [ ] Failures logged in `results/failures.md` and kept in tables

**Done when:** all 20 run folders are complete and `results/summary.csv` rebuilds from them.

## P5. Evaluation, figures and understanding

- [ ] Metrics in PINN_PLAN section 4; figures and tables FB1 to FB5, TB2
- [ ] Classical comparison: solve_bvp time to reach each arm's median error; repeat 5 times on one thread
- [ ] Answer the questions in PINN_PLAN section 7, each with figure and run IDs

**Done when:** the results memo is written and every number traces to a run ID.
