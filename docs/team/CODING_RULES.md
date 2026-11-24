# Coding rules

Date: 2026-11-19. These rules add to [AGENTS.md](../../AGENTS.md), which always wins: writing rule, honesty rules, research rules and gates. Items marked **proposed** are new suggestions to confirm; the rest come from AGENTS.md or [PINN_PLAN.md](../pinn/PINN_PLAN.md).

## 1. Repository layout

| Path | Rule |
|---|---|
| `src/bvp_pinn/` | All BVP code. Rewrite useful Lorenz patterns; never import from `prior-work/` ([prior-work/README.md](../../prior-work/README.md)) |
| `src/bvp_pinn/config.py` | Loads the locked case (with its SHA-256) and the reference settings |
| `src/bvp_pinn/reference/` | NumPy equations, solve_bvp driver, checks, data I/O, run script, plots ([SPEC_reference_solutions.md](../reference-solver/SPEC_reference_solutions.md)) |
| `src/bvp_pinn/pinn/` | The PINN, one file per job ([IMPLEMENTATION_NOTES.md](../pinn/IMPLEMENTATION_NOTES.md) section 1). It calls the same `reference/equations.py` functions on tensors instead of a second transcription; the SymPy test checks those functions |
| `tests/` | pytest tests, one file per topic |
| `configs/` | JSON configs. `physical-case-pending.json` is locked; never edit it, not even its notes: its SHA-256 is part of the reference lock |
| `reference/<case_id>/` | Accepted reference data (SPEC section 6) |
| `runs/<run_id>/`, `results/` | PINN run folders and summaries (format below) |
| `tools/` | Record checks, builders, the step timer, the manufactured-solution check, the PINN error estimate, the reference error bars |
| `sources/` | Data only. Never run code from it |
| `docs/` | Plans and records, one folder per topic, indexed in [docs/README.md](../README.md). `docs/status-log.md` is updated at the end of each session. No loose docs in the repository root |
| `engine/` | The C++ box-scheme check solver and its error-estimate mode. Standard library only |

## 2. Language and library versions

- **Python 3.12.** The Lorenz project's environment runs 3.12.13. The system `python3` on our main laptop has no NumPy, so always work inside the project environment.
- **Environment manager: uv** (0.10.9 installed). `pyproject.toml` and `uv.lock` exist; run `uv sync` once, then `uv run pytest` and `uv run python -m bvp_pinn.reference.run`. Commit `uv.lock` so every member installs the same versions.
- **Versions seen in the Lorenz environment on 2026-11-19:**

  | Library | Version |
  |---|---|
  | NumPy | 2.5.3 |
  | SciPy | 1.18.1 |
  | PyTorch | 2.14.0+cpu |
  | Matplotlib | 3.11.2 |

  This repo's environment on 2026-11-19: NumPy 2.5.5, SciPy 1.18.1, Matplotlib 3.11.2, SymPy 1.14.0, pytest 9.2.0. PyTorch is added when the PINN stage starts.
- **SymPy and pytest** are needed for tests. Neither is installed system-wide; add them to the project environment.
- Record the exact versions in every run's `meta.json`.

## 3. Style and formatting

- Code comments, docstrings, messages and docs follow the AGENTS.md writing rule: plain words, short sentences, no banned words, no em dashes.
- **Proposed:** format and lint with `ruff format` and `ruff check` at default settings.
- Use the names from [equations.md](../problem/equations.md): `f1..f4`, `th1`, `ph1`, `ch1` and so on. NN weights are `w`, never `theta`.
- Keep the five residuals and twelve boundary errors as separate named functions or columns, never summed before logging.
- No silent changes to Eqs. 14 to 19. A changed term needs a dated note in equations.md first.

## 4. Configs

- **One physics config, locked:** [configs/physical-case-pending.json](../../configs/physical-case-pending.json). Code reads it and checks its hash; it never writes it.
- **Numerical settings** live in separate JSON configs (start from [configs/training-proposal.json](../../configs/training-proposal.json)). A null setting must be filled with a dated note before a final run.
- **Each run saves its full resolved config** as `config.json`. The run ID includes the first 8 hex characters of that file's SHA-256.

## 5. Seeding and determinism

- **Seeds:** 0 to 4 for every arm, paired across arms ([PINN_PLAN.md](../pinn/PINN_PLAN.md) section 3).
- **Where seeds are set:** NumPy through `numpy.random.default_rng(seed)`, PyTorch through `torch.manual_seed(seed)`. Data sampling uses a seeded generator, never global state.
- **Precision:** float64 everywhere for the BVP.
- **Threads:** `torch.set_num_threads(1)` for campaign runs, with the thread count saved in `meta.json`.
- **Proposed:** call `torch.use_deterministic_algorithms(True)` and record it.
- **Limits:** bit-for-bit repeats across different machines are not promised. Record what was set (AGENTS.md and [strategy.md](../plans/strategy.md) section 13).

## 6. Testing

- **Framework:** pytest. Gate G1 tests must pass before any training run counts:
  - derivatives up to order 4
  - residuals against SymPy
  - R(H) - R(0) = M*H
  - B4 stable form
  - NumPy against PyTorch evaluators
- **Each new loss term gets a test.** For example, the replay loss is zero for unchanged weights, and no gradient reaches stored targets.
- **Before every commit:** run `python3 -I tools/check_records.py`. Its writing check covers the docs.
- **Tests must not need the reference data,** except the loader's own tests.

## 7. Experiment logging

- **Run folder:** `runs/<run_id>/` with `config.json`, `history.csv` (every loss term per step), `metrics.json`, `pred_eval.npz`, the final checkpoint and `meta.json` (git hash, versions, hardware, threads, wall time). Run ID: `bvp-{arm}-s{seed}-{confighash8}`.
- **Summaries:** `results/summary.csv` is rebuilt by one script from run folders. Figures are made only from `summary.csv` and `.npz` files.
- **Failed and diverged runs stay in the tables.** Each gets one line in `results/failures.md`.
- **The reference is read only by evaluation code,** after model selection (gate G5).

## 8. Commits and branches

- **Commit messages are plain and say what changed** (AGENTS.md). End with the attribution lines your tooling requires.
- **Branches:** one branch per TODO phase, named `bvp/<phase>` (for example `bvp/r1-system`). Merge to `main` after review by the other member. **Proposed** naming.
- **Nobody merges their own code** (PINN_PLAN.md section 8).
- **Never commit large generated files.** Reference `.npz` files and checkpoints go to a release or archive, not git, unless the user decides otherwise. **Proposed.**
- **The remote is private** while `sources/` holds supervisor documents ([README.md](../../README.md)).
