# C++ reference engine

A small, separate solver for Eqs. 14 to 19, for checking the PINN and for plots. One file, standard library only. It is a second independent check next to `solve_bvp` and the Runge-Kutta shooting in `src/bvp_pinn/reference/`. Like them, it never feeds training (AGENTS.md section 3).

## Build and run

From the repository root:

```bash
g++ -O2 -std=c++17 -o engine/bvp_engine engine/bvp_engine.cpp
```

```bash
engine/bvp_engine
```

```bash
uv run python engine/plot_compare.py runs/<run_id>/pred_eval.npz
```

The engine reads the 22 values from `configs/physical-case-pending.json`, so it always uses the locked case. Optional arguments: `engine/bvp_engine <case.json> <out_dir> <L>`. The plot script also accepts a CSV with columns `eta,f,f1,...,ch1`. With no argument it plots the engine alone.

Error estimate for a PINN, without the reference (added 2026-11-23):

```bash
engine/bvp_engine --estimate <pinn.csv> <out.csv>
```

`pinn.csv` has a header and the columns eta and the 12 states, on any mesh from 0 to L. The engine builds the box-scheme residual and Jacobian at that state and takes one Newton step. The step is the estimated error (PINN minus solution), written per node to `out.csv`. `tools/pinn_error_estimate.py` does this for a run folder. Self-test (`--self-test` in that tool): a known theta change of 1e-3 added to the reference is recovered to 99.9 % at the wall.

Outputs go to `engine/out/` (not committed, except `engine_convergence.csv`, which `tools/reference_error_bars.py` reads):

| File | What it holds |
|---|---|
| engine_solution.csv | Richardson value on 3201 nodes: eta, the 12 states, the 5 highest derivatives |
| engine_box_finest.csv | Plain box-scheme solution on 6401 nodes |
| engine_convergence.csv | Wall values and change per mesh level |
| compare.png | Profiles, PINN against the engine, and the PINN error |

## Method

| Step | What | Why |
|---|---|---|
| 1 | Write Eqs. 14 to 18 as 12 first-order ODEs `z' = F(eta, z)`: solve `M*H = -r0` for the five highest derivatives | Same exact rewrite as `reference/equations.py`; see docs/reference-solver/first-order-form.md |
| 2 | Keller box scheme: `(z[j+1] - z[j])/h = F(eta_mid, (z[j]+z[j+1])/2)` on each interval | Standard second-order scheme for boundary-layer ODEs |
| 3 | Newton with step halving; Jacobian of F by forward differences | Nonlinear algebraic system, 12 unknowns per node |
| 4 | Banded LU with partial pivoting, bandwidth 17 each side | The conditions are separated (6 at the wall, 6 at L), so the Jacobian is banded and each Newton step costs O(N) |
| 5 | Halve the mesh four times (401 to 6401 nodes); Richardson `(4*fine - coarse)/3` | Gives fourth-order values and the observed order |

Far-field conditions are the same as the reference: B7 to B9 and B12 at eta = L, asymptotic conditions for theta and phi (Lentini and Keller 1980; `equations.far_conditions`).

## Measured on 2026-11-22 (this laptop, one core)

| Check | Result |
|---|---|
| Observed order of the box scheme | 2.00 |
| Wall values, Richardson vs locked `solve_bvp` reference | max difference 2.7e-10 |
| All 12 states on the reference grid (cubic-spline interpolation) | max difference 1.2e-9 |
| Total run time, five mesh levels | 0.48 s, 53 MB |
| Newton steps per level | 5 at the first level from the smooth guess, then 2 to 3 |

The start is the smooth guess with rate 1 and theta(0) = 0.3, which reaches branch A in the reference scan. Another start may reach branch B; check theta(0) (A: 0.2502, B: 0.5924 with the asymptotic far conditions at L = 20; 0.7423 was the old zero-at-L = 10 value).

## Sources

- Box scheme: H. B. Keller's box method, as used for boundary-layer equations in the Cebeci and Bradshaw texts. Named from memory; not re-opened on 2026-11-22.
- Richardson extrapolation for a second-order scheme: textbook method; the 2.00 order above is our own measurement.
- Far conditions: Lentini and Keller 1980, SIAM J. Numer. Anal., doi 10.1137/0717049 (checked in the 2026-11-19 session).
