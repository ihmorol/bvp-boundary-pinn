# Why the PINN is slow, what the equations really need, and what to change

Date: 2026-11-22. A dated study; its timings are of the code at that date. **Outcome (2026-11-24):** jets are the default derivative path; the per-step boundary errors are skipped in exact mode; Levenberg-Marquardt later became fast enough with a per-point Jacobian (24 s per step at 2000 points, [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) section 5.2); the reduced-order rewrite of section 2.3 is not used. Branch: `speedup/taylor-jets-and-reference-engine`. Code read: `main` at `a8cf833` (`src/bvp_pinn/pinn/`, `src/bvp_pinn/reference/equations.py`, `configs/pinn.json`). Timings: this laptop, 4 cores, 7 GB RAM, other programs open, PyTorch 2.15.0+cpu, float64, 1 thread unless stated. Every number below is either measured here (marked **measured**), taken from the pilot `meta.json` files (marked **pilot**), or worked out (marked **estimate** or **inference**).

## 1. Short diagnosis

**What is more complicated than it needs to be.** Not the equations. The code that takes the derivatives is. `model.states` gets f', f'', f''', f'''' and the second derivatives of the other four fields with 12 nested `autograd.grad` calls, one output and one order at a time. Each call works on the graph of the one before, so the cost grows about 3.5 to 4.3 times per derivative order (**measured**, section 3.1). The network has one input, so all of this can be done in a single forward pass instead. That is now done in `src/bvp_pinn/pinn/jets.py`. It gives the same values and the same weight gradients to 1e-14.

**What is necessary.** Every term of Eqs. 14 to 18 is in the supervisor-confirmed equations, and no term is small enough to drop (section 2.2). The derivative orders are what the physics needs. The one real choice is f'''': it comes from removing the pressure from the two momentum equations of a curved sheet. Section 2.3 proves an exact rewrite that puts a pressure-like unknown back, so f needs only f'''. That changes the network (one more output), so it is a proposal for the supervisor, not a default.

**Main runtime bottlenecks, largest first.**

| # | Cause | Where | Share of one step (old code, exact mode, 2000 points) |
|---|---|---|---|
| 1 | Nested reverse-mode derivatives, 12 calls, up to order 4 | `pinn/model.py`, `states` loop | About 85 % (equation forward plus its backward) |
| 2 | Boundary errors computed every step, though exact mode never uses them in the loss | `pinn/train.py`, `loss_parts`, line `bc = boundary_errors(...)` | 15 % (99 of 678 ms) |
| 3 | The constraint layer recomputes the network and two derivatives at both ends inside every forward call | `pinn/constraints.py`, `Constrained.forward` and `end_values` | Inside 1 and 2 above |
| 4 | Fixed budgets: 20 000 Adam steps plus L-BFGS with up to 6250 loss evaluations | `configs/pinn.json`; PyTorch's default `max_eval = 1.25 * max_iter` | Sets the number of steps, not the step cost |
| 5 | Training does not reach the solution yet (wall layer, weight blow-up) | docs/reviews/PINN_REVIEW_2026-11-21.md, H1 and H2 | Not a step cost, but it is why runs get longer |

The number of points, the network size, float64 and the single thread are not the main causes (section 3).

**The honest bottom line.** For this one fixed-parameter 1D boundary problem, a classical solver is much faster and much more accurate. `solve_bvp` solves the locked case in 0.74 s to tol 1e-8 (**measured**, reference/study.json). The new C++ box solver takes 0.48 s for five mesh levels and matches it to 2.7e-10 at the wall (**measured**). The B0 pilots took 1200 to 2800 s for 3000 Adam steps and did not reach the solution (**pilot**). Section 6 says when a PINN would still be the better tool.

## 2. The equations, term by term

### 2.1 Derivative orders

| Field | Highest derivative in Eqs. 14 to 18 | Needed by the physics? |
|---|---|---|
| f | f'''' (Eq. 14) | Yes, after the pressure has been removed. Section 2.3 shows how to bring it down to f''' exactly |
| g | g'' (Eqs. 14, 15) | Yes. Micro-rotation diffusion |
| theta | theta'' (Eqs. 16, 17) | Yes. Heat conduction plus radiation |
| phi | phi'' (Eqs. 17, 18) | Yes. Mass diffusion |
| chi | chi'' (Eq. 18) | Yes. Diffusion of chi |

The first-order rewrite in docs/reference-solver/first-order-form.md (12 states, `M*H = -r0`) is needed only by `solve_bvp`, which accepts first-order systems only. The PINN never uses it, so it does not slow the PINN down.

### 2.2 Which terms could go

Sizes on the branch-A reference (**measured**, script in the 2026-11-22 status entry):

| Equation | Largest term at the wall | Smallest nonzero term (share of largest) | Exactly zero |
|---|---|---|---|
| R14 | viscous (1+Delta)(f'''' + ...), 1269 | -kappa/q^3 f f', 0.1 % | delta_star term |
| R15 | (1+Delta/2)(g'' + g'/q), 0.80 | -Delta B(...), 50 % | delta_star term |
| R16 | dipole heating, 3.44 | Brownian Nb/Pr theta' phi', 1.6 % | delta_star term |
| R17 | (phi'' + phi'/q)/(Le Pr), 9.1 | kappa/q f phi', 3.5 % | delta_star term |
| R18 | (chi'' + chi'/q)/Lb, 24.3 | kappa/q f phi, 0.04 % | delta_star term |

- **delta_star = 0** in the locked case, so the five delta_star terms are exactly zero. They use derivatives we compute anyway, so they cost almost nothing. They stay in the code so it matches the printed equations for any delta_star.
- **No other term is negligible.** The smallest are 4e-4 of the largest, and we need residuals near 1e-6 or below. Far from the wall the small terms set the slow tails.
- **The wall layer is physics.** In Eq. 14 the magnetic dipole force 2*beta_m/(eta+b)^4 * (...) is about 1250 at the wall and balances (1+Delta) f''''. That is why f''''(0) = -1728 and why the PINN struggles. It is not a code artifact.

### 2.3 An exact rewrite of Eq. 14 that lowers f'''' to f'''

Multiply R14 by q = eta + kappa. Every part becomes an exact derivative except one:

```text
q * R14 = d/deta[ Phi ] - 2*kappa*f'^2/q

Phi = (1+Delta)*(q f''' + f'' - f'/q)
    + kappa*(f f'' - f'^2 + f f'/q)
    - delta_star/2*(q eta f'' + 2 q f' - eta f'/2 - f/2)
    - q f'/Da
    - Delta q g'
    - 2 beta_m q theta/(eta+b)^4
```

So Eq. 14 is the same as two lower-order equations with one new unknown P(eta):

```text
Ra = Phi - P = 0                    (f up to f''')
Rb = P' - 2*kappa*f'^2/q = 0        (first order in P)
```

**Proof of equivalence.** If R14 = 0, set P = Phi. Then Ra = 0, and P' = Phi' = q R14 + 2 kappa f'^2/q = 2 kappa f'^2/q, so Rb = 0. If Ra = Rb = 0, then Phi = P is differentiable and q R14 = Phi' - 2 kappa f'^2/q = P' - 2 kappa f'^2/q = 0. Since q >= kappa = 0.2 > 0 on [0, L], R14 = 0. P has no boundary condition. f keeps B1, B2, B7 and B8, so the count of conditions does not change (f order 3 plus P order 1 = 4).

**Checks.** `tests/test_equations_sympy.py::test_reduced_momentum_is_an_exact_rewrite_of_eq14` proves the identity in SymPy with every parameter left symbolic. `test_reduced_momentum_holds_on_the_reference` checks it on the branch-A reference with finite differences: the gap is 2e-5 at the wall, which is the difference-formula error on that grid.

**Physical meaning (inference).** P plays the role of the pressure that the curved-sheet derivation removes when it combines the two momentum equations. In the classical curved stretching-sheet model the pressure gradient across the sheet balances the centripetal term f'^2/(eta+kappa), which has the same form as Rb. I recall this from the curved-sheet literature (for example Sajid et al. 2010) but did not re-open those papers in this session.

**Cost (measured, raw MLP, 2000 points, forward plus backward).** Highest order 4: 77 ms with jets and 1682 ms with nested autograd. Highest order 3: 57 ms with jets and 490 ms with nested autograd. So with jets the saving is about a quarter of the derivative cost. It is much larger only if you stay with nested autograd.

**Code.** `equations.momentum_flux` and `equations.reduced_momentum`. Not wired into training: it needs a sixth network output, and AGENTS.md section 3 asks for supervisor agreement before the model changes.

### 2.4 Two more exact forms, for explaining the physics

- `(g'' + g'/q) = (1/q) d/deta(q g')`, and the same for phi and chi: diffusion across a curved surface.
- The heat terms of Eq. 16 are `(1/(Pr q)) d/deta[ q (1 + Nr A^3) theta' ]`: conduction plus radiation, with a conductivity that grows with temperature. SymPy confirms it (2026-11-22, scratch check).

These change no derivative order. They are useful only if you later move to a first-order (mixed) PINN, where q(1 + Nr A^3) theta' would be a network output.

## 3. Where the time goes

### 3.1 Derivative cost by order (measured)

Raw MLP (width 40, depth 4), 2000 points, the k-th derivative of all five outputs, forward and backward:

| Highest order | Nested autograd | Forward jets |
|---|---:|---:|
| 0 | 6 ms | 15 ms |
| 1 | 35 ms | 14 ms |
| 2 | 114 ms | 27 ms |
| 3 | 490 ms | 57 ms |
| 4 | 1682 ms | 77 ms |

Nested autograd grows by 3.5 to 4.3 times per order. The reason: each `autograd.grad(..., create_graph=True)` builds a new graph that contains the previous one, and the final `backward` has to go through all of them. Reverse mode also needs one pass per output, and there are five outputs but only one input. Forward propagation of derivatives (Taylor mode) carries all orders through each layer at once. A linear layer passes each order through the same matrix, and tanh needs the Faa di Bruno formula, which costs order K^2 small elementwise products. That is the textbook Taylor-mode method (Griewank and Walther, "Evaluating Derivatives", 2nd ed., SIAM 2008, Chapter 13; and Bettencourt, Johnson and Duvenaud, "Taylor-mode automatic differentiation for higher-order derivatives in JAX", NeurIPS 2019 workshop). I cite both from memory; I did not re-open them in this session. The 6 ms against 15 ms at order 0 is fixed overhead of the jet code; it does not matter for training, where order 4 is always needed.

IMPLEMENTATION_NOTES.md section 4 measured `torch.func.jvp` at 1.63 s against 0.36 s for reverse mode. That result still holds for `torch.func`, whose transforms carry their own overhead. `jets.py` is plain tensor arithmetic, which is why it is faster.

### 3.2 One full training step (measured, B0, configs/pinn.json)

| Mode | Engine | Equation loss forward | Boundary errors | Full step (loss + backward) |
|---|---|---:|---:|---:|
| exact | nested autograd | 350 ms | 99 ms | 678 ms |
| exact | jets | 43 ms | 17 ms | 125 ms |
| soft | nested autograd | 230 ms | 15 ms | 454 ms |
| soft | jets | 34 ms | 4 ms | 127 ms |

The old 363 ms per step at 1000 points matches the pilots: 3000 Adam steps took 1212 to 2214 s with three pilots sharing the machine, about 0.4 to 0.7 s per step (**pilot**).

**Peak memory of one step, exact mode (measured, process start 229 MB):** 2000 points: 554 MB against 345 MB. 4000 points: 871 MB against 455 MB. An 8000-point run with nested autograd was killed by the system for lack of memory on this 7 GB laptop.

### 3.3 Number of points (measured, exact mode, full step)

| Points | Nested autograd | Jets |
|---:|---:|---:|
| 250 | 247 ms | 54 ms |
| 1000 | 363 ms | 80 ms |
| 2000 | 643 ms | 121 ms |
| 4000 | 1110 ms | 213 ms |

Most of the cost at small N is fixed per-step overhead (Python, many small kernels, the boundary and end-value graphs). Cutting from 2000 to 250 points saves only about 2.2 times with jets, and it risks missing the wall layer. Points are not the main cost.

### 3.4 Other causes I checked

| Cause | Where | Finding |
|---|---|---|
| Network size | `configs/pinn.json`: width 40, depth 4, about 5000 weights | Small. The 2000 x 40 matrix products take microseconds; overhead and derivative graphs dominate |
| Optimizer budget | `adam_steps` 20 000, `lbfgs_iters` 5000 | About 26 000 loss evaluations per run. PyTorch's `LBFGS` allows up to `1.25 * max_iter` evaluations by default; the pilots used 577 to 625 for 500 iterations (**pilot**) |
| Loss weights | `losses.GradNormWeights.update` every 1000 steps | Cheap. In exact mode there is only one weighted term, so the weights do nothing for B0. They matter for B1 and the soft modes, where they blow up (review H2). That hurts convergence, not step time |
| Monitor loss | `train.adam_stage`, every `log_every` = 100 steps | One extra forward on 1000 points every 100 steps: about 1 % |
| Batching | `losses.equation_mse` loops over window groups | One group for B0, B2, B3; four for B1. Full batch is right for a 1D problem this small |
| Recomputation | `Constrained.forward` calls `end_values` at both ends every time | Three times per step in exact mode (training points, wall, far end). Jets compute the end values in one 2-point pass |
| Device and precision | `run.py`: CPU, `threads` 1, float64 | Inference: a GPU will not help much, since the work is many tiny operations, and consumer GPUs (Colab T4) run float64 far slower than float32. float64 is worth keeping for L-BFGS (Kiyani et al. 2025, our analysis in sources/literature/optimizer-optimizing-2025). Measure with `tools/time_pinn_step.py --threads 4` and `--float32` before deciding |

## 4. Changes ranked by impact against effort

| Rank | Change | Status | Impact | Effort | Evidence | Risk to accuracy |
|---|---|---|---|---|---|---|
| 1 | Forward jets instead of nested autograd (`pinn/jets.py`, used by `model.states`) | Done on this branch, tests pass | 5.4x per bare step in exact mode, 3.6x in soft mode, 3.9x in a real 200-step Adam loop; about 40 % less peak memory | Done | Section 3.2; equal values and gradients to 1e-14 (`test_forward_jets_equal_nested_autograd`); 200-step training check in section 4.1 | None. Same function, same gradients, up to rounding |
| 2 | Compute boundary errors only on logging steps when nothing in the loss needs them | Done: `train.loss_parts` computes them only on logging steps when nothing in the loss needs them | About 14 % per step in exact mode (17 of 125 ms with jets) | Small | Section 3.2 | None. Only logging changes |
| 3 | Make training converge: log input map, capped weights, warm start (review H1, H2) | Done 2026-11-23 as config options (`start`, `input_map`, `weight_cap`, `residual_scale`); measured in [IMPLEMENTATION_NOTES](IMPLEMENTATION_NOTES.md) section 7.1 | Largest effect on time to a usable answer: a run that never converges wastes all its time | Medium | Pilots miss B4 by 2.3 and sit near theta(0) = 0 (review H1). The effect of each fix is not measured yet | Each fix must be tested; a warm start from the smooth guess is allowed, never from the reference |
| 4 | Stop on a target instead of a fixed budget; report time to reach a given error | Partly: time to reach an error is reported against solve_bvp (figure FX1, table TX1, 2026-11-24); stopping on a target is not built | Inference: could cut many of the 26 000 evaluations once a run has converged | Small | Pilot histories show plateaus; not measured as a rule | Low, if the stop rule uses the monitor loss and never the reference |
| 5 | Reduced-order Eq. 14 with the pressure-like unknown P (section 2.3) | Equations and proofs done; not in the network | About a quarter of the derivative cost with jets (77 to 57 ms, raw MLP); inference: lower-order residuals may train better | Medium (sixth output, constraint layer, tests) | Section 2.3; the same idea of lower-order residuals as the first-order PINN of Gladstone, Nabian and Meidani (arXiv:2210.14320, 2022; named from memory, not re-opened) | Low in theory (exact rewrite). P is only tied to f through the loss, so check f'''' after training |
| 6 | `torch.compile` on the jets path | Not tried | Unknown. Possible now because the step has no double backward | Small to try | None yet | None if results match |
| 7 | Fewer points, more threads, float32 | Measure first | Fewer points: about 2x at most; threads and float32: not measured | Small | Section 3.3 | Fewer points can miss the wall layer; float32 limits L-BFGS |

**Projected full run (estimate).** About 26 000 loss evaluations at 0.59 s each (old, from the 200-step check in section 4.1) is about 4.3 hours on one thread; at 0.15 s (jets) it is about 1.1 hours. Rank 2 would take a little more off. This is arithmetic from measured step times, not a measured full run.

### 4.1 Training check: same seed, 200 Adam steps

B0, default config, seed 0, 200 Adam steps with the real `train.adam_stage` (logging every 50 steps, monitor loss included), 1 thread (**measured**):

| Engine | Wall time | Loss at steps 0, 50, 100, 150, 199 |
|---|---:|---|
| nested autograd | 118.7 s | 2546.9261721, 802.68271898, 779.52769095, 769.06885230, 764.07865197 |
| jets | 30.1 s | the same to all 11 printed digits |

The largest weight difference after 200 steps is 2.4e-15. So training follows the same path, 3.9 times faster in the real loop. That is less than the 5.4 times of the bare step because Adam, logging, the monitor loss and the still-computed boundary errors do not shrink as much.

## 5. How to measure the real numbers yourself

```bash
uv run python tools/time_pinn_step.py
```

```bash
uv run python tools/time_pinn_step.py --mode soft --points 1000 --threads 4
```

```bash
uv run python tools/time_pinn_step.py --profile
```

The first prints the median time of the equation loss, the boundary errors and a full step, for both engines. `--profile` adds a `torch.profiler` table of the slowest operations. For a whole run, read `adam_seconds`, `lbfgs_seconds` and `lbfgs_evaluations` in `runs/<run_id>/meta.json`. Run timings with nothing else heavy open, and use the median of several repeats: on this laptop the same step measured 1851 ms with other programs busy and 678 ms with them idle.

## 6. Is a PINN the right tool here?

For this problem, no, if the only goal is the solution. Measured on this laptop:

| Solver | Time | Accuracy |
|---|---|---|
| `solve_bvp` (reference/solver.py) | 0.74 s | residual 1e-8, agrees with RK shooting to 2e-10 |
| C++ box scheme with Richardson (engine/) | 0.48 s for five levels; 0.02 s for the first | 2.7e-10 at the wall against `solve_bvp` |
| B0 PINN pilots, 3000 Adam steps | 1200 to 2800 s | Did not reach the solution (theta(0) about 0.1 or less, against 0.250) |

This matches the published comparisons we hold: Grossmann et al. 2024 (IMA J. Appl. Math., arXiv:2302.04107) found PINNs did not beat FEM on solve time or accuracy in their tests, and Chuang and Barba 2022 (SciPy proceedings) report 32 hours of PINN training to match a 20-second finite-difference run. Those are their results on other problems, not ours.

A PINN can still be the better choice when:

- **Many parameter values are needed.** Put Pr, Nr or beta_m in as extra inputs and train once; then each new case is a forward pass. A classical solver must re-solve for each value. That pays off only after many cases.
- **Some values are unknown and there are measurements.** An inverse problem fits naturally into the PINN loss.
- **Meshing is the hard part**, as in higher dimensions or complex shapes. Not our case: this is 1D.

For the FYDP, the PINN is the object of study, so the classical solvers stay as the yardstick and the PINN is judged against them, as docs/pinn/PINN_PLAN.md already says.

## 7. How to explain this to the supervisor

- "The equations are right and every term is needed. The only term that is exactly zero is the unsteady one, because delta_star is 0."
- "f has a fourth derivative because the pressure was removed from the momentum equations. I can show an exact rewrite with a pressure-like unknown that needs only the third derivative. It is proven in SymPy. It changes the network, so I would like your agreement before using it."
- "Training was slow mainly because of how the derivatives were computed: 12 nested passes, each bigger than the last, so the fourth derivative cost about 280 times the plain output. The network has one input, so I now push all derivatives forward in one pass. Same numbers to 1e-14, and about four times faster in a real training loop."
- "A step is not the only problem. The pilots do not reach the solution yet, because of the very thin wall layer and loss weights that blow up. Fixing that matters more than speed."
- "For one fixed case a classical solver is far better: under a second, with 1e-10 accuracy, against hours. We use it as the reference. The PINN is worth it if we want many parameter values from one training, or an inverse problem."
