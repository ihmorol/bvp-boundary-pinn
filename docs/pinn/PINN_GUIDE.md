# How our PINN works: a reading guide

Date: 2026-11-21, updated 2026-11-24. Audience: the team. Read it top to bottom once. Each section answers one question.

The training code is in `src/bvp_pinn/pinn/`; the comparison with the reference is in `src/bvp_pinn/evaluation/`. The equations come from `src/bvp_pinn/reference/equations.py`, which the PINN shares with the classical solver. File-by-file details: [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md).

---

## 1. What problem are we solving?

We have five unknown curves. Each one is a function of a single number, eta (the distance from the wall):

| Name | Meaning in short | Highest derivative in the equations |
|---|---|---|
| f | flow | 4th (f'''') |
| g | micro-rotation | 2nd |
| theta | temperature | 2nd |
| phi | nanoparticle fraction | 2nd |
| chi | micro-organisms | 2nd |

Five equations (Eqs. 14 to 18) link them. Twelve boundary conditions (Eq. 19) fix their values at the wall (eta = 0) and far away (eta = 20 in our code). All 22 physical numbers (Pr = 8.8, Delta = 0.5, and so on) are fixed. They come from the supervisor and live in `configs/physical-case-pending.json`.

This is a **boundary value problem (BVP)**: conditions sit at both ends, so we cannot just start at the wall and step outward.

## 2. What is a PINN, in one paragraph?

A PINN (physics-informed neural network) is a small neural network that takes eta and returns guesses for f, g, theta, phi and chi. We never show it the answer. Instead, we check how badly its guesses break the equations and the boundary conditions, and we change the network weights to make that error smaller. When the error is close to zero everywhere, the network output is the solution.

## 3. Tools we use

| Tool | What we use it for |
|---|---|
| Python 3.12 | Language |
| PyTorch | The network, automatic derivatives (autograd), the Adam and L-BFGS optimizers |
| float64 numbers | All PINN maths. Fourth derivatives lose too much accuracy in float32 |
| NumPy | Arrays, the reference equations |
| SciPy `solve_bvp` | The classical reference solution (for checking only) |
| SciPy `solve_ivp` (DOP853) | A second, independent reference by Runge-Kutta shooting |
| SymPy | A separate copy of the equations, used in tests to catch typos |
| pytest | Tests (23 pass on 2026-11-21) |
| matplotlib | Figures |
| uv | Installs packages and runs commands (`uv run ...`) |
| Google Colab | Planned place for the long final runs |

## 4. The big picture

```text
configs/pinn.json ──> campaign.py / run.py ──> model.py (network, start profile, boundary layer)
                                    │
                                    ▼
                points.py ──> train.py ──> losses.py, lm.py ──> physics.py ──> reference/equations.py
                                    │
                                    ▼
                             runs/<run_id>/  (model, history, meta)
                                    │
                                    ▼
                             estimate.py ──> selection.json   (acceptance, no reference)
                                    │
reference/locked-branch*/ ──> evaluation/evaluate.py ──> metrics.json, results/<campaign>/ ──> evaluation/plots.py
```

Training and acceptance never read the reference. Only the `evaluation` package loads it, after both. A test reads every import of the training package to check this.

## 5. Input and output of the network

**Input:** one number, eta, between 0 and 20.

**Output:** five numbers, the guesses for f, g, theta, phi, chi at that eta.

**Inside:** a plain stack of layers with tanh between them. Default size: 4 hidden layers, 40 neurons each, about 5200 weights.

Before eta enters the network, it is squeezed into the range [-1, 1], because tanh networks learn best there:

```python
# model.py, MLP.forward
s = eta - self.lo
if self.input_map == "log":
    s = torch.log1p(s / self.scale) / math.log1p((self.hi - self.lo) / self.scale)
else:
    s = s / (self.hi - self.lo)
return self.net(2 * s - 1)
```

**Read:** with the "linear" map, eta = 0 becomes -1 and eta = 20 becomes +1. The "log" map stretches the region near the wall, so the network has more room for the thin wall layer.

## 6. How we get the derivatives

The equations need f', f'', f''', f'''' and the first and second derivatives of the other four fields. These are exact derivatives of the network, not finite differences.

The network has one input, so the derivatives can be pushed forward through the layers with the chain rule, all orders at once. This is **Taylor mode** (`jets.py`). A linear layer passes every order through the same matrix; a tanh layer uses the known derivatives of tanh:

```python
# jets.py, tanh_jet (orders 0 to 2 shown)
t = torch.tanh(a[0])
D = 1 - t * t
out = [t, D * a[1], -2 * t * D * a[1] ** 2 + D * a[2]]
```

**Read:** `a[k]` is the k-th derivative of the layer's input with respect to eta; `out[k]` is the k-th derivative after tanh. One forward pass gives all five fields and all orders.

The result is two lists:

- `z`, 12 columns: f, f', f'', f''', g, g', theta, theta', phi, phi', chi, chi'.
- `H`, 5 columns: f'''', g'', theta'', phi'', chi''.

The older way, nested `torch.autograd.grad` calls, gives the same numbers and is about four times slower ([RUNTIME_STRATEGY.md](RUNTIME_STRATEGY.md)). It stays in `model.states_autograd`, and a test checks that both give the same values and the same weight gradients. A test also checks the derivatives against sin, exp and other known functions up to order 4.

## 7. How the physics is enforced

### 7.1 The residual

Each equation is written as "left side = 0". For any guess we compute the left side. We call that number the **residual**. If the guess were perfect, all five residuals would be zero at every eta.

For example, Eq. 15 in code:

```python
R15 = ((1 + Delta / 2) * (g2 + g1 / q)
       - Delta * p["B"] * (f2 + f1 / q + 2 * g)
       - kappa / q * (f1 * g - f * g1)
       - ds / 2 * (eta * g1 + 3 * g))
```

Here `q = eta + kappa`, `g1` is g', `g2` is g'', and so on. The same code serves NumPy (for the reference) and PyTorch (for the PINN). We checked every term against the source page of the draft.

### 7.2 Making the five residuals comparable

The five equations have very different sizes. So `physics.residuals` divides each one by the number in front of its highest derivative:

| Equation | Divided by | Value |
|---|---|---|
| R14 | 1 + Delta | 1.5 |
| R15 | 1 + Delta/2 | 1.25 |
| R16 | (1 + Nr·A³)/Pr | about 0.14 to 0.16 |
| R17 | 1/(Le·Pr) | 0.114 |
| R18 | 1/Lb | 1 |

After this, each residual is in "units of its highest derivative". Dividing by a non-zero number does not change where the residual is zero, so the solution is unchanged.

### 7.3 The equation loss

We pick about 1000 to 2000 points in [0, 20], called **collocation points**. At each point we compute the five residuals, square them, and take the mean:

```python
# losses.py, equation_mse
z, H = states(model[k], eta)
total = total + residuals(eta, z, H, p).pow(2).sum(0)
...
return total / n          # five numbers: mean square of R14, ..., R18
```

The points are packed closer near the wall, because that is where the curves change fastest:

```python
# points.py: one random point per equal cell of s in [0, 1], then stretch
eta = L * (exp(stretch * s) - 1) / (exp(stretch) - 1)
```

With `stretch = 5`, about 31 % of the points fall in [0, 0.5].

A second, separate set of **monitor points** is never used for training. It only tells us whether the network is good between the training points too.

## 8. How the boundary conditions are enforced

Eq. 19 gives 12 conditions, B1 to B12. Each one is written as "something = 0":

| Where | Conditions |
|---|---|
| Wall, eta = 0 | B1 slip on f'; B2 links f(0) and phi'(0); B3 links g(0) and f''(0); B4 heat condition with radiation; B5 links phi'(0) and theta'(0); B6 chi(0) = 1 |
| Far end, eta = 20 | B7 f' = 0; B8 f'' = 0; B9 g = 0; B10 theta far condition; B11 phi far condition; B12 chi = 0 |

The code can impose them in three ways. A config switch, `bc_mode`, picks one.

### 8.1 Soft mode: as loss terms

Compute each B at the ends, square it, add it to the loss. Simple, but the conditions are never met exactly, and the optimizer must balance 12 extra terms against the equations.

### 8.2 Exact modes: built into the output (TFC)

The idea comes from the **theory of functional connections (TFC)**. We take the raw network output N and add a small correction that fixes exactly the amount by which N misses the condition.

**Simplest example: B6, chi(0) = 1.**

```text
chi(eta) = N(eta) + (1 - N(0)) * exp(-eta / 0.5)
```

At eta = 0 the exponential is 1, so chi(0) = N(0) + 1 - N(0) = 1. This holds for **any** weights. Far from the wall the exponential is almost 0, so the correction does not disturb the rest of the curve. `0.5` is `tfc_decay` in the config.

**General recipe (`constraints.py`).** One field may carry several conditions, some on values and some on slopes, at both ends. For each condition we:

1. Pick a shape: `d^n * exp(-d / 0.5)`, where d is the distance to the end that carries the condition and n = 0, 1, 2 counts the conditions at that end.
2. Build a small square matrix: row i says what condition i gives when applied to shape j.
3. Invert that matrix once, when the model is built.
4. At every forward pass, compute each "miss" = target minus what the raw network gives, then add the right mix of shapes:

```python
# constraints.py, Field.shape
S = torch.cat([self.support(eta, e, n) for e, n in self.supports], 1)
return S @ (self.inv @ miss).reshape(-1, 1)
```

Because the matrix is inverted exactly, every condition is met exactly, whatever the weights are.

**The order of corrections matters.** Some targets depend on other fields. For example, B3 says g(0) = -ell·f''(0). So the code first corrects f, then reads the corrected f''(0), then corrects g. The same holds for B2, which needs phi'(0), which B5 sets from theta'(0).

**The nonlinear ones: B4, B10, B11.** These are not linear, but each fixes one slope from one value:

- B4 gives theta'(0) from theta(0): `theta'(0) = -Bi*(1 - theta(0)) - Rd*Qw(theta(0))`.
- B10 gives theta'(20) from theta(20) and f(20).
- B11 gives phi'(20) from phi(20), theta(20) and f(20).

So the layer adds a "keep" condition that leaves the value alone (miss = 0) and then sets the slope from that value. This step is our own; the team found no paper that does it.

**The modes:**

| `bc_mode` | Exact in the output | Left as loss terms |
|---|---|---|
| soft | none | all 12 |
| linear | the 9 linear ones | B4, B10, B11 |
| exact (default) | all 12 | none |

A test builds a random network in each mode and checks that the exact conditions hold to 1e-12.

### 8.3 Why "far conditions" and not just zero at eta = 20?

The source says theta and phi go to 0 as eta goes to infinity. But these curves fall very slowly, like a power of eta. In the reference, at eta = 20, theta is still 0.068 and phi is 0.18. Forcing them to 0 at eta = 20 would bend the whole solution.

So both the reference and the PINN use **asymptotic** conditions at eta = 20 (Lentini and Keller, 1980). Far out, f' and f'' have died away, and Eq. 16 becomes simple. Its decaying solution is theta = C·(eta + kappa)^(-m), with m = Pr·kappa·f(20)/(1 + Nr·A³). That shape has a known slope:

```text
B10:  theta'(20) + m * theta(20) / (20 + kappa) = 0
```

B11 is built the same way from Eq. 17. This keeps only the decaying part.

chi has no such limit, because Eq. 18 as printed has `kappa/q * f * phi` and no chi convection term. So we set chi(20) = 0 and say clearly that chi results are for the eta = 20 problem. This is supervisor question 9.

## 9. How physics and boundary conditions work together

The equations alone have many solutions. Any curve that satisfies Eqs. 14 to 18 works, whatever it does at the ends. A fourth-order f plus four second-order fields means 12 free constants. The 12 boundary conditions pin down those 12 constants. Only together do they give one answer (in our case, more precisely, a small number of answers: the reference found two branches, A and B).

What each part gives:

| Part | What it does | What happens without it |
|---|---|---|
| Equation loss | Makes the shape of every curve obey the physics between the ends | Curves can be anything that meets the end conditions |
| Boundary conditions | Picks the one physical solution out of the family | The network can settle on a trivial or wrong member of the family, for example a "cold wall" with theta near 0 |

Why exact boundary conditions should help: with soft conditions, the optimizer must trade the equations against 12 extra terms, and their sizes differ a lot. With exact conditions, every network the optimizer can reach already meets all 12. The optimizer only has to make the equation residual small. The search space is smaller and the loss has one job.

The cost: exact mode is slower per step, and the extra correction shapes can make the loss harder to minimise near the ends. So the team ran a pilot to choose the mode. Pilot P2a (2026-11-22) kept exact mode: soft and linear runs settled near theta = 0 at the wall. All arms now use it.

## 10. The total loss and the loss weights

```text
loss = w0 * (sum of the five equation MSEs)
     + w_i * B_i^2      for each condition left soft
     + w_I * interface  (arm B1 only)
     + replay_weight * replay   (arms B2 and B3 only)
```

The weights w come from the **gradient-norm rule** of Wang, Sankaran, Wang and Perdikaris (2023). Every 1000 steps:

```text
lambda_hat_i = (sum over all terms of |gradient of term|) / |gradient of term i|
w_i <- 0.9 * w_i + 0.1 * lambda_hat_i
```

**Read:** a term whose gradient is small gets a big weight, so every term pulls on the weights with the same strength.

**Small example.** Two terms with gradient sizes 18 and 6. Sum = 24. Then lambda_hat = 24/18 = 1.33 and 24/6 = 4. After one update: w = 0.9 + 0.1·1.33 = 1.03 and 0.9 + 0.1·4 = 1.3. The unit test uses exactly these numbers.

**Watch out.** When a condition is almost met, its gradient is almost zero, so its weight grows without limit. In the pilots weights reached 1e9 to 1e10. In exact mode there is only one term, so the rule has no effect there.

## 11. Training steps

1. **Start profile.** A fixed smooth curve, the same analytic guess the reference solvers start from, sits under the network. The last layer starts at zero, so training starts on that curve, not on a random one.
2. **Adam** (a gradient method with momentum), learning rate from 1e-3 down to 1e-4, 20 000 steps by default, on the log residual scale.
3. **L-BFGS** (a quasi-Newton method that uses curvature), up to 5000 iterations, weights frozen, still on the log scale.
4. **Levenberg-Marquardt** (a Gauss-Newton method for sums of squares), 260 steps, on the balanced scale. In exact mode the loss is a plain sum of squares, which this method uses directly. For B1 it works on four networks at once, one block each.
5. **Rollback:** each refine stage is kept only if the monitor loss improves and stays finite.

```python
# train.py, refine_once
accepted = after == after and after <= before   # after == after is False for NaN
if not accepted:
    model.load_state_dict(saved)
```

Why two scales: the log scale brings the far tail of theta into the decaying shape; the balanced scale then lets Levenberg-Marquardt finish near the wall. The balanced scale from the start left the tail flat (run X4, IMPLEMENTATION_NOTES section 7.2).

## 12. The four arms (training set-ups we compare)

The domain is cut into windows, by default [0, 0.5], [0.5, 2], [2, 6], [6, 20]. Small windows near the wall, where things change fast.

| Arm | Networks | How it trains |
|---|---|---|
| B0 | One | Everything at once. The basic PINN |
| B1 | One per window | All trained together. An interface loss keeps the 12 states continuous where windows meet |
| B2 | One | Window by window from the wall outward. Earlier windows keep their equation residual in the loss (physics replay) |
| B3 | One | Window by window. Earlier windows are held to the network's own outputs from the stage before (self-replay) |

All arms meet all 12 boundary conditions at every step, and all end with the same refine chain (L-BFGS, then Levenberg-Marquardt) on the full problem. So differences between arms come from the training path, not from the optimizer.

## 13. What comes out, and how we judge it

Each run writes `runs/<run_id>/` with the config, every loss term per step (`history.csv`), the model, and `meta.json` (versions, times, final weights).

**First, without the reference: accept or reject.** `estimate.py` checks the finished run:

- the far decay powers m and mp must be positive, or theta and phi cannot reach zero;
- theta must have no constant part on [10, 20] (the tail share);
- every boundary error must be tiny;
- the C++ engine takes one Newton step of a classical solver from the PINN's curves. That step is close to the PINN's error, so its size estimates how wrong the wall values are. The score must be below 1e-3.

A small loss alone is never enough: one run had a very small loss and a wrong solution (run E8). Rejected runs are kept and reported. The rule is in `configs/pinn.json`.

**Then, against the reference.** `evaluation/evaluate.py` compares with the reference (branch A, L = 20):

- **Relative L2 error** of each field and derivative: the size of the error curve divided by the size of the reference curve, measured over [0, 20].
- **Wall values:** f''(0), theta(0), theta'(0), phi(0), phi'(0), g'(0), chi'(0), each with its relative error.
- **Residuals** R14 to R18 and boundary errors B1 to B12, each on its own.
- **Nearest branch:** whether the run is closer to reference branch A or B.
- For B2 and B3, the error in each window after each stage, to see if old windows are forgotten.
- **Cost:** wall time, and how long solve_bvp needs to reach the same error.

The reference wall values we aim for (branch A):

| f''(0) | g'(0) | theta(0) | theta'(0) | phi(0) | phi'(0) | chi'(0) |
|---:|---:|---:|---:|---:|---:|---:|
| -4.2033 | -2.8713 | 0.2502 | -1.2594 | -0.1202 | 3.7783 | -2.0087 |

## 14. Where we stand

**2026-11-24.** All four arms train end to end with the same refine chain, every run is judged without the reference, and evaluation builds every planned table and figure. 57 tests pass.

**Result so far (B0 only, two seeds, 1 000 points):** from scratch, every wall value except phi(0) within 4e-4 of the reference (seed 0: theta(0) 0.2501906 against 0.2501853); phi(0) is off by 4.3e-4 in seed 0 and 6.2e-3 in seed 1. The acceptance rule, applied to all 13 pilot runs, accepts exactly the three that match the reference. Numbers and run IDs: [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) section 7.

**Not run yet:** the pilots that fix the point count, budgets and windows, and the 20-run campaign.

How we got here: the first pilots (2026-11-21) ended with theta near 0 at the wall ([PINN_REVIEW_2026-11-21.md](../reviews/PINN_REVIEW_2026-11-21.md)). The start profile, the log input map, the residual scales and Levenberg-Marquardt fixed that, one change at a time (IMPLEMENTATION_NOTES sections 7.1 to 7.3). None of them changes an equation, a boundary condition or a value.

## 15. Short glossary

| Word | Meaning here |
|---|---|
| eta | Distance from the wall, the only input |
| residual | Left side of an equation for the current guess; zero when exact |
| collocation points | The eta values where we check the residuals during training |
| monitor points | Extra eta values used only to watch for overfitting |
| jets (Taylor mode) | All derivatives of the network in one forward pass |
| autograd | PyTorch's exact derivative of the network, by repeated backward passes (kept as a check) |
| soft condition | A boundary condition added to the loss as a squared error |
| exact (hard) condition | A boundary condition built into the output so it always holds |
| TFC | Theory of functional connections: the method behind the exact conditions |
| miss | Target value minus the raw network value at a boundary |
| Adam, L-BFGS, Levenberg-Marquardt | The three optimizers, used one after the other |
| acceptance rule | The checks without the reference that accept or reject a finished run |
| arm | One training set-up (B0, B1, B2, B3) |
| branch | One of the solutions the reference found (A is the valid one) |
| reference | The classical solve_bvp solution, used only for checking |
