# Glossary

Every term the methodology pages and the step files use, in plain words. [Back to the index](README.md).

## The problem

| Term | Meaning |
|---|---|
| BVP | Boundary value problem: differential equations plus conditions at both ends of the domain. Ours has conditions at the wall (η = 0) and at the far end (η = L) |
| η (eta) | The one input. The distance from the wall, in scaled units. Everything is a function of η |
| f, g, θ, φ, χ | The five unknown functions (fields). f (flow), g (micro-rotation), θ (temperature), φ (concentration), χ (the fifth field of Eq. 18). We only need their math, not their physics |
| f′, f″, f‴, f⁽⁴⁾ | First, second, third and fourth derivative with respect to η. The prime counts derivatives |
| Field | One of the five unknown functions |
| Residual R14 to R18 | Put a guess into an equation, move everything to one side, and what is left is the residual. The true solution makes it zero everywhere |
| Boundary condition B1 to B12 | A rule a solution must meet at one end. Written as a quantity that must be zero |
| Wall | The end η = 0 |
| Far end, far field | The end η = L = 20, which stands in for η → ∞ |
| Wall layer | The thin region near η = 0 where the fields change very fast (f⁽⁴⁾(0) is about −1 728) |
| Wall values | The numbers at η = 0 we report: f″(0), θ(0), θ′(0), φ(0), φ′(0), g′(0), χ′(0) |
| Locked values | The 22 supervisor numbers (Pr = 8.8, κ = 0.2 and so on). Never changed by any code |
| q, h, A | Short names: q = h = η + 0.2, A = 1 − 0.8θ |
| L | The far end of the computed domain, 20. Our numerical choice, not a supervisor value |
| L error | How far the L = 20 answer is from the η → ∞ answer |
| Asymptotic far condition | A condition at a finite L that keeps only the parts of the solution that decay to zero at infinity (B10, B11 in the form we use) |
| m, m_p | Decay powers: far out, θ ≈ C q^(−m) and φ ≈ E q^(−m_p). A real decaying solution needs m > 0 |
| Branch | One of several solutions of the same equations and conditions. We found two: A (decays, our target) and B (does not decay) |
| Linear condition | A fixed weighted sum of values and slopes equals a known number. Doubling the change doubles the effect |
| Nonlinear condition | Contains products or powers of unknowns, such as θ(0)⁴ in B4 |

## How conditions are imposed

| Term | Meaning |
|---|---|
| Hard (exact) condition | Built into the network output, so it holds for any weights. Not part of the loss |
| Soft condition | Added to the loss as a penalty. It only holds as well as training makes it |
| TFC | Theory of functional connections: output = network + fixed shapes × (target − what the network gives). This makes conditions exact |
| Correction shape ψ | A fixed smooth function, such as η e^(−η/0.5), used to correct the network near one end |
| Functional C | A rule that reads one number from a function, such as "value at 0" or "1.5 × slope at 0 − 0.1 × curvature at 0" |
| Miss | Target minus what the raw network gives for one condition |
| Kept value | A value the layer leaves as the network gives it, such as θ(0); its shape adds nothing there |

## Numbers and solvers

| Term | Meaning |
|---|---|
| First-order form | Rewriting the equations as 12 first-order equations z′ = F(η, z), which most solvers need |
| State z | The 12 values f, f′, f″, f‴, g, g′, θ, θ′, φ, φ′, χ, χ′ at one η |
| Highest derivatives H | f⁽⁴⁾, g″, θ″, φ″, χ″: what the equations decide once z is known |
| M H = −r₀ | The small 5 × 5 linear system that gives H from z |
| solve_bvp | SciPy's boundary value solver: collocation on a mesh that it refines until the error is below the tolerance |
| Collocation | Making the equations hold exactly at chosen points of each mesh interval |
| Mesh, node | The points where a solver stores the solution |
| Tolerance (tol) | The error a solver aims for. Smaller is more accurate and slower |
| Ladder | Solving again with a smaller tolerance or a longer L, each time starting from the last answer, to see the answer settle |
| Shooting | Guess the missing start values, integrate across, and correct the guess until the far end is right |
| Multiple shooting | Shooting on many short pieces at once, so errors cannot grow too much on any piece |
| Runge-Kutta, DOP853 | A family of step-by-step integrators; DOP853 is an 8th-order one in SciPy |
| Newton's method | Repeatedly solve the linear version of a nonlinear problem to correct the guess |
| Jacobian | The matrix of first derivatives of all residuals with respect to all unknowns |
| Box scheme | A classic second-order method for boundary layers: each mesh interval is a small box with the equations at its midpoint |
| Richardson value | Combine answers on two meshes to cancel the main error term |
| GCI | Grid convergence index: an error bar from three meshes |
| Deflation | A trick that stops Newton from finding the same solution twice, to search for other branches |
| Spline | A smooth piecewise polynomial through saved points, used to take their slope |
| Gate G1 to G6 | The checks a stage must pass before the next stage starts |

## The network and training

| Term | Meaning |
|---|---|
| PINN | Physics-informed neural network: a network trained so the equations hold, with no solution data |
| MLP | Multilayer perceptron: layers of weighted sums followed by a smooth function |
| tanh | The smooth S-shaped function used in every hidden layer |
| Weights | The numbers inside the network that training changes |
| float64 | Double precision numbers, about 16 digits |
| Seed | A number that fixes every random choice, so a run can be repeated |
| Collocation (training) points | The η values where the residuals are computed in the loss |
| Monitor points | A second set of points never used for gradients, to catch overfitting and to decide if a refine stage helped |
| Input map | The function that turns η into the network input x in [−1, 1] |
| Start profile | A fixed smooth guess added to the network output, so training starts from a sensible shape |
| Taylor jets | Pushing all derivatives (orders 0 to 4) forward through the network in one pass |
| Autograd | PyTorch's automatic differentiation, used for the weight gradients |
| Loss J | The single number training makes small: the mean of the squared scaled residuals |
| Residual scale s(η) | A fixed positive factor on each residual so no region dominates the loss. It never changes where the residual is zero |
| Adam | A gradient method with per-weight step sizes. Good far from the answer |
| Learning rate | The step size of Adam |
| L-BFGS | A quasi-Newton method that builds curvature from recent gradients. Good closer to the answer |
| Strong Wolfe line search | A rule that picks a step length that lowers the loss enough and flattens the slope enough |
| Levenberg-Marquardt (LM) | A least-squares method: a Gauss-Newton step with a damping term that keeps steps safe |
| Gauss-Newton | Use KᵀK, built from the residuals' Jacobian K, as the curvature |
| Damping μ | The LM term that blends a careful gradient step (large μ) and a full Gauss-Newton step (small μ) |
| Cholesky | A fast way to solve a symmetric positive definite linear system |
| Woodbury identity | A formula to add a few extra rows to a solve that is already factored, cheaply |
| Gradient-norm weights | Loss weights set so every loss term pulls on the weights equally hard |
| Rollback | Undo a refine stage if it made the monitor loss worse |
| Checkpoint | A saved copy of the weights |

## Arms and runs

| Term | Meaning |
|---|---|
| Arm | One way of training: B0, B1, B2, B3 |
| Window | A piece of [0, 20]: [0, 0.5], [0.5, 2], [2, 6], [6, 20] |
| B0 | One network on the whole domain |
| B1 | One network per window, trained together, with continuity between windows |
| Interface jump | The difference of the 12 states of two neighbouring window networks at their shared edge |
| B2, B3 | One network, trained window by window from the wall outward |
| Replay | Keeping earlier windows right while training a new one: B2 keeps their residual (physics replay), B3 keeps the network's own earlier outputs (self replay) |
| Stage | One training phase on a set of windows |
| Run | One arm with one seed, trained from start to end |
| Campaign | All runs of the study: 4 arms × 5 seeds = 20 runs |
| Paired seeds | The same seed number in two arms, so they start from the same points |
| Run folder | The folder that holds everything one run made |
| Run ID | bvp-arm-seed-hash; the hash is of the run's settings |

## Checks and results

| Term | Meaning |
|---|---|
| Acceptance rule | The fixed test a finished run must pass, without looking at the reference |
| Newton-step estimate | One Newton step of the box scheme from the PINN's answer; its size estimates the PINN's error |
| Score | The largest estimated relative wall error over f″(0), θ(0), θ′(0), φ′(0) |
| Tail share | How much of θ on [10, 20] is a constant that does not decay. Near 0 for a good run |
| Failure share | The share of monitor points where some residual is above 10⁻³ |
| Relative L2 error | The size of the error curve divided by the size of the reference curve, both measured over [0, 20] |
| Trapezoid rule | Adding up a curve by trapezoids between grid points |
| Rule G6 | "Arm X beats arm Y" only if it holds in all five paired seeds |
| TB2, TB3, TX1 | Result tables: wall values, accuracy and paired seeds, cost |
| FB1 to FB5, FX1 | Result figures: profiles, errors, residuals, training curves, windows, error against time |
| SHA-256, hash | A fingerprint of a file. Any change of the file changes it |

## Running on Colab

| Term | Meaning |
|---|---|
| Colab | Google's hosted notebooks, with optional GPU |
| GPU, CUDA | Graphics card used for maths; CUDA is NVIDIA's software for it |
| Drive | Google Drive, mounted as a folder so results survive a disconnect |
| uv, lock file | A Python package tool; uv.lock lists the exact version of every package |
| Virtual environment (venv) | A private Python install with its own packages |
| Dry run, smoke test | The whole pipeline with tiny settings, to find errors in a minute |
| Marker | A small file that says a step finished, so a restart skips it |
| Pin | Recording the commit and settings, so a restart uses the same ones |
