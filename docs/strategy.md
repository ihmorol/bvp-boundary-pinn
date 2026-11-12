# PINN strategy and implementation plan

Status: **strategy ready; physical simulation blocked by seven missing coefficients**. All work is in this new repository. No solver training or physical-reference simulation has been run. Missing values stay null; they are not silently replaced by zero or by paper values.

Start with [the parameter record](parameters.md), [the exact equation sheet](equations.md), and [the checked reading list](reading-list.md). The source equations are confirmed as written. A copying error in the older strategy is corrected here; the original Lorenz file remains unchanged.

## 1. What we will solve

The network takes one position, eta, and returns five predictions: f, g, theta, phi and chi. All physical coefficients remain fixed during that case. Predictions form smooth functions, so automatic differentiation gives the needed eta derivatives.

The complete check uses:

- Five equation residuals, one from each of Eqs. 14–18, at interior points.
- Twelve boundary errors from Eq. 19, evaluated at eta=0 and a finite far endpoint eta=L.
- No labeled reference solution in the baseline loss.

A single network serves the whole interval. We do not carry a guessed state from one spatial window to the next using the old Lorenz time-rollout method.

## 2. Separate the target from an easier numerical check

The requested target uses kappa=b=Da=0.1, among the other recorded rounded choices. Near the wall, the inverse-power coefficients are large. Treat this as a numerical challenge, not proof that the parameters are invalid.

Once the missing coefficients arrive, an optional debug case uses kappa=b=Da=1 and theta_w=1.5, keeping the same remaining physical coefficients. First test whether the solver and PINN work on this less extreme case. Then approach the target by parameter continuation if needed. For example, use rounded kappa/b/Da stages 1, 0.5 and 0.1. Keep theta_w's target choice separate; crossing theta_w=1 requires the stable radiation expression and confirmation that the full normalization permits the limit. Do not claim a continuous physical branch through that point automatically.

The easy case is never relabeled as the requested result. Both target and debug cases are currently blocked by the same missing coefficients.

## 3. Architecture and module responsibilities

| Module to implement later | Input | Output | Main test |
|---|---|---|---|
| Parameter validation | Complete physical case and numerical settings | Validated case or clear failure | Missing values, range membership, denominators |
| Network | Eta tensor, shape N by 1 | Five physical predictions, N by 5 | Shape and dependence on each input row |
| Derivatives | Network predictions and eta | f1–f4; other first/second derivatives | Analytic derivatives and chain rule |
| Torch equations | Eta, predictions, derivatives, fixed parameters | R14–R18, N by 5 | Independent transcription and signs |
| Boundary checker | Prediction/derivative traces at 0 and L | B1–B12, length 12 | All twelve conditions and no extras |
| Loss | Original residuals, fixed scales, positive weights | Scalar and component diagnostics | Scaling/weight application and gradient connection |
| Classical reference | Same complete coefficients, original equations and BCs | Mesh-refined twelve-state BVP solution | Tolerance/domain/guess sensitivity |
| Trainer | Model, points, loss, optimizer settings | Checkpoints and explicit run status | Deterministic closure and resume |
| Evaluation | Selected model, fresh points, approved reference | Profiles, residuals, BCs, errors and wall quantities | Independent grids and derivative accuracy |

Proposed solver structure, not existing implementation:

```text
src/bvp_pinn/
    config.py       complete case and training settings
    model.py        one smooth five-output network
    derivatives.py  reusable componentwise eta differentiation
    equations.py    Torch residuals from the checked specification
    boundaries.py   exactly twelve errors
    losses.py       scales, positive weights and diagnostics
    reference.py    separately coded NumPy BVP and convergence checks
    sampling.py     interior, endpoint, monitor and final sets
    train.py        Adam and L-BFGS stages
    evaluate.py     original residuals, profiles and wall errors
    io.py           source/case hashes, checkpoints, result tables
    tests/          analytic, algebraic and manufactured tests
```

Do not introduce a second trainable “physics network.” The physics part is the fixed equation calculation. Physical parameter values are not in the optimizer parameter list.

## 4. Network and derivatives

The starting model is a pointwise multilayer perceptron: one input, four hidden tanh layers of width 64, five linear outputs. Use float64, Xavier-style weight initialization, zero biases and recorded seeds. This is a testable starting choice, not an optimal architecture claim.

Normalize the coordinate inside the model with x=2*eta/L-1. Differentiate the complete model with respect to eta, so the factor 2/L is included once by autograd. If differentiating with respect to x instead, the kth eta derivative needs (2/L)^k. Test up to order four; do not apply both methods at once.

Extract columns separately as N by 1 tensors. Compute f1, f2, f3, f4 and two derivatives for each other field. Keep the graph through the highest derivative so the final loss can differentiate with respect to NN weights. This nested differentiation is the key change from our old first-order Lorenz PINN.

Do not sum all five outputs before asking for their derivatives. A vector-Jacobian product with all ones over every output sums their contributions. Summing samples for a single column gives the desired pointwise derivatives only when sample rows are independent.

Avoid ReLU, dropout and training-mode BatchNorm for this direct high-order baseline. Do not constrain all outputs between zero and one. Do not multiply every output by a factor vanishing at L: f itself is not prescribed zero there.

## 5. Boundary enforcement and loss

Use all twelve source boundary errors in [equations.md](equations.md). At every step, run the model at eta=0 and eta=L, differentiate the required fields there, and calculate B1–B12. Square each error and include it in the same loss used for the interior equations.

This is soft enforcement. We reduce the error by updating NN weights; we do not replace an output with its required value. Each final error must be checked individually.

For example, B3=g(0)+ell*f2(0). There is no fixed label for g(0); the given relation itself supplies the target zero. B6=chi(0)-1 directly checks a known endpoint value. These two types of constraints both work in the loss.

Define fixed positive equation scales s_i and boundary scales t_j. Let r_i=R_i/s_i and b_j=B_j/t_j. The full loss is:

```text
J = sum over the 5 equations of lambda_i * mean(r_i**2)
    + sum over the 12 boundaries of mu_j * b_j**2
```

**Read:** square each scaled equation error and average over the interior positions. Multiply each equation average by its weight. Add the twelve weighted squared boundary errors.

All weights start positive. Begin with weights of one after documenting characteristic scales; this does not guarantee the gradients are balanced. Scales must come from declared field/derivative magnitudes and equation terms, not a denominator equal to the current residual. An absolute loss threshold changes meaning when a scale changes, so save both the original and scaled errors.

If there is a 2000-sized wall coefficient, a convenient formulation may scale that equation by a fixed characteristic magnitude. It must still report the original equation error. Dividing an equation by a tiny supplied parameter can also amplify numerical gradients. Record the scaling choice rather than hiding it.

No initial-value term, temporal causal weights, Lorenz invariants or solution-data term belongs in this baseline. Do not train unconstrained loss weights that can reduce the objective by shrinking to zero.

## 6. Classical reference simulation

Before quantitative accuracy claims, solve the same BVP conventionally. Use the twelve first-order states and the highest-derivative matrix documented in [equations.md](equations.md). Solve the matrix system, do not form its inverse. Check its scaled conditioning and effective theta-second-derivative coefficient along the state.

Use SciPy solve_bvp, or MATLAB bvp4c if requested. The regular first-order representation requires nonzero highest-derivative coefficients; the direct PINN's residual form does not remove a mathematical degeneracy when they vanish.

Reference steps:

1. Reject null or invalid parameters and ensure valid eta+kappa and eta+b denominators.
2. Construct a nonuniform initial mesh with wall coverage. A trial L=10 and 200 initial nodes are software starting choices, not certified far-field values.
3. Use smooth initial guesses and consistent derivative-state guesses. Try more than one wall-temperature/flow guess.
4. Solve and record success status, final mesh, original BC errors and residual estimates.
5. Tighten tolerance, for example from 1e-5 to 1e-7, and compare wall quantities and profiles. These values are pilots, not guaranteed error bounds.
6. Increase L, for example 10, 15 and 20, extending the previous guess smoothly. Compare only the common interval and the wall quantities.
7. Increase mesh resolution independently of L. If solutions change materially, the reference is not converged yet.
8. Keep all converged branches found and name them. A converged solver does not prove uniqueness or temporal stability.

There is an optional structural cross-check: solve the first four fields as a ten-state core, then solve chi as a linear BVP with those fixed curves, using the printed Eq. 18. Compare this with the monolithic twelve-state reference. Errors from interpolation and core residuals must also be checked. This hierarchy is not a reason to leave chi out of the PINN's full loss.

Do not differentiate a piecewise-cubic interpolant for f four times and call that a reference f4. Use the appropriate derivative-state interpolants and check compatibility. Do not obtain every derivative from the RHS and then claim an independent residual check by substituting it back into the same RHS.

If the reference fails, diagnose guesses, transcription, singularity, branch and domain. Failure is not a proof of nonexistence. A PINN could still be explored, but reference-based accuracy then remains unverified. No reference values enter the baseline training loss.

## 7. Point sets and near-wall coverage

Maintain four separate sets:

- Interior training coordinates.
- Endpoint coordinates, always checked separately.
- Fixed monitoring points used to select a model and diagnose fitting.
- Fresh final evaluation coordinates after design choices stop.

For the first pilot, use about 1024 interior points. Half can cover the full interval using a seeded stratified uniform design; half can cover a wall subinterval chosen from min(kappa,b), while still preserving outer coverage. For kappa=b=0.1, a trial wall region of a few multiples of 0.1 is reasonable to test, not a claimed layer thickness. The reference and residual plots should guide changes.

An unweighted mean over deliberately wall-biased points changes the emphasis of the objective. Say so. If the objective should approximate a uniform-domain integral, use appropriate quadrature/importance weights. Do not call these objectives identical.

Use at least a dense monitoring grid and a fresh random final set, with extra checks between training points near the wall. Endpoint value agreement cannot show that an entire curve is correct.

## 8. Testing before any long run

Tests should run without the missing physical values where possible, using manufactured fixtures clearly marked as synthetic test coefficients. Such fixtures are not Case 1 and cannot silently become real-run defaults.

Required tests:

1. Parameter schema: exactly 22 physical coefficients, missing/null rejection, finite values and documented ranges.
2. Source transcription: corrected R14 convective and unsteady brackets, all signs, printed R18 terms and all twelve source BCs.
3. Analytic derivatives: exponentials and polynomial functions through fourth order; handle mathematically constant/zero derivatives deliberately. Do not mask a graph break by treating an unused input as zero automatically.
4. Coordinate scaling: a known function evaluated with internal normalization has the same eta derivatives as its analytic formula.
5. Batch independence: adding another coordinate does not change previous rows' predictions.
6. Shapes: every residual is N by 1 before stacking; prevent N by N broadcasting.
7. Algebraic rearrangement: R(H)-R(0)=M*H at finite artificial states; independent NumPy and Torch evaluations agree.
8. Boundary examples: value, derivative, slip, flux and nonlinear radiation tests, including nonzero slopes that cancel.
9. Gradient connectivity: representative loss components produce finite gradients on NN parameters. Never NumPy-convert or detach values that enter training loss.
10. Manufactured solution: smooth known curves, matching source forcing and boundary targets, recovered to declared test tolerances. This tests implementation, not original physical assumptions.
11. L-BFGS closure: repeated evaluation at unchanged weights/points returns the same objective, and reevaluation creates a new graph.
12. Checkpoint round-trip: predictions and optimizer/RNG state are restored under matching configuration.
13. Short smoke run: finite updates, component logs and intentional early-stop behavior. No physical result claim.

## 9. Adam training flow

For each seed:

1. Create the model, complete case, fixed scales/weights and point sets. Save all settings and source hashes.
2. Create Adam with a trial learning rate 1e-3; no weight decay for the baseline. A declared decay towards 1e-4 and a budget around 20000 steps are pilots to test, not accuracy guarantees.
3. At every step, clear previous NN gradients.
4. Make fresh eta coordinate leaves requiring gradients.
5. Predict fields at interior points; compute the needed derivatives once and reuse them.
6. Calculate R14–R18 from the confirmed source specification.
7. Predict endpoints and required derivatives; calculate all twelve boundary errors.
8. Scale, square and combine the loss components.
9. Check residuals and loss for NaN/infinity before backpropagation.
10. Backpropagate through the derivatives and equation arithmetic to the network weights.
11. Check gradients and record the unmodified norm. Then update weights.
12. At a fixed monitoring interval, evaluate all original errors on independent points and save an eligible checkpoint.
13. Continue until the declared budget or transition condition is reached. A flat loss is a reason to diagnose scaling or graph connection, not assume convergence.

History stores detached numbers, not live graphs. Use weighted deterministic chunks if full-batch fourth-order autograd runs out of memory. Each chunk contributes its residual sum divided by the total training count; add boundaries once. Do not give a tiny final chunk the same aggregate weight as a full chunk accidentally.

If any value is nonfinite, stop the update and record the first failing term. Do not hide it with nan_to_num, arbitrary epsilons or zero residual substitution. Gradient clipping in Adam is a separately recorded option if justified; never clip physical outputs to conceal an invalid solution.

## 10. L-BFGS refinement flow

Save the Adam checkpoint and freeze parameters, domain, points, scales and weights. Create a fresh L-BFGS optimizer with a deterministic closure.

The closure clears NN gradients, creates fresh coordinates, recomputes predictions/derivatives/residuals/boundaries/loss, checks finite values, backpropagates and returns the loss. The optimizer can call it multiple times for a single step. Count function evaluations and accepted progress separately from Adam steps.

A trial strong-Wolfe line search and a history of about 50 vectors can be tested. Its memory cost and installed-library settings must be recorded. Keep all parameters on one device in PyTorch's standard implementation.

Do not resample or change weights/scheduler inside the closure. Do not give line search clipped gradients that no longer match the objective. If the objective changes, reset the optimizer history. Recompute post-step diagnostics; the returned step loss need not describe the final weights.

If refinement fails, preserve the Adam result and failure reason. Compare it honestly instead of automatically declaring the hybrid run better.

## 11. Optional improvements, in a controlled order

1. Fix equations, scaling, graph or sampling mistakes first. Larger models do not fix wrong arithmetic.
2. Compare fixed points with periodic uniform resampling during Adam; freeze the selected set for refinement.
3. Compare custom residual enrichment: score scaled errors on a candidate pool, add difficult coordinates, retain uniform coverage. Include scoring time. Do not call it RAD/RAR-D without following their checked definitions.
4. Compare fixed and gradient-balanced positive weights during Adam. Bound extreme updates, stop gradient through weight estimates, freeze for refinement, retain original errors.
5. Compare five-output direct and twelve-output first-order formulations using the same soft BCs. The first-order method adds seven derivative helpers and seven agreement errors. Naming a helper f2 does not make it the physical second derivative. Check original equations/BCs with the physical-head derivatives.
6. Compare a proven simple exact endpoint construction, such as chi=1-x+x(1-x)*N_chi(x) with x=eta/L, separately from other changes. This enforces two chi values exactly, not all twelve coupled nonlinear BCs.
7. Only after a verified fixed-parameter baseline, consider a parametric input vector [eta,p1,...] for many cases. Every parameter sample must be admissible, boundaries apply to every sample, and held-out parameter cases need independent reference checks. This is not the first implementation.

No causal time-window rollout, multi-expert architecture or mixed precision is an automatic requirement for this small eta BVP. Fourth-order accuracy and measured cost decide whether to investigate them.

## 12. Final evaluation and results

The trained output is a function on the tested interval. Input an eta and obtain five predictions; differentiate when needing velocity f-prime and wall quantities. No accuracy claim outside the tested domain or for unseen physical parameter sets.

Save:

- eta and all five PINN curves, reference curves and absolute errors;
- f-prime velocity-related curve and derivative error;
- RMS/max original and scaled residuals for each equation;
- every signed original and scaled boundary error;
- required wall derivatives and approved derived quantities;
- independent dense-grid and random-point results;
- L/mesh/point-count sensitivity;
- all seed outcomes, time, evaluations, memory and stopping reasons;
- weights, config, source/case hashes and checkpoints.

Use absolute or scale-normalized errors when reference fields/quantities approach zero. Use quadrature weights for nonuniform-grid norms. Reference discrepancies should be interpreted against estimated numerical uncertainty of the reference itself.

Eq. 23's physical output formulas are a separate checked post-processing task. Do not copy a possibly inconsistent wall-shear formula into the loss, and do not claim wall-transfer accuracy merely because the profiles look smooth. Ask which derived quantities the supervisor needs before those acceptance targets are fixed.

Candidate pilot targets could include scaled boundary max <=1e-4 and field L2 error <=1e-3 where a reliable reference exists. These are proposed numerical goals, not supplied requirements or promises. Choose final tolerances before interpreting experiments and keep reference error appreciably smaller than the reported PINN error. All acceptance thresholds must be tied to recorded scales.

## 13. Controlled research comparisons

| ID | What changes | What is held comparable |
|---|---|---|
| B0 | Five outputs, fixed scaled soft BCs, wall/full-domain points, Adam then L-BFGS | Confirmed equations, case, precision, evaluation sets, seeds |
| B1 | Adam only | Same declared wall-time/evaluation budget, not equal method names |
| B2 | Uniform resampling during Adam | Architecture, scales, budget and final grid |
| B3 | Residual enrichment | Coverage, objective definition, budget including scoring |
| B4 | Gradient-based weights | Same sampling/formulation; freeze weights for refinement |
| B5 | Twelve-output first-order method | Original BCs, precision and declared matching budget/parameter-count policy |
| B6 | Selected exact simple endpoint values | Other architecture/training choices unchanged |

First compare one change at a time. Report median/spread and failures over at least three pilot seeds, with more for serious quantitative comparisons. Exact reproducibility depends on hardware/libraries; record deterministic settings rather than promise bit-for-bit identity across machines.

Our research question is which change improves reliable solution accuracy and cost on the tested cases, especially the inverse-power wall region. It is not a predeclared claim that the PINN outperforms solve_bvp. Classical collocation may remain faster for a single 1D case.

## 14. Algorithm

```text
INPUT: confirmed equations and BCs; complete parameter case;
       proposed network, point, scale, optimizer and acceptance settings
OUTPUT: verified solution record or explicit failed/blocked run

if any physical coefficient is null:
    return BLOCKED_MISSING_PARAMETERS
validate source specification, ranges and denominators
run analytic, algebraic, boundary, gradient and checkpoint tests
if tests fail:
    return FAILED_IMPLEMENTATION_TESTS

build classical reference; vary guesses, mesh, tolerance and L
if reference unavailable:
    record UNVERIFIED_REFERENCE (do not manufacture target curves)

for seed in the declared seed list:
    initialize one five-output network and point sets
    train Adam:
        predict interiors -> derivatives -> original equation errors
        predict endpoints -> derivatives -> all twelve BC errors
        scale/square/weight -> total loss
        check finite -> backprop -> check gradients -> update NN weights
        monitor independently -> checkpoint -> transition/budget rule
    freeze objective and refine with fresh deterministic L-BFGS
    keep a valid checkpoint even if refinement fails
    evaluate original equations, all BCs, fields and f-prime on fresh points
    compare approved wall quantities and same-branch reference if available
    check acceptance and save explicit status/settings/errors

repeat domain and sample sensitivity; report all failures
compare controlled method variants; claim only measured improvements
```

No physical computation is executed by this document. The record-check tool in this repository checks documentation/config integrity, not BVP convergence or NN correctness.

## 15. Failure handling and implementation gates

| Failure | Action |
|---|---|
| Missing coefficients | Stop; obtain supervisor values; no auto-defaults |
| Invalid denominator or thermal derivative coefficient | Diagnose specified case/normalization; no epsilon patches |
| NaN, infinite derivative or disconnected graph | Identify first failing expression and restore known-good state |
| Small combined loss, one large boundary error | Inspect separate components, scales and gradients |
| Good endpoints, poor interior/fluxes | Improve independent checking and wall resolution |
| Large L sensitivity | Re-solve on longer domain and verify coverage |
| Different seeds/reference branches | Keep branches separate; no averaging as one solution |
| L-BFGS instability | Check frozen objective, closure gradients and line search |
| Good helpers, bad physical derivatives | Check original derivatives and compatibility gradients |
| Reference fails | Diagnose; do not claim no solution or fake accuracy |

Implementation should proceed in stages: parameter/schema and source fixtures; tested original residuals; reference simulation; direct PINN smoke test; full fixed-case runs; controlled improvements. User approval for this strategy does not mean publishing a remote repository, committing or starting expensive runs without the required input.
